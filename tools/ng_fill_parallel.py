#!/usr/bin/env python
"""Seed-striped PARALLEL ARC-GEN corpus filler — many cores on ONE task.

WHY THIS EXISTS
---------------
`ng_fill_hang_safe.py` is one-worker-per-TASK: a single slow generator (task157
~0.6 pairs/s) takes ~40h for its 200k tail because the seed stream is consumed
sequentially. But every pair is independently reproducible: pair `s` is fully
determined by `random.seed(SEED_OFFSET + s)` (see corpus.py docstring), with NO
dependence on prior pairs. So the remaining seed range can be split into K
disjoint stripes, generated concurrently on K cores, then MERGED back in seed
order. The merged cache is byte-identical to what a sequential fill produces —
consumers (`corpus.build(t, N)` = "first N valid pairs in seed order") see no
difference.

CORRECTNESS INVARIANT
---------------------
Sequential fill appends valid pairs in increasing seed index, skipping
oversize (>30x30), exception, and known-hang seeds. The striped fill:
  1. generates ALL valid pairs over [start, end) across stripes (same skip rules),
  2. sorts them by seed index and concatenates AFTER the existing base pairs
     (whose seed indices are all < start = base n_raw),
  3. takes the first `target` pairs and sets n_raw = (seed index of the
     target-th pair) + 1 — exactly sequential's stop condition.
This reproduces the identical pair SEQUENCE and n_raw. Proven by --selftest
(byte-equality vs the canonical cache on a fast task, from-scratch and from a
truncated base) before any real run.

HANG SAFETY
-----------
Each stripe is a watchdog+subprocess pair (same mechanism as ng_fill_hang_safe):
the worker heartbeats the current seed before each gen(); if the frontier stalls
past --hang-timeout (and it is not a long checkpoint write), the watchdog kills
the worker, records the seed in the shared skip-set, and resumes the stripe from
its checkpoint. Known-hang seeds produce no pair in BOTH sequential and striped
fills, so skipping them preserves the invariant.

USAGE
-----
  python tools/ng_fill_parallel.py --selftest              # prove correctness, exit
  python tools/ng_fill_parallel.py --task 157 --stripes 10 # fill task157 to 200k
  python tools/ng_fill_parallel.py --task 157 --status     # stripe progress, exit
  python tools/ng_fill_parallel.py --stop                  # signal manager to halt
Resumable + idempotent: stripe partials checkpoint; a re-run continues them. The
canonical task{t}.npz is only written by the final merge (atomic), never mid-fill.
"""
import argparse, json, os, re, subprocess, sys, tempfile, threading, time, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ng_harness import corpus, registry, encode

HBDIR = pathlib.Path(tempfile.gettempdir()) / "ng_fill_par_hb"
_skip_lock = threading.Lock()


# ----------------------------------------------------------------- paths/state
def _state_dir():
    d = corpus.CACHE_DIR / "_fill_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _skip_path(t):
    return _state_dir() / f"skip_{t}.json"


def load_skip(t):
    try:
        return set(int(x) for x in json.load(open(_skip_path(t))))
    except Exception:
        return set()


def add_skip(t, seed):
    with _skip_lock:
        s = load_skip(t)
        s.add(int(seed))
        tmp = _skip_path(t).with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(sorted(s), f)
        os.replace(tmp, _skip_path(t))
        return s


def _stripe_path(t, a, b):
    return corpus.CACHE_DIR / f"task{t:03d}.stripe_{a}_{b}.npz"


def _stripe_hb(t, a):
    HBDIR.mkdir(parents=True, exist_ok=True)
    return HBDIR / f"hb_{t}_{a}.json"


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [par] {msg}"
    print(line, flush=True)
    try:
        with open(_state_dir() / "fill_parallel.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ----------------------------------------------------------------- stripe i/o
def _save_stripe(path, seeds, in_idx, in_sh, out_idx, out_sh, next_seed, a, b):
    d = {"seeds": np.asarray(seeds, np.int64),
         "in_idx": np.asarray(in_idx, np.uint8), "in_shapes": np.asarray(in_sh, np.uint16),
         "out_idx": np.asarray(out_idx, np.uint8), "out_shapes": np.asarray(out_sh, np.uint16),
         "next_seed": np.int64(next_seed), "seed_a": np.int64(a), "seed_b": np.int64(b)}
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **d)
    tmp.replace(path)


def _load_stripe(path):
    if not path.exists():
        return None
    try:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    except Exception:
        return None


# ----------------------------------------------------------------- fake gen (selftest injection only)
def _fake_gen_factory(hang_seed, sleep_s, oversize_seeds=frozenset()):
    import random as _r

    def gen():
        _r.randint(0, 10**9)  # consume seeded RNG so each seed is distinct+deterministic
        cur = _CUR_SEED[0]
        if hang_seed is not None and cur == hang_seed:
            time.sleep(sleep_s)
        if cur in oversize_seeds:
            row = [0] * 31
            big = [row[:] for _ in range(31)]  # 31x31 -> grid_to_index rejects (oversize)
            return {"input": big, "output": big}
        v = cur % 9
        return {"input": [[v]], "output": [[(v + 1) % 9]]}
    return gen


_CUR_SEED = [None]  # set by run_stripe before each gen() when in fake mode


# ----------------------------------------------------------------- stripe worker (subprocess)
def run_stripe(t, a, b, ckpt):
    import random
    fake_hang = os.environ.get("NG_FAKE_HANG_SEED")
    fake_oversize = os.environ.get("NG_FAKE_OVERSIZE")
    if fake_hang is not None or fake_oversize is not None:
        ov = frozenset(int(x) for x in fake_oversize.split(",")) if fake_oversize else frozenset()
        hs = int(fake_hang) if fake_hang is not None else None
        gen = _fake_gen_factory(hs, float(os.environ.get("NG_FAKE_SLEEP", "30")), ov)
        err = None
    else:
        gen, err = registry.generator(t)
    if err:
        print(f"UNAVAILABLE {err}", flush=True)
        return 2
    path = _stripe_path(t, a, b)
    d = _load_stripe(path)
    seeds = list(d["seeds"]) if d is not None else []
    in_idx = list(d["in_idx"]) if d is not None else []
    in_sh = list(d["in_shapes"]) if d is not None else []
    out_idx = list(d["out_idx"]) if d is not None else []
    out_sh = list(d["out_shapes"]) if d is not None else []
    n = int(d["next_seed"]) if d is not None else a
    skip = load_skip(t)
    hbp = _stripe_hb(t, a)
    last = len(seeds)
    last_t = time.time()
    while n < b:
        if n in skip:
            n += 1
            continue
        try:
            with open(hbp, "w") as f:
                json.dump({"n_raw": n, "valid": len(seeds), "t": time.time()}, f)
        except Exception:
            pass
        random.seed(corpus.SEED_OFFSET + n)
        _CUR_SEED[0] = n
        cur = n
        n += 1
        try:
            ex = gen()
        except Exception:
            continue
        if not isinstance(ex, dict) or "input" not in ex or "output" not in ex:
            continue
        ii, ish = encode.grid_to_index(ex["input"])
        oi, osh = encode.grid_to_index(ex["output"])
        if ii is None or oi is None:
            continue
        seeds.append(cur)
        in_idx.append(ii); in_sh.append(ish); out_idx.append(oi); out_sh.append(osh)
        if len(seeds) - last >= ckpt or (len(seeds) > last and time.time() - last_t >= 300):
            _save_stripe(path, seeds, in_idx, in_sh, out_idx, out_sh, n, a, b)
            last = len(seeds); last_t = time.time()
    _save_stripe(path, seeds, in_idx, in_sh, out_idx, out_sh, b, a, b)  # next_seed=b => complete
    print(f"STRIPE_DONE {len(seeds)}", flush=True)
    return 0


# ----------------------------------------------------------------- watchdog (one stripe)
def fill_stripe(t, a, b, hang_timeout, ckpt, stop_evt, child_env=None):
    py = sys.executable
    self = os.path.abspath(__file__)
    while not stop_evt.is_set():
        d = _load_stripe(_stripe_path(t, a, b))
        if d is not None and int(d["next_seed"]) >= b:
            return  # complete
        try:
            os.remove(_stripe_hb(t, a))
        except OSError:
            pass
        env = dict(os.environ)
        if child_env:
            env.update(child_env)
        w = subprocess.Popen(
            [py, self, "--mode", "stripe", "--task", str(t), "--seed-a", str(a),
             "--seed-b", str(b), "--ckpt", str(ckpt), "--cache-dir", str(corpus.CACHE_DIR)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        last_seed = None
        last_change = time.time()
        while not stop_evt.is_set():
            if w.poll() is not None:
                break
            try:
                hb = json.load(open(_stripe_hb(t, a)))
            except Exception:
                hb = None
            now = time.time()
            if hb is not None:
                if hb["n_raw"] != last_seed:
                    last_seed = hb["n_raw"]; last_change = now
                elif now - last_change > hang_timeout:
                    tmp_p = _stripe_path(t, a, b).with_name(_stripe_path(t, a, b).stem + ".tmp.npz")
                    try:
                        saving = tmp_p.exists() and (now - tmp_p.stat().st_mtime) < hang_timeout
                    except OSError:
                        saving = False
                    if saving:
                        last_change = now
                    else:
                        w.kill()
                        add_skip(t, last_seed)
                        log(f"task{t:03d} stripe[{a},{b}): HANG seed {last_seed} -> skipped")
                        break
            time.sleep(2)
        if stop_evt.is_set():
            w.kill()
            return
        w.wait()


# ----------------------------------------------------------------- merge
def merge_into_canonical(t, target):
    """Concatenate base pairs + seed-sorted stripe pairs, take first `target`,
    set n_raw to sequential's stop point. Returns {ok, valid, n_raw}."""
    base = corpus._load(t)
    if base is not None:
        b_in = list(base["in_idx"]); b_ish = list(base["in_shapes"])
        b_oi = list(base["out_idx"]); b_osh = list(base["out_shapes"])
        b_nraw = int(base["n_raw"])
    else:
        b_in = b_ish = b_oi = b_osh = []
        b_nraw = 0
    base_valid = len(b_in)

    new = []
    for p in sorted(corpus.CACHE_DIR.glob(f"task{t:03d}.stripe_*.npz")):
        d = _load_stripe(p)
        if not d:
            continue
        for i in range(len(d["seeds"])):
            new.append((int(d["seeds"][i]), d["in_idx"][i], d["in_shapes"][i],
                        d["out_idx"][i], d["out_shapes"][i]))
    new.sort(key=lambda x: x[0])
    if new and new[0][0] < b_nraw:
        raise RuntimeError(f"stripe overlaps base: min new seed {new[0][0]} < base n_raw {b_nraw}")
    seen = set(); uniq = []
    for rec in new:
        if rec[0] in seen:
            continue
        seen.add(rec[0]); uniq.append(rec)
    new = uniq

    total = base_valid + len(new)
    if total < target:
        return {"ok": False, "valid": total, "n_raw": (new[-1][0] + 1 if new else b_nraw)}

    need_new = target - base_valid
    sel = new[:need_new]
    fin_in = b_in + [r[1] for r in sel]
    fin_ish = b_ish + [r[2] for r in sel]
    fin_oi = b_oi + [r[3] for r in sel]
    fin_osh = b_osh + [r[4] for r in sel]
    n_raw_final = (sel[-1][0] + 1) if sel else b_nraw
    d = {"in_idx": np.asarray(fin_in, np.uint8), "in_shapes": np.asarray(fin_ish, np.uint16),
         "out_idx": np.asarray(fin_oi, np.uint8), "out_shapes": np.asarray(fin_osh, np.uint16),
         "n_raw": np.int64(n_raw_final), "seed_offset": np.int64(corpus.SEED_OFFSET)}
    corpus._save(t, d)
    return {"ok": True, "valid": len(fin_in), "n_raw": int(n_raw_final)}


def _cleanup_stripes(t):
    for p in corpus.CACHE_DIR.glob(f"task{t:03d}.stripe_*.npz"):
        try:
            os.remove(p)
        except OSError:
            pass
    for p in corpus.CACHE_DIR.glob(f"task{t:03d}.stripe_*.tmp.npz"):
        try:
            os.remove(p)
        except OSError:
            pass


# ----------------------------------------------------------------- helpers
def _partition(start, end, k):
    k = max(1, int(k))
    n = end - start
    if n <= 0:
        return []
    step = (n + k - 1) // k
    out = []
    a = start
    while a < end:
        out.append((a, min(a + step, end)))
        a += step
    return out


def _next_free_seed(t, n0):
    mx = n0
    for p in corpus.CACHE_DIR.glob(f"task{t:03d}.stripe_*.npz"):
        m = re.search(r"stripe_(\d+)_(\d+)\.npz$", p.name)
        if m:
            mx = max(mx, int(m.group(2)))
    return mx


def _stripe_pairs_on_disk(t, bounds):
    tot = 0
    done = 0
    for (a, b) in bounds:
        d = _load_stripe(_stripe_path(t, a, b))
        if d is not None:
            tot += len(d["seeds"])
            if int(d["next_seed"]) >= b:
                done += 1
    return done, tot


# ----------------------------------------------------------------- manager
def manager(t, target, stripes, hang_timeout, ckpt, margin, child_env=None):
    base = corpus._load(t)
    valid0 = len(base["in_idx"]) if base is not None else 0
    if valid0 >= target:
        log(f"task{t:03d}: already >= target ({valid0}); nothing to do")
        return
    stop_evt = threading.Event()

    def _stop_watch():
        while not stop_evt.is_set():
            if (_state_dir() / "STOP_PARALLEL").exists():
                log("STOP_PARALLEL seen -> stopping")
                stop_evt.set()
                return
            time.sleep(3)
    threading.Thread(target=_stop_watch, daemon=True).start()

    round_no = 0
    while not stop_evt.is_set():
        base = corpus._load(t)
        valid0 = len(base["in_idx"]) if base is not None else 0
        n0 = int(base["n_raw"]) if base is not None else 0
        if valid0 >= target:
            break
        need = target - valid0
        span = need + margin + len(load_skip(t))
        start = _next_free_seed(t, n0)
        end = start + span
        bounds = _partition(start, end, stripes)
        log(f"task{t:03d} round {round_no}: base valid={valid0} n_raw={n0}; "
            f"seeds [{start},{end}) over {len(bounds)} stripes (need {need} + {margin} margin)")
        threads = []
        for (a, b) in bounds:
            th = threading.Thread(target=fill_stripe,
                                  args=(t, a, b, hang_timeout, ckpt, stop_evt, child_env),
                                  daemon=True)
            th.start()
            threads.append(th)
        while any(th.is_alive() for th in threads):
            time.sleep(15)
            done, tot = _stripe_pairs_on_disk(t, bounds)
            log(f"task{t:03d} round {round_no}: stripes {done}/{len(bounds)} complete, "
                f"+{tot} new pairs (~{valid0 + tot}/{target})")
        if stop_evt.is_set():
            log(f"task{t:03d}: stopped; stripe partials preserved for resume")
            return
        res = merge_into_canonical(t, target)
        log(f"task{t:03d} round {round_no}: merge -> valid={res['valid']} "
            f"n_raw={res['n_raw']} ok={res['ok']}")
        if res["ok"]:
            _cleanup_stripes(t)
            break
        round_no += 1
        if round_no > 20:
            log(f"task{t:03d}: too many extend rounds; aborting at valid={res['valid']}")
            return
    fin = corpus._load(t)
    log(f"task{t:03d}: DONE valid={len(fin['in_idx'])} n_raw={int(fin['n_raw'])}")


# ----------------------------------------------------------------- selftest
def _load_npz_path(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _arrays_equal(d1, d2, n):
    for k in ("in_idx", "in_shapes", "out_idx", "out_shapes"):
        if not np.array_equal(np.asarray(d1[k])[:n], np.asarray(d2[k])[:n]):
            return False, k
    return True, None


def selftest():
    import shutil
    real_dir = corpus.CACHE_DIR
    print(f"== selftest (canonical cache: {real_dir}) ==", flush=True)

    # pick a fast, full task with an EMPTY skip-set
    cand = [1, 2, 3, 4, 5, 8, 11, 20, 30, 50]
    best = None
    for t in cand:
        cpath = real_dir / f"task{t:03d}.npz"
        if not cpath.exists():
            continue
        if (real_dir / "_fill_state" / f"skip_{t}.json").exists():
            continue
        gen, err = registry.generator(t)
        if err:
            continue
        import random
        t0 = time.time()
        ok = True
        for s in range(2000, 2010):
            random.seed(corpus.SEED_OFFSET + s)
            try:
                ex = gen()
                encode.grid_to_index(ex["input"]); encode.grid_to_index(ex["output"])
            except Exception:
                ok = False; break
        dt = (time.time() - t0) / 10
        if ok and (best is None or dt < best[1]):
            best = (t, dt)
    assert best is not None, "no fast full task found for selftest"
    F, per = best
    print(f"  chose task{F:03d} ({per*1000:.1f} ms/pair)", flush=True)
    canon = _load_npz_path(real_dir / f"task{F:03d}.npz")
    canon_valid = len(canon["in_idx"])
    N = min(1200, canon_valid)
    print(f"  canonical task{F:03d}: {canon_valid} pairs; testing first {N}", flush=True)

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ng_par_selftest_"))
    results = []
    try:
        # ---- TEST A: from-scratch striped fill == canonical[:N]
        corpus.CACHE_DIR = tmp / "A"
        corpus.CACHE_DIR.mkdir(parents=True)
        manager(F, N, stripes=4, hang_timeout=60, ckpt=250, margin=300,
                child_env={"NEUROGOLF_ARGEN_CACHE": str(corpus.CACHE_DIR)})
        got = corpus._load(F)
        eq, badk = _arrays_equal(got, canon, N)
        nraw_ok = int(got["n_raw"]) == int(canon["n_raw"]) if canon_valid == N else True
        a_ok = eq and len(got["in_idx"]) >= N and nraw_ok
        results.append(("A from-scratch byte-equality", a_ok,
                        f"valid={len(got['in_idx'])} n_raw={int(got['n_raw'])} "
                        f"(canon n_raw={int(canon['n_raw'])}) badkey={badk}"))

        # ---- TEST B: truncated base + striped extend == canonical[:N]
        corpus.CACHE_DIR = tmp / "B"
        corpus.CACHE_DIR.mkdir(parents=True)
        half = N // 2
        # build a truncated base = canonical[:half] with its correct n_raw.
        # n_raw for the prefix: the seed index after the half-th valid pair. For a
        # yield-1 fast task that's `half`, but to be exact we recompute by streaming.
        import random
        seen_valid = 0
        nraw_half = 0
        gen, _ = registry.generator(F)
        s = 0
        while seen_valid < half:
            random.seed(corpus.SEED_OFFSET + s)
            s += 1
            try:
                ex = gen()
            except Exception:
                continue
            if not isinstance(ex, dict) or "input" not in ex or "output" not in ex:
                continue
            ii, ish = encode.grid_to_index(ex["input"])
            oi, osh = encode.grid_to_index(ex["output"])
            if ii is None or oi is None:
                continue
            seen_valid += 1
            nraw_half = s
        base_d = {"in_idx": np.asarray(canon["in_idx"][:half], np.uint8),
                  "in_shapes": np.asarray(canon["in_shapes"][:half], np.uint16),
                  "out_idx": np.asarray(canon["out_idx"][:half], np.uint8),
                  "out_shapes": np.asarray(canon["out_shapes"][:half], np.uint16),
                  "n_raw": np.int64(nraw_half), "seed_offset": np.int64(corpus.SEED_OFFSET)}
        corpus._save(F, base_d)
        manager(F, N, stripes=4, hang_timeout=60, ckpt=250, margin=300,
                child_env={"NEUROGOLF_ARGEN_CACHE": str(corpus.CACHE_DIR)})
        got = corpus._load(F)
        eq, badk = _arrays_equal(got, canon, N)
        nraw_ok = int(got["n_raw"]) == int(canon["n_raw"]) if canon_valid == N else True
        b_ok = eq and len(got["in_idx"]) >= N and nraw_ok
        results.append(("B base+extend byte-equality", b_ok,
                        f"base[:{half}] n_raw={nraw_half} -> valid={len(got['in_idx'])} "
                        f"n_raw={int(got['n_raw'])} badkey={badk}"))

        # ---- TEST C: hang injection (watchdog kills + skips + resumes; pair omitted)
        corpus.CACHE_DIR = tmp / "C"
        corpus.CACHE_DIR.mkdir(parents=True)
        FT = 999  # synthetic task id (fake gen ignores it)
        a, b = 100, 130
        hang_seed = 115
        stop_evt = threading.Event()
        fenv = {"NG_FAKE_HANG_SEED": str(hang_seed), "NG_FAKE_SLEEP": "30",
                "NEUROGOLF_ARGEN_CACHE": str(corpus.CACHE_DIR)}
        t_start = time.time()
        fill_stripe(FT, a, b, hang_timeout=5, ckpt=5, stop_evt=stop_evt, child_env=fenv)
        elapsed = time.time() - t_start
        d = _load_stripe(_stripe_path(FT, a, b))
        got_seeds = sorted(int(x) for x in d["seeds"]) if d is not None else []
        expect = [s for s in range(a, b) if s != hang_seed]
        skipped = hang_seed in load_skip(FT)
        resumed = elapsed < 30  # if it had NOT killed the 30s sleep, it'd take >=30s
        c_ok = got_seeds == expect and skipped and resumed
        results.append(("C hang kill+skip+resume", c_ok,
                        f"seeds={len(got_seeds)} expect={len(expect)} "
                        f"hang_skipped={skipped} elapsed={elapsed:.1f}s(<30 ok)"))
    finally:
        corpus.CACHE_DIR = real_dir
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n== RESULTS ==", flush=True)
    allok = True
    for name, ok, detail in results:
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  --  {detail}", flush=True)
    print(f"\nSELFTEST: {'ALL PASS' if allok else 'FAILURE'}", flush=True)
    return 0 if allok else 1


# ----------------------------------------------------------------- status
def status(t, target):
    base = corpus._load(t)
    valid = len(base["in_idx"]) if base is not None else 0
    n0 = int(base["n_raw"]) if base is not None else 0
    print(f"task{t:03d}: canonical {valid}/{target} (n_raw={n0})")
    sp = sorted(corpus.CACHE_DIR.glob(f"task{t:03d}.stripe_*.npz"))
    if not sp:
        print("  no stripe partials on disk")
        return
    tot = 0
    for p in sp:
        d = _load_stripe(p)
        if not d:
            continue
        m = re.search(r"stripe_(\d+)_(\d+)\.npz$", p.name)
        a, b = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        cnt = len(d["seeds"]); nxt = int(d["next_seed"])
        tot += cnt
        pct = 100.0 * (nxt - a) / max(1, b - a)
        print(f"  stripe[{a},{b}): {cnt} pairs, frontier {nxt} ({pct:.0f}% of range)"
              + ("  COMPLETE" if nxt >= b else ""))
    print(f"  total stripe pairs: {tot}  (merged projection: {valid + tot})")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Seed-striped parallel ARC-GEN corpus filler")
    ap.add_argument("--mode", default="manager", choices=["manager", "stripe"])
    ap.add_argument("--task", type=int)
    ap.add_argument("--target", type=int, default=200000)
    ap.add_argument("--stripes", type=int, default=10)
    ap.add_argument("--hang-timeout", type=float, default=300.0)
    ap.add_argument("--ckpt", type=int, default=2000)
    ap.add_argument("--margin", type=int, default=3000)
    ap.add_argument("--cache-dir", type=str, help="override NEUROGOLF_ARGEN_CACHE")
    ap.add_argument("--seed-a", type=int)
    ap.add_argument("--seed-b", type=int)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    a = ap.parse_args()

    if a.cache_dir:
        corpus.CACHE_DIR = pathlib.Path(a.cache_dir)

    if a.selftest:
        sys.exit(selftest())
    if a.stop:
        (_state_dir() / "STOP_PARALLEL").write_text("stop")
        print("wrote STOP_PARALLEL; manager halts within ~3s")
        return
    if a.mode == "stripe":
        sys.exit(run_stripe(a.task, a.seed_a, a.seed_b, a.ckpt))
    if a.status:
        status(a.task, a.target)
        return
    assert a.task, "--task required"
    try:
        (_state_dir() / "STOP_PARALLEL").unlink()
    except OSError:
        pass
    manager(a.task, a.target, a.stripes, a.hang_timeout, a.ckpt, a.margin)


if __name__ == "__main__":
    main()
