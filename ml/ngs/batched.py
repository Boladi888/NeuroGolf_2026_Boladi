"""batched.py — the GPU-batched discrete argument search engine (for the long run over the hard bucket).

The cost-0 Einsum grammar (einsum_search.py) enumerates a *tiny* set of parameter-free equations. The
hard bucket needs the other half of the arg-inference lever: search a LARGE discrete space of small
operand tensors (low-rank Einsum factors, tiny kernels, selector bits) for the one that makes a cheap
graph exact. That space is millions of candidates — too many for a Python loop, perfectly suited to the
B70. This is the teammate's `gpu_batch_domain` mechanism, ported to torch-xpu:

  1. DiscreteDomain(cardinalities): a mixed-radix odometer over the argument tuple. total = prod(card).
     `decode(ids)` peels digits from a 1-D int64 id tensor -> [batch, n_vars] choice indices (pure tensor
     math, no per-candidate Python).
  2. batched_topk(domain, score_fn, chunk): stream arange(total) in chunks, decode each chunk, call
     score_fn(choices)->[chunk] residual (LOWER = better), keep a running global top-k. Shardable via
     (start, limit) for resume/multi-GPU.
  3. score_fn is task-specific (a batched exact-fit residual, e.g. ridge-fit + sign-mismatch). GPU
     proposes; the CPU/ONNX path + argen gate dispose (verify only the survivors).

Device: prefers the discovered B70 torch-xpu env (ml.ngs.backends). Falls back to CPU torch, then to a
pure-numpy path, so it imports and runs anywhere — only the *throughput* needs the GPU.

This module is the ENGINE (domain + sweep + device). Task-specific scorers/emitters (which template to
fit) plug in on top; a demo low-rank scorer is included and self-tested. Reach grows as templates land.
"""
import numpy as np


# ---- device -----------------------------------------------------------------------------
def get_device(prefer_xpu=True):
    """Return (torch_module_or_None, device_str). Prefers B70 xpu, then cpu torch, then (None,'numpy')."""
    try:
        import torch
    except Exception:
        return None, "numpy"
    if prefer_xpu and getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return torch, "xpu"
    return torch, "cpu"


# ---- mixed-radix discrete domain --------------------------------------------------------
class DiscreteDomain:
    """Odometer over an argument tuple with per-variable cardinalities. Candidate id in [0,total)
    decodes to a choice vector without any Python per-candidate loop."""

    def __init__(self, cardinalities):
        self.card = [int(c) for c in cardinalities]
        self.n = len(self.card)
        self.total = 1
        for c in self.card:
            self.total *= c
        # radix place-values (little-endian): place[i] = prod(card[:i])
        self.place = [1] * self.n
        for i in range(1, self.n):
            self.place[i] = self.place[i - 1] * self.card[i - 1]

    def decode(self, ids, xp=np):
        """ids: 1-D int64 tensor/array of candidate ids -> [len(ids), n] choice indices."""
        out = []
        for i in range(self.n):
            out.append((ids // self.place[i]) % self.card[i])
        return xp.stack(out, axis=-1)


def batched_topk(domain, score_fn, k=8, chunk=65536, start=0, limit=None, torch=None, device="numpy",
                 log=None):
    """Sweep the domain in chunks; keep the global top-k LOWEST-residual candidates.

    score_fn(choices) -> residual[chunk] (lower is better). `choices` is [chunk, n] on the active
    backend (torch tensor on xpu/cpu, else numpy). Returns list[(residual, id, choice_list)] sorted
    ascending. (start, limit) shard/resume the sweep."""
    end = domain.total if limit is None else min(domain.total, start + limit)
    best = []                                     # list of (residual_float, id_int, choice_tuple)
    pos = start
    use_torch = torch is not None
    while pos < end:
        hi = min(pos + chunk, end)
        if use_torch:
            ids = torch.arange(pos, hi, dtype=torch.int64, device=device)
            choices = domain.decode(ids, xp=torch)
            res = score_fn(choices)               # [chunk] torch tensor
            res_cpu = res.detach().to("cpu").numpy()
            ids_cpu = ids.detach().to("cpu").numpy()
            ch_cpu = choices.detach().to("cpu").numpy()
        else:
            ids = np.arange(pos, hi, dtype=np.int64)
            choices = domain.decode(ids, xp=np)
            res_cpu = np.asarray(score_fn(choices))
            ids_cpu, ch_cpu = ids, choices
        # merge this chunk's local top-k into the global list
        m = min(k, len(res_cpu))
        loc = np.argpartition(res_cpu, m - 1)[:m] if len(res_cpu) > m else np.arange(len(res_cpu))
        for j in loc:
            best.append((float(res_cpu[j]), int(ids_cpu[j]), tuple(int(x) for x in ch_cpu[j])))
        best.sort(key=lambda t: t[0])
        del best[k:]
        pos = hi
        if log:
            log(pos, end, best[0][0] if best else None)
    return best


# ---- demo scorer: low-rank exact-fit residual (the shape task108/task380 want) ----------
def demo_lowrank_scorer(target, basis, torch=None, device="numpy"):
    """Return a score_fn scoring candidate factor tuples by how well a low-rank product fits `target`.

    This is a *shape demo* of the teammate's ridge-fit + residual kernel: each candidate selects small
    integer factor entries; we form the reconstruction and score by summed squared error to `target`.
    A real template plugs its own reconstruction + exactness residual here. target/basis are numpy."""
    if torch is not None:
        tgt = torch.tensor(target, dtype=torch.float32, device=device)
        bas = torch.tensor(basis, dtype=torch.float32, device=device)

        def score_fn(choices):
            vals = (choices.to(torch.float32) - 1.0)          # map {0,1,2}->{-1,0,1}
            recon = vals @ bas                                # [chunk, len(target)]
            return ((recon - tgt) ** 2).sum(dim=1)
        return score_fn

    def score_fn(choices):
        vals = choices.astype(np.float32) - 1.0
        recon = vals @ basis
        return ((recon - target) ** 2).sum(axis=1)
    return score_fn


# ---- template registry + solver stage --------------------------------------------------
# A TEMPLATE is a task-specific (or family-wide) recipe with three parts:
#   propose(tid, X, Y) -> (domain, score_fn, meta) | None   # what discrete space to sweep + how to score
#   emit(choice, meta) -> onnx.ModelProto                    # turn the winning candidate into a graph
#   (verification is generic: the argen truth gate). Registering templates here makes them run in the
# longrun 'batched' stage. This is the plug point to grow the hard-bucket reach — e.g. a low-rank
# Einsum factor fit (task380=66 / task108=300 shape): domain = discretized factor entries, score_fn =
# batched ridge-fit residual (demo_lowrank_scorer is the kernel shape), emit = the Einsum + factor inits.
class TernaryKernel:
    """Broad GPU-swept family: output = grader-thresholded depthwise Conv(input, ternary 3x3 kernel).

    All 3^9 = 19,683 ternary 3x3 kernels are swept in one batched conv2d on the B70 (channels folded
    into the batch, kernels into the conv's out-channels, so ONE conv computes every kernel's result).
    The grader decodes the output with `>0`, so the emitted graph is a single depthwise Conv -> output
    (cost = the 90-element weight; the [1,10,30,30] output is exempt).

    MEASURED (2026-07-10, B70): the sweep is fast (~1.2 s/task for all 19,683 kernels) and the
    emit/score pipeline is correct, but this NAIVE form solved 0/100 tasks. Reason: the conv bleeds
    into the 0-padding, so `Conv>0` can't equal a top-left-embedded one-hot output. To give it reach it
    needs a boundary mask (e.g. `Conv * (ReduceSum_ch(input) > 0)`, confining output to input cells) or
    a growth-aware decode — at the cost of extra ops. The engine + interface are proven; this template
    is the honest v1 that shows a single ternary Conv is too blunt. See LONGRUN.md 'batched' notes."""
    name = "ternary_kernel"

    def propose(self, tid, X, Y):
        torch, device = get_device()
        n, C, H, W = X.shape
        dom = DiscreteDomain([3] * 9)
        Yb = (Y > 0.0)
        if torch is not None:
            import torch.nn.functional as Fn
            Xr = torch.tensor(X, dtype=torch.float32, device=device).reshape(n * C, 1, H, W)
            Yt = torch.tensor(Yb, device=device)                     # [n,C,H,W] bool

            def score_fn(choices):
                k = choices.shape[0]
                w = (choices.to(torch.float32) - 1.0).reshape(k, 1, 3, 3)   # ternary {-1,0,1}
                out = Fn.conv2d(Xr, w, padding=1)                    # [n*C, k, H, W]
                pred = (out.reshape(n, C, k, H, W).permute(2, 0, 1, 3, 4) > 0.0)   # [k,n,C,H,W]
                return (pred != Yt.unsqueeze(0)).reshape(k, -1).sum(dim=1)
            return dom, score_fn, None

        # numpy fallback (correct but slow — real sweeps want the B70)
        P = np.pad(X.reshape(n * C, 1, H, W), ((0, 0), (0, 0), (1, 1), (1, 1)))

        def score_fn(choices):
            k = choices.shape[0]
            w = choices.astype(np.float32) - 1.0                     # [k,9]
            out = np.zeros((n * C, k, H, W), np.float32)
            for p in range(9):
                dr, dc = divmod(p, 3)
                out += P[:, 0, dr:dr + H, dc:dc + W][:, None] * w[:, p][None, :, None, None]
            pred = (out.reshape(n, C, k, H, W).transpose(2, 0, 1, 3, 4) > 0.0)
            return (pred != Yb[None]).reshape(k, -1).sum(axis=1)
        return dom, score_fn, None

    def emit(self, choice, meta):
        import onnx
        from onnx import helper, numpy_helper
        from . import spec
        k = (np.array(choice, np.float32) - 1.0).reshape(3, 3)
        w = np.broadcast_to(k, (spec.CHANNELS, 1, 3, 3)).astype(np.float32).copy()
        weight = numpy_helper.from_array(w, "K")
        node = helper.make_node("Conv", [spec.IN_NAME, "K"], [spec.OUT_NAME],
                                group=spec.CHANNELS, pads=[1, 1, 1, 1], kernel_shape=[3, 3])
        g = helper.make_graph([node], "tk",
                              [helper.make_tensor_value_info(spec.IN_NAME, spec.FLOAT, list(spec.IN_SHAPE))],
                              [helper.make_tensor_value_info(spec.OUT_NAME, spec.FLOAT, list(spec.IN_SHAPE))],
                              [weight])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
        return m


TEMPLATES = [TernaryKernel()]   # objects with .name, .propose, .emit — the batched-stage recipes


def solve_batched(tid, pairs=None, infer=None, verify=None, budget_s=None,
                  n_filter=64, k_verify=8, chunk=1 << 18):
    """Run every registered template's GPU sweep on `tid`; ONNX-verify the top-k survivors with the
    argen gate; return the cheapest confirmed longrun result dict, or None. No-ops (returns None) until
    templates are registered — the engine + interface are ready; the templates are the next build."""
    if not TEMPLATES:
        return None
    from . import argen, score, spec
    s = argen.sample(tid, n_filter)
    if s is None:
        return None
    X = s["X"]
    Y = argen.encode_batch(s["out_idx"], s["out_shapes"])
    torch, device = get_device()
    best = None
    for tpl in TEMPLATES:
        proposal = tpl.propose(tid, X, Y)
        if proposal is None:
            continue
        domain, score_fn, meta = proposal
        cands = batched_topk(domain, score_fn, k=k_verify, chunk=chunk, torch=torch, device=device)
        for _res, _id, choice in cands:
            model = tpl.emit(choice, meta)
            if model is None:
                continue
            p, t = score.argen_gate(model, tid, n=512)
            if t > 0 and p == t:
                cost, _ = score.exact_cost(model)
                if best is None or cost < best["cost"]:
                    best = {"cost": cost, "points": spec.points(cost),
                            "label": f"batched[{tpl.name}]", "method": "batched", "argen": (p, t)}
                break                                       # cheapest-first within a template
    return best


def _selftest():
    torch, device = get_device()
    print(f"batched engine device: {device}" + (f" ({torch.xpu.get_device_name(0)})"
          if device == "xpu" else ""))
    # a small domain: 8 factor vars, each in {0,1,2} -> 3^8 = 6561 candidates
    dom = DiscreteDomain([3] * 8)
    print(f"domain: {dom.card} -> {dom.total} candidates")
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((8, 12)).astype(np.float32)
    true = rng.integers(0, 3, size=8).astype(np.float32) - 1.0
    target = true @ basis                                     # a candidate that fits exactly exists
    sf = demo_lowrank_scorer(target, basis, torch=torch, device=device)
    best = batched_topk(dom, sf, k=5, chunk=2048, torch=torch, device=device)
    top = best[0]
    print(f"best residual {top[0]:.6f}  choices {top[2]}  (0.0 == exact fit found)")
    ok = top[0] < 1e-4
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
