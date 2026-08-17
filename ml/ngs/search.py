"""search.py — DAG program search over the op DSL, scored by gates + cost.

Each search node carries a GState (the partial graph). Expansion tries every op in the vocab and,
per op, every bounded operand tuple it offers (`op.candidates`) — this is what lets the search build
branches and merge them (Where/Add/...). Ranking: solved-fraction first (correctness), then cost.
The optional `policy` orders/prunes the op choices per node (the neural-guided knob); a learned
PolicyModel is a drop-in for the heuristic one.
"""
from . import score
from .ops import base_vocab, GState
from .builder import to_model


def _eval(g, pairs):
    model = to_model(g)
    if model is None:
        return None
    ok, tot = score.evaluate(model, pairs)
    return model, (ok / tot if tot else 0.0), score.static_cost(model)


def beam_search(pairs, vocab=None, beam_width=8, max_depth=3, policy=None, top_k=None,
                exact_final=True, max_evals=8000, infer_pairs=None, verify=None):
    """Search programs that reproduce `pairs` (the visible set) exactly, ranked by (frac, cost).

    infer_pairs: pairs the data-aware ops read to INFER their arguments (a color map, a coordinate
      map, ...). Defaults to `pairs`; pass the large ARC-GEN corpus so inference is robust while the
      cheap `pairs` set still drives the correctness ranking.
    verify: optional callback (model)->bool. When set, a program that solves every visible pair is
      recorded as `best` only if verify(model) also passes — the ARC-GEN truth gate that turns a
      visible-pair solve into a trustworthy one. verify_info holds the last gate result for reporting.
    """
    vocab = vocab or base_vocab()
    ip = infer_pairs if infer_pairs is not None else pairs
    seen, evals, best = set(), 0, None
    verify_info = {}

    def consider(model, ops, path):
        nonlocal best
        if verify is not None:
            vi = verify(model)
            verify_info["last"] = vi
            if not vi[0]:
                verify_info["rejected"] = vi            # a visible-solve that flunked the truth gate
                return                                  # failed the truth gate — not a real solve
        cost, kind = (score.exact_cost(model) if exact_final else (score.static_cost(model), "static"))
        if best is None or cost < best["cost"]:
            best = {"program": ops, "label": path, "cost": cost, "cost_kind": kind, "frac": 1.0,
                    "verify": verify_info.get("last")}

    r = _eval(GState.start(), pairs)
    if r is None:
        return None, {"evals": 0, "vocab": len(vocab)}
    evals += 1
    model, frac, cost = r
    if frac == 1.0:
        consider(model, [], "(identity)")
    frontier = [{"g": GState.start(), "ops": [], "path": "(identity)", "frac": frac, "cost": cost}]

    for _ in range(max_depth):
        scored = []
        for node in frontier:
            g = node["g"]
            order = policy(pairs, node["ops"], vocab) if policy else range(len(vocab))
            if top_k is not None:
                order = list(order)[:top_k]
            for i in order:
                op = vocab[i]
                for operands in op.candidates(g):
                    move = f"{op.label}({','.join(o.name for o in operands)})"
                    path = node["path"] + " -> " + move
                    if path in seen:
                        continue
                    seen.add(path)
                    ng = op.build(g, operands, ip)
                    if ng is None:
                        continue
                    r = _eval(ng, pairs)
                    if r is None:
                        continue
                    evals += 1
                    model, frac, cost = r
                    ops2 = node["ops"] + [op]
                    if frac == 1.0:
                        consider(model, ops2, path)
                    scored.append({"g": ng, "ops": ops2, "path": path, "frac": frac, "cost": cost})
                    if evals >= max_evals:
                        break
                if evals >= max_evals:
                    break
            if evals >= max_evals:
                break
        scored.sort(key=lambda n: (-n["frac"], n["cost"]))
        frontier = scored[:beam_width]
        if evals >= max_evals or not frontier:
            break

    return best, {"evals": evals, "vocab": len(vocab), "verify": verify_info}
