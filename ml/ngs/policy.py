"""policy.py — the guidance layer that turns brute-force search into *neural-guided* search.

Two implementations behind one interface (`.rank(pairs, program, vocab) -> list[int]`,
the order/prune hook search.beam_search already calls):

  HeuristicPolicy  — rule-based prior from task features. Needs NO training, runs in the
                     plain .venv today. Already prunes the branching factor (e.g. boosts
                     Transpose when out.shape == in.shape.T). The honest baseline policy.
  PolicyModel      — a small torch MLP: task+program features -> softmax over the vocab.
                     Trained by imitation on mined (winning-graph, task) pairs (train.py),
                     then refined by search-bootstrap RL. Optional import (needs .venv-ml).

`task_features` is shared by both and by mine.py, so the learned model sees the same state
the heuristic does.
"""
import numpy as np

FEATURE_NAMES = [
    "same_shape", "transpose_shape", "out_bigger", "out_smaller", "area_ratio",
    "in_colors", "out_colors", "new_colors", "lost_colors", "out_eq_hflip", "out_eq_vflip",
]


def task_features(pairs):
    """Cheap, shape/color/symmetry features of a task's example pairs -> np.float32 vector."""
    f = {k: 0.0 for k in FEATURE_NAMES}
    if not pairs:
        return np.zeros(len(FEATURE_NAMES), np.float32)
    n = len(pairs)
    for ig, og in pairs:
        ig = np.asarray(ig); og = np.asarray(og)
        f["same_shape"] += (ig.shape == og.shape)
        f["transpose_shape"] += (ig.shape == og.shape[::-1])
        f["out_bigger"] += (og.size > ig.size)
        f["out_smaller"] += (og.size < ig.size)
        f["area_ratio"] += og.size / max(1, ig.size)
        ic, oc = set(np.unique(ig).tolist()), set(np.unique(og).tolist())
        f["in_colors"] += len(ic); f["out_colors"] += len(oc)
        f["new_colors"] += len(oc - ic); f["lost_colors"] += len(ic - oc)
        if ig.shape == og.shape:
            f["out_eq_hflip"] += np.array_equal(og, ig[:, ::-1])
            f["out_eq_vflip"] += np.array_equal(og, ig[::-1, :])
    for k in f:
        f[k] /= n
    return np.array([f[k] for k in FEATURE_NAMES], np.float32)


class HeuristicPolicy:
    """Hand-written prior over the vocab from task features. Drop-in; no training required."""
    def rank(self, pairs, program, vocab):
        feat = dict(zip(FEATURE_NAMES, task_features(pairs)))
        prior = np.zeros(len(vocab), np.float32)
        for i, op in enumerate(vocab):
            lbl = op.label
            if lbl.startswith("transpose"):
                prior[i] += 3.0 * feat["transpose_shape"]
            if lbl.startswith("flip"):
                prior[i] += 3.0 * (feat["out_eq_hflip"] + feat["out_eq_vflip"]) + 0.5 * feat["same_shape"]
            if lbl == "colorremap":
                # the size-agnostic recolor op — must not be pruned on recolor tasks (colors change
                # but shapes match). This is the family the engine already solves; rank it high.
                prior[i] += 2.0 * (feat["new_colors"] + feat["lost_colors"]) + 0.5 * feat["same_shape"] + 0.5
            if lbl == "coordremap":
                # data-aware pixel rearrangement (flip/rot/crop/mirror). Worth trying whenever the
                # geometry hints at a rearrange: symmetry, a size change, or a transposed shape.
                prior[i] += (2.0 * (feat["out_eq_hflip"] + feat["out_eq_vflip"])
                             + 1.0 * feat["out_smaller"] + 1.0 * feat["out_bigger"]
                             + 0.5 * feat["transpose_shape"] + 0.5 * feat["same_shape"] + 0.3)
            if lbl == "identity":
                prior[i] += 1.0 * feat["same_shape"]
            if lbl.startswith("cast"):
                prior[i] += 0.3                          # cheap dtype trick, always mildly worth trying late
            if lbl in ("reducesum_ch", "greater0"):
                prior[i] += 0.4 * feat["same_shape"]
        return list(np.argsort(-prior))

    # used by mine.py / train.py as the "no-model" featurizer
    def features(self, pairs, program=None):
        return task_features(pairs)


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


class PolicyModel:
    """Learnable policy. Real torch model; only instantiable where torch is installed (.venv-ml).
    Identical .rank() interface as HeuristicPolicy, so search.py is agnostic to which is used."""
    def __init__(self, n_vocab, hidden=64):
        torch = _torch()
        if torch is None:
            raise RuntimeError("PolicyModel needs torch (.venv-ml). Use HeuristicPolicy in the plain venv.")
        import torch.nn as nn
        self.torch = torch
        self.n_vocab = n_vocab
        in_dim = len(FEATURE_NAMES) + n_vocab          # task features + program op-count histogram
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_vocab),
        )

    def _state_vec(self, pairs, program, vocab):
        hist = np.zeros(len(vocab), np.float32)
        idx = {op.label: i for i, op in enumerate(vocab)}
        for op in (program or []):
            if op.label in idx:
                hist[idx[op.label]] += 1.0
        return np.concatenate([task_features(pairs), hist]).astype(np.float32)

    def rank(self, pairs, program, vocab):
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(self._state_vec(pairs, program, vocab))[None, :]
            logits = self.net(x)[0].cpu().numpy()
        return list(np.argsort(-logits))

    def save(self, path):
        self.torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(self.torch.load(path)); self.net.eval(); return self
