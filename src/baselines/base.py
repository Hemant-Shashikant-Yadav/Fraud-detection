"""
src/baselines/base.py
=====================

Shared contract for the Phase-6 state-of-the-art comparison baselines
(BP1 - BP4).  Every baseline subclasses :class:`BaselineModel` and therefore
exposes exactly the same public surface:

    fit(X, y)                       -> self
    predict_proba(X)                -> np.ndarray (N, 2), rows sum to 1
    predict(X)                      -> np.ndarray (N,), values in {0, 1}
    measure_latency(X, n_runs=100)  -> {"p50", "p95", "p99", ...} in ms

The base class also owns everything that must be *identical* across baselines
so that the comparison against MG-ECFM is fair:

* reproducibility (SEED = 42 for Python / NumPy / PyTorch, deterministic ops);
* input validation (2-D finite float matrix, binary labels, fixed width);
* the shared heavy-tail preprocessing ``z = standardise(sign(x) log1p|x|)``
  fitted on training rows only (no look-ahead);
* the chronological tail hold-out used for early stopping / calibration:
  rows are assumed to arrive in time order (Phase-1 frames are sorted by
  ``step``) so the *last* ``fraction`` of the training rows is used and no
  future row ever trains a model evaluated on an earlier one;
* cost-sensitive class weights and the single-transaction latency harness.
"""
from __future__ import annotations

import math
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, fields
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn

from src.utils.device import resolve_device

SEED = 42
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = SEED) -> None:
    """Fix Python, NumPy and PyTorch RNGs and request deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_from_mapping(cls, mapping: Optional[Mapping[str, Any]]):
    """Build dataclass ``cls`` from a (possibly partial / superset) mapping.

    Unknown keys raise ``KeyError`` so that typos in experiment configs are
    caught instead of silently ignored.
    """
    if mapping is None:
        return cls()
    if isinstance(mapping, cls):
        return mapping
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {unknown}")
    return cls(**dict(mapping))


# --------------------------------------------------------------------------- #
# Validation & preprocessing                                                  #
# --------------------------------------------------------------------------- #
def check_X(X: Any, n_features: Optional[int] = None) -> np.ndarray:
    """Return ``X`` as a C-contiguous float64 2-D array or raise ``ValueError``."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("X has no rows")
    if arr.shape[1] == 0:
        raise ValueError("X has no feature columns")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X contains NaN or infinite values")
    if n_features is not None and arr.shape[1] != n_features:
        raise ValueError(
            f"X has {arr.shape[1]} features but the model was fitted with {n_features}"
        )
    return np.ascontiguousarray(arr)


def check_y(y: Any, n_rows: int) -> np.ndarray:
    """Binary {0, 1} label vector of length ``n_rows`` with both classes present."""
    arr = np.asarray(y).reshape(-1)
    if arr.shape[0] != n_rows:
        raise ValueError(f"y has {arr.shape[0]} labels for {n_rows} rows")
    if not np.all(np.isin(arr, (0, 1))):
        raise ValueError("y must contain only 0 (legitimate) and 1 (fraud)")
    arr = arr.astype(np.int64)
    if arr.min() == arr.max():
        raise ValueError("y must contain both classes to train a fraud classifier")
    return arr


class SignedLogStandardScaler:
    """``z = clip((sign(x) log1p|x| - mu) / sigma, -c, c)`` fitted on training rows.

    PaySim / IEEE-CIS amounts and velocity aggregates span 6+ orders of
    magnitude; the signed log compresses them while preserving sign
    (balance deltas can be negative).  Constant columns get ``sigma = 1``.
    """

    def __init__(self, clip: float = 10.0) -> None:
        self.clip = float(clip)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    @staticmethod
    def _slog(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, X: np.ndarray) -> "SignedLogStandardScaler":
        s = self._slog(X)
        self.mean_ = s.mean(axis=0)
        std = s.std(axis=0)
        self.scale_ = np.where(std > 1e-9, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("SignedLogStandardScaler is not fitted")
        z = (self._slog(X) - self.mean_) / self.scale_
        return np.clip(z, -self.clip, self.clip).astype(np.float32)


def chronological_holdout(n: int, fraction: float, min_rows: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Indices ``(head, tail)``: the last ``fraction`` of ``n`` time-ordered rows.

    If ``fraction <= 0`` the tail is empty.  The head always keeps at least
    ``min_rows`` rows.
    """
    if fraction <= 0.0:
        return np.arange(n), np.arange(0)
    n_tail = int(round(n * fraction))
    n_tail = min(max(n_tail, 1), max(n - min_rows, 0))
    cut = n - n_tail
    return np.arange(cut), np.arange(cut, n)


def stratified_tail_ok(y_tail: np.ndarray) -> bool:
    """A hold-out is usable for validation only if it holds both classes."""
    return y_tail.size > 0 and 0 < int(y_tail.sum()) < y_tail.size


def cost_sensitive_class_weights(y: np.ndarray, fn_cost: float, fp_cost: float,
                                 balance: bool = True) -> np.ndarray:
    """Per-class loss weights ``[w_0, w_1]``.

    With ``balance=True`` the inverse-frequency balance factor is multiplied by
    the asymmetric misclassification costs (missing a fraud costs ``fn_cost``,
    a false alarm ``fp_cost``); weights are normalised so that the mean
    per-sample weight on the training set is 1.
    """
    n = y.size
    n1 = max(int(y.sum()), 1)
    n0 = max(n - n1, 1)
    w0 = float(fp_cost)
    w1 = float(fn_cost)
    if balance:
        w0 *= n / (2.0 * n0)
        w1 *= n / (2.0 * n1)
    mean_w = (w0 * n0 + w1 * n1) / n
    return np.array([w0 / mean_w, w1 / mean_w], dtype=np.float32)


def two_column_proba(p_fraud: np.ndarray) -> np.ndarray:
    """``[1 - p, p]`` with ``p`` clipped into [0, 1] (guards float round-off)."""
    p = np.clip(np.asarray(p_fraud, dtype=np.float64).reshape(-1), 0.0, 1.0)
    return np.column_stack([1.0 - p, p])


def iterate_minibatches(n: int, batch_size: int, rng: np.random.Generator,
                        shuffle: bool = True):
    """Yield index arrays covering ``range(n)`` in mini-batches."""
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


# --------------------------------------------------------------------------- #
# Base class                                                                  #
# --------------------------------------------------------------------------- #
class BaselineModel(ABC):
    """Abstract unified-interface baseline.

    Subclasses implement :meth:`_fit` (on already validated arrays) and
    :meth:`_predict_fraud_proba` (returning ``P(y = 1 | x)`` of shape ``(N,)``).
    """

    #: short identifier used by the factory / reports (e.g. ``"bp1"``)
    name: str = "baseline"
    #: bibliographic reference reproduced by the subclass
    reference: str = ""

    def __init__(self, config: Any = None, seed: int = SEED) -> None:
        self.seed = int(seed)
        self.config = config
        self.threshold_: float = 0.5
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False
        self.fit_report_: Dict[str, Any] = {}
        self.device = resolve_device()

    # ------------------------------------------------------------ public API
    def fit(self, X: Any, y: Any) -> "BaselineModel":
        """Train on time-ordered rows ``X`` with binary labels ``y``."""
        X_arr = check_X(X)
        y_arr = check_y(y, X_arr.shape[0])
        seed_everything(self.seed)
        self.n_features_in_ = X_arr.shape[1]
        t0 = time.perf_counter()
        self._fit(X_arr, y_arr)
        self.is_fitted_ = True
        self.fit_report_["fit_seconds"] = time.perf_counter() - t0
        self.fit_report_["n_train"] = int(X_arr.shape[0])
        self.fit_report_["n_features"] = int(X_arr.shape[1])
        self.fit_report_["fraud_rate"] = float(y_arr.mean())
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """``(N, 2)`` array ``[P(legit), P(fraud)]``; every row sums to 1."""
        self._check_fitted()
        X_arr = check_X(X, self.n_features_in_)
        p = self._predict_fraud_proba(X_arr)
        return two_column_proba(p)

    def predict(self, X: Any) -> np.ndarray:
        """Hard decision ``1{P(fraud) >= threshold_}`` as int64 ``(N,)``."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold_).astype(np.int64)

    def measure_latency(self, X: Any, n_runs: int = 100, batch_size: int = 1,
                        warmup: int = 5) -> Dict[str, float]:
        """Wall-clock latency of ``predict_proba`` on ``batch_size`` rows.

        Rows are taken cyclically from ``X`` so each timed call scores a
        different transaction, mimicking a real-time authorisation stream.
        Returns milliseconds: ``p50``, ``p95``, ``p99``, ``mean``, ``max`` plus
        the run metadata.
        """
        self._check_fitted()
        X_arr = check_X(X, self.n_features_in_)
        if n_runs < 1:
            raise ValueError("n_runs must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        n = X_arr.shape[0]

        def rows(i: int) -> np.ndarray:
            idx = (np.arange(batch_size) + i * batch_size) % n
            return X_arr[idx]

        for i in range(max(int(warmup), 0)):
            self.predict_proba(rows(i))
        timings = np.empty(n_runs, dtype=np.float64)
        for i in range(n_runs):
            batch = rows(i)
            t0 = time.perf_counter()
            self.predict_proba(batch)
            timings[i] = (time.perf_counter() - t0) * 1000.0
        return {
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
            "p99": float(np.percentile(timings, 99)),
            "mean": float(timings.mean()),
            "max": float(timings.max()),
            "n_runs": int(n_runs),
            "batch_size": int(batch_size),
        }

    def get_config(self) -> Dict[str, Any]:
        """Plain-dict view of the (dataclass) configuration."""
        return asdict(self.config) if self.config is not None else {}

    def count_parameters(self) -> int:
        """Number of trainable PyTorch parameters (0 for non-neural parts)."""
        total = 0
        for value in vars(self).values():
            if isinstance(value, nn.Module):
                total += sum(p.numel() for p in value.parameters() if p.requires_grad)
        return int(total)

    # --------------------------------------------------------------- hooks
    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on validated arrays."""

    @abstractmethod
    def _predict_fraud_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``P(fraud)`` with shape ``(N,)``."""

    # ------------------------------------------------------------- helpers
    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(X, y) first")

    def _rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def _to_tensor(self, a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=self.device)


class EarlyStopper:
    """Keep the best ``state_dict`` by validation loss; signal after ``patience``."""

    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best = math.inf
        self.bad_epochs = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch = -1

    def step(self, loss: float, module: nn.Module, epoch: int) -> bool:
        """Record ``loss``; return ``True`` when training should stop."""
        if loss < self.best - 1e-6:
            self.best = loss
            self.bad_epochs = 0
            self.best_epoch = epoch
            self.best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            return False
        self.bad_epochs += 1
        return self.patience > 0 and self.bad_epochs >= self.patience

    def restore(self, module: nn.Module) -> None:
        if self.best_state is not None:
            module.load_state_dict(self.best_state)


# --------------------------------------------------------------------------- #
# Inductive k-NN neighbourhoods (BP2 / BP3 graph construction)                #
# --------------------------------------------------------------------------- #
class KNNNeighborIndex:
    """Euclidean k-NN graph over a fixed *reference* (training) node set.

    ``train_neighbors()`` returns, for every reference node, its ``k`` nearest
    *other* reference nodes; ``query(Z)`` links unseen transactions to their
    ``k`` nearest reference nodes, which is how the graph baselines score a new
    transaction inductively in O(k) message passing without re-training.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k_requested = int(k)
        self.k: int = int(k)
        self._nn = None
        self.n_ref: int = 0

    def fit(self, Z_ref: np.ndarray) -> "KNNNeighborIndex":
        from sklearn.neighbors import NearestNeighbors

        self.n_ref = int(Z_ref.shape[0])
        if self.n_ref < 2:
            raise ValueError("a k-NN graph needs at least two reference nodes")
        self.k = min(self.k_requested, self.n_ref - 1)
        self._nn = NearestNeighbors(n_neighbors=self.k + 1, algorithm="auto").fit(Z_ref)
        return self

    def train_neighbors(self, Z_ref: np.ndarray) -> np.ndarray:
        """``(n_ref, k)`` neighbour ids excluding each node itself."""
        nbrs = self._nn.kneighbors(Z_ref, n_neighbors=self.k + 1, return_distance=False)
        own = nbrs == np.arange(nbrs.shape[0])[:, None]
        no_self = ~own.any(axis=1)
        own[no_self, -1] = True              # duplicates hid self: drop farthest
        first = np.argmax(own, axis=1)       # keep exactly one dropped entry per row
        drop = np.zeros_like(own)
        drop[np.arange(own.shape[0]), first] = True
        return nbrs[~drop].reshape(nbrs.shape[0], self.k).astype(np.int64)

    def query(self, Z: np.ndarray) -> np.ndarray:
        """``(M, k)`` reference-node ids nearest to each row of ``Z``."""
        return self._nn.kneighbors(Z, n_neighbors=self.k, return_distance=False).astype(np.int64)
