"""
src/detection/tabular_engine.py
===============================

Cost-sensitive, calibrated XGBoost risk engine (MG-ECFM Stage 1A scorer).

Cost sensitivity
----------------
* ``scale_pos_weight = "auto"`` resolves to the misclassification cost ratio
  ``c_FN / c_FP`` (Elkan, 2001) when cost-sensitive learning is enabled and to
  the class-imbalance ratio ``n_neg / n_pos`` otherwise.
* With ``amount_weighted = true`` every fraud row additionally receives the
  sample weight ``A_t / mean(A_fraud)`` clipped to ``[0.25, 10]`` (mean-one
  over training frauds), so high-value frauds dominate the gradient in line
  with the monetary loss ``L(Approve, 1, A_t) = A_t`` of CS-ACRC.

Calibration
-----------
Margins are mapped to ``p_hat_t in [0, 1]`` on the chronologically later
*calibration split* (never the training split) by either

* **isotonic regression** (PAV, clipped to the observed range), or
* **inductive Venn-Abers** (Vovk & Petej, 2014): for a test score ``s`` two
  isotonic fits on ``calibration ∪ {(s, 0)}`` and ``calibration ∪ {(s, 1)}``
  give the multi-probability ``(p0, p1)`` and the merged point estimate
  ``p = p1 / (1 - p0 + p1)``. The IVAP is exact; because ``(p0, p1)`` only
  depends on the rank of ``s`` among calibration scores it is pre-computed
  once per rank position (O(1) lookup at inference). If the calibration set
  has more than ``max_unique_scores`` distinct scores, scores are first
  quantised to that many quantile bins.

Early stopping uses the chronologically last ``val_fraction`` of the training
rows unless an explicit validation split is supplied.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

SEED = 42

try:  # imported lazily so calibrators remain usable without XGBoost
    import xgboost as xgb

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - depends on the environment
    xgb = None
    HAS_XGBOOST = False


def _require_xgboost() -> None:
    if not HAS_XGBOOST:
        raise ImportError("xgboost is required for CalibratedXGBoostEngine: pip install xgboost")


# --------------------------------------------------------------------------- #
# Calibrators                                                                 #
# --------------------------------------------------------------------------- #
class IsotonicCalibrator:
    """Monotone PAV map from scores to probabilities, clipped to [0, 1]."""

    name = "isotonic"

    def __init__(self) -> None:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                       out_of_bounds="clip")
        self.fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if s.shape[0] != y.shape[0] or s.shape[0] == 0:
            raise ValueError("scores and labels must be non-empty and aligned")
        self._iso.fit(s, y)
        self.fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator is not fitted")
        p = self._iso.predict(np.asarray(scores, dtype=np.float64))
        return np.clip(p, 0.0, 1.0)


class VennAbersCalibrator:
    """Inductive Venn-Abers predictor with pre-computed rank tables.

    ``predict_interval`` returns ``(p0, p1)`` with ``p0 <= p1``; ``predict``
    returns the merged probability ``p1 / (1 - p0 + p1)``.
    """

    name = "venn_abers"

    def __init__(self, max_unique_scores: int = 400) -> None:
        if max_unique_scores < 2:
            raise ValueError("max_unique_scores must be >= 2")
        self.max_unique_scores = int(max_unique_scores)
        self.fitted = False
        self.quantized = False
        self._points: np.ndarray = np.zeros(0)
        self._edges: Optional[np.ndarray] = None
        self._tie_p0 = self._tie_p1 = np.zeros(0)
        self._gap_p0 = self._gap_p1 = np.zeros(0)

    @staticmethod
    def _iso_value(x: np.ndarray, y: np.ndarray, w: np.ndarray, x_new: float, y_new: float) -> float:
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        xs = np.append(x, x_new)
        ys = np.append(y, y_new)
        ws = np.append(w, 1.0)
        iso.fit(xs, ys, sample_weight=ws)
        return float(iso.predict(np.asarray([x_new]))[0])

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "VennAbersCalibrator":
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if s.shape[0] != y.shape[0] or s.shape[0] == 0:
            raise ValueError("scores and labels must be non-empty and aligned")
        uniq = np.unique(s)
        self.quantized = uniq.shape[0] > self.max_unique_scores
        if self.quantized:
            qs = np.linspace(0.0, 1.0, self.max_unique_scores + 1)[1:-1]
            self._edges = np.unique(np.quantile(s, qs))
            bins = np.searchsorted(self._edges, s, side="right")
            n_bins = self._edges.shape[0] + 1
            counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
            pos = np.bincount(bins, weights=y, minlength=n_bins)
            ssum = np.bincount(bins, weights=s, minlength=n_bins)
            keep = counts > 0
            # Representative point of bin b is its index; ordering is preserved.
            self._bin_ids = np.flatnonzero(keep)
            x = self._bin_ids.astype(np.float64)
            w = counts[keep]
            ybar = pos[keep] / w
            self._points = x
        else:
            self._edges = None
            inv = np.searchsorted(uniq, s)
            w = np.bincount(inv, minlength=uniq.shape[0]).astype(np.float64)
            ybar = np.bincount(inv, weights=y, minlength=uniq.shape[0]) / w
            x = uniq
            self._points = uniq
        k = x.shape[0]
        tie_p0 = np.empty(k)
        tie_p1 = np.empty(k)
        for j in range(k):
            tie_p0[j] = self._iso_value(x, ybar, w, x[j], 0.0)
            tie_p1[j] = self._iso_value(x, ybar, w, x[j], 1.0)
        # Gap g in [0, k]: strictly below x[0], between x[g-1] and x[g], above x[-1].
        gap_x = np.empty(k + 1)
        gap_x[0] = x[0] - 1.0
        gap_x[k] = x[-1] + 1.0
        if k > 1:
            gap_x[1:k] = 0.5 * (x[:-1] + x[1:])
        gap_p0 = np.empty(k + 1)
        gap_p1 = np.empty(k + 1)
        for g in range(k + 1):
            gap_p0[g] = self._iso_value(x, ybar, w, gap_x[g], 0.0)
            gap_p1[g] = self._iso_value(x, ybar, w, gap_x[g], 1.0)
        self._tie_p0, self._tie_p1 = tie_p0, tie_p1
        self._gap_p0, self._gap_p1 = gap_p0, gap_p1
        self.fitted = True
        return self

    def predict_interval(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("VennAbersCalibrator is not fitted")
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        if self.quantized:
            bins = np.searchsorted(self._edges, s, side="right").astype(np.float64)
            pts = self._points
        else:
            bins = s
            pts = self._points
        pos = np.searchsorted(pts, bins, side="left")
        k = pts.shape[0]
        pos_c = np.minimum(pos, k - 1)
        is_tie = (pos < k) & (pts[pos_c] == bins)
        p0 = np.where(is_tie, self._tie_p0[pos_c], self._gap_p0[pos])
        p1 = np.where(is_tie, self._tie_p1[pos_c], self._gap_p1[pos])
        return np.clip(p0, 0.0, 1.0), np.clip(p1, 0.0, 1.0)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        p0, p1 = self.predict_interval(scores)
        denom = 1.0 - p0 + p1
        return np.clip(np.where(denom > 0, p1 / np.where(denom > 0, denom, 1.0), 0.5), 0.0, 1.0)


def make_calibrator(method: str, **kwargs):
    if method == "isotonic":
        return IsotonicCalibrator()
    if method in ("venn_abers", "venn-abers", "ivap"):
        return VennAbersCalibrator(**kwargs)
    raise ValueError(f"Unknown calibration method '{method}'")


# --------------------------------------------------------------------------- #
# Calibration metrics                                                         #
# --------------------------------------------------------------------------- #
def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width-bin ECE."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


# --------------------------------------------------------------------------- #
# Engine                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class TabularEngineConfig:
    """XGBoost + cost-sensitivity + calibration hyper-parameters."""

    params: Dict[str, object] = field(default_factory=lambda: {
        "objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
        "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 5, "gamma": 0.0, "reg_lambda": 1.0, "reg_alpha": 0.0,
        "max_bin": 256,
    })
    n_estimators: int = 600
    early_stopping_rounds: Optional[int] = 50
    scale_pos_weight: object = "auto"
    cost_sensitive: bool = True
    fn_weight: float = 25.0
    fp_weight: float = 1.0
    amount_weighted: bool = True
    amount_weight_clip: Tuple[float, float] = (0.25, 10.0)
    calibration_method: str = "isotonic"
    venn_abers_max_unique: int = 400
    val_fraction: float = 0.1
    seed: int = SEED
    n_jobs: int = -1
    inference_nthread: int = 1

    _PARAM_KEYS = ("objective", "eval_metric", "tree_method", "max_depth", "learning_rate",
                   "subsample", "colsample_bytree", "min_child_weight", "gamma",
                   "reg_lambda", "reg_alpha", "max_bin", "device")

    @classmethod
    def from_hyperparams(cls, hyperparams: Mapping, **overrides) -> "TabularEngineConfig":
        from src.utils.device import resolve_xgb_device
        xg = dict(hyperparams.get("xgboost", {})) if hyperparams else {}
        cs = xg.get("cost_sensitive", {})
        cal = xg.get("calibration", {})
        base = cls()
        params = dict(base.params)
        params.update({k: xg[k] for k in cls._PARAM_KEYS if k in xg})
        if "device" not in params:
            params["device"] = resolve_xgb_device()
        kwargs = dict(
            params=params,
            n_estimators=int(xg.get("n_estimators", 600)),
            early_stopping_rounds=xg.get("early_stopping_rounds", 50),
            scale_pos_weight=xg.get("scale_pos_weight", "auto"),
            cost_sensitive=bool(cs) if cs else False,
            fn_weight=float(cs.get("fn_weight", 25.0)) if cs else 25.0,
            fp_weight=float(cs.get("fp_weight", 1.0)) if cs else 1.0,
            amount_weighted=bool(cs.get("amount_weighted", True)) if cs else False,
            calibration_method=str(cal.get("method", "isotonic")),
            seed=int(xg.get("random_state", hyperparams.get("seed", SEED) if hyperparams else SEED)),
            n_jobs=int(xg.get("n_jobs", -1)),
        )
        kwargs.update(overrides)
        return cls(**kwargs)


class CalibratedXGBoostEngine:
    """Cost-sensitive XGBoost classifier with post-hoc calibration."""

    def __init__(self, config: Optional[TabularEngineConfig] = None) -> None:
        _require_xgboost()
        self.config = config or TabularEngineConfig()
        self.booster = None
        self.best_iteration: Optional[int] = None
        self.calibrator = None
        self.feature_names: Optional[List[str]] = None
        self.resolved_scale_pos_weight: Optional[float] = None
        self.fit_report: Dict[str, float] = {}

    # ------------------------------------------------------------ weights
    def resolve_scale_pos_weight(self, labels: np.ndarray) -> float:
        cfg = self.config
        if cfg.scale_pos_weight not in (None, "auto"):
            return float(cfg.scale_pos_weight)
        if cfg.cost_sensitive:
            return float(cfg.fn_weight / cfg.fp_weight)
        y = np.asarray(labels)
        pos = float((y == 1).sum())
        return float((y.shape[0] - pos) / max(pos, 1.0))

    def sample_weights(self, labels: np.ndarray, amounts: Optional[np.ndarray]) -> np.ndarray:
        """Per-row weights: ``fp_weight`` for legit rows, amount factor for frauds."""
        cfg = self.config
        y = np.asarray(labels).astype(int)
        w = np.ones(y.shape[0], dtype=np.float64)
        if not cfg.cost_sensitive:
            return w
        w[y == 0] = cfg.fp_weight
        w[y == 1] = cfg.fp_weight
        if cfg.amount_weighted and amounts is not None and (y == 1).any():
            a = np.asarray(amounts, dtype=np.float64)
            mean_pos = a[y == 1].mean()
            if mean_pos > 0:
                factor = np.clip(a / mean_pos, *cfg.amount_weight_clip)
                w = np.where(y == 1, w * factor, w)
        return w

    # ---------------------------------------------------------------- fit
    def _params(self, labels: np.ndarray) -> Dict[str, object]:
        cfg = self.config
        params = dict(cfg.params)
        self.resolved_scale_pos_weight = self.resolve_scale_pos_weight(labels)
        params["scale_pos_weight"] = self.resolved_scale_pos_weight
        params["seed"] = int(cfg.seed)
        params["nthread"] = int(os.cpu_count() or 1) if cfg.n_jobs in (-1, None) else int(cfg.n_jobs)
        params["verbosity"] = 0
        return params

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_cal: np.ndarray, y_cal: np.ndarray,
            amounts_train: Optional[np.ndarray] = None,
            x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            feature_names: Optional[Sequence[str]] = None) -> "CalibratedXGBoostEngine":
        """Fit XGBoost on ``train`` (early stopping on ``val``) and calibrate on ``cal``."""
        cfg = self.config
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train).astype(int)
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train / y_train length mismatch")
        if not (y_train == 1).any() or not (y_train == 0).any():
            raise ValueError("Training labels must contain both classes")
        self.feature_names = list(feature_names) if feature_names is not None else None
        weights = self.sample_weights(y_train, amounts_train)
        if x_val is None and cfg.early_stopping_rounds and cfg.val_fraction > 0:
            n_val = int(round(x_train.shape[0] * cfg.val_fraction))
            if n_val > 0 and (y_train[-n_val:] == 1).any() and (y_train[:-n_val] == 1).any():
                x_val, y_val = x_train[-n_val:], y_train[-n_val:]
                x_train, y_train, weights = x_train[:-n_val], y_train[:-n_val], weights[:-n_val]
        params = self._params(y_train)
        dtrain = xgb.DMatrix(x_train, label=y_train, weight=weights)
        evals = [(dtrain, "train")]
        es_rounds = None
        if x_val is not None and y_val is not None and cfg.early_stopping_rounds:
            dval = xgb.DMatrix(np.asarray(x_val, dtype=np.float32), label=np.asarray(y_val).astype(int))
            evals.append((dval, "val"))
            es_rounds = int(cfg.early_stopping_rounds)
        t0 = time.perf_counter()
        self.booster = xgb.train(params, dtrain, num_boost_round=int(cfg.n_estimators),
                                 evals=evals, early_stopping_rounds=es_rounds,
                                 verbose_eval=False)
        best = getattr(self.booster, "best_iteration", None)
        self.best_iteration = int(best) if (es_rounds and best is not None) else int(cfg.n_estimators) - 1
        self.booster.set_param({"nthread": int(cfg.inference_nthread)})
        train_seconds = time.perf_counter() - t0
        self.calibrate(x_cal, y_cal)
        self.fit_report = {
            "train_seconds": train_seconds,
            "best_iteration": float(self.best_iteration),
            "scale_pos_weight": float(self.resolved_scale_pos_weight),
        }
        return self

    def calibrate(self, x_cal: np.ndarray, y_cal: np.ndarray,
                  method: Optional[str] = None) -> "CalibratedXGBoostEngine":
        """(Re-)fit the calibration map on a held-out, later calibration split."""
        y = np.asarray(y_cal).astype(int)
        if y.shape[0] == 0:
            raise ValueError("Calibration split is empty")
        margins = self.predict_margin(x_cal)
        self.calibrator = make_calibrator(method or self.config.calibration_method,
                                          **({"max_unique_scores": self.config.venn_abers_max_unique}
                                             if (method or self.config.calibration_method) != "isotonic"
                                             else {}))
        self.calibrator.fit(margins, y)
        return self

    # ------------------------------------------------------------ predict
    def predict_margin(self, x: np.ndarray) -> np.ndarray:
        """Raw boosted margin (log-odds) using trees up to the best iteration."""
        if self.booster is None:
            raise RuntimeError("CalibratedXGBoostEngine is not fitted")
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        rng = (0, int(self.best_iteration) + 1)
        try:
            return np.asarray(self.booster.inplace_predict(arr, iteration_range=rng,
                                                           predict_type="margin"),
                              dtype=np.float64).reshape(-1)
        except (AttributeError, TypeError):
            return np.asarray(self.booster.predict(xgb.DMatrix(arr), output_margin=True,
                                                   iteration_range=rng),
                              dtype=np.float64).reshape(-1)

    def predict_uncalibrated(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid of the margin (cost-weighted, hence *not* a calibrated probability)."""
        m = self.predict_margin(x)
        return 1.0 / (1.0 + np.exp(-m))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Calibrated risk ``p_hat_t in [0, 1]``."""
        if self.calibrator is None:
            raise RuntimeError("Engine has not been calibrated")
        return np.clip(self.calibrator.predict(self.predict_margin(x)), 0.0, 1.0)

    def predict_interval(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Venn-Abers ``(p0, p1)`` (requires a Venn-Abers calibrator)."""
        if not isinstance(self.calibrator, VennAbersCalibrator):
            raise RuntimeError("predict_interval requires the venn_abers calibrator")
        return self.calibrator.predict_interval(self.predict_margin(x))

    def predict_one(self, x_row: np.ndarray) -> float:
        return float(self.predict_proba(np.asarray(x_row, dtype=np.float32)[None, :])[0])

    def feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        if self.booster is None:
            raise RuntimeError("CalibratedXGBoostEngine is not fitted")
        raw = self.booster.get_score(importance_type=importance_type)
        if self.feature_names is None:
            return dict(raw)
        out: Dict[str, float] = {}
        for key, val in raw.items():
            if key.startswith("f") and key[1:].isdigit() and int(key[1:]) < len(self.feature_names):
                out[self.feature_names[int(key[1:])]] = float(val)
            else:
                out[key] = float(val)
        return out
