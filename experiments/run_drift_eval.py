"""
experiments/run_drift_eval.py
=============================

Phase 7 - 50-window chronological drift evaluation (Menezes & Filho, 2025).

Protocol
--------
1. A causal (leakage-free) Phase-1 feature stream is split strictly by time
   into ``train < calibration < operating < test``.
2. Every model is trained once on the past (``train`` [+ ``calibration`` for
   the baselines, which hold out their own chronological tail]) and its binary
   operating threshold ``tau`` is selected on the ``operating`` split only
   (F1-optimal), never on test data.
3. The ``test`` period is partitioned into ``n_windows = 50`` consecutive,
   step-aligned monitoring windows of (approximately) equal transaction count.
4. The windows are processed sequentially (prequential "test-then-learn"):
   every model scores window ``w`` using only information available before
   ``w``; the verified labels of window ``j`` are released to the online
   components of MG-ECFM (CS-ACRC thresholds, martingale knowledge gate) only
   before window ``j + 1 + label_delay_windows`` is scored.  Baselines are
   frozen after training, exactly as in the Menezes & Filho drift protocol.
5. Per window and per model the following are recorded: PR-AUC (primary),
   ROC-AUC, Precision, Recall, F1, F2, Brier score, ECE and Expected Monetary
   Loss (EML) under the asymmetric CS-ACRC cost matrix of
   :mod:`src.conformal.loss_functions`.

EML semantics
-------------
``expected_monetary_loss`` is the mean realised loss per transaction under the
model's *own* decision policy: CS-ACRC four-action routing for MG-ECFM,
approve/block at ``tau`` for the binary baselines.  ``eml_binary_policy``
applies the identical approve/block policy at ``tau`` to every model, which
isolates the effect of the risk scores from the effect of the action space.

Data sources
------------
``auto`` (default) resolves to ``paysim`` and raises if the raw file is
missing - it never falls back to synthetic data silently.
``synthetic`` (explicit ``--source synthetic`` only): the schema-faithful
PaySim generator of Phase 1, with optional *camouflage concept drift* - in the
second part of the stream fraud amounts migrate geometrically towards the
legitimate amount distribution, so amount-driven detectors degrade over the
windows.  ``paysim`` / ``ieee_cis``: the real raw files configured in
``configs/project_config.json`` (``data/raw/...``), streamed through the
Phase-1 loaders and causal feature engineer.

Outputs
-------
``reports/drift_evaluation_results.csv``      one row per (model, window)
``reports/drift_evaluation_predictions.csv``  one row per (model, test transaction)
``reports/drift_evaluation_manifest.json``    configuration and split summary

Run from the repository root::

    python experiments/run_drift_eval.py --source synthetic  # generated data, fast profile
    python experiments/run_drift_eval.py --profile full
    python experiments/run_drift_eval.py --source paysim --max-rows 2000000
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.conformal.adaptive_crc import OnlineAdaptiveCRC  # noqa: E402
from src.conformal.loss_functions import Action, AsymmetricLoss, CostParameters  # noqa: E402
from src.conformal.router import PreTransactionRouter  # noqa: E402
from src.data.causal_preprocessor import (  # noqa: E402
    CausalFeatureConfig,
    CausalFeatureEngineer,
    set_global_seed,
)
from src.data.dataset_loaders import build_loader, make_synthetic_paysim_frame  # noqa: E402
from src.data.perturbation_generator import (  # noqa: E402
    CandidateRule,
    PerturbationConfig,
    PerturbationGenerator,
)

SEED = 42
CONFIG_DIR = ROOT / "configs"
REPORTS_DIR = ROOT / "reports"
DEFAULT_RESULTS_PATH = REPORTS_DIR / "drift_evaluation_results.csv"
DEFAULT_PREDICTIONS_PATH = REPORTS_DIR / "drift_evaluation_predictions.csv"
DEFAULT_MANIFEST_PATH = REPORTS_DIR / "drift_evaluation_manifest.json"

PROPOSED_MODEL = "mg_ecfm"
BASELINE_KEYS: Tuple[str, ...] = ("bp1", "bp2", "bp3", "bp4")
MODEL_ORDER: Tuple[str, ...] = (PROPOSED_MODEL,) + BASELINE_KEYS
MODEL_LABELS: Dict[str, str] = {
    "mg_ecfm": "MG-ECFM (proposed)",
    "bp1": "BP1: EARN+RXT-J (Almazroi & Ayub, 2023)",
    "bp2": "BP2: Lambda GNN+AE (Alarfaj & Shahzadi, 2025)",
    "bp3": "BP3: CARE-GNN (Dou et al., 2020)",
    "bp4": "BP4: ARGUS (Nayak & Bushara, 2026)",
}
PRIMARY_METRIC = "pr_auc"
METRIC_COLUMNS: Tuple[str, ...] = (
    "pr_auc", "roc_auc", "precision", "recall", "f1", "f2", "brier", "ece",
    "expected_monetary_loss", "eml_binary_policy", "total_monetary_loss",
)
RESULT_COLUMNS: Tuple[str, ...] = (
    "model", "model_label", "window", "start_step", "end_step", "n_transactions", "n_fraud",
    "fraud_rate", "threshold") + METRIC_COLUMNS + (
    "intervention_rate", "step_up_rate", "hold_rate", "block_rate", "n_active_rules",
    "lambda_low", "lambda_mid", "lambda_high", "latency_ms_per_txn", "data_source")

# Small, CPU-friendly configurations (identical to the Phase-6 test profile) so the
# whole evaluation runs in minutes; ``--profile full`` uses the published defaults.
FAST_BASELINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "bp1": {"earn_epochs": 4, "rxt_epochs": 4, "batch_size": 128, "ae_hidden": 32,
            "resnet_width": 32, "resnet_embedding": 16, "rxt_tokens": 4, "rxt_channels": 16,
            "cardinality": 4, "bottleneck_width": 4, "gru_hidden": 32, "smote_ratio": 0.5,
            "jaya_population": 6, "jaya_iterations": 5},
    "bp2": {"k_neighbors": 5, "gnn_hidden": 32, "gnn_epochs": 20, "gnn_batch_size": 128,
            "ae_epochs": 10, "ae_batch_size": 256, "patience": 8},
    "bp3": {"k_neighbors": 5, "embed_dim": 16, "epochs": 6, "batch_size": 256,
            "learning_rate": 0.01},
    "bp4": {"d_model": 16, "n_heads": 2, "n_blocks": 1, "ssm_state": 8, "epochs": 25,
            "batch_size": 64, "mc_samples": 8, "patience": 8},
}


# --------------------------------------------------------------------------- #
# Configuration helpers                                                       #
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(hyperparams, project_config)`` from ``configs/``."""
    return load_json(CONFIG_DIR / "hyperparams.json"), load_json(CONFIG_DIR / "project_config.json")


def fast_hyperparams(hyperparams: Mapping[str, Any]) -> Dict[str, Any]:
    """Reduced MG-ECFM detector capacity (same reductions as the Phase-2 tests)."""
    hp = copy.deepcopy(dict(hyperparams))
    hp["temporal_memory"]["memory_dim"] = 16
    hp["similarity_filter"]["hidden_dim"] = 0
    hp["similarity_filter"]["min_neighbors_kept"] = 0
    hp["graphsage"].update({"hidden_dim": 32, "out_dim": 16, "fanouts": [5, 3],
                            "epochs": 2, "batch_size": 256})
    hp["autoencoder"].update({"epochs": 5, "batch_size": 256})
    hp["xgboost"].update({"n_estimators": 120, "early_stopping_rounds": 30, "max_depth": 4})
    return hp


def profile_settings(profile: str, hyperparams: Mapping[str, Any]
                     ) -> Tuple[Dict[str, Any], Optional[Dict[str, Dict[str, Any]]]]:
    """``(mg_ecfm_hyperparams, baseline_configs)`` for ``fast`` / ``full``."""
    if profile == "fast":
        return fast_hyperparams(hyperparams), copy.deepcopy(FAST_BASELINE_CONFIGS)
    if profile == "full":
        return copy.deepcopy(dict(hyperparams)), None
    raise ValueError("profile must be 'fast' or 'full'")


# --------------------------------------------------------------------------- #
# Data: causal streams and chronological splits                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvaluationWindow:
    """One chronological monitoring window (row range of the test frame)."""

    index: int
    row_start: int
    row_end: int
    start_step: int
    end_step: int

    @property
    def n_rows(self) -> int:
        return self.row_end - self.row_start


@dataclass
class StreamSplits:
    """Chronological splits of one causal feature stream."""

    train: pd.DataFrame
    calibration: pd.DataFrame
    operating: pd.DataFrame
    test: pd.DataFrame
    feature_names: List[str]
    windows: List[EvaluationWindow]
    source: str
    canonical: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def window_frame(self, window: EvaluationWindow) -> pd.DataFrame:
        return self.test.iloc[window.row_start:window.row_end].reset_index(drop=True)

    def summary(self) -> Dict[str, Any]:
        def part(frame: pd.DataFrame) -> Dict[str, Any]:
            return {"transactions": int(len(frame)), "frauds": int(frame["label"].sum()),
                    "start_step": int(frame["step"].min()), "end_step": int(frame["step"].max())}
        sizes = [w.n_rows for w in self.windows]
        return {"source": self.source, "n_features": len(self.feature_names),
                "train": part(self.train), "calibration": part(self.calibration),
                "operating": part(self.operating), "test": part(self.test),
                "n_windows": len(self.windows), "window_rows_min": int(min(sizes)),
                "window_rows_max": int(max(sizes)), "window_rows_mean": float(np.mean(sizes)),
                **self.metadata}


def raw_paysim_to_canonical(raw: pd.DataFrame, step_seconds: int = 3600) -> pd.DataFrame:
    """Phase-1 canonical frame from a PaySim-schema log (leakage columns are dropped)."""
    steps = raw["step"].to_numpy(dtype=np.int64)
    return pd.DataFrame({
        "tx_id": np.arange(len(raw), dtype=np.int64),
        "step": steps,
        "event_time": steps.astype(np.float64) * float(step_seconds),
        "src": raw["nameOrig"].astype(str).to_numpy(),
        "dst": raw["nameDest"].astype(str).to_numpy(),
        "amount": raw["amount"].to_numpy(dtype=np.float64),
        "tx_type": raw["type"].astype(str).to_numpy(),
        "label": raw["isFraud"].to_numpy(dtype=np.int8),
    })


def inject_camouflage_drift(raw: pd.DataFrame, start_step: int, strength: float,
                            seed: int = SEED, legit_log_mean: float = 9.0,
                            legit_log_sigma: float = 1.2) -> pd.DataFrame:
    """Gradual concept drift: fraud amounts migrate towards legitimate amounts.

    For a fraud row at step ``s >= start_step`` the weight
    ``w(s) = strength * (s - start_step + 1) / (s_max - start_step + 1)`` ramps
    linearly to ``strength`` and the amount becomes the geometric interpolation
    ``exp((1 - w) log A + w log A_legit)`` with ``A_legit`` drawn from the
    legitimate log-normal amount law of the generator. Legitimate rows are not
    modified. ``strength = 0`` returns an unchanged copy.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("drift strength must lie in [0, 1]")
    out = raw.copy()
    if strength == 0.0:
        return out
    rng = np.random.default_rng([seed, 7_919])
    steps = out["step"].to_numpy(dtype=np.int64)
    s_max = int(steps.max())
    if start_step > s_max:
        return out
    mask = (out["isFraud"].to_numpy() == 1) & (steps >= start_step)
    if not mask.any():
        return out
    w = strength * (steps[mask] - start_step + 1) / float(s_max - start_step + 1)
    a = np.maximum(out.loc[mask, "amount"].to_numpy(dtype=np.float64), 1.0)
    legit = rng.lognormal(mean=legit_log_mean, sigma=legit_log_sigma, size=int(mask.sum()))
    out.loc[mask, "amount"] = np.round(np.exp((1.0 - w) * np.log(a) + w * np.log(legit)), 2)
    return out


def build_synthetic_stream(n_steps: int = 400, tx_per_step: Tuple[int, int] = (50, 90),
                           n_customers: int = 1500, n_merchants: int = 150,
                           fraud_rate: float = 0.03, drift_strength: float = 0.5,
                           drift_start_fraction: float = 0.6, seed: int = SEED
                           ) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """``(feature_frame, feature_names, canonical_frame)`` for a synthetic PaySim stream."""
    raw = make_synthetic_paysim_frame(n_steps=n_steps, tx_per_step=tx_per_step,
                                      n_customers=n_customers, n_merchants=n_merchants,
                                      fraud_rate=fraud_rate, seed=seed)
    start = int(round(1 + drift_start_fraction * (n_steps - 1)))
    raw = inject_camouflage_drift(raw, start, drift_strength, seed=seed)
    canon = raw_paysim_to_canonical(raw)
    engineer = CausalFeatureEngineer()
    frame = engineer.transform(canon)
    return frame, list(engineer.feature_names), canon


def build_real_stream(dataset: str, project_config: Mapping[str, Any],
                      max_rows: Optional[int] = None
                      ) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Stream a configured raw dataset through the Phase-1 loader + causal engineer."""
    loader = build_loader(dataset, project_config, repo_root=ROOT, max_rows=max_rows)
    engineer = CausalFeatureEngineer(CausalFeatureConfig.from_project_config(project_config, dataset))
    frame = pd.concat(list(engineer.transform_stream(loader.iter_chunks())), ignore_index=True)
    # The engineer carries the canonical columns through unchanged and in input
    # order, so the canonical stream is a column selection of ``frame`` (shared
    # buffers under copy-on-write) instead of a second copy of every chunk.
    canon = frame[[c for c in ("tx_id", "step", "event_time", "src", "dst", "amount", "tx_type", "label")
                   if c in frame.columns]]
    return frame, list(engineer.feature_names), canon


def raw_dataset_available(dataset: str, project_config: Mapping[str, Any]) -> bool:
    ds = project_config.get("datasets", {}).get(dataset)
    return bool(ds) and (ROOT / ds["transaction_path"]).is_file()


def resolve_source(source: str, project_config: Mapping[str, Any]) -> str:
    """Resolve ``auto`` to ``paysim``; never fall back to synthetic data silently.

    A missing raw file raises instead of switching to the synthetic generator, so
    a run reported as real data can never be a synthetic one. Synthetic runs must
    be requested explicitly with ``source="synthetic"``.
    """
    resolved = "paysim" if source == "auto" else source
    if resolved in ("paysim", "ieee_cis") and not raw_dataset_available(resolved, project_config):
        path = project_config.get("datasets", {}).get(resolved, {}).get("transaction_path", "data/raw/...")
        raise FileNotFoundError(
            f"raw {resolved} file not found: {ROOT / path}\n"
            f"Download the dataset and place it there (see README section 10), "
            f"or pass --source synthetic to run on generated data (not valid for paper results).")
    return resolved


def partition_windows(steps: np.ndarray, n_windows: int) -> List[EvaluationWindow]:
    """Step-aligned chronological windows with approximately equal row counts.

    ``steps`` must be sorted. Window boundaries never split a step, so a
    transaction batch that shares a timestamp is scored in one window.
    """
    steps = np.asarray(steps)
    if steps.size == 0:
        raise ValueError("cannot partition an empty test split")
    if np.any(np.diff(steps) < 0):
        raise ValueError("steps must be sorted chronologically")
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    uniq, counts = np.unique(steps, return_counts=True)
    n_u = uniq.size
    n_windows = min(n_windows, n_u)
    if n_windows < 1:
        raise ValueError(f"test split has {n_u} distinct steps; cannot form step-aligned windows")
    cum = np.concatenate([[0], np.cumsum(counts)])        # cum[j] = rows in steps[:j]
    total = int(cum[-1])
    bounds = [0]
    for k in range(1, n_windows):
        target = k * total / n_windows
        j = int(np.searchsorted(cum, target))
        if j > 0 and abs(cum[j - 1] - target) <= abs(cum[min(j, n_u)] - target):
            j -= 1
        lo = bounds[-1] + 1
        hi = n_u - (n_windows - k)
        bounds.append(int(min(max(j, lo), hi)))
    bounds.append(n_u)
    windows = []
    for k in range(n_windows):
        a, b = bounds[k], bounds[k + 1]
        windows.append(EvaluationWindow(k + 1, int(cum[a]), int(cum[b]), int(uniq[a]), int(uniq[b - 1])))
    return windows


def chronological_splits(frame: pd.DataFrame, feature_names: Sequence[str], n_windows: int = 50,
                         fractions: Tuple[float, float, float] = (0.45, 0.10, 0.10),
                         split_steps: Optional[Mapping[str, Mapping[str, int]]] = None,
                         source: str = "synthetic", canonical: Optional[pd.DataFrame] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> StreamSplits:
    """Split a step-sorted causal frame into train / calibration / operating / test.

    ``fractions`` are the shares of *distinct steps* given to train,
    calibration and operating; the remainder is the test period. When
    ``split_steps`` (``project_config['temporal_splits'][dataset]``) is given and
    every split is non-empty in ``frame`` it is used instead.
    """
    steps = frame["step"].to_numpy()
    if np.any(np.diff(steps) < 0):
        raise ValueError("feature frame must be sorted by step")
    parts: Dict[str, pd.DataFrame] = {}

    def step_range(lo: int, hi_exclusive: int) -> pd.DataFrame:
        # ``steps`` is sorted, so a step range is a contiguous row range: an iloc
        # slice shares the parent's buffers (copy-on-write) instead of copying
        # the frame like a boolean mask does. Same rows, same order.
        a = int(np.searchsorted(steps, lo, side="left"))
        b = int(np.searchsorted(steps, hi_exclusive, side="left"))
        return frame.iloc[a:b].reset_index(drop=True)

    if split_steps:
        for name in ("train", "calibration", "operating", "test"):
            spec = split_steps[name]
            parts[name] = step_range(int(spec["start_step"]), int(spec["end_step"]) + 1)
    if not parts or any(p.empty for p in parts.values()):
        uniq = np.unique(steps)
        f_tr, f_cal, f_op = fractions
        if not (f_tr > 0 and f_cal > 0 and f_op > 0 and f_tr + f_cal + f_op < 1.0):
            raise ValueError("fractions must be positive and leave a non-empty test period")
        n_u = uniq.size
        c1 = int(round(f_tr * n_u))
        c2 = c1 + max(1, int(round(f_cal * n_u)))
        c3 = c2 + max(1, int(round(f_op * n_u)))
        if not (0 < c1 < c2 < c3 < n_u):
            raise ValueError("stream has too few distinct steps for a four-way split")
        edges = [uniq[0], uniq[c1], uniq[c2], uniq[c3], uniq[-1] + 1]
        for name, lo, hi in zip(("train", "calibration", "operating", "test"), edges[:-1], edges[1:]):
            parts[name] = step_range(int(lo), int(hi))
    for name in ("train", "calibration"):
        y = parts[name]["label"].to_numpy()
        if y.min() == y.max():
            raise ValueError(f"{name} split must contain both classes")
    windows = partition_windows(parts["test"]["step"].to_numpy(), n_windows)
    return StreamSplits(parts["train"], parts["calibration"], parts["operating"], parts["test"],
                        list(feature_names), windows, source, canonical, dict(metadata or {}))


def prepare_splits(source: str = "auto", n_windows: int = 50, project_config: Optional[Mapping] = None,
                   max_rows: Optional[int] = None, synthetic_kwargs: Optional[Mapping[str, Any]] = None
                   ) -> StreamSplits:
    """Resolve the data source and return chronological splits with test windows."""
    if project_config is None:
        _, project_config = load_configs()
    source = resolve_source(source, project_config)
    if source == "synthetic":
        kwargs = dict(synthetic_kwargs or {})
        frame, names, canon = build_synthetic_stream(**kwargs)
        label = f"synthetic_paysim(drift={kwargs.get('drift_strength', 0.5)})"
        return chronological_splits(frame, names, n_windows, source=label, canonical=canon,
                                    metadata={"synthetic_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                                                                   for k, v in kwargs.items()}})
    if source in ("paysim", "ieee_cis"):
        frame, names, canon = build_real_stream(source, project_config, max_rows=max_rows)
        split_steps = project_config.get("temporal_splits", {}).get(source)
        return chronological_splits(frame, names, n_windows, split_steps=split_steps,
                                    source=source, canonical=canon,
                                    metadata={"max_rows": max_rows})
    raise ValueError("source must be 'auto', 'synthetic', 'paysim' or 'ieee_cis'")


def candidate_feed(project_config: Mapping[str, Any], hyperparams: Mapping[str, Any],
                   feature_names: Sequence[str], poison_rate: float = 0.10,
                   n_candidates: Optional[int] = None, current_step: Optional[int] = None,
                   seed: int = SEED) -> List[CandidateRule]:
    """Contaminated candidate-rule feed restricted to rules executable on the stream."""
    cfg = PerturbationConfig.from_configs(project_config, hyperparams)
    gen = PerturbationGenerator(cfg, seed=seed)
    feed = gen.generate_feed(poison_rate, n_candidates=n_candidates, current_step=current_step)
    available = set(feature_names) | {"tx_type"}
    return [r for r in feed.rules if all(c.feature in available for c in r.conditions)]


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def f_beta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    den = b2 * precision + recall
    return float((1.0 + b2) * precision * recall / den) if den > 0 else 0.0


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width-bin ECE ``sum_b (n_b / n) |mean(y_b) - mean(p_b)|``."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, p: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    """Ranking, threshold and calibration metrics for one window."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    yhat = np.asarray(yhat).astype(int)
    both = 0 < y.sum() < y.size
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": f_beta(precision, recall, 1.0),
        "f2": f_beta(precision, recall, 2.0),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(p, y),
    }


def select_operating_threshold(y: np.ndarray, p: np.ndarray, beta: float = 1.0) -> float:
    """F_beta-optimal decision threshold on the operating split (0.5 if undefined)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    if y.sum() == 0 or y.sum() == y.size:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    b2 = beta * beta
    den = b2 * prec + rec
    score = np.where(den > 0, (1 + b2) * prec * rec / np.where(den > 0, den, 1.0), 0.0)
    best = int(np.argmax(score))
    return float(thr[best])


def binary_policy_actions(yhat: np.ndarray) -> np.ndarray:
    """Approve / block policy of a binary detector."""
    return np.where(np.asarray(yhat).astype(int) == 1, int(Action.BLOCK), int(Action.APPROVE))


def make_loss(project_config: Mapping[str, Any]) -> AsymmetricLoss:
    return AsymmetricLoss(CostParameters.from_config(project_config))


# --------------------------------------------------------------------------- #
# Systems under evaluation                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class WindowOutput:
    """Scores and decisions of one system on one window."""

    p: np.ndarray
    yhat: np.ndarray
    actions: np.ndarray
    threshold: float
    latency_ms_per_txn: float
    extras: Dict[str, float] = field(default_factory=dict)


class PlattCalibrator:
    """Static sigmoid (Platt) calibration ``p = 1 / (1 + exp(-(a m + b)))`` on margins."""

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression
        s = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(labels).astype(int)
        if y.min() == y.max():
            raise ValueError("Platt calibration needs both classes")
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(s, y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        z = self.a * np.asarray(scores, dtype=np.float64) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


class MGECFMSystem:
    """MG-ECFM online pipeline (also parameterised for the Phase-7 ablations).

    Components: :class:`StackedFusionClassifier` (tabular + camouflage-filtered
    temporal GraphSAGE + DAE residual + phi_KB), the martingale e-process
    knowledge gate fed with the empirical firing outcomes of candidate rules,
    and the CS-ACRC pre-transaction router. All online state is updated only
    from *released* labels (see :meth:`release_feedback`).

    Parameters
    ----------
    gate_mode : ``"martingale"`` (Gate ON), ``"auto_admit"`` (Gate OFF: every
        candidate enters the KB) or ``"none"`` (no knowledge base).
    crc_mode : ``"adaptive"`` (CS-ACRC) or ``"static"`` (frozen thresholds).
    routing : ``"crc"`` (four actions) or ``"binary"`` (approve / block at tau).
    calibration : ``"isotonic"`` (Phase-2 default), ``"venn_abers"`` or
        ``"platt"`` (static sigmoid; tabular-only configurations).
    label_delay_windows : labels of window ``j`` are released before window
        ``j + 1 + label_delay_windows`` is scored.
    gate_cycle_steps : training / operating history is replayed through the
        gate in blocks of this many steps (one update cycle per block).
    """

    def __init__(self, feature_names: Sequence[str], hyperparams: Mapping[str, Any],
                 project_config: Mapping[str, Any], candidate_rules: Sequence[CandidateRule] = (),
                 gate_mode: str = "martingale", crc_mode: str = "adaptive", routing: str = "crc",
                 use_relational: bool = True, use_ae: bool = True, calibration: str = "isotonic",
                 label_delay_windows: int = 1, gate_cycle_steps: Optional[int] = None,
                 graph_epochs: Optional[int] = None, ae_epochs: Optional[int] = None,
                 name: str = PROPOSED_MODEL, seed: int = SEED) -> None:
        if gate_mode not in ("martingale", "auto_admit", "none"):
            raise ValueError("gate_mode must be 'martingale', 'auto_admit' or 'none'")
        if crc_mode not in ("adaptive", "static"):
            raise ValueError("crc_mode must be 'adaptive' or 'static'")
        if routing not in ("crc", "binary"):
            raise ValueError("routing must be 'crc' or 'binary'")
        if calibration not in ("isotonic", "venn_abers", "platt"):
            raise ValueError("calibration must be 'isotonic', 'venn_abers' or 'platt'")
        if calibration == "platt" and use_relational:
            raise ValueError("Platt calibration is only supported for the non-relational scorer")
        if label_delay_windows < 0:
            raise ValueError("label_delay_windows must be >= 0")
        from src.detection.fusion_classifier import StackedFusionClassifier
        from src.detection.tabular_engine import CalibratedXGBoostEngine, TabularEngineConfig
        from src.epistemic_gate.martingale_eprocess import MartingaleEProcessGate

        self.name = name
        self.seed = int(seed)
        self.hyperparams = copy.deepcopy(dict(hyperparams))
        self.project_config = project_config
        self.feature_names = list(feature_names)
        self.gate_mode, self.crc_mode, self.routing = gate_mode, crc_mode, routing
        self.calibration = calibration
        self.label_delay_windows = int(label_delay_windows)
        self.gate_cycle_steps = int(gate_cycle_steps or self.hyperparams.get("closed_loop", {})
                                    .get("rollback_check_interval_steps", 6))
        self.graph_epochs, self.ae_epochs = graph_epochs, ae_epochs
        self.candidates: List[CandidateRule] = list(candidate_rules)
        self._rules_by_id = {str(r.rule_id): r for r in self.candidates}
        use_kb = gate_mode != "none"
        tab_method = "isotonic" if calibration == "platt" else calibration
        tabular = CalibratedXGBoostEngine(TabularEngineConfig.from_hyperparams(
            self.hyperparams, calibration_method=tab_method))
        self.classifier = StackedFusionClassifier(
            self.feature_names, hyperparams=self.hyperparams, tabular=tabular,
            use_relational_embeddings=use_relational, use_ae_residual=use_ae,
            use_kb_triggers=use_kb, seed=self.seed)
        self.gate = MartingaleEProcessGate.from_config(self.hyperparams) if gate_mode == "martingale" else None
        controller = OnlineAdaptiveCRC.from_config(self.hyperparams, project_config, mode=crc_mode,
                                                   record_history=False)
        self.router = PreTransactionRouter.from_config(self.hyperparams, project_config,
                                                       controller=controller)
        self.loss_fn = controller.loss_fn
        self.threshold_: float = 0.5
        self.is_fitted_ = False
        self._pending: List[Tuple[int, pd.DataFrame, np.ndarray]] = []
        self.gate_log: List[Dict[str, Any]] = []
        self.fit_report: Dict[str, Any] = {}

    # ------------------------------------------------------------ knowledge base
    def active_rules(self) -> List[CandidateRule]:
        if self.gate_mode == "none":
            return []
        if self.gate_mode == "auto_admit":
            return list(self.candidates)
        return [self._rules_by_id[rid] for rid in self.gate.active_rule_ids]

    def rule_weights(self) -> Dict[str, float]:
        """Subjective-Logic expected probability for gated rules; claimed precision otherwise."""
        if self.gate_mode == "martingale":
            return {rid: float(self.gate.records[rid].opinion.expected_probability)
                    for rid in self.gate.active_rule_ids}
        return {str(r.rule_id): float(r.historical_precision) for r in self.active_rules()}

    def _sync_encoder(self) -> None:
        self.classifier.kb_encoder.set_rules(self.active_rules(), self.rule_weights())

    def _gate_cycle(self, frame: pd.DataFrame) -> None:
        """One gate update cycle from the verified labels of ``frame`` (chronological firings)."""
        if self.gate is None or frame.empty:
            return
        y = frame["label"].to_numpy().astype(int)
        outcomes: Dict[str, List[int]] = {}
        for rid, rec in self.gate.records.items():
            if rec.status not in ("quarantined", "active"):
                continue
            fires = np.asarray(rec.rule.evaluate(frame), dtype=bool)
            if fires.any():
                outcomes[rid] = y[fires].tolist()
        report = self.gate.run_update_cycle(outcomes)
        self.gate_log.append({"cycle": report.cycle, "n_observations": report.n_observations,
                              "promoted": len(report.promoted), "rolled_back": len(report.rolled_back),
                              "rejected": len(report.rejected),
                              "n_active": len(self.gate.active_rule_ids)})

    def _replay_history(self, frame: pd.DataFrame) -> None:
        steps = frame["step"].to_numpy()
        if steps.size == 0:
            return
        lo = int(steps.min())
        block = ((steps - lo) // self.gate_cycle_steps).astype(int)
        for b in np.unique(block):
            self._gate_cycle(frame.loc[block == b])

    # ------------------------------------------------------------------- fit
    def fit(self, splits: StreamSplits) -> "MGECFMSystem":
        """Train on ``train``/``calibration``; warm CS-ACRC and select ``tau`` on ``operating``."""
        set_global_seed(self.seed)
        t0 = time.perf_counter()
        if self.gate is not None:
            now = int(splits.train["step"].max())
            for rule in self.candidates:
                self.gate.submit_candidate(rule, now)
            self._replay_history(splits.train)
        self._sync_encoder()
        report = self.classifier.fit(splits.train, splits.calibration,
                                     rules=self.active_rules(), rule_weights=self.rule_weights(),
                                     graph_epochs=self.graph_epochs, ae_epochs=self.ae_epochs)
        if self.calibration == "platt":
            z_cal = self.classifier.transform(splits.calibration)
            margins = self.classifier.tabular.predict_margin(z_cal)
            self.classifier.tabular.calibrator = PlattCalibrator().fit(
                margins, splits.calibration["label"].to_numpy())
        if self.gate is not None:
            self._replay_history(splits.calibration)
            self._sync_encoder()
        # Operating split: past data with known labels, processed block by block
        # (score -> route -> feedback) so the controller and gate enter the test
        # period in their warmed-up state; tau is selected here.
        op = splits.operating
        p_all: List[np.ndarray] = []
        steps = op["step"].to_numpy()
        lo = int(steps.min()) if steps.size else 0
        block = ((steps - lo) // self.gate_cycle_steps).astype(int) if steps.size else np.array([], int)
        for b in np.unique(block):
            part = op.loc[block == b].reset_index(drop=True)
            p = self.classifier.predict_proba(part)
            actions = self._route(part, p)
            self._feedback(part, actions)
            p_all.append(p)
        p_op = np.concatenate(p_all) if p_all else np.zeros(0)
        self.threshold_ = select_operating_threshold(op["label"].to_numpy(), p_op)
        self.is_fitted_ = True
        self.fit_report = {"fit_seconds": time.perf_counter() - t0,
                           "n_fusion_features": report.get("n_fusion_features"),
                           "threshold": self.threshold_,
                           "n_active_rules_after_fit": len(self.active_rules()),
                           "crc_thresholds_after_operating": list(self.router.controller.thresholds)}
        return self

    # --------------------------------------------------------------- routing
    def _route(self, frame: pd.DataFrame, p: np.ndarray) -> np.ndarray:
        amounts = frame["amount"].to_numpy(dtype=np.float64)
        if self.routing == "binary":
            return binary_policy_actions(p >= self.threshold_)
        hits = (self.classifier.kb_encoder.rule_hit_matrix(frame).sum(axis=1)
                if self.classifier.use_kb else np.zeros(len(frame)))
        return np.fromiter((int(self.router.route(float(p[i]), 0.0, float(amounts[i]),
                                                  int(hits[i])).action)
                            for i in range(len(frame))), dtype=np.int64, count=len(frame))

    def _feedback(self, frame: pd.DataFrame, actions: np.ndarray) -> None:
        y = frame["label"].to_numpy().astype(int)
        amounts = frame["amount"].to_numpy(dtype=np.float64)
        if self.routing == "crc":
            self.router.controller.update_batch(actions, y, amounts)
        if self.gate is not None:
            self._gate_cycle(frame)
            self._sync_encoder()

    # ------------------------------------------------------------ evaluation
    def release_feedback(self, window_index: int) -> int:
        """Release every pending window whose label-availability time has come."""
        due = [item for item in self._pending if item[0] <= window_index]
        self._pending = [item for item in self._pending if item[0] > window_index]
        for _, frame, actions in sorted(due, key=lambda it: it[0]):
            self._feedback(frame, actions)
        return len(due)

    def score_window(self, frame: pd.DataFrame, window_index: int) -> WindowOutput:
        if not self.is_fitted_:
            raise RuntimeError("system is not fitted")
        self.release_feedback(window_index)
        lam = self.router.controller.thresholds
        n_rules = len(self.active_rules())
        t0 = time.perf_counter()
        p = np.clip(self.classifier.predict_proba(frame), 0.0, 1.0)
        actions = self._route(frame, p)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._pending.append((window_index + 1 + self.label_delay_windows, frame, actions))
        yhat = (p >= self.threshold_).astype(int)
        n = max(len(frame), 1)
        extras = {"n_active_rules": float(n_rules), "lambda_low": lam[0], "lambda_mid": lam[1],
                  "lambda_high": lam[2]}
        return WindowOutput(p, yhat, actions, self.threshold_, elapsed / n, extras)


class BaselineSystem:
    """Frozen Phase-6 baseline (BP1 - BP4) scored window by window."""

    def __init__(self, key: str, config: Optional[Mapping[str, Any]] = None, seed: int = SEED) -> None:
        from src.baselines import get_baseline_model, resolve_baseline_name
        self.name = resolve_baseline_name(key)
        self.model = get_baseline_model(self.name, config, seed=seed)
        self.seed = int(seed)
        self.feature_names: List[str] = []
        self.threshold_: float = 0.5
        self.is_fitted_ = False
        self.fit_report: Dict[str, Any] = {}

    def _X(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.feature_names].to_numpy(dtype=np.float64)

    def fit(self, splits: StreamSplits) -> "BaselineSystem":
        set_global_seed(self.seed)
        self.feature_names = list(splits.feature_names)
        # train + calibration feature matrix without concatenating the full frames
        # (which would also copy the string / carry columns).
        n_tr, n_cal = len(splits.train), len(splits.calibration)
        X_past = np.empty((n_tr + n_cal, len(self.feature_names)), dtype=np.float64)
        X_past[:n_tr] = self._X(splits.train)
        X_past[n_tr:] = self._X(splits.calibration)
        y_past = np.concatenate([splits.train["label"].to_numpy(),
                                 splits.calibration["label"].to_numpy()]).astype(int)
        t0 = time.perf_counter()
        self.model.fit(X_past, y_past)
        del X_past
        p_op = self.model.predict_proba(self._X(splits.operating))[:, 1]
        self.threshold_ = select_operating_threshold(splits.operating["label"].to_numpy(), p_op)
        self.is_fitted_ = True
        self.fit_report = {"fit_seconds": time.perf_counter() - t0, "threshold": self.threshold_,
                           **{k: v for k, v in self.model.fit_report_.items()
                              if isinstance(v, (int, float, str, bool))}}
        return self

    def release_feedback(self, window_index: int) -> int:
        return 0  # frozen after training (Menezes & Filho protocol)

    def score_window(self, frame: pd.DataFrame, window_index: int) -> WindowOutput:
        if not self.is_fitted_:
            raise RuntimeError("system is not fitted")
        t0 = time.perf_counter()
        p = np.clip(self.model.predict_proba(self._X(frame))[:, 1], 0.0, 1.0)
        elapsed = (time.perf_counter() - t0) * 1000.0
        yhat = (p >= self.threshold_).astype(int)
        return WindowOutput(p, yhat, binary_policy_actions(yhat), self.threshold_,
                            elapsed / max(len(frame), 1))


def clone_system(system: Any) -> Any:
    """Independent deep copy of a fitted system (including its SQLite knowledge store).

    ``sqlite3`` connections and locks cannot be deep-copied, so the gate's
    in-memory :class:`KnowledgeStore` is duplicated with the SQLite online
    backup API and substituted through the ``deepcopy`` memo. The clone and the
    original then evolve independently (used to branch one fitted pipeline into
    ablation arms and to run leakage audits on identical starting states).
    """
    import sqlite3
    import threading

    memo: Dict[int, Any] = {}
    gate = getattr(system, "gate", None)
    if gate is not None:
        store = gate.store
        if store.path != ":memory:":
            raise ValueError("clone_system supports in-memory knowledge stores only")
        new_store = copy.copy(store)
        conn = sqlite3.connect(":memory:", timeout=30.0, isolation_level=None,
                               check_same_thread=False)
        with store._lock:
            store._conn.backup(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        new_store._conn = conn
        new_store._lock = threading.RLock()
        memo[id(store)] = new_store
    return copy.deepcopy(system, memo)


# --------------------------------------------------------------------------- #
# Evaluator                                                                   #
# --------------------------------------------------------------------------- #
class DriftEvaluator:
    """Sequential window-by-window evaluation of several systems."""

    def __init__(self, splits: StreamSplits, systems: Mapping[str, Any],
                 loss_fn: AsymmetricLoss, labels: Optional[Mapping[str, str]] = None) -> None:
        self.splits = splits
        self.systems = dict(systems)
        self.loss_fn = loss_fn
        self.labels = dict(MODEL_LABELS)
        self.labels.update(labels or {})
        self.fit_reports: Dict[str, Dict[str, Any]] = {}

    def fit_all(self, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        for name, system in self.systems.items():
            t0 = time.perf_counter()
            system.fit(self.splits)
            self.fit_reports[name] = dict(system.fit_report)
            if verbose:
                print(f"[fit] {name:<18s} {time.perf_counter() - t0:7.1f}s  "
                      f"tau={system.threshold_:.4f}", flush=True)
        return self.fit_reports

    def window_record(self, name: str, window: EvaluationWindow, frame: pd.DataFrame,
                      out: WindowOutput) -> Dict[str, Any]:
        y = frame["label"].to_numpy().astype(int)
        amounts = frame["amount"].to_numpy(dtype=np.float64)
        losses = self.loss_fn.batch(out.actions, y, amounts)
        losses_bin = self.loss_fn.batch(binary_policy_actions(out.yhat), y, amounts)
        n = len(frame)
        rec: Dict[str, Any] = {
            "model": name, "model_label": self.labels.get(name, name), "window": window.index,
            "start_step": window.start_step, "end_step": window.end_step, "n_transactions": n,
            "n_fraud": int(y.sum()), "fraud_rate": float(y.mean()) if n else float("nan"),
            "threshold": float(out.threshold),
        }
        rec.update(classification_metrics(y, out.p, out.yhat))
        acts = np.asarray(out.actions)
        rec.update({
            "expected_monetary_loss": float(losses.mean()) if n else float("nan"),
            "eml_binary_policy": float(losses_bin.mean()) if n else float("nan"),
            "total_monetary_loss": float(losses.sum()),
            "intervention_rate": float((acts >= int(Action.STEP_UP)).mean()) if n else float("nan"),
            "step_up_rate": float((acts == int(Action.STEP_UP)).mean()) if n else float("nan"),
            "hold_rate": float((acts == int(Action.HOLD)).mean()) if n else float("nan"),
            "block_rate": float((acts == int(Action.BLOCK)).mean()) if n else float("nan"),
            "latency_ms_per_txn": float(out.latency_ms_per_txn),
            "data_source": self.splits.source,
        })
        for key in ("n_active_rules", "lambda_low", "lambda_mid", "lambda_high"):
            rec[key] = float(out.extras.get(key, float("nan")))
        return rec

    def evaluate(self, windows: Optional[Sequence[EvaluationWindow]] = None, verbose: bool = True
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process windows in chronological order; returns ``(metrics, predictions)``."""
        windows = list(self.splits.windows if windows is None else windows)
        if any(b.index <= a.index for a, b in zip(windows, windows[1:])):
            raise ValueError("windows must be processed in chronological order")
        records: List[Dict[str, Any]] = []
        preds: List[pd.DataFrame] = []
        for window in windows:
            frame = self.splits.window_frame(window)
            for name, system in self.systems.items():
                out = system.score_window(frame, window.index)
                records.append(self.window_record(name, window, frame, out))
                preds.append(pd.DataFrame({
                    "model": name, "window": window.index,
                    "tx_id": frame["tx_id"].to_numpy() if "tx_id" in frame.columns else
                    np.arange(window.row_start, window.row_end),
                    "step": frame["step"].to_numpy(), "y": frame["label"].to_numpy().astype(int),
                    "amount": frame["amount"].to_numpy(dtype=np.float64),
                    "p": out.p, "yhat": out.yhat, "action": np.asarray(out.actions).astype(int),
                }))
            if verbose and (window.index % 10 == 0 or window.index == windows[-1].index):
                last = {r["model"]: r["pr_auc"] for r in records if r["window"] == window.index}
                txt = "  ".join(f"{k}={v:.3f}" for k, v in last.items() if np.isfinite(v))
                print(f"[window {window.index:3d}/{windows[-1].index}] {txt}", flush=True)
        metrics = pd.DataFrame(records)
        metrics = metrics[[c for c in RESULT_COLUMNS if c in metrics.columns]]
        predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
        return metrics, predictions


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / median over windows per model for the headline metrics."""
    cols = [c for c in METRIC_COLUMNS if c in metrics.columns]
    agg = metrics.groupby("model", sort=False)[cols].agg(["mean", "std", "median"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def build_systems(splits: StreamSplits, hyperparams: Mapping[str, Any], project_config: Mapping[str, Any],
                  baseline_configs: Optional[Mapping[str, Any]], models: Sequence[str] = MODEL_ORDER,
                  poison_rate: float = 0.10, label_delay_windows: int = 1,
                  seed: int = SEED) -> Dict[str, Any]:
    """Instantiate MG-ECFM and the requested baselines (unfitted)."""
    systems: Dict[str, Any] = {}
    for name in models:
        if name == PROPOSED_MODEL:
            rules = candidate_feed(project_config, hyperparams, splits.feature_names, poison_rate,
                                   current_step=int(splits.train["step"].max()), seed=seed)
            systems[name] = MGECFMSystem(splits.feature_names, hyperparams, project_config, rules,
                                         label_delay_windows=label_delay_windows, seed=seed)
        else:
            cfg = None if baseline_configs is None else baseline_configs.get(name)
            systems[name] = BaselineSystem(name, cfg, seed=seed)
    return systems


def run_drift_evaluation(source: str = "auto", profile: str = "fast", n_windows: int = 50,
                         models: Sequence[str] = MODEL_ORDER, poison_rate: float = 0.10,
                         label_delay_windows: int = 1, max_rows: Optional[int] = None,
                         synthetic_kwargs: Optional[Mapping[str, Any]] = None,
                         hyperparams_override: Optional[Mapping[str, Any]] = None,
                         baseline_configs_override: Optional[Mapping[str, Any]] = None,
                         results_path: Optional[Path] = DEFAULT_RESULTS_PATH,
                         predictions_path: Optional[Path] = DEFAULT_PREDICTIONS_PATH,
                         manifest_path: Optional[Path] = DEFAULT_MANIFEST_PATH,
                         seed: int = SEED, verbose: bool = True
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """End-to-end Phase-7 drift evaluation; writes the CSV / JSON outputs."""
    set_global_seed(seed)
    hyperparams, project_config = load_configs()
    hp, baseline_cfgs = profile_settings(profile, hyperparams)
    if hyperparams_override is not None:
        hp = copy.deepcopy(dict(hyperparams_override))
    if baseline_configs_override is not None:
        baseline_cfgs = copy.deepcopy(dict(baseline_configs_override))
    unknown = [m for m in models if m not in MODEL_ORDER]
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {MODEL_ORDER}")
    t0 = time.perf_counter()
    splits = prepare_splits(source, n_windows, project_config, max_rows, synthetic_kwargs)
    if verbose:
        s = splits.summary()
        print(f"[data] {s['source']}: train={s['train']['transactions']} cal={s['calibration']['transactions']} "
              f"op={s['operating']['transactions']} test={s['test']['transactions']} "
              f"({s['test']['frauds']} frauds) in {s['n_windows']} windows", flush=True)
    systems = build_systems(splits, hp, project_config, baseline_cfgs, models, poison_rate,
                            label_delay_windows, seed)
    evaluator = DriftEvaluator(splits, systems, make_loss(project_config))
    fit_reports = evaluator.fit_all(verbose=verbose)
    metrics, predictions = evaluator.evaluate(verbose=verbose)
    manifest = {
        "protocol": "Menezes & Filho (2025) chronological drift evaluation",
        "n_windows": n_windows, "profile": profile, "seed": seed, "models": list(models),
        "label_delay_windows": label_delay_windows, "candidate_poison_rate": poison_rate,
        "threshold_selection": "F1-optimal on the operating split (no test data)",
        "baselines": "frozen after training; MG-ECFM online components updated from released labels only",
        "splits": splits.summary(), "fit_reports": _jsonable(fit_reports),
        "summary": _jsonable(summarise(metrics).to_dict(orient="records")),
        "runtime_seconds": time.perf_counter() - t0,
    }
    for path, writer in ((results_path, lambda p: metrics.to_csv(p, index=False)),
                         (predictions_path, lambda p: predictions.to_csv(p, index=False)),
                         (manifest_path, lambda p: Path(p).write_text(json.dumps(manifest, indent=2),
                                                                      encoding="utf-8"))):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            writer(path)
    if verbose:
        print(summarise(metrics)[["model", "pr_auc_mean", "roc_auc_mean", "f1_mean", "f2_mean",
                                  "brier_mean", "expected_monetary_loss_mean"]].to_string(index=False))
        if results_path is not None:
            print(f"[saved] {results_path}")
    return metrics, predictions, manifest


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="auto", choices=["auto", "synthetic", "paysim", "ieee_cis"])
    ap.add_argument("--profile", default="fast", choices=["fast", "full"])
    ap.add_argument("--n-windows", type=int, default=50)
    ap.add_argument("--models", nargs="+", default=list(MODEL_ORDER), choices=list(MODEL_ORDER))
    ap.add_argument("--poison-rate", type=float, default=0.10)
    ap.add_argument("--label-delay-windows", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--drift-strength", type=float, default=0.5,
                    help="synthetic camouflage drift strength in [0, 1]")
    ap.add_argument("--n-steps", type=int, default=400, help="synthetic stream length (steps)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = ap.parse_args(argv)
    run_drift_evaluation(source=args.source, profile=args.profile, n_windows=args.n_windows,
                         models=args.models, poison_rate=args.poison_rate,
                         label_delay_windows=args.label_delay_windows, max_rows=args.max_rows,
                         synthetic_kwargs={"drift_strength": args.drift_strength,
                                           "n_steps": args.n_steps, "seed": args.seed},
                         results_path=args.output,
                         predictions_path=args.output.with_name("drift_evaluation_predictions.csv"),
                         manifest_path=args.output.with_name("drift_evaluation_manifest.json"),
                         seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
