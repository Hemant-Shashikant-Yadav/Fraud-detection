"""
src/detection/fusion_classifier.py
==================================

Stacked tabular-relational-knowledge fusion scorer (MG-ECFM Stage 1A / C3).

For every transaction ``e_t = (u_t, v_t, x_t, A_t, tau_t)`` the fusion vector

    z_t = [ x_t^clean || h_{u_t} || h_{v_t} || r_AE(x_t) || phi_KB(e_t) ]

is scored by the cost-sensitive XGBoost engine and mapped to a calibrated
risk ``p_hat_t in [0, 1]``:

* ``x_t^clean``  leakage-free causal features from Phase 1;
* ``h_u, h_v``   camouflage-filtered temporal GraphSAGE embeddings;
* ``r_AE``       denoising-autoencoder reconstruction residual;
* ``phi_KB``     fixed-width summary of the *active* knowledge-base rules
  firing on ``e_t``: hit count, any-hit flag, belief-weighted score, max and
  mean weight. The width is independent of how many rules the Martingale
  gate admits or rolls back, so the stacker never needs re-shaping.

Ablation switches (``use_relational_embeddings``, ``use_ae_residual``,
``use_kb_triggers``) follow ``configs/hyperparams.json['fusion']``.

Training is strictly chronological: ``train < (val) < calibration`` by step.
The autoencoder and GraphSAGE are fit on the training split, embeddings are
produced by replaying the stream through the trained graph (memory carried
forward across splits), XGBoost is fit on training ``z_t`` and calibrated on
the calibration split.
"""
from __future__ import annotations

import json
import operator
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.causal_preprocessor import FORBIDDEN_FEATURE_COLUMNS, LeakageError
from src.detection.autoencoder_residual import AutoencoderConfig, AutoencoderResidualEngine
from src.detection.tabular_engine import CalibratedXGBoostEngine, TabularEngineConfig
from src.graph.camouflage_graphsage import CamouflageResistantGraphSAGE, GraphSAGEConfig

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPERPARAMS_PATH = ROOT / "configs" / "hyperparams.json"

_SCALAR_OPS: Mapping[str, Callable] = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt,
    "<=": operator.le, "==": operator.eq, "!=": operator.ne,
}


def load_hyperparams(path: Optional[Path] = None) -> Dict:
    """Read ``configs/hyperparams.json``."""
    with open(path or DEFAULT_HYPERPARAMS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Knowledge-base trigger encoder                                              #
# --------------------------------------------------------------------------- #
class KnowledgeBaseTriggerEncoder:
    """``phi_KB(e_t)``: fixed-width encoding of active-rule hits.

    A rule is any object with ``rule_id`` and ``evaluate(frame) -> bool mask``
    (e.g. :class:`src.data.perturbation_generator.CandidateRule`). Rules that
    expose ``conditions`` made of ``(feature, op, value)`` predicates are
    evaluated on a single event without building a DataFrame (fast path).

    Rule weights (e.g. Subjective-Logic expected probabilities supplied by the
    knowledge gate) default to ``rule.weight`` -> ``rule.historical_precision``
    -> ``1.0``.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "kb_hit_count", "kb_any_hit", "kb_weighted_score", "kb_max_weight", "kb_mean_weight",
    )

    def __init__(self, rules: Iterable = (), weights: Optional[Mapping[str, float]] = None) -> None:
        self.rules: List = []
        self.weights: Dict[str, float] = {}
        self.set_rules(rules, weights)

    @property
    def n_features(self) -> int:
        return len(self.FEATURE_NAMES)

    def set_rules(self, rules: Iterable, weights: Optional[Mapping[str, float]] = None) -> None:
        """Install the active KB snapshot (called after every admission / rollback)."""
        self.rules = list(rules)
        resolved: Dict[str, float] = {}
        for rule in self.rules:
            rid = str(getattr(rule, "rule_id"))
            if weights is not None and rid in weights:
                w = float(weights[rid])
            elif hasattr(rule, "weight"):
                w = float(getattr(rule, "weight"))
            elif hasattr(rule, "historical_precision"):
                w = float(getattr(rule, "historical_precision"))
            else:
                w = 1.0
            resolved[rid] = w
        self.weights = resolved

    def _weight_vector(self) -> np.ndarray:
        return np.asarray([self.weights[str(r.rule_id)] for r in self.rules], dtype=np.float64)

    def _summarise(self, hits: np.ndarray) -> np.ndarray:
        """``hits``: (n, R) bool -> (n, 5) feature block."""
        n = hits.shape[0]
        out = np.zeros((n, self.n_features), dtype=np.float64)
        if hits.shape[1] == 0:
            return out
        w = self._weight_vector()
        hf = hits.astype(np.float64)
        count = hf.sum(axis=1)
        weighted = hf @ w
        max_w = np.where(hits, w[None, :], 0.0).max(axis=1)
        out[:, 0] = count
        out[:, 1] = (count > 0).astype(np.float64)
        out[:, 2] = weighted
        out[:, 3] = max_w
        out[:, 4] = np.where(count > 0, weighted / np.maximum(count, 1.0), 0.0)
        return out

    def rule_hit_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.rules:
            return np.zeros((len(frame), 0), dtype=bool)
        return np.column_stack([np.asarray(r.evaluate(frame), dtype=bool) for r in self.rules])

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return self._summarise(self.rule_hit_matrix(frame))

    @staticmethod
    def _eval_condition(cond, event: Mapping[str, object]) -> bool:
        value = cond.value
        raw = event[cond.feature]
        op = _SCALAR_OPS[cond.op]
        if isinstance(value, str):
            return bool(op(str(raw), value))
        return bool(op(float(raw), float(value)))

    def transform_event(self, event: Mapping[str, object]) -> np.ndarray:
        """Single-event ``phi_KB`` without DataFrame construction where possible."""
        if not self.rules:
            return np.zeros(self.n_features, dtype=np.float64)
        hits = np.zeros((1, len(self.rules)), dtype=bool)
        frame: Optional[pd.DataFrame] = None
        for j, rule in enumerate(self.rules):
            conds = getattr(rule, "conditions", None)
            if conds is not None and all(hasattr(c, "op") for c in conds):
                hits[0, j] = all(self._eval_condition(c, event) for c in conds)
            else:
                if frame is None:
                    frame = pd.DataFrame([dict(event)])
                hits[0, j] = bool(np.asarray(rule.evaluate(frame), dtype=bool)[0])
        return self._summarise(hits)[0]


# --------------------------------------------------------------------------- #
# Fusion classifier                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class FusionColumns:
    """Column names of the Phase-1 processed frame."""

    src: str = "src"
    dst: str = "dst"
    amount: str = "amount"
    step: str = "step"
    label: str = "label"


class StackedFusionClassifier:
    """End-to-end Stage-2 scorer producing calibrated ``p_hat_t``."""

    def __init__(
        self,
        feature_names: Sequence[str],
        hyperparams: Optional[Mapping] = None,
        graph: Optional[CamouflageResistantGraphSAGE] = None,
        autoencoder: Optional[AutoencoderResidualEngine] = None,
        tabular: Optional[CalibratedXGBoostEngine] = None,
        kb_encoder: Optional[KnowledgeBaseTriggerEncoder] = None,
        use_relational_embeddings: Optional[bool] = None,
        use_ae_residual: Optional[bool] = None,
        use_kb_triggers: Optional[bool] = None,
        columns: Optional[FusionColumns] = None,
        seed: int = SEED,
    ) -> None:
        self.feature_names = list(feature_names)
        forbidden = set(FORBIDDEN_FEATURE_COLUMNS)
        leaked = [f for f in self.feature_names if f in forbidden or f == "label"]
        if leaked:
            raise LeakageError(f"Forbidden columns in fusion features: {leaked}")
        hp = dict(hyperparams) if hyperparams is not None else load_hyperparams()
        self.hyperparams = hp
        fusion = hp.get("fusion", {})
        self.use_relational = bool(fusion.get("use_relational_embeddings", True)
                                   if use_relational_embeddings is None else use_relational_embeddings)
        self.use_ae = bool(fusion.get("use_ae_residual", True)
                           if use_ae_residual is None else use_ae_residual)
        self.use_kb = bool(fusion.get("use_kb_triggers", True)
                           if use_kb_triggers is None else use_kb_triggers)
        self.columns = columns or FusionColumns()
        self.seed = int(seed)
        d = len(self.feature_names)
        self.graph = graph if graph is not None or not self.use_relational else \
            CamouflageResistantGraphSAGE(GraphSAGEConfig.from_hyperparams(hp, d), seed=self.seed)
        self.autoencoder = autoencoder if autoencoder is not None or not self.use_ae else \
            AutoencoderResidualEngine(AutoencoderConfig.from_hyperparams(hp, d), seed=self.seed)
        self.tabular = tabular or CalibratedXGBoostEngine(TabularEngineConfig.from_hyperparams(hp))
        self.kb_encoder = kb_encoder or KnowledgeBaseTriggerEncoder()
        self.fit_report: Dict[str, object] = {}

    # ------------------------------------------------------------ schema
    @property
    def fusion_feature_names(self) -> List[str]:
        names = list(self.feature_names)
        if self.use_relational:
            k = self.graph.out_dim
            names += [f"h_u_{i}" for i in range(k)] + [f"h_v_{i}" for i in range(k)]
        if self.use_ae:
            names.append("r_ae")
        if self.use_kb:
            names += list(KnowledgeBaseTriggerEncoder.FEATURE_NAMES)
        return names

    def assemble(self, x_clean: np.ndarray, h_src: Optional[np.ndarray], h_dst: Optional[np.ndarray],
                 r_ae: Optional[np.ndarray], kb: Optional[np.ndarray]) -> np.ndarray:
        """``z_t = [x_clean || h_u || h_v || r_AE || phi_KB]`` (row-wise)."""
        x = np.asarray(x_clean, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        parts = [x]
        if self.use_relational:
            parts += [np.asarray(h_src, dtype=np.float64).reshape(x.shape[0], -1),
                      np.asarray(h_dst, dtype=np.float64).reshape(x.shape[0], -1)]
        if self.use_ae:
            parts.append(np.asarray(r_ae, dtype=np.float64).reshape(x.shape[0], 1))
        if self.use_kb:
            parts.append(np.asarray(kb, dtype=np.float64).reshape(x.shape[0], -1))
        return np.concatenate(parts, axis=1)

    # ----------------------------------------------------------- helpers
    def _x(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise KeyError(f"Frame missing feature columns: {missing}")
        return frame[self.feature_names].to_numpy(dtype=np.float64)

    def _graph_args(self, frame: pd.DataFrame):
        c = self.columns
        dst = frame[c.dst].tolist() if c.dst in frame.columns else [""] * len(frame)
        return (self._x(frame), frame[c.amount].to_numpy(dtype=np.float64),
                frame[c.step].to_numpy(), frame[c.src].tolist(), dst)

    @staticmethod
    def _assert_chronological(*frames_and_names: Tuple[Optional[pd.DataFrame], str],
                              step_col: str = "step") -> None:
        prev_max, prev_name = None, None
        for frame, name in frames_and_names:
            if frame is None or frame.empty:
                continue
            steps = frame[step_col].to_numpy()
            if np.any(np.diff(steps) < 0):
                raise ValueError(f"{name} split is not sorted by step")
            if prev_max is not None and steps.min() <= prev_max:
                raise ValueError(f"{name} split overlaps / precedes the {prev_name} split")
            prev_max, prev_name = steps.max(), name

    def _components(self, frame: pd.DataFrame, record_labels: bool = False) -> np.ndarray:
        """Compute ``z_t`` for a chronological frame (advances graph memory)."""
        x = self._x(frame)
        h_src = h_dst = r = kb = None
        if self.use_relational:
            labels = frame[self.columns.label].to_numpy() if (
                record_labels and self.columns.label in frame.columns) else None
            out = self.graph.transform_stream(*self._graph_args(frame), labels=labels)
            h_src, h_dst = out["h_src"], out["h_dst"]
        if self.use_ae:
            r = self.autoencoder.residual(x)
        if self.use_kb:
            kb = self.kb_encoder.transform_frame(frame)
        return self.assemble(x, h_src, h_dst, r, kb)

    # --------------------------------------------------------------- fit
    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame,
            validation: Optional[pd.DataFrame] = None,
            rules: Optional[Iterable] = None, rule_weights: Optional[Mapping[str, float]] = None,
            graph_epochs: Optional[int] = None, ae_epochs: Optional[int] = None,
            verbose: bool = False) -> Dict[str, object]:
        """Fit all components chronologically and calibrate on ``calibration``."""
        c = self.columns
        self._assert_chronological((train, "train"), (validation, "validation"),
                                   (calibration, "calibration"), step_col=c.step)
        t_start = time.perf_counter()
        if rules is not None:
            self.kb_encoder.set_rules(rules, rule_weights)
        y_train = train[c.label].to_numpy().astype(int)
        y_cal = calibration[c.label].to_numpy().astype(int)
        x_train = self._x(train)
        report: Dict[str, object] = {}

        if self.use_ae:
            self.autoencoder.fit(x_train, y_train, epochs=ae_epochs, verbose=verbose)
            report["ae_best_epoch"] = self.autoencoder.best_epoch

        if self.use_relational:
            feats, amts, steps, src, dst = self._graph_args(train)
            self.graph.reset_state()
            self.graph.fit_normalizer(feats, amts)
            hist = self.graph.fit_stream(feats, amts, steps, src, dst, y_train,
                                         epochs=graph_epochs, verbose=verbose)
            report["graph_history"] = hist
            self.graph.reset_state()

        z_train = self._components(train, record_labels=True)
        z_val = y_val = None
        if validation is not None and not validation.empty:
            z_val = self._components(validation)
            y_val = validation[c.label].to_numpy().astype(int)
        z_cal = self._components(calibration)
        self.tabular.fit(z_train, y_train, z_cal, y_cal,
                         amounts_train=train[c.amount].to_numpy(dtype=np.float64),
                         x_val=z_val, y_val=y_val, feature_names=self.fusion_feature_names)
        report["tabular"] = dict(self.tabular.fit_report)
        report["n_fusion_features"] = len(self.fusion_feature_names)
        report["fit_seconds"] = time.perf_counter() - t_start
        self.fit_report = report
        return report

    # ----------------------------------------------------------- predict
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Fusion matrix ``Z`` for a later chronological frame."""
        return self._components(frame)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated ``p_hat_t`` for every row of a chronological frame."""
        return self.tabular.predict_proba(self.transform(frame))

    def score_event(self, event: Mapping[str, object], commit: bool = True) -> Dict[str, object]:
        """Real-time single-transaction scoring with per-component latency."""
        c = self.columns
        t0 = time.perf_counter()
        x = np.fromiter((float(event[n]) for n in self.feature_names), dtype=np.float64,
                        count=len(self.feature_names))
        timings: Dict[str, float] = {}
        h_src = h_dst = r = kb = None
        gnn_proba = None
        if self.use_relational:
            t = time.perf_counter()
            g = self.graph.embed_event(event[c.src], event.get(c.dst, ""), x,
                                       float(event[c.amount]), float(event[c.step]), commit=commit)
            h_src, h_dst, gnn_proba = g["h_src"], g["h_dst"], g["gnn_proba"]
            timings["graph_ms"] = (time.perf_counter() - t) * 1000.0
        if self.use_ae:
            t = time.perf_counter()
            r = np.asarray([self.autoencoder.residual_one(x)])
            timings["ae_ms"] = (time.perf_counter() - t) * 1000.0
        if self.use_kb:
            t = time.perf_counter()
            kb = self.kb_encoder.transform_event(event)
            timings["kb_ms"] = (time.perf_counter() - t) * 1000.0
        t = time.perf_counter()
        z = self.assemble(x, h_src, h_dst, r, kb)
        margin = float(self.tabular.predict_margin(z)[0])
        p_hat = float(self.tabular.calibrator.predict(np.asarray([margin]))[0])
        timings["tabular_ms"] = (time.perf_counter() - t) * 1000.0
        latency = (time.perf_counter() - t0) * 1000.0
        return {"p_hat": float(np.clip(p_hat, 0.0, 1.0)), "margin": margin,
                "gnn_proba": gnn_proba, "latency_ms": latency, "timings": timings}

    def latency_profile(self, events: Iterable[Mapping[str, object]], warmup: int = 10,
                        commit: bool = True) -> Dict[str, float]:
        """P50 / P95 / P99 / max end-to-end latency over ``events``."""
        events = list(events)
        for ev in events[:warmup]:
            self.score_event(ev, commit=False)
        lat = np.asarray([self.score_event(ev, commit=commit)["latency_ms"] for ev in events])
        return {"n": float(lat.size), "p50_ms": float(np.percentile(lat, 50)),
                "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99)),
                "max_ms": float(lat.max())}


def frame_to_events(frame: pd.DataFrame) -> List[Dict[str, object]]:
    """Row dictionaries for the real-time path (column -> scalar)."""
    return frame.to_dict(orient="records")
