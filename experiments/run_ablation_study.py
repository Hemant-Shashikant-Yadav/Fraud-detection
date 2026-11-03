"""
experiments/run_ablation_study.py
=================================

Phase 7 - systematic 5-point internal ablation of MG-ECFM.

All arms are evaluated on the *same* chronological stream, the *same* 50
test windows and the *same* contaminated candidate-rule feed as the drift
evaluation (:mod:`experiments.run_drift_eval`), with the same causal label
release (labels of window ``j`` reach the online components only before window
``j + 1 + label_delay_windows``).

======================  ====================================================================
Arm                     Configuration
======================  ====================================================================
B1_xgboost_only         Causal tabular features -> cost-sensitive XGBoost, static sigmoid
                        (Platt) calibration, approve / block at tau. No graph, no gate,
                        no recovery.
B2_graph_gate_off       Stacked XGBoost + camouflage-filtered temporal GraphSAGE + DAE
                        residual + phi_KB with the evidence gate OFF (Auto-Admit: every
                        candidate rule enters the KB), CS-ACRC routing, no recovery.
B3_static_crc           Full gated pipeline; thresholds learned on the operating split and
                        then frozen (static conformal risk control).
B3_adaptive_crc         Same fitted pipeline with online adaptive CS-ACRC.
B4_full_mg_ecfm         Gate ON (Subjective-Logic fusion + martingale e-process admission /
                        rollback), adaptive CS-ACRC and TD-TPF post-incident recovery.
B5_assist_off           B4 with the Analyst-Assist layer OFF.
B5_assist_on            B4 with the Analyst-Assist layer ON (decoupled, off the decision path).
======================  ====================================================================

B3_static_crc, B3_adaptive_crc, B4 and B5 branch from **one** fitted pipeline
(:func:`experiments.run_drift_eval.clone_system`), so their differences are
caused only by the ablated component. TD-TPF recovery is post-incident: it
does not change any authorisation decision, it changes the realised monetary
outcome of frauds that were not intercepted before settlement.

Analyst-Assist layer
--------------------
:class:`AnalystAssistLayer` is a retrieval-based case-evidence assistant: for
every transaction routed to Hold-and-Verify it retrieves the ``k`` most
similar *already verified* historical cases (cosine similarity over
signed-log standardised causal features; the index only ever contains cases
whose labels were released) and assembles an explanation (neighbour fraud
share, firing knowledge-base rules with their Subjective-Logic expected
probabilities, most anomalous features). It runs after the routing decision
has been committed, so it cannot alter decisions; the ablation certifies this
(decision agreement, identical scores) and quantifies its utility through a
simulated analyst triage of the hold queue (score-only rule ``p >= 0.5`` vs.
evidence-assisted rule ``(p + neighbour fraud share) / 2 >= 0.5``).

Output: ``reports/ablation_study_results.csv`` (one row per arm) plus
``reports/ablation_study_windows.csv`` (per-window metrics of every arm).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_drift_eval import (  # noqa: E402
    SEED,
    DriftEvaluator,
    MGECFMSystem,
    StreamSplits,
    WindowOutput,
    candidate_feed,
    classification_metrics,
    clone_system,
    load_configs,
    make_loss,
    prepare_splits,
    profile_settings,
    _jsonable,
)
from src.conformal.loss_functions import Action, AsymmetricLoss  # noqa: E402
from src.data.causal_preprocessor import set_global_seed  # noqa: E402

REPORTS_DIR = ROOT / "reports"
DEFAULT_RESULTS_PATH = REPORTS_DIR / "ablation_study_results.csv"
DEFAULT_WINDOWS_PATH = REPORTS_DIR / "ablation_study_windows.csv"

ARM_ORDER: Tuple[str, ...] = (
    "B1_xgboost_only", "B2_graph_gate_off", "B3_static_crc", "B3_adaptive_crc",
    "B4_full_mg_ecfm", "B5_assist_off", "B5_assist_on",
)
ARM_DESCRIPTIONS: Dict[str, str] = {
    "B1_xgboost_only": "XGBoost-only baseline: no graph, no gate, no recovery (static sigmoid calibration)",
    "B2_graph_gate_off": "Graph-enhanced detector, evidence gate OFF (Auto-Admit), CS-ACRC routing",
    "B3_static_crc": "Full gated pipeline with static conformal risk control (frozen thresholds)",
    "B3_adaptive_crc": "Full gated pipeline with online adaptive CS-ACRC",
    "B4_full_mg_ecfm": "Full MG-ECFM: Gate ON (SL + martingale), adaptive CS-ACRC, TD-TPF recovery",
    "B5_assist_off": "Full MG-ECFM with Analyst-Assist layer OFF",
    "B5_assist_on": "Full MG-ECFM with Analyst-Assist layer ON (decoupled)",
}
ARM_FLAGS: Dict[str, Dict[str, Any]] = {
    "B1_xgboost_only": dict(use_graph=False, use_ae=False, gate_mode="none", crc_mode="none",
                            routing="binary", calibration="platt", recovery=False, assist=False),
    "B2_graph_gate_off": dict(use_graph=True, use_ae=True, gate_mode="auto_admit", crc_mode="adaptive",
                              routing="crc", calibration="isotonic", recovery=False, assist=False),
    "B3_static_crc": dict(use_graph=True, use_ae=True, gate_mode="martingale", crc_mode="static",
                          routing="crc", calibration="isotonic", recovery=False, assist=False),
    "B3_adaptive_crc": dict(use_graph=True, use_ae=True, gate_mode="martingale", crc_mode="adaptive",
                            routing="crc", calibration="isotonic", recovery=False, assist=False),
    "B4_full_mg_ecfm": dict(use_graph=True, use_ae=True, gate_mode="martingale", crc_mode="adaptive",
                            routing="crc", calibration="isotonic", recovery=True, assist=False),
    "B5_assist_off": dict(use_graph=True, use_ae=True, gate_mode="martingale", crc_mode="adaptive",
                          routing="crc", calibration="isotonic", recovery=True, assist=False),
    "B5_assist_on": dict(use_graph=True, use_ae=True, gate_mode="martingale", crc_mode="adaptive",
                         routing="crc", calibration="isotonic", recovery=True, assist=True),
}


# --------------------------------------------------------------------------- #
# Analyst-Assist layer                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class AssistExplanation:
    tx_id: int
    p_hat: float
    neighbour_fraud_share: float
    neighbour_ids: List[int]
    neighbour_similarity: List[float]
    firing_rules: List[Dict[str, Any]]
    top_features: List[Tuple[str, float]]
    latency_ms: float


class AnalystAssistLayer:
    """Retrieval-based case-evidence assistant for the Hold-and-Verify queue.

    The case index is append-only and receives a case only when its verified
    label is released, so retrieval is causal. Explanations are produced after
    the routing decision is committed (shadow mode): nothing the assistant
    computes is ever read by the scorer, the router, the gate or the recovery.
    """

    def __init__(self, feature_names: Sequence[str], k: int = 10, explain_actions: Sequence[int] = (2,),
                 max_index_size: int = 50_000, top_features: int = 3) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.feature_names = list(feature_names)
        self.k = int(k)
        self.explain_actions = tuple(int(a) for a in explain_actions)
        self.max_index_size = int(max_index_size)
        self.n_top_features = int(top_features)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self._vectors = np.zeros((0, len(self.feature_names)), dtype=np.float64)
        self._labels = np.zeros(0, dtype=np.int64)
        self._ids = np.zeros(0, dtype=np.int64)
        self.explanations: List[AssistExplanation] = []

    @staticmethod
    def _slog(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, history: pd.DataFrame) -> "AnalystAssistLayer":
        """Standardiser on historical rows; index seeded with their verified labels."""
        s = self._slog(history[self.feature_names].to_numpy(dtype=np.float64))
        self.mean_ = s.mean(axis=0)
        std = s.std(axis=0)
        self.scale_ = np.where(std > 1e-9, std, 1.0)
        self._vectors = np.zeros((0, len(self.feature_names)), dtype=np.float64)
        self._labels = np.zeros(0, dtype=np.int64)
        self._ids = np.zeros(0, dtype=np.int64)
        self.ingest(history)
        return self

    def _embed(self, frame: pd.DataFrame) -> np.ndarray:
        z = (self._slog(frame[self.feature_names].to_numpy(dtype=np.float64)) - self.mean_) / self.scale_
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        return z / np.where(norm > 0, norm, 1.0)

    @property
    def index_size(self) -> int:
        return int(self._labels.size)

    def ingest(self, frame: pd.DataFrame) -> None:
        """Append verified cases (called only when labels are released)."""
        if frame.empty:
            return
        ids = frame["tx_id"].to_numpy(dtype=np.int64) if "tx_id" in frame.columns else \
            np.arange(len(frame), dtype=np.int64) + self._ids.size
        self._vectors = np.vstack([self._vectors, self._embed(frame)])[-self.max_index_size:]
        self._labels = np.concatenate([self._labels, frame["label"].to_numpy().astype(np.int64)])[
            -self.max_index_size:]
        self._ids = np.concatenate([self._ids, ids])[-self.max_index_size:]

    def explain(self, frame: pd.DataFrame, out: WindowOutput, kb_encoder=None,
                rule_weights: Optional[Mapping[str, float]] = None) -> List[AssistExplanation]:
        """Explanations for every row whose committed action is in ``explain_actions``."""
        acts = np.asarray(out.actions).astype(int)
        rows = np.flatnonzero(np.isin(acts, self.explain_actions))
        produced: List[AssistExplanation] = []
        if rows.size == 0 or self.index_size == 0:
            return produced
        sub = frame.iloc[rows]
        q = self._embed(sub)
        hits = (kb_encoder.rule_hit_matrix(sub) if kb_encoder is not None and kb_encoder.rules
                else np.zeros((len(sub), 0), dtype=bool))
        z_raw = (self._slog(sub[self.feature_names].to_numpy(dtype=np.float64)) - self.mean_) / self.scale_
        k = min(self.k, self.index_size)
        for j, i in enumerate(rows):
            t0 = time.perf_counter()
            sims = self._vectors @ q[j]
            nn = np.argpartition(-sims, k - 1)[:k]
            nn = nn[np.argsort(-sims[nn])]
            share = float((self._labels[nn].sum() + 0.5) / (k + 1.0))      # Jeffreys-smoothed
            rules = []
            if hits.shape[1]:
                for r_idx in np.flatnonzero(hits[j]):
                    rule = kb_encoder.rules[int(r_idx)]
                    rid = str(rule.rule_id)
                    rules.append({"rule_id": rid, "predicate": rule.describe(),
                                  "expected_probability": float((rule_weights or {}).get(rid, float("nan")))})
            order = np.argsort(-np.abs(z_raw[j]))[:self.n_top_features]
            feats = [(self.feature_names[int(f)], float(z_raw[j, int(f)])) for f in order]
            produced.append(AssistExplanation(
                int(sub["tx_id"].iloc[j]) if "tx_id" in sub.columns else int(i), float(out.p[i]),
                share, [int(v) for v in self._ids[nn]], [float(v) for v in sims[nn]], rules, feats,
                (time.perf_counter() - t0) * 1000.0))
        self.explanations.extend(produced)
        return produced


class AssistedSystem:
    """Wraps a fitted MG-ECFM system with the decoupled Analyst-Assist layer.

    The core system is called first and its output is returned untouched; the
    assistant then consumes the committed output. The assistant's own label
    release mirrors the core system's delay (causal case index).
    """

    def __init__(self, core: MGECFMSystem, assist: AnalystAssistLayer) -> None:
        self.core = core
        self.assist = assist
        self.name = core.name
        self._pending: List[Tuple[int, pd.DataFrame]] = []
        self.assist_latency_ms: List[float] = []
        self.triage: List[Tuple[int, float, float]] = []    # (y, p_hat, neighbour share)

    @property
    def threshold_(self) -> float:
        return self.core.threshold_

    @property
    def fit_report(self) -> Dict[str, Any]:
        return self.core.fit_report

    def fit(self, splits: StreamSplits) -> "AssistedSystem":
        if not self.core.is_fitted_:
            self.core.fit(splits)
        return self

    def release_feedback(self, window_index: int) -> int:
        due = [f for w, f in self._pending if w <= window_index]
        self._pending = [(w, f) for w, f in self._pending if w > window_index]
        for frame in due:
            self.assist.ingest(frame)
        return len(due)

    def score_window(self, frame: pd.DataFrame, window_index: int) -> WindowOutput:
        out = self.core.score_window(frame, window_index)          # decision committed here
        self.release_feedback(window_index)
        t0 = time.perf_counter()
        expl = self.assist.explain(frame, out, self.core.classifier.kb_encoder, self.core.rule_weights())
        self.assist_latency_ms.append((time.perf_counter() - t0) * 1000.0)
        y = frame["label"].to_numpy().astype(int)
        pos = {int(t): i for i, t in enumerate(frame["tx_id"].to_numpy())}
        for e in expl:
            i = pos[e.tx_id]
            self.triage.append((int(y[i]), e.p_hat, e.neighbour_fraud_share))
        self._pending.append((window_index + 1 + self.core.label_delay_windows, frame))
        return out


# --------------------------------------------------------------------------- #
# TD-TPF recovery accounting                                                  #
# --------------------------------------------------------------------------- #
def recovery_accounting(predictions: pd.DataFrame, canonical: Optional[pd.DataFrame],
                        hyperparams: Mapping[str, Any], project_config: Mapping[str, Any],
                        loss_fn: AsymmetricLoss, alert_delay_hours: float = 1.0,
                        schema: str = "paysim", seed: int = SEED) -> Dict[str, float]:
    """Post-incident TD-TPF recovery for frauds that were not intercepted pre-settlement.

    * ``STEP_UP`` frauds bypass the challenge with probability ``beta_bypass``;
      the bypass raises an immediate alert (analyst triage after
      ``alert_delay_hours``), so the expected recovered value is
      ``beta * R(e_0, delay)``.
    * ``APPROVE`` frauds are only discovered through the verification channel
      (customer dispute / bank chargeback latency of
      :class:`src.closed_loop.verification_queue.LatencyModel`, 1 - 14 days), so
      the 24 h TD-TPF window has usually expired.
    * ``HOLD`` / ``BLOCK`` frauds are intercepted before settlement.
    * Frauds whose first hop already leaves the ledger (e.g. PaySim
      ``CASH_OUT``) cannot be traced by TD-TPF; they are counted as
      ``n_untraceable`` with zero recovery (``cash_out_recovery_rate = 0``).

    Returns totals in monetary units: fraud value, value intercepted before
    settlement, value recovered after settlement, operational / collateral
    recovery cost, and the derived Prevented-Loss Ratio and Recovery Rate.
    """
    from src.closed_loop.verification_queue import LatencyModel
    from src.recovery.action_simulator import RecoverySimulator
    from src.recovery.tainted_flow import EDGE_INTERNAL, EdgeSchema, TaintedFlowTracker

    beta = loss_fn.costs.beta_bypass
    y = predictions["y"].to_numpy().astype(int)
    acts = predictions["action"].to_numpy().astype(int)
    amt = predictions["amount"].to_numpy(dtype=np.float64)
    fraud = y == 1
    fraud_value = float(amt[fraud].sum())
    intercepted = float(amt[fraud & (acts >= int(Action.HOLD))].sum()
                        + (1.0 - beta) * amt[fraud & (acts == int(Action.STEP_UP))].sum())
    out = {"fraud_value": fraud_value, "intercepted_value": intercepted, "recovered_value": 0.0,
           "recovery_cost": 0.0, "n_recovery_cases": 0.0, "n_recovered_within_window": 0.0,
           "n_untraceable": 0.0}
    eligible = fraud & (acts <= int(Action.STEP_UP))
    unprevented = float(amt[fraud & (acts == int(Action.APPROVE))].sum()
                        + beta * amt[fraud & (acts == int(Action.STEP_UP))].sum())
    if canonical is not None and eligible.any() and {"src", "dst", "event_time"} <= set(canonical.columns):
        tracker = TaintedFlowTracker.from_frame(canonical, "canonical", schema=EdgeSchema.by_name(schema),
                                                **TaintedFlowTracker.config_kwargs(hyperparams, project_config))
        sim = RecoverySimulator.from_config(tracker, hyperparams, project_config)
        latency = LatencyModel.from_config(hyperparams, project_config)
        rng = np.random.default_rng([seed, 24])
        tx_ids = predictions["tx_id"].to_numpy(dtype=np.int64)
        for i in np.flatnonzero(eligible):
            stepped = acts[i] == int(Action.STEP_UP)
            delay = alert_delay_hours if stepped else latency.sample(1, rng)[0]
            edge = tracker.edge(int(tx_ids[i]))
            out["n_recovery_cases"] += 1.0
            if tracker.schema.classify(edge, tracker.external_accounts) != EDGE_INTERNAL or edge.amount <= 0:
                out["n_untraceable"] += 1.0
                continue
            res = sim.simulate(edge, edge.tau + float(delay))
            scale = beta if stepped else 1.0
            out["recovered_value"] += scale * res.recovered
            out["recovery_cost"] += scale * (res.operational_cost + res.collateral_penalty)
            out["n_recovered_within_window"] += float(res.within_window and res.recovered > 0)
    out["unprevented_value"] = unprevented
    out["recovery_rate_pct"] = 100.0 * out["recovered_value"] / unprevented if unprevented > 0 else 0.0
    out["prevented_loss_ratio_pct"] = (100.0 * (intercepted + out["recovered_value"]) / fraud_value
                                       if fraud_value > 0 else float("nan"))
    return out


# --------------------------------------------------------------------------- #
# Knowledge-base diagnostics                                                  #
# --------------------------------------------------------------------------- #
def kb_diagnostics(system: Any, test: pd.DataFrame, p0: float, min_firings: int = 5) -> Dict[str, float]:
    """KB size and empirical (test-period) quality of admitted rules."""
    if not isinstance(system, MGECFMSystem) or system.gate_mode == "none":
        return {"kb_rules_active_end": float("nan"), "kb_rules_ever_admitted": float("nan"),
                "kb_admitted_bad_rate": float("nan"), "kb_active_mean_precision": float("nan")}
    if system.gate_mode == "auto_admit":
        ever = list(system.candidates)
    else:
        ever = [rec.rule for rec in system.gate.records.values() if rec.n_admissions > 0]
    y = test["label"].to_numpy().astype(int)

    def precision(rule) -> Optional[float]:
        fires = np.asarray(rule.evaluate(test), dtype=bool)
        return float(y[fires].mean()) if fires.sum() >= min_firings else None

    ever_prec = [p for p in (precision(r) for r in ever) if p is not None]
    active_prec = [p for p in (precision(r) for r in system.active_rules()) if p is not None]
    return {"kb_rules_active_end": float(len(system.active_rules())),
            "kb_rules_ever_admitted": float(len(ever)),
            "kb_admitted_bad_rate": float(np.mean([p <= p0 for p in ever_prec])) if ever_prec else float("nan"),
            "kb_active_mean_precision": float(np.mean(active_prec)) if active_prec else float("nan")}


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #
def pooled_metrics(predictions: pd.DataFrame, loss_fn: AsymmetricLoss) -> Dict[str, float]:
    y = predictions["y"].to_numpy().astype(int)
    p = predictions["p"].to_numpy(dtype=np.float64)
    yhat = predictions["yhat"].to_numpy().astype(int)
    acts = predictions["action"].to_numpy().astype(int)
    amt = predictions["amount"].to_numpy(dtype=np.float64)
    losses = loss_fn.batch(acts, y, amt)
    ell = np.minimum(losses / loss_fn.costs.loss_normalizer, loss_fn.costs.loss_cap)
    m = classification_metrics(y, p, yhat)
    m.update({
        "expected_monetary_loss": float(losses.mean()),
        "total_monetary_loss": float(losses.sum()),
        "normalized_risk": float(ell.mean()),
        "intervention_rate": float((acts >= int(Action.STEP_UP)).mean()),
        "step_up_rate": float((acts == int(Action.STEP_UP)).mean()),
        "hold_rate": float((acts == int(Action.HOLD)).mean()),
        "block_rate": float((acts == int(Action.BLOCK)).mean()),
        "false_alarm_ratio": float(((acts >= int(Action.HOLD)) & (y == 0)).sum()
                                   / max(int((acts >= int(Action.HOLD)).sum()), 1)),
    })
    return m


def triage_accuracy(triage: Sequence[Tuple[int, float, float]]) -> Tuple[float, float, int]:
    """(score-only accuracy, evidence-assisted accuracy, queue size) on the hold queue."""
    if not triage:
        return float("nan"), float("nan"), 0
    arr = np.asarray(triage, dtype=np.float64)
    y, p, share = arr[:, 0].astype(int), arr[:, 1], arr[:, 2]
    unassisted = float(((p >= 0.5).astype(int) == y).mean())
    assisted = float((((p + share) / 2.0 >= 0.5).astype(int) == y).mean())
    return unassisted, assisted, int(arr.shape[0])


@dataclass
class ArmRun:
    name: str
    system: Any
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    fit_seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)


def build_arm_system(arm: str, splits: StreamSplits, hp: Mapping[str, Any], pc: Mapping[str, Any],
                     rules: Sequence, label_delay_windows: int, seed: int) -> MGECFMSystem:
    f = ARM_FLAGS[arm]
    return MGECFMSystem(splits.feature_names, hp, pc, rules if f["gate_mode"] != "none" else (),
                        gate_mode=f["gate_mode"],
                        crc_mode="adaptive" if f["crc_mode"] == "none" else f["crc_mode"],
                        routing=f["routing"], use_relational=f["use_graph"], use_ae=f["use_ae"],
                        calibration=f["calibration"], label_delay_windows=label_delay_windows,
                        name=arm, seed=seed)


def run_ablation_study(source: str = "auto", profile: str = "fast", n_windows: int = 50,
                       poison_rate: float = 0.10, label_delay_windows: int = 1,
                       max_rows: Optional[int] = None,
                       synthetic_kwargs: Optional[Mapping[str, Any]] = None,
                       splits: Optional[StreamSplits] = None, assist_k: int = 10,
                       hyperparams_override: Optional[Mapping[str, Any]] = None,
                       results_path: Optional[Path] = DEFAULT_RESULTS_PATH,
                       windows_path: Optional[Path] = DEFAULT_WINDOWS_PATH,
                       seed: int = SEED, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, ArmRun]]:
    """Run B1 - B5 and return ``(summary_frame, runs)``; writes the CSV outputs."""
    set_global_seed(seed)
    hyperparams, pc = load_configs()
    hp, _ = profile_settings(profile, hyperparams)
    if hyperparams_override is not None:
        hp = copy.deepcopy(dict(hyperparams_override))
    if splits is None:
        splits = prepare_splits(source, n_windows, pc, max_rows, synthetic_kwargs)
    loss_fn = make_loss(pc)
    p0 = float(hp["martingale_eprocess"]["admission"]["p0"])
    rules = candidate_feed(pc, hp, splits.feature_names, poison_rate,
                           current_step=int(splits.train["step"].max()), seed=seed)

    # ---- fit the three independent pipelines ---------------------------------
    fitted: Dict[str, Tuple[MGECFMSystem, float]] = {}
    for arm in ("B1_xgboost_only", "B2_graph_gate_off", "B4_full_mg_ecfm"):
        system = build_arm_system(arm, splits, hp, pc, rules, label_delay_windows, seed)
        t0 = time.perf_counter()
        system.fit(splits)
        fitted[arm] = (system, time.perf_counter() - t0)
        if verbose:
            print(f"[fit] {arm:<18s} {fitted[arm][1]:7.1f}s  tau={system.threshold_:.4f} "
                  f"active_rules={len(system.active_rules())}", flush=True)

    # ---- branch the full pipeline into the B3 / B4 / B5 arms -----------------
    full, full_fit = fitted["B4_full_mg_ecfm"]
    static = clone_system(full)
    static.name = "B3_static_crc"
    static.crc_mode = "static"
    static.router.controller.mode = "static"
    assisted_core = clone_system(full)
    assisted_core.name = "B5_assist_on"
    assist = AnalystAssistLayer(splits.feature_names, k=assist_k).fit(
        pd.concat([splits.train, splits.calibration, splits.operating], ignore_index=True))
    assisted = AssistedSystem(assisted_core, assist)

    systems = {"B1_xgboost_only": fitted["B1_xgboost_only"][0],
               "B2_graph_gate_off": fitted["B2_graph_gate_off"][0],
               "B3_static_crc": static,
               "B4_full_mg_ecfm": full,
               "B5_assist_on": assisted}
    labels = {k: ARM_DESCRIPTIONS[k] for k in systems}
    evaluator = DriftEvaluator(splits, systems, loss_fn, labels=labels)
    metrics, predictions = evaluator.evaluate(verbose=verbose)

    runs: Dict[str, ArmRun] = {}
    fit_s = {"B1_xgboost_only": fitted["B1_xgboost_only"][1],
             "B2_graph_gate_off": fitted["B2_graph_gate_off"][1]}
    source_arm = {"B1_xgboost_only": "B1_xgboost_only", "B2_graph_gate_off": "B2_graph_gate_off",
                  "B3_static_crc": "B3_static_crc", "B3_adaptive_crc": "B4_full_mg_ecfm",
                  "B4_full_mg_ecfm": "B4_full_mg_ecfm", "B5_assist_off": "B4_full_mg_ecfm",
                  "B5_assist_on": "B5_assist_on"}
    for arm in ARM_ORDER:
        src = source_arm[arm]
        m = metrics[metrics["model"] == src].copy()
        m["model"] = arm
        m["model_label"] = ARM_DESCRIPTIONS[arm]
        runs[arm] = ArmRun(arm, systems[src], m.reset_index(drop=True),
                           predictions[predictions["model"] == src].reset_index(drop=True),
                           fit_s.get(src, full_fit))

    # ---- summary rows --------------------------------------------------------
    ref_pred = runs["B5_assist_off"].predictions
    rows: List[Dict[str, Any]] = []
    for arm in ARM_ORDER:
        run = runs[arm]
        flags = ARM_FLAGS[arm]
        row: Dict[str, Any] = {"arm": arm, "description": ARM_DESCRIPTIONS[arm],
                               "evaluated_run": source_arm[arm], **flags}
        row.update(pooled_metrics(run.predictions, loss_fn))
        prw = run.metrics["pr_auc"].to_numpy(dtype=np.float64)
        row["pr_auc_window_mean"] = float(np.nanmean(prw))
        row["pr_auc_window_std"] = float(np.nanstd(prw, ddof=1)) if np.isfinite(prw).sum() > 1 else float("nan")
        row["f2_window_mean"] = float(np.nanmean(run.metrics["f2"].to_numpy(dtype=np.float64)))
        row["eml_window_mean"] = float(np.nanmean(run.metrics["expected_monetary_loss"].to_numpy()))
        row["latency_ms_per_txn"] = float(run.metrics["latency_ms_per_txn"].mean())
        row["latency_ms_per_txn_p95"] = float(run.metrics["latency_ms_per_txn"].quantile(0.95))
        row["target_alpha"] = float(hp["conformal_crc"]["target_risk_alpha"]) if flags["routing"] == "crc" \
            else float("nan")
        schema = "generic" if "ieee" in splits.source else "paysim"
        rec = recovery_accounting(run.predictions, splits.canonical, hp, pc, loss_fn, schema=schema,
                                  seed=seed) if flags["recovery"] else None
        if rec is None:
            fraud = run.predictions["y"].to_numpy() == 1
            acts = run.predictions["action"].to_numpy()
            amt = run.predictions["amount"].to_numpy(dtype=np.float64)
            beta = loss_fn.costs.beta_bypass
            fv = float(amt[fraud].sum())
            inter = float(amt[fraud & (acts >= int(Action.HOLD))].sum()
                          + (1 - beta) * amt[fraud & (acts == int(Action.STEP_UP))].sum())
            rec = {"fraud_value": fv, "intercepted_value": inter, "recovered_value": 0.0,
                   "recovery_cost": 0.0, "recovery_rate_pct": 0.0,
                   "prevented_loss_ratio_pct": 100.0 * inter / fv if fv > 0 else float("nan")}
        row["fraud_value"] = rec["fraud_value"]
        row["intercepted_value"] = rec["intercepted_value"]
        row["recovered_value"] = rec["recovered_value"]
        row["recovery_rate_pct"] = rec["recovery_rate_pct"]
        row["n_recovery_cases"] = rec.get("n_recovery_cases", 0.0)
        row["n_untraceable_frauds"] = rec.get("n_untraceable", 0.0)
        row["prevented_loss_ratio_pct"] = rec["prevented_loss_ratio_pct"]
        row["net_monetary_loss"] = row["total_monetary_loss"] - rec["recovered_value"] + rec["recovery_cost"]
        row.update(kb_diagnostics(run.system.core if isinstance(run.system, AssistedSystem) else run.system,
                                  splits.test, p0))
        same = (run.predictions["tx_id"].to_numpy() == ref_pred["tx_id"].to_numpy()).all() \
            if len(run.predictions) == len(ref_pred) else False
        if arm == "B5_assist_on" and same:
            row["decision_agreement_with_assist_off"] = float(
                (run.predictions["action"].to_numpy() == ref_pred["action"].to_numpy()).mean())
            row["max_abs_score_diff_vs_assist_off"] = float(
                np.max(np.abs(run.predictions["p"].to_numpy() - ref_pred["p"].to_numpy())))
        else:
            row["decision_agreement_with_assist_off"] = float("nan")
            row["max_abs_score_diff_vs_assist_off"] = float("nan")
        if arm == "B5_assist_on":
            un, asd, nq = triage_accuracy(assisted.triage)
            lat = np.asarray(assisted.assist_latency_ms, dtype=np.float64)
            row.update({"assist_explanations": float(len(assist.explanations)),
                        "assist_hold_queue_size": float(nq),
                        "triage_accuracy_score_only": un, "triage_accuracy_assisted": asd,
                        "assist_latency_ms_per_window_mean": float(lat.mean()) if lat.size else float("nan"),
                        "assist_index_size_end": float(assist.index_size)})
        else:
            row.update({"assist_explanations": float("nan"), "assist_hold_queue_size": float("nan"),
                        "triage_accuracy_score_only": float("nan"), "triage_accuracy_assisted": float("nan"),
                        "assist_latency_ms_per_window_mean": float("nan"),
                        "assist_index_size_end": float("nan")})
        row["fit_seconds"] = run.fit_seconds
        row["n_test_transactions"] = int(len(run.predictions))
        row["n_windows"] = int(run.metrics["window"].nunique())
        row["data_source"] = splits.source
        rows.append(row)
    summary = pd.DataFrame(rows)
    per_window = pd.concat([runs[a].metrics for a in ARM_ORDER], ignore_index=True)
    if results_path is not None:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(results_path, index=False)
    if windows_path is not None:
        Path(windows_path).parent.mkdir(parents=True, exist_ok=True)
        per_window.to_csv(windows_path, index=False)
    if verbose:
        cols = ["arm", "pr_auc", "f1", "f2", "brier", "expected_monetary_loss", "net_monetary_loss",
                "prevented_loss_ratio_pct", "kb_rules_active_end", "decision_agreement_with_assist_off"]
        print(summary[cols].to_string(index=False))
        if results_path is not None:
            print(f"[saved] {results_path}")
    return summary, runs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MG-ECFM Phase-7 5-point ablation study")
    ap.add_argument("--source", default="auto", choices=["auto", "synthetic", "paysim", "ieee_cis"])
    ap.add_argument("--profile", default="fast", choices=["fast", "full"])
    ap.add_argument("--n-windows", type=int, default=50)
    ap.add_argument("--poison-rate", type=float, default=0.10)
    ap.add_argument("--label-delay-windows", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--drift-strength", type=float, default=0.5)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = ap.parse_args(argv)
    run_ablation_study(source=args.source, profile=args.profile, n_windows=args.n_windows,
                       poison_rate=args.poison_rate, label_delay_windows=args.label_delay_windows,
                       max_rows=args.max_rows,
                       synthetic_kwargs={"drift_strength": args.drift_strength, "n_steps": args.n_steps,
                                         "seed": args.seed},
                       results_path=args.output,
                       windows_path=args.output.with_name("ablation_study_windows.csv"), seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
