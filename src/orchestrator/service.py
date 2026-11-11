"""
src/orchestrator/service.py
===========================

Phase 8 - the live MG-ECFM service: one fitted pipeline plus the state needed
to score new transactions, explain decisions and run post-incident recovery.

The service wraps the Phase-7 :class:`experiments.run_drift_eval.MGECFMSystem`
(Stacked fusion scorer + martingale knowledge gate + CS-ACRC router) and keeps

* the **causal feature engineer** replayed over the whole history, so a new
  transaction is featurised exactly as it would be in the stream (only events
  with an earlier step contribute - Phase-1 contract);
* the **history** of scored transactions with their MG-ECFM decisions (used by
  the batch scorer, the mule-network visualiser and TD-TPF recovery);
* the **TD-TPF tracker** over the transaction ledger.

Scoring a single transaction is a *what-if* evaluation: the engineer, the graph
memory and the router are not advanced (``commit=False``), so repeated queries
are reproducible and never contaminate the live state.

Persistence: :meth:`MGECFMService.save` / :meth:`load` pickle the service; the
in-memory SQLite knowledge store is serialised with ``sqlite3.Connection.serialize``
(Python >= 3.11) and restored on load.
"""
from __future__ import annotations

import copy
import math
import os
import pickle
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.run_drift_eval import (
    SEED,
    MGECFMSystem,
    StreamSplits,
    candidate_feed,
    classification_metrics,
    fast_hyperparams,
    load_configs,
    prepare_splits,
)
from src.conformal.loss_functions import Action
from src.data.causal_preprocessor import (
    FORBIDDEN_FEATURE_COLUMNS,
    CausalFeatureConfig,
    CausalFeatureEngineer,
    set_global_seed,
    strip_leakage_columns,
)

ACTION_LABELS: Dict[int, str] = {int(Action.APPROVE): "Approve", int(Action.STEP_UP): "Step-Up",
                                 int(Action.HOLD): "Hold", int(Action.BLOCK): "Block"}
PAYSIM_RENAME: Dict[str, str] = {"nameOrig": "src", "nameDest": "dst", "type": "tx_type", "isFraud": "label"}
LABEL_ALIASES: Tuple[str, ...] = ("label", "isFraud", "is_fraud", "y")
SERVICE_FORMAT_VERSION = 1
REPLAY_CHUNK_ROWS = 250_000   # causal-engineer replay chunk (memory bound, result-neutral)


class _ServiceUnpickler(pickle.Unpickler):
    """Resolve classes pickled under ``__main__`` (a script run directly) to their real modules."""

    _FALLBACK_MODULES = ("experiments.run_drift_eval", "experiments.run_ablation_study",
                         "src.orchestrator.service")

    def find_class(self, module: str, name: str):
        if module == "__main__":
            import importlib
            for mod in self._FALLBACK_MODULES:
                found = getattr(importlib.import_module(mod), name, None)
                if found is not None:
                    return found
        return super().find_class(module, name)


def _tolerant_booster_setstate(original):
    """Booster unpickling that survives a config block the local XGBoost cannot parse.

    A model pickled on macOS/Linux stores ``max_cached_hist_node = 2**64 - 1`` in its config, which
    the Windows build (32-bit ``unsigned long``) rejects with "input stream corrupted". The
    ``Model`` section of the same buffer loads fine and holds every tree, so restore from that.
    """

    def setstate(self, state):
        import struct

        import xgboost as xgb
        try:
            return original(self, state)
        except xgb.core.XGBoostError:
            buf = bytes(state["handle"])
            marker = struct.pack(">Q", 5) + b"Model"      # UBJSON key "Model": 8-byte length + text
            key = buf.find(marker)
            if key < 0:
                raise
            booster = xgb.Booster()
            booster.load_model(bytearray(buf[key + len(marker):-1]))
            self.__dict__.update(booster.__dict__)
            booster.handle = None          # ownership moved to ``self``; avoid a double free

    return setstate


def utc_clock() -> str:
    """Default knowledge-store timestamp (picklable replacement for the store's lambda clock)."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class ServiceConfig:
    """How the live service is built.

    ``profile``: ``demo`` (small synthetic stream, seconds), ``fast`` (Phase-7
    fast profile on the evaluation stream) or ``full`` (published defaults).
    """

    source: str = "synthetic"
    profile: str = "demo"
    n_steps: int = 160
    tx_per_step: Tuple[int, int] = (40, 70)
    n_customers: int = 900
    n_merchants: int = 90
    fraud_rate: float = 0.03
    drift_strength: float = 0.5
    n_windows: int = 10
    poison_rate: float = 0.10
    label_delay_windows: int = 1
    max_rows: Optional[int] = None
    seed: int = SEED

    def synthetic_kwargs(self) -> Dict[str, Any]:
        return {"n_steps": int(self.n_steps), "tx_per_step": tuple(int(v) for v in self.tx_per_step),
                "n_customers": int(self.n_customers), "n_merchants": int(self.n_merchants),
                "fraud_rate": float(self.fraud_rate), "drift_strength": float(self.drift_strength),
                "seed": int(self.seed)}

    @classmethod
    def for_profile(cls, profile: str, source: str = "synthetic", **overrides) -> "ServiceConfig":
        if profile == "demo":
            cfg = cls(source=source, profile="demo")
        elif profile in ("fast", "full"):
            cfg = cls(source=source, profile=profile, n_steps=400, tx_per_step=(50, 90), n_customers=1500,
                      n_merchants=150, n_windows=50)
        else:
            raise ValueError("profile must be 'demo', 'fast' or 'full'")
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


def profile_hyperparams(profile: str, hyperparams: Mapping[str, Any]) -> Dict[str, Any]:
    """MG-ECFM detector hyper-parameters for a service profile."""
    if profile == "full":
        return copy.deepcopy(dict(hyperparams))
    hp = fast_hyperparams(hyperparams)
    if profile == "demo":
        hp["xgboost"].update({"n_estimators": 100, "early_stopping_rounds": 25})
        hp["autoencoder"].update({"epochs": 4})
    return hp


# --------------------------------------------------------------------------- #
# Service                                                                     #
# --------------------------------------------------------------------------- #
class MGECFMService:
    """Fitted MG-ECFM pipeline + live state for scoring, explanation and recovery."""

    def __init__(self, system: MGECFMSystem, splits: StreamSplits, history: pd.DataFrame,
                 config: ServiceConfig, hyperparams: Mapping[str, Any], project_config: Mapping[str, Any],
                 build_report: Optional[Dict[str, Any]] = None) -> None:
        self.system = system
        self.config = config
        self.hyperparams = copy.deepcopy(dict(hyperparams))
        self.project_config = copy.deepcopy(dict(project_config))
        self.feature_names: List[str] = list(splits.feature_names)
        self.source = splits.source
        self.split_summary = splits.summary()
        self.canonical = splits.canonical.reset_index(drop=True) if splits.canonical is not None else None
        if self.canonical is None:
            raise ValueError("the service requires the canonical transaction stream")
        self.history = history.reset_index(drop=True)
        self.build_report: Dict[str, Any] = dict(build_report or {})
        # Causal feature state at the end of the history (Phase-1 engineer replay).
        self.engineer = CausalFeatureEngineer(self._feature_config())
        # Only the engineer's end state is needed: replay in chunks and drop each
        # output instead of materialising the whole feature frame (streaming is
        # exact, see CausalFeatureEngineer guarantee 4).
        chunk = REPLAY_CHUNK_ROWS
        for _ in self.engineer.transform_stream(
                self.canonical.iloc[i:i + chunk] for i in range(0, len(self.canonical), chunk)):
            pass
        if list(self.engineer.feature_names) != self.feature_names:
            raise RuntimeError("feature schema of the replayed engineer differs from the fitted model")
        self.last_step = int(self.canonical["step"].max())
        # Reference distributions for explanation (training split only).
        train = splits.train
        x_train = train[self.feature_names].to_numpy(dtype=np.float64)
        # sign(x) * log1p(|x|) in one buffer (no sign / abs / log1p temporaries)
        slog = np.abs(x_train)
        np.log1p(slog, out=slog)
        np.negative(slog, out=slog, where=x_train < 0)
        self._feat_mu = slog.mean(axis=0)
        sd = slog.std(axis=0)
        del slog
        self._feat_sd = np.where(sd > 1e-9, sd, 1.0)
        clf = self.system.classifier
        self._ae_ref = (np.sort(clf.autoencoder.residual(x_train)) if clf.use_ae else np.zeros(0))
        self._fraud_accounts = self._labelled_accounts()
        self._tracker = None
        self._simulator = None

    # ------------------------------------------------------------ building
    def _feature_config(self) -> CausalFeatureConfig:
        if self.source in ("paysim", "ieee_cis"):
            return CausalFeatureConfig.from_project_config(self.project_config, self.source)
        return CausalFeatureConfig()

    @classmethod
    def build(cls, config: Optional[ServiceConfig] = None, verbose: bool = True,
              splits: Optional[StreamSplits] = None) -> "MGECFMService":
        """Fit MG-ECFM on the chronological splits and replay the test period prequentially."""
        config = config or ServiceConfig()
        set_global_seed(config.seed)
        hyperparams, project_config = load_configs()
        hp = profile_hyperparams(config.profile, hyperparams)
        t0 = time.perf_counter()
        if splits is None:
            kwargs = config.synthetic_kwargs() if config.source == "synthetic" else None
            splits = prepare_splits(config.source, config.n_windows, project_config, config.max_rows, kwargs)
        rules = candidate_feed(project_config, hp, splits.feature_names, config.poison_rate,
                               current_step=int(splits.train["step"].max()), seed=config.seed)
        system = MGECFMSystem(splits.feature_names, hp, project_config, rules,
                              label_delay_windows=config.label_delay_windows, seed=config.seed)
        t_fit = time.perf_counter()
        system.fit(splits)
        fit_s = time.perf_counter() - t_fit
        if verbose:
            print(f"[service] fitted MG-ECFM in {fit_s:.1f}s (tau={system.threshold_:.4f}, "
                  f"active rules={len(system.active_rules())})", flush=True)
        history = cls._replay_test(system, splits)
        report = {"profile": config.profile, "source": splits.source, "fit_seconds": fit_s,
                  "build_seconds": time.perf_counter() - t0, "threshold": system.threshold_,
                  "n_candidates": len(rules), "test_metrics": cls._history_metrics(history)}
        if verbose:
            m = report["test_metrics"]
            print(f"[service] replayed {len(history)} test transactions: PR-AUC={m['pr_auc']:.3f} "
                  f"F1={m['f1']:.3f} (build {report['build_seconds']:.1f}s)", flush=True)
        return cls(system, splits, cls._history_frame(splits, history), config, hp, project_config, report)

    @classmethod
    def from_evaluated(cls, system: MGECFMSystem, splits: StreamSplits, predictions: pd.DataFrame,
                       config: Optional[ServiceConfig] = None,
                       hyperparams: Optional[Mapping[str, Any]] = None) -> "MGECFMService":
        """Wrap a system that :class:`DriftEvaluator` already ran over every test window."""
        hp_all, pc = load_configs()
        pred = predictions[predictions["model"] == system.name] if "model" in predictions.columns else predictions
        report = {"profile": (config.profile if config else "evaluated"), "source": splits.source,
                  "fit_seconds": float(system.fit_report.get("fit_seconds", float("nan"))),
                  "threshold": system.threshold_, "n_candidates": len(system.candidates),
                  "test_metrics": cls._history_metrics(pred)}
        return cls(system, splits, cls._history_frame(splits, pred), config or ServiceConfig(),
                   hyperparams if hyperparams is not None else system.hyperparams, pc, report)

    @staticmethod
    def _replay_test(system: MGECFMSystem, splits: StreamSplits) -> pd.DataFrame:
        parts = []
        for w in splits.windows:
            frame = splits.window_frame(w)
            out = system.score_window(frame, w.index)
            parts.append(pd.DataFrame({"tx_id": frame["tx_id"].to_numpy(), "window": w.index,
                                       "step": frame["step"].to_numpy(),
                                       "y": frame["label"].to_numpy().astype(int),
                                       "amount": frame["amount"].to_numpy(dtype=np.float64),
                                       "p": out.p, "yhat": out.yhat,
                                       "action": np.asarray(out.actions).astype(int)}))
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _history_metrics(pred: pd.DataFrame) -> Dict[str, float]:
        y = pred["y"].to_numpy().astype(int)
        return classification_metrics(y, pred["p"].to_numpy(dtype=np.float64), pred["yhat"].to_numpy())

    @staticmethod
    def _history_frame(splits: StreamSplits, pred: pd.DataFrame) -> pd.DataFrame:
        canon = splits.canonical
        cols = [c for c in ("tx_id", "src", "dst", "tx_type") if c in canon.columns]
        out = pred[[c for c in ("tx_id", "window", "step", "y", "amount", "p", "yhat", "action")
                    if c in pred.columns]].merge(canon[cols], on="tx_id", how="left")
        out["action_label"] = out["action"].map(ACTION_LABELS)
        return out

    def _labelled_accounts(self) -> Dict[str, str]:
        """Accounts with *verified* fraud involvement (labels released before the last window)."""
        canon = self.canonical
        released_upto = self.last_step
        if "window" in self.history.columns and not self.history.empty:
            delay = int(self.config.label_delay_windows)
            last_released_window = int(self.history["window"].max()) - 1 - delay
            hw = self.history[self.history["window"] <= last_released_window]
            test_start = int(self.history["step"].min())
            released_upto = int(hw["step"].max()) if not hw.empty else test_start - 1
        fraud = canon[(canon["label"] == 1) & (canon["step"] <= released_upto)]
        out: Dict[str, str] = {}
        for v in fraud["dst"].astype(str):
            out[v] = "fraud beneficiary (mule)"
        for u in fraud["src"].astype(str):
            out.setdefault(u, "fraud victim")
        return out

    # ------------------------------------------------------------ features
    def _canonical_rows(self, rows: pd.DataFrame, step: int) -> pd.DataFrame:
        n = len(rows)
        return pd.DataFrame({
            "tx_id": np.arange(-n, 0, dtype=np.int64),
            "step": np.full(n, int(step), dtype=np.int64),
            "event_time": np.full(n, float(step) * 3600.0),
            "src": rows["src"].astype(str).to_numpy(),
            "dst": rows["dst"].astype(str).to_numpy(),
            "amount": rows["amount"].to_numpy(dtype=np.float64),
            "tx_type": rows["tx_type"].astype(str).str.upper().to_numpy(),
            "label": np.full(n, -1, dtype=np.int8),
        })

    def featurize(self, src: str, dst: str, amount: float, tx_type: str, step: Optional[int] = None
                  ) -> Tuple[Dict[str, Any], float]:
        """Causal feature event for one hypothetical transaction (engineer state untouched)."""
        if not (amount >= 0 and math.isfinite(float(amount))):
            raise ValueError("amount must be a finite non-negative number")
        step = int(self.last_step + 1 if step is None else step)
        if step <= self.last_step:
            raise ValueError(f"step must be after the last finalised step {self.last_step}")
        t0 = time.perf_counter()
        # State of just these two entities (exact; a deepcopy of every account is O(n_entities)).
        eng = self.engineer.restricted_copy([str(src), str(dst)])
        row = pd.DataFrame([{"src": str(src), "dst": str(dst), "amount": float(amount),
                             "tx_type": str(tx_type).upper()}])
        canon = self._canonical_rows(row, step)
        # Single complete step: validate + order-check + process directly (same code path as
        # transform(), without building the empty "completed steps" frame of transform_chunk).
        eng._validate_input(canon)
        eng._check_order(canon["step"].to_numpy(dtype=np.int64))
        feats = eng._process_frame(canon)
        ms = (time.perf_counter() - t0) * 1000.0
        rec = feats.iloc[0]
        event: Dict[str, Any] = {n: float(rec[n]) for n in self.feature_names}
        event.update({"src": str(src), "dst": str(dst), "amount": float(amount), "step": step,
                      "tx_type": str(tx_type).upper()})
        return event, ms

    # ------------------------------------------------------------- scoring
    def _fired_rules(self, event: Mapping[str, Any]) -> List[Any]:
        from src.detection.fusion_classifier import KnowledgeBaseTriggerEncoder
        enc = self.system.classifier.kb_encoder
        fired = []
        for rule in enc.rules:
            conds = getattr(rule, "conditions", ())
            if conds and all(KnowledgeBaseTriggerEncoder._eval_condition(c, event) for c in conds):
                fired.append(rule)
        return fired

    def score_transaction(self, src: str, dst: str, amount: float, tx_type: str = "TRANSFER",
                          step: Optional[int] = None, explain: bool = True) -> Dict[str, Any]:
        """Real-time authorisation of one transaction with reason attribution.

        ``latency_ms`` is the authorisation path (fusion scoring incl. graph
        embedding, DAE residual, phi_KB, calibrated XGBoost, plus CS-ACRC
        routing); causal featurisation and the explanation are timed separately.
        """
        event, feat_ms = self.featurize(src, dst, amount, tx_type, step)
        clf = self.system.classifier
        router = copy.copy(self.system.router)            # shallow: shares the live thresholds,
        router.action_counts = list(router.action_counts)  # but does not alter the live counters
        t0 = time.perf_counter()
        score = clf.score_event(event, commit=False)
        fired = self._fired_rules(event) if clf.use_kb else []
        decision = router.route(score["p_hat"], 0.0, float(amount), len(fired))
        latency = (time.perf_counter() - t0) * 1000.0
        weights = self.system.rule_weights()
        p = float(score["p_hat"])
        exp_loss = self.system.loss_fn.expected_losses(p, float(amount))
        out: Dict[str, Any] = {
            "src": str(src), "dst": str(dst), "amount": float(amount), "tx_type": str(tx_type).upper(),
            "step": int(event["step"]), "p_hat": p, "risk_score": float(decision.score),
            "gnn_proba": score.get("gnn_proba"), "margin": float(score["margin"]),
            "action": int(decision.action), "action_label": ACTION_LABELS[int(decision.action)],
            "route_reason": decision.reason, "thresholds": tuple(float(v) for v in decision.thresholds),
            "binary_threshold": float(self.system.threshold_),
            "latency_ms": latency, "component_ms": dict(score.get("timings", {})),
            "route_ms": float(decision.latency_ms), "featurize_ms": feat_ms,
            "expected_loss": {ACTION_LABELS[i]: float(v) for i, v in enumerate(exp_loss)},
            "fired_rules": [{"rule_id": str(r.rule_id), "predicate": r.describe(),
                             "weight": float(weights.get(str(r.rule_id), float("nan")))} for r in fired],
            "src_flag": self._fraud_accounts.get(str(src), ""),
            "dst_flag": self._fraud_accounts.get(str(dst), ""),
        }
        if explain:
            out.update(self._explain(event, score))
        return out

    def _explain(self, event: Mapping[str, Any], score: Mapping[str, Any]) -> Dict[str, Any]:
        """TreeSHAP attribution on the fusion vector + feature deviations (off the hot path)."""
        import xgboost as xgb
        clf = self.system.classifier
        x = np.fromiter((float(event[n]) for n in self.feature_names), dtype=np.float64,
                        count=len(self.feature_names))
        h_src = h_dst = r = kb = None
        if clf.use_relational:
            g = copy.deepcopy(clf.graph).embed_event(event["src"], event.get("dst", ""), x,
                                                     float(event["amount"]), float(event["step"]), commit=False)
            h_src, h_dst = g["h_src"], g["h_dst"]
        if clf.use_ae:
            r = np.asarray([clf.autoencoder.residual_one(x)])
        if clf.use_kb:
            kb = clf.kb_encoder.transform_event(event)
        z = clf.assemble(x, h_src, h_dst, r, kb).astype(np.float32)
        eng = clf.tabular
        contrib = np.asarray(eng.booster.predict(xgb.DMatrix(z), pred_contribs=True,
                                                 iteration_range=(0, int(eng.best_iteration) + 1)))[0]
        names = clf.fusion_feature_names
        bias = float(contrib[-1])
        groups: Dict[str, float] = {}
        for name, c in zip(names, contrib[:-1]):
            if name.startswith("h_u_"):
                key = "graph embedding of sender (h_u)"
            elif name.startswith("h_v_"):
                key = "graph embedding of receiver (h_v)"
            elif name == "r_ae":
                key = "autoencoder residual r_AE"
            elif name.startswith("kb_"):
                key = "knowledge-base triggers phi_KB"
            else:
                key = name
            groups[key] = groups.get(key, 0.0) + float(c)
        attrib = (pd.DataFrame({"component": list(groups), "shap_margin": list(groups.values())})
                  .assign(abs_=lambda d: d["shap_margin"].abs())
                  .sort_values("abs_", ascending=False).drop(columns="abs_").reset_index(drop=True))
        slog = np.sign(x) * np.log1p(np.abs(x))
        zdev = (slog - self._feat_mu) / self._feat_sd
        dev = (pd.DataFrame({"feature": self.feature_names, "value": x, "z_vs_train": zdev})
               .assign(abs_=lambda d: d["z_vs_train"].abs()).sort_values("abs_", ascending=False)
               .drop(columns="abs_").reset_index(drop=True))
        r_pct = (float(np.searchsorted(self._ae_ref, float(r[0])) / max(len(self._ae_ref), 1) * 100.0)
                 if r is not None and self._ae_ref.size else float("nan"))
        return {"attribution": attrib, "base_margin": bias, "feature_deviation": dev,
                "ae_residual": float(r[0]) if r is not None else float("nan"),
                "ae_residual_percentile": r_pct}

    def latency_benchmark(self, n: int = 200, warmup: int = 20, seed: int = SEED) -> Dict[str, float]:
        """Authorisation-path latency over ``n`` transactions resampled from the history."""
        rng = np.random.default_rng([seed, 5])
        hist = self.history.dropna(subset=["src", "dst"])
        idx = rng.choice(len(hist), size=min(n + warmup, len(hist)), replace=len(hist) < n + warmup)
        lat, feat = [], []
        for k, i in enumerate(idx):
            r = hist.iloc[int(i)]
            out = self.score_transaction(r["src"], r["dst"], float(r["amount"]), r["tx_type"], explain=False)
            if k >= warmup:
                lat.append(out["latency_ms"])
                feat.append(out["featurize_ms"])
        lat_a, feat_a = np.asarray(lat), np.asarray(feat)
        return {"n": float(lat_a.size), "p50_ms": float(np.percentile(lat_a, 50)),
                "p95_ms": float(np.percentile(lat_a, 95)), "p99_ms": float(np.percentile(lat_a, 99)),
                "max_ms": float(lat_a.max()), "mean_ms": float(lat_a.mean()),
                "featurize_p50_ms": float(np.percentile(feat_a, 50)),
                "featurize_p95_ms": float(np.percentile(feat_a, 95))}

    # ------------------------------------------------------------- batches
    @staticmethod
    def canonicalize_upload(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[str]]:
        """Normalise a PaySim-schema or canonical CSV; returns ``(canonical, labels, dropped_columns)``."""
        df = frame.copy()
        dropped = [c for c in FORBIDDEN_FEATURE_COLUMNS if c in df.columns]
        df = strip_leakage_columns(df)
        df = df.rename(columns={k: v for k, v in PAYSIM_RENAME.items() if k in df.columns and v not in df.columns})
        missing = [c for c in ("src", "dst", "amount", "tx_type") if c not in df.columns]
        if missing:
            raise ValueError(f"upload is missing columns {missing}; expected PaySim (type, amount, nameOrig, "
                             "nameDest[, step, isFraud]) or canonical (tx_type, amount, src, dst[, step, label])")
        labels = None
        for c in LABEL_ALIASES:
            if c in df.columns:
                labels = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int).to_numpy()
                break
        amounts = pd.to_numeric(df["amount"], errors="coerce")
        if amounts.isna().any() or (amounts < 0).any():
            raise ValueError("amount must be numeric and non-negative in every row")
        df["amount"] = amounts.astype(np.float64)
        if "step" in df.columns:
            df["step"] = pd.to_numeric(df["step"], errors="coerce")
            if df["step"].isna().any():
                raise ValueError("step must be numeric in every row")
            df["step"] = df["step"].astype(np.int64)
        return df, labels, dropped

    def score_frame(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Score an uploaded batch after the history (live state is not modified).

        Rows are ordered chronologically (stable within a step); upload steps are
        shifted so that the first one directly follows the history. Features are
        computed causally *within* the batch as well, so a later row of the
        batch sees the earlier ones.
        """
        canon_in, labels, dropped = self.canonicalize_upload(frame)
        n = len(canon_in)
        if n == 0:
            raise ValueError("the uploaded file contains no transactions")
        order = np.arange(n)
        if "step" in canon_in.columns:
            order = np.argsort(canon_in["step"].to_numpy(), kind="stable")
            raw_steps = canon_in["step"].to_numpy()[order]
            steps = raw_steps - raw_steps.min() + self.last_step + 1
        else:
            steps = np.full(n, self.last_step + 1, dtype=np.int64)
        rows = canon_in.iloc[order].reset_index(drop=True)
        canon = self._canonical_rows(rows, 0)
        canon["step"] = steps.astype(np.int64)
        canon["event_time"] = canon["step"].astype(np.float64) * 3600.0
        canon["tx_id"] = np.arange(n, dtype=np.int64) + 10_000_000
        t0 = time.perf_counter()
        # Only the batch's own entities are needed (exact, see restricted_copy).
        eng = self.engineer.restricted_copy(pd.unique(pd.concat([canon["src"], canon["dst"]]).astype(str)))
        feats = eng.transform(canon)
        clf = copy.deepcopy(self.system.classifier)
        p = np.clip(clf.predict_proba(feats), 0.0, 1.0)
        hits = (clf.kb_encoder.rule_hit_matrix(feats).sum(axis=1) if clf.use_kb else np.zeros(n))
        router = copy.deepcopy(self.system.router)
        decisions = [router.route(float(p[i]), 0.0, float(feats["amount"].iloc[i]), int(hits[i])) for i in range(n)]
        elapsed = time.perf_counter() - t0
        scored = frame.iloc[order].reset_index(drop=True).copy()
        scored["mg_ecfm_step"] = steps
        scored["p_hat"] = p
        scored["risk_score"] = [d.score for d in decisions]
        scored["action"] = [int(d.action) for d in decisions]
        scored["action_label"] = [ACTION_LABELS[int(d.action)] for d in decisions]
        scored["route_reason"] = [d.reason for d in decisions]
        scored["kb_rules_fired"] = hits.astype(int)
        scored["binary_flag"] = (p >= self.system.threshold_).astype(int)
        summary: Dict[str, Any] = {"n": n, "seconds": elapsed, "ms_per_txn": 1000.0 * elapsed / n,
                                   "dropped_leakage_columns": dropped,
                                   "actions": scored["action_label"].value_counts().to_dict(),
                                   "thresholds": tuple(float(v) for v in self.system.router.controller.thresholds)}
        if labels is not None:
            y = labels[order]
            ok = y >= 0
            if ok.any() and 0 < y[ok].sum() < ok.sum():
                summary["metrics"] = classification_metrics(y[ok], p[ok], scored["binary_flag"].to_numpy()[ok])
                acts = scored["action"].to_numpy()[ok]
                summary["fraud_intercepted_share"] = float((acts[y[ok] == 1] >= int(Action.HOLD)).mean())
        return scored, summary

    # --------------------------------------------------- camouflage graph
    def explain_graph(self, src: str, dst: str, amount: float, tx_type: str = "TRANSFER",
                      step: Optional[int] = None) -> Dict[str, Any]:
        """Two-hop neighbourhood of the transaction and the camouflage filter's keep / prune decisions."""
        import torch
        clf = self.system.classifier
        if not clf.use_relational:
            raise RuntimeError("the fitted pipeline has no relational component")
        event, _ = self.featurize(src, dst, amount, tx_type, step)
        x = np.fromiter((float(event[n]) for n in self.feature_names), dtype=np.float64,
                        count=len(self.feature_names))
        graph = copy.deepcopy(clf.graph)
        graph.eval()
        known_src = graph.memory.lookup(str(src)) is not None
        known_dst = graph.memory.lookup(str(dst)) is not None
        with torch.no_grad():
            u = graph.memory.register_one(str(src))
            v = graph.memory.register_one(str(dst))
            msg = graph.memory.build_message_input(x[None, :], np.asarray([float(amount)]))
            times = torch.tensor([float(event["step"])], dtype=torch.float64)
            batch = graph.build_subgraph(torch.tensor([u]), torch.tensor([v]), msg, times)
            filt = graph.similarity.filter_edges(batch.states, batch.edge_index, n_nodes=batch.states.shape[0])
            out = graph.forward_subgraph(batch)
        n_ev = batch.n_events
        keys: Dict[int, str] = {0: str(src), n_ev: str(dst)}
        roles: Dict[int, str] = {0: "sender", n_ev: "receiver"}
        for j, g in enumerate(batch.hist_global):
            keys[2 * n_ev + j] = str(graph.memory.key_of(g))
        ei = batch.edge_index.cpu().numpy()
        sim = filt.similarity.detach().cpu().numpy() if filt.similarity.numel() else np.zeros(0)
        keep = filt.keep_mask.detach().cpu().numpy() if filt.keep_mask.numel() else np.zeros(0, dtype=bool)
        hop: Dict[int, int] = {0: 0, n_ev: 0}
        anchor: Dict[int, int] = {}
        for k in range(ei.shape[1]):
            a, c = int(ei[0, k]), int(ei[1, k])
            if c in (0, n_ev) and a not in (0, n_ev):
                hop.setdefault(a, 1)
                anchor.setdefault(a, c)
        for k in range(ei.shape[1]):
            a, c = int(ei[0, k]), int(ei[1, k])
            if a not in hop:
                hop[a] = hop.get(c, 1) + 1
                anchor.setdefault(a, anchor.get(c, c))
        edges = []
        for k in range(ei.shape[1]):
            a, c = int(ei[0, k]), int(ei[1, k])
            kind = "transaction" if {a, c} == {0, n_ev} else f"hop-{max(hop.get(a, 1), hop.get(c, 0))}"
            edges.append({"neighbour": keys.get(a, str(a)), "centre": keys.get(c, str(c)),
                          "neighbour_local": a, "centre_local": c, "kind": kind,
                          "similarity": float(sim[k]) if k < sim.size else float("nan"),
                          "kept": bool(keep[k]) if k < keep.size else True})
        edges_df = pd.DataFrame(edges)
        nodes = []
        for loc, key in keys.items():
            nodes.append({"local": loc, "account": key,
                          "role": roles.get(loc, f"hop-{hop.get(loc, 1)} neighbour"),
                          "hop": hop.get(loc, 1), "attached_to": keys.get(anchor.get(loc, loc), key),
                          "verified_flag": self._fraud_accounts.get(key, ""),
                          "is_merchant": key.startswith("M")})
        nodes_df = pd.DataFrame(nodes).sort_values(["hop", "local"]).reset_index(drop=True)
        nb = edges_df[edges_df["kind"] != "transaction"] if not edges_df.empty else edges_df
        stats = {"n_nodes": int(len(nodes_df)), "n_edges": int(len(edges_df)),
                 "n_neighbour_edges": int(len(nb)),
                 "n_pruned": int((~nb["kept"]).sum()) if len(nb) else 0,
                 "kept_fraction": float(nb["kept"].mean()) if len(nb) else 1.0,
                 "gnn_proba": float(torch.sigmoid(out["logits"][0])),
                 "sender_known": known_src, "receiver_known": known_dst,
                 "similarity_threshold": float(getattr(graph.similarity.config, "threshold_p", float("nan"))),
                 "filter_mode": str(getattr(graph.similarity.config, "mode", "threshold"))}
        return {"nodes": nodes_df, "edges": edges_df, "stats": stats}

    # ------------------------------------------------------------- examples
    def example_transactions(self) -> Dict[str, Dict[str, Any]]:
        """Presets for the dashboard: a mule-network fraud, a legitimate payment, a cold-start account."""
        h = self.history.dropna(subset=["src", "dst"])
        out: Dict[str, Dict[str, Any]] = {}
        fraud = h[(h["y"] == 1) & (h["tx_type"] == "TRANSFER")]
        if fraud.empty:
            fraud = h[h["y"] == 1]
        if not fraud.empty:
            deg = pd.concat([self.canonical["src"], self.canonical["dst"]]).value_counts()
            f = fraud.assign(deg=fraud["dst"].map(deg).fillna(0)).sort_values(["deg", "amount"], ascending=False)
            r = f.iloc[0]
            out["Known mule-network transfer (verified fraud pattern)"] = {
                "src": r["src"], "dst": r["dst"], "amount": float(r["amount"]), "tx_type": r["tx_type"]}
        legit = h[(h["y"] == 0) & (h["tx_type"] == "PAYMENT")]
        if not legit.empty:
            r = legit.iloc[len(legit) // 2]
            out["Routine merchant payment"] = {"src": r["src"], "dst": r["dst"], "amount": float(r["amount"]),
                                               "tx_type": r["tx_type"]}
        any_src = str(h.iloc[0]["src"]) if not h.empty else "C000000001"
        out["Large transfer to a never-seen account"] = {"src": any_src, "dst": "C999999999",
                                                         "amount": 250000.0, "tx_type": "TRANSFER"}
        return out

    # ------------------------------------------------------- knowledge base
    def kb_table(self) -> pd.DataFrame:
        """Every candidate rule with its gate status, Dirichlet opinion and live e-process wealth."""
        gate = self.system.gate
        rows = []
        if gate is None:
            for r in self.system.active_rules():
                rows.append({"rule_id": str(r.rule_id), "category": getattr(r, "category", ""),
                             "predicate": r.describe(), "status": "active (auto-admit)",
                             "belief": float("nan"), "disbelief": float("nan"), "uncertainty": float("nan"),
                             "base_rate": float("nan"), "expected_probability": float(r.historical_precision),
                             "admission_wealth": float("nan"), "rollback_wealth": float("nan"),
                             "n_admissions": 0, "n_rollbacks": 0, "simulated_harmful": bool(r.is_harmful),
                             "last_event": ""})
            return pd.DataFrame(rows)
        for rid, rec in sorted(gate.records.items()):
            op = rec.opinion
            rule = rec.rule
            rows.append({
                "rule_id": rid, "category": getattr(rule, "category", ""), "predicate": rule.describe(),
                "status": rec.status, "belief": float(op.belief), "disbelief": float(op.disbelief),
                "uncertainty": float(op.uncertainty), "base_rate": float(op.base_rate),
                "expected_probability": float(op.expected_probability),
                "admission_wealth": float(rec.admission.wealth) if rec.admission is not None else float("nan"),
                "rollback_wealth": float(rec.rollback.wealth) if rec.rollback is not None else float("nan"),
                "n_admissions": int(rec.n_admissions), "n_rollbacks": int(rec.n_rollbacks),
                "simulated_harmful": bool(getattr(rule, "is_harmful", False)),
                "true_precision": float(getattr(rule, "true_precision", float("nan"))),
                "last_event": rec.events[-1][1] if rec.events else "",
            })
        return pd.DataFrame(rows)

    def kb_summary(self) -> Dict[str, Any]:
        gate = self.system.gate
        if gate is None:
            return {"mode": self.system.gate_mode, "n_active": len(self.system.active_rules())}
        s = gate.summary()
        s["mode"] = "martingale"
        s["integrity"] = gate.store.verify_integrity()
        s["alpha_admit"] = gate.config.alpha_admit
        s["alpha_roll"] = gate.config.alpha_roll
        s["p0"], s["q0"] = gate.config.p0, gate.config.q0
        return s

    def kb_snapshots(self) -> pd.DataFrame:
        gate = self.system.gate
        return pd.DataFrame(gate.store.list_snapshots()) if gate is not None else pd.DataFrame()

    # -------------------------------------------------------------- recovery
    def _recovery_engine(self):
        if self._simulator is None:
            from src.recovery.action_simulator import RecoverySimulator
            from src.recovery.tainted_flow import EdgeSchema, TaintedFlowTracker
            schema = EdgeSchema.paysim() if "ieee" not in str(self.source) else EdgeSchema.generic()
            self._tracker = TaintedFlowTracker.from_frame(
                self.canonical, "canonical", schema=schema,
                **TaintedFlowTracker.config_kwargs(self.hyperparams, self.project_config))
            self._simulator = RecoverySimulator.from_config(self._tracker, self.hyperparams, self.project_config)
        return self._tracker, self._simulator

    def recovery_candidates(self) -> pd.DataFrame:
        """Fraudulent test transactions MG-ECFM did not intercept that TD-TPF can trace."""
        from src.recovery.tainted_flow import EDGE_INTERNAL
        tracker, _ = self._recovery_engine()
        h = self.history[(self.history["y"] == 1)].copy()
        if h.empty:
            return h
        kinds = [tracker.schema.classify(tracker.edge(int(t)), tracker.external_accounts) for t in h["tx_id"]]
        h["edge_kind"] = kinds
        h["traceable"] = h["edge_kind"] == EDGE_INTERNAL
        h["intercepted"] = h["action"] >= int(Action.HOLD)
        cols = ["tx_id", "step", "src", "dst", "amount", "tx_type", "p", "action_label", "intercepted",
                "edge_kind", "traceable"]
        return h[cols].sort_values(["intercepted", "traceable", "amount"], ascending=[True, False, False]
                                   ).reset_index(drop=True)

    def recovery_trace(self, tx_id: int, delay_hours: float,
                       curve_delays: Sequence[float] = (0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 20, 23, 24, 30)
                       ) -> Dict[str, Any]:
        """TD-TPF trace + freeze / block / chargeback protocol for one fraud detected after ``delay_hours``."""
        tracker, sim = self._recovery_engine()
        edge = tracker.edge(int(tx_id))
        outcome = sim.simulate(edge, edge.tau + float(delay_hours))
        trace = outcome.trace
        return {
            "edge": asdict(edge), "metrics": outcome.metrics(),
            "trace_metrics": trace.metrics() if trace is not None else {},
            "nodes": trace.nodes_frame() if trace is not None else pd.DataFrame(),
            "edges": trace.edges_frame() if trace is not None else pd.DataFrame(),
            "actions": outcome.actions_frame(),
            "curve": sim.recovery_curve(edge, list(curve_delays)),
            "window_hours": float(tracker.window_hours), "kappa": float(tracker.kappa),
        }

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> Path:
        """Pickle the service (the SQLite knowledge store is serialised to bytes)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        store = self.system.gate.store if self.system.gate is not None else None
        conn = lock = clock = None
        blob = None
        if store is not None:
            with store._lock:
                blob = store._conn.serialize()
            conn, lock, clock = store._conn, store._lock, store._clock
            store._conn, store._lock, store._clock = None, None, None
        tracker, simulator = self._tracker, self._simulator
        self._tracker = self._simulator = None
        try:
            payload = {"format": SERVICE_FORMAT_VERSION, "service": self, "kb_sqlite": blob}
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as fh:          # atomic: a failed dump never leaves a truncated model
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        finally:
            if store is not None:
                store._conn, store._lock, store._clock = conn, lock, clock
            self._tracker, self._simulator = tracker, simulator
        return path

    @classmethod
    def load(cls, path: Path) -> "MGECFMService":
        with open(Path(path), "rb") as fh:
            import xgboost as xgb
            original = xgb.core.Booster.__setstate__
            xgb.core.Booster.__setstate__ = _tolerant_booster_setstate(original)
            try:
                payload = _ServiceUnpickler(fh).load()
            finally:
                xgb.core.Booster.__setstate__ = original
        if payload.get("format") != SERVICE_FORMAT_VERSION:
            raise ValueError(f"unsupported service file format {payload.get('format')}")
        service: MGECFMService = payload["service"]
        blob = payload.get("kb_sqlite")
        if service.system.gate is not None and blob is not None:
            conn = sqlite3.connect(":memory:", timeout=30.0, isolation_level=None, check_same_thread=False)
            conn.deserialize(blob)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 30000")
            store = service.system.gate.store
            store._conn, store._lock = conn, threading.RLock()
            if getattr(store, "_clock", None) is None:
                store._clock = utc_clock
        return service

    def describe(self) -> Dict[str, Any]:
        return {"source": self.source, "profile": self.config.profile, "last_step": self.last_step,
                "n_history": int(len(self.history)), "n_accounts": int(self.engineer.n_entities),
                "threshold": float(self.system.threshold_),
                "crc_thresholds": tuple(float(v) for v in self.system.router.controller.thresholds),
                "n_active_rules": len(self.system.active_rules()), "build_report": self.build_report,
                "splits": self.split_summary}


__all__ = ["ACTION_LABELS", "MGECFMService", "ServiceConfig", "profile_hyperparams"]
