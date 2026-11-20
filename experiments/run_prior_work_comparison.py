"""
experiments/run_prior_work_comparison.py
========================================

Reproduces the detectors of the earlier "Evidence-Gated Risk-Aware Fraud
Prevention" project (the earlier M.Tech notebook implementation) on the
*identical* PaySim stream, splits and 50 test windows of the Phase-7 drift
evaluation, and compares them with MG-ECFM and BP1-BP4 on the same test rows.

Prior-work detectors (as documented in the project's assessment report)
----------------------------------------------------------------------
* ``prior_b1_raw``  - Day 6 B1: XGBoost on 7 features
  ``[step, amount, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT,
  type_TRANSFER]`` with ``scale_pos_weight = n_neg / n_pos`` and seed 42.
* ``prior_b1_cal``  - Day 7: the same model with Platt (sigmoid) calibration
  fitted on the calibration split (steps 521-575).
* ``prior_b1_graph`` - Day 10: B1 + the six history-based graph counting
  features (TRANSFER / CASH_OUT edges, strictly earlier steps only), Platt
  calibrated.

Every threshold is F1-optimal on the operating split (steps 576-631); no test
label is used for fitting, calibration or threshold selection.

Outputs
-------
``reports/prior_work_predictions.csv``   one row per (model, test transaction)
``reports/prior_work_fit.json``          thresholds and fit details
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_drift_eval import (  # noqa: E402
    PlattCalibrator,
    SEED,
    load_configs,
    prepare_splits,
    select_operating_threshold,
)

REPORTS = ROOT / "reports"
TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
B1_FEATURES = ["step", "amount"] + [f"type_{t}" for t in TYPES]
GRAPH_FEATURES = ["src_graph_out_count_before", "dst_graph_in_count_before",
                  "src_dst_pair_repeat_before", "src_recency_steps", "dst_recency_steps",
                  "src_dst_pair_seen_before"]
NO_HISTORY = 9999.0


def _count_before(df: pd.DataFrame, key: str, edge: np.ndarray) -> np.ndarray:
    """Number of edge rows with the same ``key`` at a strictly earlier step."""
    e = pd.Series(edge.astype(np.int64), index=df.index)
    incl = e.groupby(df[key]).cumsum()
    same_step = e.groupby([df[key], df["step"]]).cumsum()
    return (incl - same_step).to_numpy(dtype=np.float64)


def _recency(df: pd.DataFrame, key: str, edge: np.ndarray) -> np.ndarray:
    """Steps since ``key`` last appeared in an edge at a strictly earlier step."""
    hist = (df.loc[edge, [key, "step"]].drop_duplicates()
            .rename(columns={"step": "last_step"}).sort_values("last_step"))
    q = df[[key, "step"]].reset_index().sort_values("step")
    m = pd.merge_asof(q, hist, left_on="step", right_on="last_step", by=key,
                      allow_exact_matches=False, direction="backward")
    m = m.set_index("index").reindex(df.index)
    return (m["step"] - m["last_step"]).fillna(NO_HISTORY).to_numpy(dtype=np.float64)


def add_prior_features(full: pd.DataFrame) -> pd.DataFrame:
    """B1 one-hot type features and the six Day-10 graph counting features."""
    out = full.copy()
    ttype = out["tx_type"].astype(str).str.upper()
    for t in TYPES:
        out[f"type_{t}"] = (ttype == t).astype(np.float64)
    edge = ttype.isin(["TRANSFER", "CASH_OUT"]).to_numpy()
    out["pair"] = out["src"].astype(str) + "|" + out["dst"].astype(str)
    out["src_graph_out_count_before"] = _count_before(out, "src", edge)
    out["dst_graph_in_count_before"] = _count_before(out, "dst", edge)
    out["src_dst_pair_repeat_before"] = _count_before(out, "pair", edge)
    out["src_recency_steps"] = _recency(out, "src", edge)
    out["dst_recency_steps"] = _recency(out, "dst", edge)
    out["src_dst_pair_seen_before"] = (out["src_dst_pair_repeat_before"] > 0).astype(np.float64)
    return out


def fit_xgb(x: pd.DataFrame, y: np.ndarray):
    from xgboost import XGBClassifier
    spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    model = XGBClassifier(scale_pos_weight=spw, random_state=SEED, n_jobs=-1,
                          tree_method="hist", eval_metric="aucpr")
    return model.fit(x, y)


def main() -> None:
    _, project_config = load_configs()
    t0 = time.perf_counter()
    splits = prepare_splits("paysim", n_windows=50, project_config=project_config,
                            sample_rows=2_000_000, sample_seed=SEED)
    print(f"[data] splits ready in {time.perf_counter() - t0:.0f}s; test rows={len(splits.test)}",
          flush=True)
    parts = {"train": splits.train, "calibration": splits.calibration,
             "operating": splits.operating, "test": splits.test}
    full = pd.concat([p.assign(_split=k) for k, p in parts.items()], ignore_index=True)
    if np.any(np.diff(full["step"].to_numpy()) < 0):
        raise RuntimeError("stream is not chronological")
    full = add_prior_features(full)
    win = np.zeros(len(splits.test), dtype=np.int64)
    for w in splits.windows:
        win[w.row_start:w.row_end] = w.index

    frames = {k: full[full["_split"] == k].reset_index(drop=True) for k in parts}
    y = {k: f["label"].to_numpy().astype(int) for k, f in frames.items()}
    configs = {"prior_b1_raw": (B1_FEATURES, False),
               "prior_b1_cal": (B1_FEATURES, True),
               "prior_b1_graph": (B1_FEATURES + GRAPH_FEATURES, True)}
    rows, fit = [], {}
    models = {}
    for name, (feats, calibrate) in configs.items():
        key = tuple(feats)
        if key not in models:
            models[key] = fit_xgb(frames["train"][feats], y["train"])
        model = models[key]
        score = lambda f: model.predict_proba(f[feats])[:, 1]  # noqa: E731
        if calibrate:
            margin = lambda f: model.predict(f[feats], output_margin=True)  # noqa: E731
            cal = PlattCalibrator().fit(margin(frames["calibration"]), y["calibration"])
            score = lambda f: cal.predict(margin(f))  # noqa: E731
        tau = select_operating_threshold(y["operating"], score(frames["operating"]))
        t1 = time.perf_counter()
        p = np.clip(score(frames["test"]), 0.0, 1.0)
        lat = (time.perf_counter() - t1) * 1000.0 / len(p)
        fit[name] = {"features": feats, "platt": calibrate, "threshold": tau,
                     "latency_ms_per_txn": lat}
        print(f"[fit] {name:<15s} tau={tau:.4f}", flush=True)
        test = frames["test"]
        rows.append(pd.DataFrame({"model": name, "window": win, "tx_id": test["tx_id"].to_numpy(),
                                  "step": test["step"].to_numpy(), "y": y["test"],
                                  "amount": test["amount"].to_numpy(), "p": p,
                                  "yhat": (p >= tau).astype(int)}))
    pd.concat(rows, ignore_index=True).to_csv(REPORTS / "prior_work_predictions.csv", index=False)
    (REPORTS / "prior_work_fit.json").write_text(json.dumps(fit, indent=1))
    print("[done] reports/prior_work_predictions.csv", flush=True)


if __name__ == "__main__":
    main()
