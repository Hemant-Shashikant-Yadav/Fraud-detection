"""
experiments/compare_with_prior_work.py
======================================

Head-to-head comparison of MG-ECFM against the earlier Evidence-Gated project
(``prior_*``, see ``run_prior_work_comparison.py``) and BP1-BP4 on the *same*
28,259 PaySim test transactions (steps 632-743, 50 chronological windows).

Inputs : reports/drift_evaluation_predictions.csv, reports/prior_work_predictions.csv
Output : reports/prior_work_comparison.json (all numbers quoted in the markdown report)

Monetary policies (cost matrix of ``configs/project_config.json``)
------------------------------------------------------------------
* ``binary``  - approve / block at the model's own operating threshold tau.
* ``bayes``   - parameter-free Bayes-optimal four-action routing
  ``a* = argmin_a E[L(a, y, A) | p, A]`` with
  E[approve] = p A, E[step-up] = c_step + p beta A, E[hold] = c_hold,
  E[block] = (1 - p) c_block.  No value is fitted on test data.
* ``logged``  - the actions actually logged by MG-ECFM's CS-ACRC router.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEED = 42
N_BOOT = 2000
ORDER = ["mg_ecfm", "prior_b1_raw", "prior_b1_cal", "prior_b1_graph", "bp1", "bp2", "bp3", "bp4"]


def costs():
    c = json.loads((ROOT / "configs" / "project_config.json").read_text())["costs"]
    return c["c_step"], c["c_hold"], c["c_block_churn"], c["beta_bypass"]


def realised_loss(action, y, a):
    c_step, c_hold, c_block, beta = costs()
    return np.select([action == 0, action == 1, action == 2],
                     [y * a, c_step + y * beta * a, np.full_like(a, c_hold)],
                     (1 - y) * c_block)


def bayes_actions(p, a):
    c_step, c_hold, c_block, beta = costs()
    exp = np.stack([p * a, c_step + p * beta * a, np.full_like(a, c_hold), (1 - p) * c_block])
    return exp.argmin(axis=0)


def holm(pvals):
    keys = sorted(pvals, key=pvals.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * pvals[k]))
        out[k] = run
    return out


def pooled(d: pd.DataFrame) -> dict:
    y, p, yh, a = d.y.to_numpy(), d.p.to_numpy(), d.yhat.to_numpy(), d.amount.to_numpy()
    tp, fp = int(((yh == 1) & (y == 1)).sum()), int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    fpr, tpr, _ = roc_curve(y, p)
    fraud_value = float(a[y == 1].sum())
    bin_loss = realised_loss(np.where(yh == 1, 3, 0), y, a)
    bay_act = bayes_actions(p, a)
    bay_loss = realised_loss(bay_act, y, a)
    out = {
        "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "f2": 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0,
        "brier": float(np.mean((p - y) ** 2)),
        "recall_at_fpr_1pct": float(np.interp(0.01, fpr, tpr)),
        "recall_at_fpr_0_1pct": float(np.interp(0.001, fpr, tpr)),
        "tp": tp, "fp": fp, "fn": fn,
        "fraud_value_blocked_pct": 100.0 * float(a[(y == 1) & (yh == 1)].sum()) / fraud_value,
        "binary_net_loss": float(bin_loss.sum()),
        "bayes_net_loss": float(bay_loss.sum()),
        "bayes_fraud_value_stopped_pct": 100.0 * float(a[(y == 1) & (bay_act >= 2)].sum()) / fraud_value,
        "bayes_intervention_rate_pct": 100.0 * float((bay_act >= 1).mean()),
        "bayes_legit_friction_events": int(((y == 0) & (bay_act >= 1)).sum()),
        "approve_all_loss": fraud_value,
    }
    if "action" in d and d["action"].notna().all():
        act = d.action.to_numpy().astype(int)
        out["logged_router_net_loss"] = float(realised_loss(act, y, a).sum())
    return out


def per_window(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w, g in d.groupby("window"):
        y, p, yh = g.y.to_numpy(), g.p.to_numpy(), g.yhat.to_numpy()
        tp = ((yh == 1) & (y == 1)).sum()
        prec = tp / yh.sum() if yh.sum() else 0.0
        rec = tp / y.sum() if y.sum() else 0.0
        rows.append({"window": w,
                     "pr_auc": average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan,
                     "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                     "recall": rec})
    return pd.DataFrame(rows).set_index("window")


def block_bootstrap_diff(ref: pd.DataFrame, other: pd.DataFrame, rng) -> dict:
    """Window-block bootstrap of pooled PR-AUC(ref) - PR-AUC(other) (paired rows)."""
    wins = ref.window.unique()
    idx = {w: np.flatnonzero(ref.window.to_numpy() == w) for w in wins}
    yr, pr, po = ref.y.to_numpy(), ref.p.to_numpy(), other.p.to_numpy()
    diffs = []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[w] for w in rng.choice(wins, size=len(wins))])
        if yr[rows].sum() == 0:
            continue
        diffs.append(average_precision_score(yr[rows], pr[rows])
                     - average_precision_score(yr[rows], po[rows]))
    diffs = np.asarray(diffs)
    return {"diff": float(average_precision_score(yr, pr) - average_precision_score(yr, po)),
            "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
            "p_one_sided": float((diffs <= 0).mean())}


def main() -> None:
    ours = pd.read_csv(REPORTS / "drift_evaluation_predictions.csv")
    prior = pd.read_csv(REPORTS / "prior_work_predictions.csv")
    allp = pd.concat([ours, prior], ignore_index=True)
    ref = allp[allp.model == "mg_ecfm"].sort_values("tx_id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    result = {"n_test": int(len(ref)), "n_fraud": int(ref.y.sum()),
              "fraud_value": float(ref.amount[ref.y == 1].sum()), "models": {}}
    pw = {}
    for m in ORDER:
        d = allp[allp.model == m].sort_values("tx_id").reset_index(drop=True)
        if not np.array_equal(d.tx_id.to_numpy(), ref.tx_id.to_numpy()):
            raise RuntimeError(f"{m}: test rows differ from MG-ECFM")
        r = pooled(d)
        pw[m] = per_window(d)
        r["window_pr_auc_mean"] = float(pw[m].pr_auc.mean())
        r["window_f1_mean"] = float(pw[m].f1.mean())
        if m != "mg_ecfm":
            r["boot_pr_auc_vs_ours"] = block_bootstrap_diff(ref, d, rng)
        result["models"][m] = r
    tests = {}
    for metric in ("pr_auc", "f1"):
        raw = {}
        for m in ORDER[1:]:
            a, b = pw["mg_ecfm"][metric], pw[m][metric]
            ok = a.notna() & b.notna() & (a != b)
            raw[m] = float(wilcoxon(a[ok], b[ok], alternative="greater").pvalue) if ok.sum() else 1.0
        adj = holm(raw)
        tests[metric] = {m: {"p_raw": raw[m], "p_holm": adj[m],
                             "windows_won": int((pw["mg_ecfm"][metric] > pw[m][metric]).sum()),
                             "windows_lost": int((pw["mg_ecfm"][metric] < pw[m][metric]).sum())}
                         for m in ORDER[1:]}
    result["wilcoxon_one_sided"] = tests
    (REPORTS / "prior_work_comparison.json").write_text(json.dumps(result, indent=1))
    t = pd.DataFrame(result["models"]).T
    cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "recall_at_fpr_1pct",
            "fraud_value_blocked_pct", "binary_net_loss", "bayes_net_loss", "bayes_fraud_value_stopped_pct"]
    print(t[cols].to_string(float_format=lambda v: f"{v:,.4f}"))
    print(json.dumps(tests, indent=1))
    for m in ORDER[1:]:
        print(m, result["models"][m]["boot_pr_auc_vs_ours"])
    print("logged router loss", result["models"]["mg_ecfm"].get("logged_router_net_loss"))


if __name__ == "__main__":
    main()
