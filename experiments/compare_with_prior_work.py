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
* ``budget``  - the budgeted Bayes CS-ACRC router (:mod:`src.conformal.budgeted_crc`)
  applied to every model's test scores under identical conditions: prequential
  routing window by window, labels released with a one-window delay, and a
  label-free warm start of the budget price on the window-1 scores. Budgets
  ``rho`` in BUDGETS; the same step sizes as the pipeline configuration.
* ``cost_curve`` - equal-friction comparison. The intervention price ``nu`` of
  the Bayes rule is swept over a fixed grid (``theta = 0``, no label is used
  by any decision). For each intervention-rate cap the lowest net loss whose
  intervention rate is within the cap is reported, like reading an ROC curve
  at a fixed false-positive rate.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
SEED = 42
N_BOOT = 2000
BUDGETS = (0.01, 0.02, 0.03, 0.05)
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


def budget_router(d: pd.DataFrame, rho: float, delay: int = 1) -> dict:
    """Prequential budgeted Bayes routing of one model's test scores (see module doc)."""
    from src.conformal.budgeted_crc import BudgetedBayesCRC
    from src.conformal.loss_functions import AsymmetricLoss, CostParameters
    hp = json.loads((ROOT / "configs" / "hyperparams.json").read_text())["conformal_crc"]["budgeted_bayes"]
    pc = json.loads((ROOT / "configs" / "project_config.json").read_text())
    ctrl = BudgetedBayesCRC(intervention_budget=rho, gamma_budget=hp["gamma_budget"],
                            gamma_calibration=hp["gamma_calibration"], eta_max=hp["eta_max"],
                            theta_max=hp["theta_max"], calibration_shift=hp["calibration_shift"],
                            loss=AsymmetricLoss(CostParameters.from_config(pc)))
    y, p, a, w = d.y.to_numpy(), d.p.to_numpy(), d.amount.to_numpy(), d.window.to_numpy()
    wins = np.unique(w)
    first = w == wins[0]
    ctrl.warm_start(p[first], a[first])
    act = np.zeros(len(d), dtype=np.int64)
    pending = []
    for k, win in enumerate(wins):
        while pending and pending[0][0] <= k:
            _, idx = pending.pop(0)
            ctrl.update_batch(act[idx], y[idx], a[idx], p[idx])
        idx = np.flatnonzero(w == win)
        act[idx] = ctrl.decide_batch(p[idx], a[idx])
        pending.append((k + 1 + delay, idx))
    loss = realised_loss(act, y, a)
    fv = a[y == 1].sum()
    return {"net_loss": float(loss.sum()),
            "intervention_rate_pct": 100.0 * float((act >= 1).mean()),
            "legit_intervention_pct": 100.0 * float((act[y == 0] >= 1).mean()),
            "fraud_value_stopped_pct": 100.0 * float(a[(y == 1) & (act >= 2)].sum() / fv),
            "frauds_approved": int(((y == 1) & (act == 0)).sum()),
            "step_ups": int((act == 1).sum()), "holds": int((act == 2).sum()),
            "blocks": int((act == 3).sum())}


def cost_curve(d: pd.DataFrame, caps=BUDGETS, n_grid: int = 400) -> dict:
    """Lowest net loss at each intervention-rate cap over a label-free sweep of nu."""
    from src.conformal.budgeted_crc import BudgetedBayesCRC
    from src.conformal.loss_functions import AsymmetricLoss, CostParameters
    pc = json.loads((ROOT / "configs" / "project_config.json").read_text())
    ctrl = BudgetedBayesCRC(calibration_shift=False, loss=AsymmetricLoss(CostParameters.from_config(pc)))
    y, p, a = d.y.to_numpy(), d.p.to_numpy(), d.amount.to_numpy()
    fv = a[y == 1].sum()
    pts = []
    for eta in np.linspace(0.0, ctrl.eta_max, n_grid):
        ctrl._eta = float(eta)
        act = ctrl.decide_batch(p, a)
        pts.append(((act >= 1).mean(), realised_loss(act, y, a).sum(),
                    a[(y == 1) & (act >= 2)].sum() / fv, (act[y == 0] >= 1).mean()))
    pts = np.asarray(pts)
    out = {}
    for cap in caps:
        ok = pts[:, 0] <= cap + 1e-12
        if not ok.any():
            out[f"{cap:.2f}"] = None
            continue
        j = int(np.flatnonzero(ok)[np.argmin(pts[ok][:, 1])])
        best = pts[j]
        ctrl._eta = float(np.linspace(0.0, ctrl.eta_max, n_grid)[j])
        act = ctrl.decide_batch(p, a)
        out[f"{cap:.2f}"] = {"net_loss": float(best[1]), "eta": ctrl._eta,
                             "_loss_by_window": pd.Series(realised_loss(act, y, a)).groupby(d.window.to_numpy()).sum(), "intervention_rate_pct": 100.0 * float(best[0]),
                             "fraud_value_stopped_pct": 100.0 * float(best[2]),
                             "legit_intervention_pct": 100.0 * float(best[3])}
    return out


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
        r["budget_router"] = {f"{rho:.2f}": budget_router(d, rho) for rho in BUDGETS}
        r["cost_curve"] = cost_curve(d)
        if m != "mg_ecfm":
            r["boot_pr_auc_vs_ours"] = block_bootstrap_diff(ref, d, rng)
        result["models"][m] = r
    # Window-block bootstrap of the net-loss difference at equal intervention caps.
    rng_l = np.random.default_rng(SEED + 1)
    wins = ref.window.unique()
    loss_boot = {}
    for cap in BUDGETS:
        k = f"{cap:.2f}"
        ours = result["models"]["mg_ecfm"]["cost_curve"][k]["_loss_by_window"].reindex(wins, fill_value=0.0).to_numpy()
        loss_boot[k] = {}
        for m in ORDER[1:]:
            other = result["models"][m]["cost_curve"][k]
            if other is None:
                continue
            oth = other["_loss_by_window"].reindex(wins, fill_value=0.0).to_numpy()
            diffs = np.array([(oth[i] - ours[i]).sum() for i in
                              (rng_l.integers(0, len(wins), len(wins)) for _ in range(N_BOOT))])
            loss_boot[k][m] = {"saving": float((oth - ours).sum()),
                               "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
                               "p_one_sided": float((diffs <= 0).mean())}
    result["cost_curve_bootstrap"] = loss_boot
    for m in ORDER:
        for v in result["models"][m]["cost_curve"].values():
            if v:
                v.pop("_loss_by_window", None)
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
    print("\n[cost-curve saving of MG-ECFM vs each model, window-block bootstrap]")
    for k, row in result["cost_curve_bootstrap"].items():
        for m, v in row.items():
            print(f"  cap {k} vs {m:<15s} saving {v['saving']/1e6:7.2f}M  CI [{v['ci95'][0]/1e6:7.2f}, {v['ci95'][1]/1e6:7.2f}]  p={v['p_one_sided']:.4f}")
    print("\n[cost curve: lowest net loss (M) at intervention rate <= cap]")
    print(pd.DataFrame({m: {k: (v["net_loss"] / 1e6 if v else np.nan)
                            for k, v in result["models"][m]["cost_curve"].items()} for m in ORDER}).T
          .to_string(float_format=lambda v: f"{v:,.2f}"))
    for rho in BUDGETS:
        k = f"{rho:.2f}"
        print(f"\n[budget {k}]")
        print(pd.DataFrame({m: result["models"][m]["budget_router"][k] for m in ORDER}).T
              .to_string(float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
