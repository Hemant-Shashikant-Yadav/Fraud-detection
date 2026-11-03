"""
experiments/statistical_significance.py
=======================================

Phase 7 - formal statistical hypothesis testing of MG-ECFM against BP1 - BP4.

Inputs (written by :mod:`experiments.run_drift_eval`)::

    reports/drift_evaluation_results.csv      per-(model, window) metrics
    reports/drift_evaluation_predictions.csv  per-(model, transaction) scores / decisions

Tests
-----
McNemar's test (paired classification)
    For MG-ECFM (A) vs. baseline (B) on the identical test transactions,
    ``b = #{A correct, B wrong}`` and ``c = #{A wrong, B correct}``.
    ``chi2 = (|b - c| - 1)^2 / (b + c)`` (Edwards continuity correction,
    1 d.o.f.), ``p = P(chi2_1 >= chi2) = erfc(sqrt(chi2 / 2))``. The exact
    two-sided binomial (mid-free) p-value ``2 P(Bin(b + c, 1/2) <= min(b, c))``
    is reported as well (recommended when ``b + c < 25``). Odds ratio
    ``OR = b / c`` with a Wald 95% CI on ``log OR`` (Haldane-Anscombe +0.5
    correction when ``b`` or ``c`` is zero).

Wilcoxon signed-rank test (paired, non-parametric, across drift windows)
    On the per-window PR-AUC differences ``d_w = PRAUC_A(w) - PRAUC_B(w)``
    (windows where either PR-AUC is undefined are dropped; zero differences
    are discarded - Wilcoxon's method). ``W+ / W-`` are the rank sums of the
    positive / negative differences, the statistic is ``T = min(W+, W-)``. The
    exact null distribution is used when ``n <= 50`` and the ``|d_w|`` have no
    ties; otherwise the normal approximation with tie-corrected variance.
    Two-sided and one-sided (``H1: A > B``) p-values, the rank-biserial
    correlation and ``r = |z| / sqrt(n)`` are reported. F2 and EML are tested
    as secondary endpoints.

Multiplicity
    Holm-Bonferroni adjustment across the four baselines within each test family.

Bootstrap confidence intervals
    95% percentile intervals from ``B = 1000`` *moving-window block bootstrap*
    resamples: the 50 monitoring windows are resampled with replacement and the
    pooled metric is recomputed, which preserves within-window dependence of
    transactions. Reported for PR-AUC, ROC-AUC, F1, F2, Brier and EML of every
    model and for the paired difference MG-ECFM - baseline.

Everything is implemented with NumPy / ``math`` only (SciPy is used solely by
the unit tests, when installed, to cross-check the implementation).

Output: ``reports/statistical_significance_results.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

SEED = 42
REPORTS_DIR = ROOT / "reports"
DEFAULT_RESULTS_CSV = REPORTS_DIR / "drift_evaluation_results.csv"
DEFAULT_PREDICTIONS_CSV = REPORTS_DIR / "drift_evaluation_predictions.csv"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "statistical_significance_results.json"
PROPOSED = "mg_ecfm"
BOOTSTRAP_METRICS: Tuple[str, ...] = ("pr_auc", "roc_auc", "f1", "f2", "brier", "expected_monetary_loss")
WILCOXON_METRICS: Dict[str, str] = {"pr_auc": "greater", "f2": "greater", "expected_monetary_loss": "less"}


# --------------------------------------------------------------------------- #
# Distribution helpers                                                        #
# --------------------------------------------------------------------------- #
def chi2_sf_df1(x: float) -> float:
    """Survival function of the chi-squared distribution with one degree of freedom."""
    if not np.isfinite(x):
        return 0.0 if x > 0 else 1.0
    if x <= 0.0:
        return 1.0
    return float(math.erfc(math.sqrt(x / 2.0)))


def normal_sf(z: float) -> float:
    """``P(Z >= z)`` for a standard normal."""
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided sign-test p-value ``min(1, 2 P(Bin(n, 1/2) <= min(k, n - k)))``."""
    if n <= 0:
        return 1.0
    m = min(k, n - k)
    log_half_n = -n * math.log(2.0)
    tail = 0.0
    for i in range(0, m + 1):
        tail += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n)
    return float(min(1.0, 2.0 * tail))


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties given their average rank."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def signed_rank_null_counts(n: int) -> np.ndarray:
    """``counts[s]`` = number of sign assignments of ranks 1..n with ``W+ = s`` (exact null)."""
    total = n * (n + 1) // 2
    counts = np.zeros(total + 1, dtype=np.float64)
    counts[0] = 1.0
    for r in range(1, n + 1):
        counts[r:] = counts[r:] + counts[:-r].copy()
    return counts


# --------------------------------------------------------------------------- #
# McNemar                                                                     #
# --------------------------------------------------------------------------- #
def mcnemar_test(correct_a: Sequence[int], correct_b: Sequence[int], continuity: bool = True,
                 confidence: float = 0.95) -> Dict[str, Any]:
    """McNemar's test for two paired classifiers (A = proposed, B = comparator)."""
    a_ok = np.asarray(correct_a).astype(bool)
    b_ok = np.asarray(correct_b).astype(bool)
    if a_ok.shape != b_ok.shape:
        raise ValueError("paired correctness vectors must have the same length")
    both = int((a_ok & b_ok).sum())
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    neither = int((~a_ok & ~b_ok).sum())
    disc = b + c
    if disc == 0:
        chi2 = 0.0
    elif continuity:
        chi2 = (max(abs(b - c) - 1.0, 0.0)) ** 2 / disc
    else:
        chi2 = (b - c) ** 2 / disc
    p = chi2_sf_df1(chi2)
    hb, hc = (b + 0.5, c + 0.5) if (b == 0 or c == 0) else (float(b), float(c))
    odds = hb / hc
    se = math.sqrt(1.0 / hb + 1.0 / hc)
    zq = _normal_quantile(0.5 + confidence / 2.0)
    return {
        "n": int(a_ok.size),
        "contingency_table": [[both, b], [c, neither]],
        "table_layout": "[[A correct & B correct, A correct & B wrong], [A wrong & B correct, both wrong]]",
        "a_both_correct": both, "b_discordant_a_correct": b, "c_discordant_b_correct": c,
        "d_both_wrong": neither, "n_discordant": disc,
        "chi2": float(chi2), "df": 1, "continuity_correction": bool(continuity),
        "p_value": float(p), "p_value_exact_binomial": binomial_two_sided_p(b, disc),
        "odds_ratio": float(odds),
        "odds_ratio_ci_low": float(math.exp(math.log(odds) - zq * se)),
        "odds_ratio_ci_high": float(math.exp(math.log(odds) + zq * se)),
        "haldane_correction": bool(b == 0 or c == 0),
        "accuracy_a": float(a_ok.mean()) if a_ok.size else float("nan"),
        "accuracy_b": float(b_ok.mean()) if b_ok.size else float("nan"),
        "favours": "A" if b > c else ("B" if c > b else "tie"),
    }


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank                                                        #
# --------------------------------------------------------------------------- #
def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float], alternative: str = "two-sided",
                         method: str = "auto", exact_max_n: int = 50) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test of ``x - y`` (Wilcoxon zero handling).

    ``alternative``: ``"two-sided"``, ``"greater"`` (``x > y``) or ``"less"``.
    ``method``: ``"auto"`` (exact if ``n <= exact_max_n`` and no tied ``|d|``),
    ``"exact"`` or ``"approx"`` (normal, tie-corrected variance, no continuity
    correction - the SciPy convention).
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must be paired (same length)")
    keep = np.isfinite(x) & np.isfinite(y)
    d_all = x[keep] - y[keep]
    d = d_all[d_all != 0.0]
    n = int(d.size)
    out: Dict[str, Any] = {"n_pairs": int(keep.sum()), "n_nonzero": n,
                           "n_zero_differences": int((d_all == 0.0).sum()),
                           "alternative": alternative,
                           "median_difference": float(np.median(d_all)) if d_all.size else float("nan"),
                           "mean_difference": float(np.mean(d_all)) if d_all.size else float("nan")}
    if n == 0:
        out.update({"w_plus": 0.0, "w_minus": 0.0, "statistic": 0.0, "p_value": 1.0, "method": "degenerate",
                    "z": 0.0, "effect_size_r": 0.0, "rank_biserial": 0.0})
        return out
    ranks = rankdata_average(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    total = n * (n + 1) / 2.0
    has_ties = np.unique(np.abs(d)).size < n
    use_exact = method == "exact" or (method == "auto" and n <= exact_max_n and not has_ties)
    if method == "exact" and has_ties:
        raise ValueError("exact Wilcoxon distribution requires untied |differences|")
    mean = total / 2.0
    _, tie_counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float(((tie_counts ** 3) - tie_counts).sum()) / 48.0
    z = (w_plus - mean) / math.sqrt(var) if var > 0 else 0.0
    if use_exact:
        counts = signed_rank_null_counts(n)
        probs = counts / counts.sum()
        s = int(round(w_plus))
        cdf = float(probs[:s + 1].sum())               # P(W+ <= s)
        sf = float(probs[s:].sum())                    # P(W+ >= s)
        if alternative == "greater":
            p = sf
        elif alternative == "less":
            p = cdf
        else:
            p = min(1.0, 2.0 * min(cdf, sf))
        method_used = "exact"
    else:
        if alternative == "greater":
            p = normal_sf(z)
        elif alternative == "less":
            p = normal_sf(-z)
        else:
            p = min(1.0, 2.0 * normal_sf(abs(z)))
        method_used = "normal_approximation"
    out.update({
        "w_plus": w_plus, "w_minus": w_minus,
        "statistic": float(min(w_plus, w_minus)) if alternative == "two-sided" else w_plus,
        "p_value": float(min(max(p, 0.0), 1.0)), "method": method_used, "z": float(z),
        "effect_size_r": float(abs(z) / math.sqrt(n)),
        "rank_biserial": float((w_plus - w_minus) / total),
    })
    return out


def holm_bonferroni(p_values: Mapping[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    items = sorted(((float(p), k) for k, p in p_values.items() if p is not None and np.isfinite(p)))
    m = len(items)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = float(running)
    for k, p in p_values.items():
        adjusted.setdefault(k, float("nan"))
    return adjusted


# --------------------------------------------------------------------------- #
# Bootstrap                                                                   #
# --------------------------------------------------------------------------- #
def pooled_metric(name: str, y: np.ndarray, p: np.ndarray, yhat: np.ndarray,
                  loss: Optional[np.ndarray] = None) -> float:
    """One pooled metric over a set of transactions (NaN when undefined)."""
    y = np.asarray(y).astype(int)
    if y.size == 0:
        return float("nan")
    both = 0 < y.sum() < y.size
    if name == "pr_auc":
        return float(average_precision_score(y, p)) if both else float("nan")
    if name == "roc_auc":
        return float(roc_auc_score(y, p)) if both else float("nan")
    if name == "brier":
        return float(np.mean((np.asarray(p, dtype=np.float64) - y) ** 2))
    if name == "expected_monetary_loss":
        if loss is None:
            raise ValueError("expected_monetary_loss needs per-transaction losses")
        return float(np.mean(loss))
    tp = float(((yhat == 1) & (y == 1)).sum())
    fp = float(((yhat == 1) & (y == 0)).sum())
    fn = float(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    beta = {"f1": 1.0, "f2": 2.0}.get(name)
    if beta is None:
        raise ValueError(f"unknown metric {name}")
    b2 = beta * beta
    den = b2 * prec + rec
    return float((1 + b2) * prec * rec / den) if den > 0 else 0.0


def block_bootstrap_ci(statistic: Callable[[np.ndarray], float], n_blocks: int, n_resamples: int = 1000,
                       confidence: float = 0.95, seed: int = SEED) -> Dict[str, Any]:
    """Percentile CI of ``statistic(block_indices)`` under block resampling with replacement."""
    if n_blocks < 1:
        raise ValueError("need at least one block")
    rng = np.random.default_rng(seed)
    estimate = statistic(np.arange(n_blocks))
    stats = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        stats[b] = statistic(rng.integers(0, n_blocks, size=n_blocks))
    valid = stats[np.isfinite(stats)]
    alpha = 1.0 - confidence
    if valid.size == 0:
        lo = hi = se = float("nan")
    else:
        lo, hi = (float(v) for v in np.quantile(valid, [alpha / 2.0, 1.0 - alpha / 2.0]))
        se = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
    return {"estimate": float(estimate), "ci_low": lo, "ci_high": hi, "bootstrap_se": se,
            "confidence": confidence, "n_resamples": int(n_resamples), "n_valid_resamples": int(valid.size),
            "resampling_unit": "monitoring window (block bootstrap)"}


class WindowedPredictions:
    """Per-window arrays of one model for fast block resampling."""

    def __init__(self, frame: pd.DataFrame, windows: Sequence[int], loss_fn=None) -> None:
        self.windows = list(windows)
        g = {w: part for w, part in frame.groupby("window", sort=True)}
        self.y = [g[w]["y"].to_numpy().astype(int) for w in self.windows]
        self.p = [g[w]["p"].to_numpy(dtype=np.float64) for w in self.windows]
        self.yhat = [g[w]["yhat"].to_numpy().astype(int) for w in self.windows]
        if loss_fn is not None:
            self.loss = [loss_fn.batch(g[w]["action"].to_numpy().astype(int), g[w]["y"].to_numpy().astype(int),
                                       g[w]["amount"].to_numpy(dtype=np.float64)) for w in self.windows]
        else:
            self.loss = None

    def metric(self, name: str, idx: np.ndarray) -> float:
        y = np.concatenate([self.y[i] for i in idx])
        p = np.concatenate([self.p[i] for i in idx])
        yhat = np.concatenate([self.yhat[i] for i in idx])
        loss = np.concatenate([self.loss[i] for i in idx]) if self.loss is not None else None
        return pooled_metric(name, y, p, yhat, loss)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run_statistical_tests(results: pd.DataFrame, predictions: pd.DataFrame, proposed: str = PROPOSED,
                          n_resamples: int = 1000, confidence: float = 0.95, alpha: float = 0.05,
                          seed: int = SEED, project_config: Optional[Mapping[str, Any]] = None
                          ) -> Dict[str, Any]:
    """All Phase-7 significance tests; returns a JSON-serialisable dictionary."""
    from src.conformal.loss_functions import AsymmetricLoss, CostParameters
    if project_config is None:
        with open(ROOT / "configs" / "project_config.json", "r", encoding="utf-8") as fh:
            project_config = json.load(fh)
    loss_fn = AsymmetricLoss(CostParameters.from_config(project_config))
    models = list(dict.fromkeys(results["model"].tolist()))
    if proposed not in models:
        raise ValueError(f"proposed model '{proposed}' not found in results")
    baselines = [m for m in models if m != proposed]
    windows = sorted(int(w) for w in results["window"].unique())

    preds = {m: predictions[predictions["model"] == m].sort_values(["window", "tx_id"]).reset_index(drop=True)
             for m in models}
    ref = preds[proposed]
    out: Dict[str, Any] = {
        "meta": {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "proposed_model": proposed, "baselines": baselines, "n_windows": len(windows),
                 "n_test_transactions": int(len(ref)), "significance_level": alpha,
                 "bootstrap_resamples": n_resamples, "confidence": confidence, "seed": seed,
                 "data_source": str(results["data_source"].iloc[0]) if "data_source" in results.columns else None,
                 "correctness_definition": "yhat == y at each model's operating threshold (F1-optimal on the operating split)"},
        "mcnemar": {}, "wilcoxon": {}, "bootstrap_ci": {}, "paired_difference_ci": {}, "holm_adjusted": {},
    }

    # --- McNemar ------------------------------------------------------------
    a_ok = (ref["yhat"].to_numpy() == ref["y"].to_numpy())
    for m in baselines:
        bm = preds[m]
        if len(bm) != len(ref) or not np.array_equal(bm["tx_id"].to_numpy(), ref["tx_id"].to_numpy()):
            raise ValueError(f"predictions of {m} are not paired with {proposed}")
        b_ok = (bm["yhat"].to_numpy() == bm["y"].to_numpy())
        res = mcnemar_test(a_ok, b_ok)
        res["significant"] = bool(res["p_value"] < alpha)
        out["mcnemar"][m] = res

    # --- Wilcoxon across windows ------------------------------------------------
    per_window = results.pivot_table(index="window", columns="model", values=list(WILCOXON_METRICS),
                                     aggfunc="first")
    for metric, one_sided in WILCOXON_METRICS.items():
        out["wilcoxon"][metric] = {}
        for m in baselines:
            xa = per_window[(metric, proposed)].reindex(windows).to_numpy(dtype=np.float64)
            xb = per_window[(metric, m)].reindex(windows).to_numpy(dtype=np.float64)
            two = wilcoxon_signed_rank(xa, xb, "two-sided")
            one = wilcoxon_signed_rank(xa, xb, one_sided)
            two["p_value_one_sided"] = one["p_value"]
            two["one_sided_alternative"] = f"{proposed} {'>' if one_sided == 'greater' else '<'} {m}"
            two["significant"] = bool(two["p_value"] < alpha)
            out["wilcoxon"][metric][m] = two

    # --- Holm-Bonferroni ----------------------------------------------------------
    out["holm_adjusted"]["mcnemar"] = holm_bonferroni({m: out["mcnemar"][m]["p_value"] for m in baselines})
    for metric in WILCOXON_METRICS:
        out["holm_adjusted"][f"wilcoxon_{metric}"] = holm_bonferroni(
            {m: out["wilcoxon"][metric][m]["p_value"] for m in baselines})
    for m in baselines:
        out["mcnemar"][m]["p_value_holm"] = out["holm_adjusted"]["mcnemar"][m]
        out["mcnemar"][m]["significant_holm"] = bool(out["mcnemar"][m]["p_value_holm"] < alpha)
        for metric in WILCOXON_METRICS:
            adj = out["holm_adjusted"][f"wilcoxon_{metric}"][m]
            out["wilcoxon"][metric][m]["p_value_holm"] = adj
            out["wilcoxon"][metric][m]["significant_holm"] = bool(np.isfinite(adj) and adj < alpha)

    # --- Block-bootstrap confidence intervals -------------------------------------
    wp = {m: WindowedPredictions(preds[m], windows, loss_fn) for m in models}
    for i, m in enumerate(models):
        out["bootstrap_ci"][m] = {}
        for j, metric in enumerate(BOOTSTRAP_METRICS):
            out["bootstrap_ci"][m][metric] = block_bootstrap_ci(
                lambda idx, _m=m, _k=metric: wp[_m].metric(_k, idx), len(windows), n_resamples, confidence,
                seed=seed + 1000 * j)          # same resampled windows for every model -> paired
    for m in baselines:
        out["paired_difference_ci"][m] = {}
        for j, metric in enumerate(BOOTSTRAP_METRICS):
            ci = block_bootstrap_ci(
                lambda idx, _m=m, _k=metric: wp[proposed].metric(_k, idx) - wp[_m].metric(_k, idx),
                len(windows), n_resamples, confidence, seed=seed + 1000 * j)
            ci["definition"] = f"{proposed} - {m}"
            ci["excludes_zero"] = bool(np.isfinite(ci["ci_low"]) and (ci["ci_low"] > 0 or ci["ci_high"] < 0))
            out["paired_difference_ci"][m][metric] = ci
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MG-ECFM Phase-7 statistical significance testing")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS_CSV)
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    for path in (args.results, args.predictions):
        if not Path(path).is_file():
            print(f"[error] {path} not found - run experiments/run_drift_eval.py first", file=sys.stderr)
            return 2
    results = pd.read_csv(args.results)
    predictions = pd.read_csv(args.predictions)
    report = run_statistical_tests(results, predictions, n_resamples=args.resamples, alpha=args.alpha,
                                   seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    for m, r in report["mcnemar"].items():
        w = report["wilcoxon"]["pr_auc"][m]
        print(f"[{m}] McNemar b={r['b_discordant_a_correct']} c={r['c_discordant_b_correct']} "
              f"chi2={r['chi2']:.3f} p={r['p_value']:.4g} OR={r['odds_ratio']:.3f} | "
              f"Wilcoxon PR-AUC W+={w['w_plus']:.1f} W-={w['w_minus']:.1f} p={w['p_value']:.4g} "
              f"(one-sided {w['p_value_one_sided']:.4g})")
    print(f"[saved] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
