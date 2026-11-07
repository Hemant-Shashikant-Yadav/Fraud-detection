#!/usr/bin/env python
"""Render the results-deck charts (matplotlib + seaborn) from reports/.

Reads only existing result files; never trains or evaluates a model.

Outputs
    reports/deck_charts/*.png        charts at 220 DPI, sized in inches to their slide slot
    reports/deck_charts/deck_data.json   headline numbers + chart sizes for the deck builder

Usage
    .venv/bin/python scripts/build_results_charts.py [--reports reports] [--out reports/deck_charts]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

DPI = 220

# Categorical palette (dataviz reference slots 1-5, validated light mode). Fixed order per entity.
MODELS = ["mg_ecfm", "bp1", "bp2", "bp3", "bp4"]
MODEL_COLORS = {"mg_ecfm": "#2a78d6", "bp1": "#eb6834", "bp2": "#1baf7a", "bp3": "#eda100", "bp4": "#e87ba4"}
SHORT = {"mg_ecfm": "MG-ECFM", "bp1": "BP1 EARN+RXT-J", "bp2": "BP2 GNN+AE", "bp3": "BP3 CARE-GNN", "bp4": "BP4 ARGUS"}
TINY = {"mg_ecfm": "MG-ECFM", "bp1": "BP1", "bp2": "BP2", "bp3": "BP3", "bp4": "BP4"}
ACTION_LABELS = {0: "Approve", 1: "Step-Up", 2: "Hold", 3: "Block"}  # src/orchestrator/service.py
ACTION_COLORS = {0: "#2a78d6", 1: "#1baf7a", 2: "#eda100", 3: "#eb6834"}
POLICY_COLORS = {"mg_ecfm_gate": "#2a78d6", "sl_screen_only": "#eb6834", "auto_admit": "#1baf7a"}
POLICY_LABELS = {"mg_ecfm_gate": "MG-ECFM martingale gate", "sl_screen_only": "SL screen only", "auto_admit": "Auto-admit"}
CATEGORY_COLORS = {"poisoned": "#2a78d6", "sybil": "#eb6834", "stale": "#1baf7a", "conflicting": "#eda100"}
GROUP_COLORS = {"genuine": "#2a78d6", "poisoned": "#eb6834", "sybil": "#1baf7a"}

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e4e3df"
SURFACE = "#ffffff"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup_style() -> None:
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "axes.labelcolor": INK2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.linewidth": 2.0,
    })


def ygrid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


class ChartWriter:
    def __init__(self, out: Path):
        self.out = out
        self.sizes: dict[str, list[float]] = {}

    def save(self, fig, name: str) -> None:
        w, h = fig.get_size_inches()
        fig.savefig(self.out / f"{name}.png", dpi=DPI)
        plt.close(fig)
        self.sizes[name] = [round(float(w), 3), round(float(h), 3)]
        print(f"  wrote {name}.png ({w:.2f} x {h:.2f} in)")


def fmt_money(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v/1e6:,.1f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:,.1f}k"
    return f"{v:,.0f}"


def fmt_p(p: float) -> str:
    if p == 0 or p < 1e-4:
        return "< 0.0001"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


# --------------------------------------------------------------------------- #
def load_predictions(path: Path) -> pd.DataFrame:
    dtypes = {"model": "category", "window": "int16", "y": "int8", "p": "float32", "yhat": "int8", "action": "int8"}
    return pd.read_csv(path, usecols=list(dtypes), dtype=dtypes, engine="c")


def chart_splits(cw: ChartWriter, man: dict) -> dict:
    sp = man["splits"]
    names = ["train", "calibration", "operating", "test"]
    labels = ["Train", "Calibration", "Operating", "Test"]
    tx = [sp[n]["transactions"] for n in names]
    fr = [sp[n]["frauds"] for n in names]
    rate = [100 * f / t for f, t in zip(fr, tx)]
    colors = ["#2a78d6", "#eb6834", "#e34948", "#1baf7a"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    wedges, _ = a1.pie(tx, colors=colors, startangle=90, counterclock=False,
                       wedgeprops={"width": 0.38, "edgecolor": SURFACE, "linewidth": 2})
    a1.text(0, 0.08, f"{sum(tx)/1e6:.2f}M", ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
    a1.text(0, -0.16, "transactions", ha="center", va="center", fontsize=10, color=INK2)
    a1.set_title("Share of rows by split")
    a1.legend(wedges, [f"{l}  {t:,}" for l, t in zip(labels, tx)], loc="upper center",
              bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=9.5, handlelength=1, columnspacing=1)

    y = np.arange(len(names))
    a2.barh(y, rate, color=colors, height=0.6)
    a2.set_xscale("log")
    a2.set_yticks(y, labels)
    a2.invert_yaxis()
    for i, (r, f) in enumerate(zip(rate, fr)):
        a2.text(r * 1.15, i, f"{r:.2f}%  ({f:,} frauds)", va="center", fontsize=10, color=INK)
    a2.set_xlim(0.03, 3000)
    a2.set_xticks([0.1, 1, 10, 100])
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    a2.set_title("Fraud rate by split (log scale)")
    ygrid(a2, "x")
    fig.tight_layout()
    cw.save(fig, "split_sizes")
    return {"tx": dict(zip(names, tx)), "frauds": dict(zip(names, fr)), "rate_pct": dict(zip(names, rate))}


def chart_pooled_ci(cw: ChartWriter, sig: dict) -> None:
    boot = sig["bootstrap_ci"]
    metrics = [("pr_auc", "PR-AUC (higher is better)"), ("roc_auc", "ROC-AUC (higher is better)"),
               ("f2", "F2 at threshold 0.5 (higher is better)"), ("brier", "Brier score (lower is better)"),
               ("expected_monetary_loss", "Expected monetary loss / txn (lower is better)"),
               ("f1", "F1 at threshold 0.5 (higher is better)")]
    fig, axes = plt.subplots(2, 3, figsize=(12.1, 5.3))
    for ax, (m, title) in zip(axes.flat, metrics):
        est = [boot[k][m]["estimate"] for k in MODELS]
        lo = [boot[k][m]["ci_low"] for k in MODELS]
        hi = [boot[k][m]["ci_high"] for k in MODELS]
        y = np.arange(len(MODELS))
        ax.barh(y, est, color=[MODEL_COLORS[k] for k in MODELS], height=0.62)
        ax.errorbar(est, y, xerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="none", ecolor=INK, elinewidth=1.1, capsize=3)
        ax.set_yticks(y, [TINY[k] for k in MODELS])
        ax.invert_yaxis()
        xmax = min(max(hi) * 1.42, 1.25) if m == "roc_auc" else max(hi) * 1.42
        ax.set_xlim(0, xmax)
        for i, (e, h) in enumerate(zip(est, hi)):
            txt = f"{e:.4f}" if m == "brier" else (f"{e:,.0f}" if m == "expected_monetary_loss" else f"{e:.3f}")
            ax.text(h + xmax * 0.02, i, txt, va="center", fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=11)
        ygrid(ax, "x")
        ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    cw.save(fig, "pooled_metrics_ci")


def chart_window_box(cw: ChartWriter, drift: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.9))
    d = drift.copy()
    d["Model"] = d["model"].map(TINY)
    order = [TINY[k] for k in MODELS]
    pal = {TINY[k]: MODEL_COLORS[k] for k in MODELS}
    for ax, (col, title) in zip(axes, [("pr_auc", "Per-window PR-AUC"), ("f2", "Per-window F2 (threshold 0.5)")]):
        sns.boxplot(data=d, x=col, y="Model", hue="Model", order=order, palette=pal, ax=ax, width=0.55,
                    fliersize=0, linewidth=1.0, boxprops={"alpha": 0.35}, legend=False)
        sns.stripplot(data=d, x=col, y="Model", hue="Model", order=order, palette=pal, ax=ax, size=3.5,
                      jitter=0.18, alpha=0.85, legend=False)
        means = d.groupby("Model")[col].mean()
        for i, name in enumerate(order):
            ax.plot(means[name], i, marker="D", color=INK, markersize=6, zorder=5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(col.upper().replace("_", "-"))
        ax.set_ylabel("")
        ax.set_title(title)
        ygrid(ax, "x")
    axes[1].set_yticklabels([])
    axes[0].plot([], [], marker="D", color=INK, linestyle="none", label="mean over windows")
    axes[0].legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    cw.save(fig, "window_boxplots")


def chart_drift_lines(cw: ChartWriter, drift: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.1, 5.3), sharex=True, gridspec_kw={"height_ratios": [3, 1.15]})
    for k in MODELS:
        s = drift[drift.model == k].sort_values("window")
        lw = 2.4 if k == "mg_ecfm" else 1.5
        a1.plot(s.window, s.pr_auc, color=MODEL_COLORS[k], lw=lw, label=SHORT[k], marker="o", markersize=3.2,
                zorder=4 if k == "mg_ecfm" else 3)
    s = drift[drift.model == "mg_ecfm"].sort_values("window")
    allfraud = s[s.n_fraud == s.n_transactions].window
    for w in allfraud:
        a1.axvspan(w - 0.5, w + 0.5, color="#f0efec", zorder=0)
        a2.axvspan(w - 0.5, w + 0.5, color="#f0efec", zorder=0)
    a1.set_ylabel("PR-AUC")
    a1.set_ylim(0, 1.0)
    a1.set_title("PR-AUC per drift window (grey bands: 100% fraud windows, PR-AUC undefined)", pad=26)
    a1.legend(ncol=5, loc="lower left", bbox_to_anchor=(0.0, 0.99), fontsize=9.5, handlelength=1.6)
    ygrid(a1)
    a2.bar(s.window, s.n_fraud, color=MUTED, width=0.75)
    a2.set_ylabel("Frauds")
    a2.set_xlabel("Chronological test window (steps 118-179)")
    a2.set_xlim(0.3, 50.7)
    ygrid(a2)
    fig.tight_layout(h_pad=0.6)
    cw.save(fig, "drift_pr_auc_windows")


def chart_eml_lines(cw: ChartWriter, drift: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for k in MODELS:
        s = drift[drift.model == k].sort_values("window")
        ax.plot(s.window, s.expected_monetary_loss.clip(lower=1), color=MODEL_COLORS[k],
                lw=2.4 if k == "mg_ecfm" else 1.4, label=TINY[k], zorder=4 if k == "mg_ecfm" else 3)
    s = drift[drift.model == "mg_ecfm"].sort_values("window")
    for w in s[s.n_fraud == s.n_transactions].window:
        ax.axvspan(w - 0.5, w + 0.5, color="#f0efec", zorder=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_money(v)))
    ax.set_ylabel("Expected monetary loss per txn (log)")
    ax.set_xlabel("Test window")
    ax.set_xlim(0.5, 50.5)
    ax.set_title("Expected monetary loss per window (grey: 100% fraud windows)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9.5, handlelength=1.4)
    ygrid(ax)
    fig.tight_layout()
    cw.save(fig, "drift_eml_windows")


def chart_curves(cw: ChartWriter, pred: pd.DataFrame) -> dict:
    out = {}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.1, 5.1))
    base = None
    for k in MODELS:
        s = pred[pred.model == k]
        y, p = s.y.to_numpy(), s.p.to_numpy().astype(np.float64)
        base = y.mean()
        ap = average_precision_score(y, p)
        auc = roc_auc_score(y, p)
        out[k] = {"pooled_pr_auc": float(ap), "pooled_roc_auc": float(auc)}
        pr, rc, _ = precision_recall_curve(y, p)
        idx = np.unique(np.linspace(0, len(pr) - 1, 4000).astype(int))
        a1.plot(rc[idx], pr[idx], color=MODEL_COLORS[k], lw=2.4 if k == "mg_ecfm" else 1.6,
                label=f"{TINY[k]}  AP={ap:.3f}", zorder=4 if k == "mg_ecfm" else 3)
        fpr, tpr, _ = roc_curve(y, p)
        idx = np.unique(np.linspace(0, len(fpr) - 1, 4000).astype(int))
        a2.plot(fpr[idx], tpr[idx], color=MODEL_COLORS[k], lw=2.4 if k == "mg_ecfm" else 1.6,
                label=f"{TINY[k]}  AUC={auc:.3f}", zorder=4 if k == "mg_ecfm" else 3)
    a1.axhline(base, color=MUTED, ls="--", lw=1)
    a1.text(0.02, base + 0.015, f"fraud base rate {100*base:.3f}%", ha="left", fontsize=9, color=INK2)
    a1.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    a1.set_title("Precision-recall curve (pooled test set)")
    a1.legend(loc="upper right", fontsize=9.5)
    a2.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    a2.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    a2.set_title("ROC curve (pooled test set)")
    a2.legend(loc="lower right", fontsize=9.5)
    ygrid(a1, "both")
    ygrid(a2, "both")
    fig.tight_layout(w_pad=2.5)
    cw.save(fig, "pooled_pr_roc_curves")
    return out


def chart_confusion(cw: ChartWriter, pred: pd.DataFrame) -> dict:
    out = {}
    fig, axes = plt.subplots(1, 5, figsize=(12.1, 3.5))
    cmap = sns.blend_palette(["#f3f7fd", "#9ec5f4", "#2a78d6", "#104281"], as_cmap=True)
    for ax, k in zip(axes, MODELS):
        s = pred[pred.model == k]
        cm = pd.crosstab(s.y, s.yhat).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
        rn = cm / cm.sum(axis=1, keepdims=True)
        out[k] = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
        annot = np.array([[f"{cm[i, j]:,}\n{100*rn[i, j]:.1f}%" for j in range(2)] for i in range(2)])
        sns.heatmap(rn, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=1, cbar=False, ax=ax, square=True,
                    linewidths=2, linecolor=SURFACE, annot_kws={"fontsize": 10})
        for t, v in zip(ax.texts, rn.flat):
            t.set_color("white" if v > 0.55 else INK)
        ax.set_title(SHORT[k], fontsize=11, loc="center")
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticklabels(["Legit", "Fraud"], rotation=0)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True" if k == "mg_ecfm" else "")
    fig.suptitle("Pooled test set, 940,304 txns, 666 frauds. Count and row-normalised rate. Threshold 0.5 (untuned).",
                 x=0.01, ha="left", fontsize=11, color=INK2, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.2)
    cw.save(fig, "confusion_matrices")
    return out


def chart_score_dist(cw: ChartWriter, pred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(12.1, 3.5), sharey=True)
    bins = np.linspace(-6, 0, 49)
    for ax, k in zip(axes, MODELS):
        s = pred[pred.model == k]
        lp = np.log10(np.clip(s.p.to_numpy(), 1e-6, 1.0))
        y = s.y.to_numpy()
        for cls, col, lab in [(0, MUTED, "Legit"), (1, MODEL_COLORS[k], "Fraud")]:
            v = lp[y == cls]
            w = np.full(len(v), 1.0 / len(v))
            ax.hist(v, bins=bins, weights=w, color=col, alpha=0.75 if cls else 0.5, label=lab)
        ax.axvline(np.log10(0.5), color=INK, ls="--", lw=1)
        ax.set_title(TINY[k], fontsize=11, loc="center")
        ax.set_xticks([-6, -4, -2, 0], ["1e-6", "1e-4", "0.01", "1"])
        ax.set_xlabel("Score p (log)")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ygrid(ax)
        ax.legend(loc="upper left", fontsize=9, handlelength=1)
    axes[0].set_ylabel("Share of class")
    fig.tight_layout(w_pad=0.8)
    cw.save(fig, "score_distributions")


def chart_actions(cw: ChartWriter, pred: pd.DataFrame) -> dict:
    s = pred[pred.model == "mg_ecfm"]
    counts = s.action.value_counts().reindex([0, 1, 2, 3], fill_value=0)
    by_cls = pd.crosstab(s.y, s.action).reindex(index=[0, 1], columns=[0, 1, 2, 3], fill_value=0)
    share = by_cls.div(by_cls.sum(axis=1), axis=0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.1, 4.6), gridspec_kw={"width_ratios": [1, 1.35]})
    inter = counts[[1, 2, 3]]
    wedges, _ = a1.pie(inter.to_numpy(), colors=[ACTION_COLORS[a] for a in [1, 2, 3]], startangle=90,
                       counterclock=False, wedgeprops={"width": 0.38, "edgecolor": SURFACE, "linewidth": 2})
    a1.text(0, 0.08, f"{int(inter.sum()):,}", ha="center", fontsize=17, fontweight="bold", color=INK, va="center")
    a1.text(0, -0.17, "interventions", ha="center", fontsize=10, color=INK2, va="center")
    a1.legend(wedges, [f"{ACTION_LABELS[a]}  {int(counts[a]):,}" for a in [1, 2, 3]], loc="center left",
              bbox_to_anchor=(0.98, 0.5), fontsize=10)
    a1.set_title(f"Non-approve actions ({100*inter.sum()/counts.sum():.2f}% of {int(counts.sum()):,} txns;\n"
                 f"Approve = {int(counts[0]):,})", fontsize=11)

    left = np.zeros(2)
    ylab = [f"Legit (n={int(by_cls.loc[0].sum()):,})", f"Fraud (n={int(by_cls.loc[1].sum()):,})"]
    for a in [0, 1, 2, 3]:
        v = share[a].to_numpy()
        a2.barh([0, 1], v, left=left, color=ACTION_COLORS[a], height=0.55, label=ACTION_LABELS[a],
                edgecolor=SURFACE, linewidth=1.5)
        for i in range(2):
            if v[i] > 0.04:
                a2.text(left[i] + v[i] / 2, i, f"{100*v[i]:.1f}%", ha="center", va="center", color="white",
                        fontsize=10, fontweight="bold")
        left += v
    a2.set_yticks([0, 1], ylab)
    a2.invert_yaxis()
    a2.set_xlim(0, 1)
    a2.xaxis.set_major_formatter(PercentFormatter(1.0))
    a2.set_title("Action share by true class (MG-ECFM CS-ACRC routing)", fontsize=11)
    a2.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
    fraud_row = by_cls.loc[1]
    a2.text(0.0, 1.6, "Fraud counts: " + ", ".join(f"{ACTION_LABELS[a]} {int(fraud_row[a])}" for a in [0, 1, 2, 3]),
            ha="left", va="center", fontsize=10, color=INK2, transform=a2.get_yaxis_transform())
    a2.set_ylim(1.75, -0.5)
    fig.tight_layout(w_pad=3)
    cw.save(fig, "action_mix")
    return {"counts": {ACTION_LABELS[a]: int(counts[a]) for a in counts.index},
            "fraud": {ACTION_LABELS[a]: int(by_cls.loc[1, a]) for a in [0, 1, 2, 3]},
            "legit": {ACTION_LABELS[a]: int(by_cls.loc[0, a]) for a in [0, 1, 2, 3]}}


def chart_binary_actions(cw: ChartWriter, pred: pd.DataFrame) -> dict:
    rates = pred.groupby("model", observed=True).action.apply(lambda a: float((a != 0).mean()))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = np.arange(len(MODELS))
    v = [rates[k] for k in MODELS]
    ax.barh(y, v, color=[MODEL_COLORS[k] for k in MODELS], height=0.6)
    ax.set_yticks(y, [TINY[k] for k in MODELS])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(min(v) / 2, 1.5)
    for i, x in enumerate(v):
        ax.text(x * 1.12, i, f"{100*x:.3f}%", va="center", fontsize=9.5, color=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{100*t:g}%"))
    ax.set_title("Non-approve rate, pooled (log)", fontsize=11)
    ygrid(ax, "x")
    fig.tight_layout()
    cw.save(fig, "intervention_rate_models")
    return {k: rates[k] for k in MODELS}


def chart_crc(cw: ChartWriter, drift: pd.DataFrame) -> None:
    s = drift[drift.model == "mg_ecfm"].sort_values("window")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    for col, c, lab in [("lambda_low", "#2a78d6", "lambda low (Step-Up)"), ("lambda_mid", "#eb6834", "lambda mid (Hold)"),
                        ("lambda_high", "#1baf7a", "lambda high (Block)")]:
        a1.step(s.window, s[col], where="mid", color=c, lw=2, label=lab)
    a1.set_ylabel("Threshold")
    a1.set_ylim(0, 1.05)
    a1.set_title("CS-ACRC routing thresholds per window")
    a1.legend(ncol=3, loc="lower right", fontsize=9.5)
    ygrid(a1)
    a2.bar(s.window, s.block_rate, color=ACTION_COLORS[3], width=0.8, label="Block")
    a2.bar(s.window, s.hold_rate, bottom=s.block_rate, color=ACTION_COLORS[2], width=0.8, label="Hold")
    a2.bar(s.window, s.step_up_rate, bottom=s.block_rate + s.hold_rate, color=ACTION_COLORS[1], width=0.8, label="Step-Up")
    a2.set_ylabel("Share of txns")
    a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a2.set_xlabel("Test window")
    a2.set_title("Intervention rate per window", fontsize=11)
    a2.legend(ncol=3, loc="upper left", fontsize=9.5)
    a2.set_xlim(0.3, 50.7)
    ygrid(a2)
    fig.tight_layout(h_pad=0.8)
    cw.save(fig, "crc_thresholds")


def chart_significance(cw: ChartWriter, sig: dict) -> dict:
    bl = ["bp1", "bp2", "bp3", "bp4"]
    rows = [("McNemar (correctness)", "mcnemar", None), ("Wilcoxon PR-AUC", "wilcoxon", "pr_auc"),
            ("Wilcoxon F2", "wilcoxon", "f2"), ("Wilcoxon EML", "wilcoxon", "expected_monetary_loss")]
    raw = np.zeros((len(rows), len(bl)))
    holm = np.zeros_like(raw)
    direction = np.empty(raw.shape, dtype=object)
    for i, (_, test, m) in enumerate(rows):
        for j, b in enumerate(bl):
            r = sig["mcnemar"][b] if test == "mcnemar" else sig["wilcoxon"][m][b]
            raw[i, j] = r["p_value"]
            holm[i, j] = r["p_value_holm"]
            if test == "mcnemar":
                direction[i, j] = "favours MG-ECFM" if r["favours"] == "A" else "favours baseline"
            else:
                md = r["median_difference"]
                better = md > 0 if m != "expected_monetary_loss" else md < 0
                direction[i, j] = "median diff. 0" if md == 0 else ("favours MG-ECFM" if better else "favours baseline")
    score = -np.log10(np.clip(holm, 1e-16, 1))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    cmap = sns.blend_palette(["#f3f7fd", "#9ec5f4", "#3987e5", "#184f95"], as_cmap=True)
    annot = np.empty(raw.shape, dtype=object)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            star = " *" if holm[i, j] < 0.05 else ""
            pv = fmt_p(holm[i, j])
            annot[i, j] = f"p {pv if pv.startswith('<') else '= ' + pv}{star}\n{direction[i, j]}"
    sns.heatmap(score, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=16, ax=ax, linewidths=2, linecolor=SURFACE,
                cbar_kws={"label": "-log10(Holm p)", "shrink": 0.85}, annot_kws={"fontsize": 9.5})
    for t, v in zip(ax.texts, score.flat):
        t.set_color("white" if v > 8 else INK)
    ax.set_xticklabels([TINY[b] for b in bl])
    ax.set_yticklabels([r[0] for r in rows], rotation=0)
    ax.set_title("MG-ECFM vs each baseline: Holm-adjusted p ( * = significant at 0.05)")
    fig.tight_layout()
    cw.save(fig, "significance_heatmap")
    return {"rows": [r[0] for r in rows], "raw": raw.tolist(), "holm": holm.tolist(), "direction": direction.tolist()}


def chart_paired_diff(cw: ChartWriter, sig: dict) -> None:
    pdc = sig["paired_difference_ci"]
    bl = ["bp1", "bp2", "bp3", "bp4"]
    metrics = [("pr_auc", "PR-AUC (+ favours MG-ECFM)"), ("f2", "F2 (+ favours MG-ECFM)"),
               ("brier", "Brier (- favours MG-ECFM)"), ("expected_monetary_loss", "EML (- favours MG-ECFM)")]
    metrics = [m for m in metrics if m[0] in pdc["bp1"]]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.1, 3.3))
    for ax, (m, title) in zip(axes, metrics):
        for j, b in enumerate(bl):
            r = pdc[b][m]
            ax.errorbar(r["estimate"], j, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                        fmt="o", color=MODEL_COLORS[b], ecolor=MODEL_COLORS[b], markersize=8, capsize=3, lw=2)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(range(len(bl)), [TINY[b] for b in bl])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5)
        ygrid(ax, "x")
        ax.tick_params(axis="x", labelsize=9)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Pooled difference MG-ECFM minus baseline, 95% block-bootstrap CI over windows",
                 x=0.01, ha="left", fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    cw.save(fig, "paired_difference_ci")


ARM_LABELS = {"B1_xgboost_only": "B1 XGBoost only", "B2_graph_gate_off": "B2 Graph, gate off",
              "B3_static_crc": "B3 Static CRC", "B3_adaptive_crc": "B3 Adaptive CRC", "B4_full_mg_ecfm": "B4 Full MG-ECFM",
              "B5_assist_off": "B5 Assist off", "B5_assist_on": "B5 Assist on"}


def chart_ablation(cw: ChartWriter, ab: pd.DataFrame) -> None:
    ab = ab.copy()
    ab["label"] = ab.arm.map(ARM_LABELS).fillna(ab.arm)
    metrics = [("pr_auc", "PR-AUC (pooled)", "{:.3f}"), ("f2", "F2 (pooled, thr 0.5)", "{:.3f}"),
               ("expected_monetary_loss", "EML per txn", "{:,.0f}"), ("prevented_loss_ratio_pct", "Prevented loss (%)", "{:.1f}")]
    fig, axes = plt.subplots(1, 4, figsize=(12.1, 3.9), sharey=True)
    colors = ["#898781", "#eb6834", "#1baf7a", "#2a78d6", "#2a78d6", "#2a78d6", "#2a78d6"]
    y = np.arange(len(ab))
    for ax, (col, title, f) in zip(axes, metrics):
        v = ab[col].to_numpy()
        ax.barh(y, v, color=colors[: len(ab)], height=0.62)
        ax.set_xlim(0, v.max() * 1.38)
        for i, x in enumerate(v):
            ax.text(x + v.max() * 0.03, i, f.format(x), va="center", fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=11)
        ygrid(ax, "x")
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_yticks(y, ab.label)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.0)
    cw.save(fig, "ablation_metrics")


def chart_ablation_switches(cw: ChartWriter, ab: pd.DataFrame) -> None:
    cols = [("use_graph", "Graph"), ("use_ae", "AE"), ("gate_mode", "KB gate"), ("crc_mode", "CRC"),
            ("routing", "Routing"), ("calibration", "Calib."), ("recovery", "Recovery"), ("assist", "Assist")]
    txt = ab[[c for c, _ in cols]].astype(str).replace({"True": "on", "False": "off", "auto_admit": "auto", "martingale": "mart."})
    on = txt.apply(lambda c: ~c.isin(["off", "none", "binary"])).astype(float)
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    cmap = sns.blend_palette(["#f0efec", "#9ec5f4"], as_cmap=True)
    sns.heatmap(on.to_numpy(), annot=txt.to_numpy(), fmt="", cmap=cmap, cbar=False, ax=ax, linewidths=2,
                linecolor=SURFACE, annot_kws={"fontsize": 9, "color": INK})
    ax.set_xticklabels([l for _, l in cols], rotation=0, fontsize=9.5)
    ax.set_yticklabels(ab.arm.map(ARM_LABELS).fillna(ab.arm), rotation=0, fontsize=9.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    fig.tight_layout()
    cw.save(fig, "ablation_switches")


def chart_adversarial(cw: ChartWriter, adv: pd.DataFrame, ttr: pd.DataFrame) -> dict:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.1, 4.3))
    pol = [p for p in POLICY_COLORS if p in adv.policy.unique()]
    for p in pol:
        s = adv[adv.policy == p].sort_values("poison_rate")
        a1.plot(100 * s.poison_rate, 100 * s.false_admission_rate, marker="o", markersize=7, color=POLICY_COLORS[p],
                label=POLICY_LABELS[p], lw=2)
    a1.set(xlabel="Poison rate (%)", ylabel="False admission rate (%)", ylim=(-3, 105))
    a1.set_title("Harmful rules admitted to the knowledge base")
    a1.legend(loc="center right", fontsize=9.5)
    ygrid(a1)
    width = 1.3
    for i, p in enumerate(pol):
        s = adv[adv.policy == p].sort_values("poison_rate")
        a2.bar(100 * s.poison_rate + (i - 1) * width, 100 * s.genuine_admission_rate, width=width * 0.92,
               color=POLICY_COLORS[p], label=POLICY_LABELS[p])
    a2.set(xlabel="Poison rate (%)", ylabel="Genuine rules admitted (%)", ylim=(0, 110))
    a2.set_xticks([5, 10, 15, 20, 25])
    a2.set_title("Genuine rules admitted (higher is better)")
    a2.legend(loc="upper center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    ygrid(a2)
    fig.tight_layout(w_pad=3)
    cw.save(fig, "adversarial_far")

    # Injected-rule rollback: category mix + time to rollback
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.1, 4.0), gridspec_kw={"width_ratios": [1, 1.5]})
    cat = ttr.category.value_counts()
    order = [c for c in CATEGORY_COLORS if c in cat.index]
    wedges, _ = a1.pie(cat[order], colors=[CATEGORY_COLORS[c] for c in order], startangle=90, counterclock=False,
                       wedgeprops={"width": 0.38, "edgecolor": SURFACE, "linewidth": 2})
    a1.text(0, 0.08, f"{len(ttr)}", ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    a1.text(0, -0.17, "injected rules", ha="center", va="center", fontsize=10, color=INK2)
    a1.legend(wedges, [f"{c.capitalize()}  {cat[c]}" for c in order], loc="center left", bbox_to_anchor=(0.98, 0.5))
    a1.set_title("Injected harmful rules by category (all poison rates)", fontsize=11)
    tab = pd.crosstab(ttr.poison_rate, ttr.time_to_rollback_cycles)
    bottom = np.zeros(len(tab))
    cyc_cols = sns.blend_palette(["#184f95", "#9ec5f4"], n_colors=len(tab.columns))
    for c, colr in zip(tab.columns, cyc_cols):
        a2.bar((100 * tab.index).astype(int).astype(str), tab[c], bottom=bottom, color=colr,
               label=f"{int(c)} cycle{'s' if c > 1 else ''}", edgecolor=SURFACE, linewidth=1)
        bottom += tab[c].to_numpy()
    a2.set(xlabel="Poison rate (%)", ylabel="Injected rules")
    rb = 100 * ttr.rolled_back.mean()
    a2.set_title(f"Cycles to rollback (8 firings per cycle); {rb:.0f}% rolled back", fontsize=11)
    a2.legend(title="Time to rollback", loc="upper left", fontsize=9.5, title_fontsize=9.5)
    ygrid(a2)
    fig.tight_layout(w_pad=3)
    cw.save(fig, "adversarial_rollback")

    # Ville bound check
    g = adv[adv.policy == "mg_ecfm_gate"].sort_values("poison_rate")
    fig, ax = plt.subplots(figsize=(5.9, 4.3))
    x = 100 * g.poison_rate.to_numpy()
    ax.axhline(g.ville_alpha_roll.iloc[0], color="#e34948", lw=1.5, ls="--")
    ax.text(2.5, g.ville_alpha_roll.iloc[0] + 0.002, f"alpha = {g.ville_alpha_roll.iloc[0]:g}", va="bottom", fontsize=9.5, color=INK2)
    ax.errorbar(x - 0.5, g.ville_mc_rate, yerr=[g.ville_mc_rate - g.ville_mc_ci_low, g.ville_mc_ci_high - g.ville_mc_rate],
                fmt="o", color="#2a78d6", capsize=3, markersize=7, label="Monte-Carlo crossing rate, 95% CI")
    ax.plot(x, g.ville_exact_crossing_prob, marker="s", ls="none", color="#eb6834", markersize=7, label="Exact crossing probability")
    ax.plot(x + 0.5, g.ville_ingate_rate, marker="^", ls="none", color="#1baf7a", markersize=8, label="False rollback in gate")
    ax.set(xlabel="Poison rate (%)", ylabel="Probability", ylim=(-0.003, 0.062), xlim=(2, 31))
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_title("Ville's bound under H0 (all below alpha)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, bbox_to_anchor=(0, 0.02))
    ygrid(ax)
    fig.tight_layout()
    cw.save(fig, "ville_bound")
    return {
        "far_gate_max_pct": float(100 * g.false_admission_rate.max()),
        "far_sl_max_pct": float(100 * adv[adv.policy == "sl_screen_only"].false_admission_rate.max()),
        "far_auto_min_pct": float(100 * adv[adv.policy == "auto_admit"].false_admission_rate.min()),
        "genuine_gate_min_pct": float(100 * g.genuine_admission_rate.min()),
        "genuine_gate_max_pct": float(100 * g.genuine_admission_rate.max()),
        "genuine_sl_max_pct": float(100 * adv[adv.policy == "sl_screen_only"].genuine_admission_rate.max()),
        "rollback_rate_gate_pct": float(100 * g.rollback_rate_harmful.min()),
        "ttr_max_cycles": float(ttr.time_to_rollback_cycles.max()),
        "ttr_mean_cycles": float(ttr.time_to_rollback_cycles.mean()),
        "n_injected": int(len(ttr)),
        "ttr_one_cycle_pct": float(100 * (ttr.time_to_rollback_cycles == 1).mean()),
        "ville_mc_max": float(g.ville_mc_rate.max()),
        "ville_exact": float(g.ville_exact_crossing_prob.iloc[0]),
        "ville_all_ok": bool(g.ville_bound_satisfied.astype(str).eq("True").all()),
    }


def chart_martingale(cw: ChartWriter, f2: pd.DataFrame) -> None:
    XMAX_FIRINGS = 60
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), sharey=True)
    tests = [("rollback", "Rollback test (H0: rule is good)"), ("admission", "Admission test (H0: rule is weak)")]
    for ax, (t, title) in zip(axes, tests):
        s = f2[f2.test == t]
        for (rid, grp), g in s.groupby(["rule_id", "group"]):
            g = g.sort_values("firing")
            stop = g.stopping_time.iloc[0]
            if pd.notna(stop) and np.isfinite(stop) and stop > 0:
                g = g[g.firing <= stop]
            ax.plot(g.firing, g.wealth.clip(lower=1e-8), color=GROUP_COLORS.get(grp, MUTED), lw=1.6, alpha=0.9)
        ax.set_xlim(0, XMAX_FIRINGS)
        thr = s.threshold.iloc[0]
        ax.axhline(thr, color=INK, ls="--", lw=1)
        ax.text(XMAX_FIRINGS, thr * 1.6, f"1/alpha = {thr:g}", ha="right", fontsize=9.5, color=INK2)
        ax.set_yscale("log")
        ax.set_ylim(5e-9, 300)
        ax.set_xlabel("Verified firings (curves clipped at 1e-8)")
        ax.set_title(title, fontsize=11)
        ygrid(ax)
    axes[0].set_ylabel("Martingale wealth (log)")
    for grp, c in GROUP_COLORS.items():
        axes[1].plot([], [], color=c, lw=2, label=grp.capitalize())
    axes[1].legend(loc="center right", fontsize=9.5)
    fig.tight_layout(w_pad=1.5)
    cw.save(fig, "martingale_wealth")


def chart_recovery(cw: ChartWriter, f4: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.9, 4.3))
    ks = sorted(f4.kappa.unique())
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    styles = ["-", "--", ":", "-."]
    for k, c, ls in zip(ks, cols, styles):
        s = f4[f4.kappa == k].sort_values("delay_hours")
        default = bool(s.is_default_kappa.astype(str).eq("True").any())
        ax.plot(s.delay_hours, s.prevented_loss_ratio_pct, color=c, ls=ls, lw=2.4 if default else 1.8,
                marker="o", markersize=4, label=f"kappa = {k:g}{' (default)' if default else ''}")
    wh = f4.window_hours.iloc[0]
    ax.axvline(wh, color=MUTED, lw=1, ls="--")
    ax.text(wh + 0.5, 50, f"{wh:g} h recovery window", fontsize=9.5, color=INK2)
    ax.set(xlabel="Detection delay (hours)", ylabel="Prevented loss (%)", ylim=(-3, 105))
    ax.set_title("TD-TPF recovery vs detection delay\n(the kappa curves overlap)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9.5)
    ygrid(ax)
    fig.tight_layout()
    cw.save(fig, "tdtpf_recovery")


def chart_latency_fit(cw: ChartWriter, drift: pd.DataFrame, man: dict) -> dict:
    lat = drift.groupby("model").latency_ms_per_txn.mean()
    fit = {k: man["fit_reports"][k]["fit_seconds"] for k in MODELS}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.1, 4.4))
    y = np.arange(len(MODELS))
    for ax, vals, title, f in [(a1, [lat[k] for k in MODELS], "Scoring latency, mean over windows (ms per txn)", "{:.3f} ms"),
                               (a2, [fit[k] for k in MODELS], "Fit time (seconds, CPU)", "{:,.0f} s")]:
        ax.barh(y, vals, color=[MODEL_COLORS[k] for k in MODELS], height=0.6)
        ax.set_yticks(y, [SHORT[k] for k in MODELS])
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.25)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.02, i, f.format(v), va="center", fontsize=10, color=INK)
        ax.set_title(title, fontsize=11)
        ygrid(ax, "x")
    fig.tight_layout(w_pad=3)
    cw.save(fig, "latency_fit")
    return {"latency_ms": {k: float(lat[k]) for k in MODELS}, "fit_seconds": fit}


def _clean(o):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--reports", type=Path, default=root / "reports")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = args.reports
    out = args.out or rep / "deck_charts"
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    cw = ChartWriter(out)

    drift = pd.read_csv(rep / "drift_evaluation_results.csv")
    man = json.loads((rep / "drift_evaluation_manifest.json").read_text())
    sig = json.loads((rep / "statistical_significance_results.json").read_text())
    ab = pd.read_csv(rep / "ablation_study_results.csv")
    adv = pd.read_csv(rep / "adversarial_robustness_results.csv")
    ttr = pd.read_csv(rep / "adversarial_time_to_rollback.csv")
    fd = rep / "figures" / "figure_data"

    data: dict = {"labels": {k: drift.loc[drift.model == k, "model_label"].iloc[0] for k in MODELS}, "short": SHORT}
    print("splits / summary charts")
    data["splits"] = chart_splits(cw, man)
    chart_pooled_ci(cw, sig)
    chart_window_box(cw, drift)
    chart_drift_lines(cw, drift)
    chart_eml_lines(cw, drift)
    chart_crc(cw, drift)

    print("reading predictions (compact dtypes)")
    pred = load_predictions(rep / "drift_evaluation_predictions.csv")
    data["n_prediction_rows"] = int(len(pred))
    data["curves"] = chart_curves(cw, pred)
    data["confusion"] = chart_confusion(cw, pred)
    chart_score_dist(cw, pred)
    data["actions"] = chart_actions(cw, pred)
    data["intervention_rate"] = chart_binary_actions(cw, pred)
    del pred

    print("significance / ablation / adversarial")
    data["significance"] = chart_significance(cw, sig)
    chart_paired_diff(cw, sig)
    chart_ablation(cw, ab)
    chart_ablation_switches(cw, ab)
    data["adversarial"] = chart_adversarial(cw, adv, ttr)
    if (fd / "fig2_martingale_trajectories.csv").exists():
        f2 = pd.read_csv(fd / "fig2_martingale_trajectories.csv")
        chart_martingale(cw, f2)
        st = f2.groupby(["test", "group", "rule_id"]).stopping_time.first().reset_index()
        data["martingale"] = {f"{t}_{g}": [int(st[(st.test == t) & (st.group == g) & (st.stopping_time > 0)].stopping_time.min()) if ((st.test == t) & (st.group == g) & (st.stopping_time > 0)).any() else None,
                                           int(st[(st.test == t) & (st.group == g)].stopping_time.max()),
                                           int(((st.test == t) & (st.group == g) & (st.stopping_time > 0)).sum()),
                                           int(((st.test == t) & (st.group == g)).sum())]
                              for t in st.test.unique() for g in st.group.unique()}
    f4 = None
    if (fd / "fig4_tdtpf_recovery_curves.csv").exists():
        f4 = pd.read_csv(fd / "fig4_tdtpf_recovery_curves.csv")
        chart_recovery(cw, f4)
    data["perf"] = chart_latency_fit(cw, drift, man)

    # ---- headline numbers -------------------------------------------------- #
    boot = sig["bootstrap_ci"]
    data["bootstrap"] = {k: {m: boot[k][m] for m in boot[k]} for k in MODELS}
    data["window_mean"] = {k: drift[drift.model == k][["pr_auc", "roc_auc", "f1", "f2", "brier", "ece",
                                                       "expected_monetary_loss"]].mean().to_dict() for k in MODELS}
    mg = drift[drift.model == "mg_ecfm"]
    data["windows"] = {"n": int(mg.window.nunique()), "all_fraud": int((mg.n_fraud == mg.n_transactions).sum()),
                       "pr_auc_defined": int(mg.pr_auc.notna().sum()),
                       "rows_min": int(mg.n_transactions.min()), "rows_max": int(mg.n_transactions.max()),
                       "lambda_start": [float(mg.sort_values("window").iloc[0][c]) for c in ["lambda_low", "lambda_mid", "lambda_high"]],
                       "lambda_median": [float(mg[c].median()) for c in ["lambda_low", "lambda_mid", "lambda_high"]],
                       "n_active_rules_max": float(mg.n_active_rules.max())}
    thr = drift.threshold.unique().tolist()
    data["thresholds"] = [float(t) for t in thr]
    mcn = sig["mcnemar"]
    data["accuracy"] = {"mg_ecfm": mcn["bp1"]["accuracy_a"], **{b: mcn[b]["accuracy_b"] for b in mcn}}
    data["ablation"] = ab[["arm", "pr_auc", "f1", "f2", "expected_monetary_loss", "net_monetary_loss",
                           "prevented_loss_ratio_pct", "recovery_rate_pct", "kb_rules_active_end",
                           "kb_rules_ever_admitted"]].to_dict(orient="records")
    if f4 is not None:
        cut = f4[f4.prevented_loss_ratio_pct < 100].delay_hours.min()
        full = f4[f4.prevented_loss_ratio_pct >= 100].delay_hours.max()
        data["recovery"] = {"full_until_h": float(full), "zero_from_h": float(cut),
                            "window_hours": float(f4.window_hours.iloc[0]), "n_frauds": int(f4.n_frauds.iloc[0])}
    data["run"] = {"runtime_seconds": man.get("runtime_seconds"), "profile": man.get("profile"), "seed": man.get("seed"),
                   "n_features": man["splits"]["n_features"], "protocol": man.get("protocol")}
    data["paired_diff"] = sig["paired_difference_ci"]
    data["chart_sizes"] = cw.sizes
    (out / "deck_data.json").write_text(json.dumps(_clean(data), indent=1, allow_nan=False))
    print(f"wrote {out / 'deck_data.json'}")


if __name__ == "__main__":
    main()
