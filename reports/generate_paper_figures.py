"""
reports/generate_paper_figures.py
=================================

Phase 8 - publication-grade figures for the MG-ECFM journal manuscript.

Every figure is written twice, as a vector PDF (fonts embedded, for LaTeX) and
as a 300-DPI PNG (for Word / slides), sized for the IEEE two-column layout
(single column 3.5 in, double column 7.16 in):

=========  =====================================================================  ============================
Figure     Content                                                                Data
=========  =====================================================================  ============================
Fig. 1     PR-AUC trajectories over the chronological drift windows               drift_evaluation_results.csv
           (MG-ECFM vs BP1-BP4), per-window values and 5-window rolling mean
Fig. 2     Martingale e-process wealth trajectories under poisoned rule           computed (Phase-4 gate
           injection: (a) rollback monitor M^roll of injected rules,              processes, Phase-1 contaminated
           (b) admission test M^admit of quarantined candidates; stopping         feed, configs/hyperparams.json)
           times at Ville's threshold 1/alpha
Fig. 3     CS-ACRC threshold dynamics (lambda_low, lambda_mid, lambda_high)       drift_evaluation_results.csv
           under concept drift, with the realised fraud / intervention rates
Fig. 4     TD-TPF fund-recovery rate vs time-to-detection (kappa sensitivity)      computed (Phase-5 TD-TPF over the
           and prevented-loss ratio inside the 24 h settlement window             evaluated synthetic stream)
=========  =====================================================================  ============================

The data behind the computed figures (2 and 4) is also saved as CSV next to the
images (``figure_data/``), so every plotted point is auditable.

Usage::

    python reports/generate_paper_figures.py
    python reports/generate_paper_figures.py --reports-dir reports/quick_demo --out-dir reports/quick_demo/figures
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SEED = 42
REPORTS_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "configs"
DEFAULT_OUT_DIR = REPORTS_DIR / "figures"
DPI = 300
SINGLE_COL_IN = 3.5
DOUBLE_COL_IN = 7.16

MODEL_ORDER: Tuple[str, ...] = ("mg_ecfm", "bp1", "bp2", "bp3", "bp4")
MODEL_LABEL: Dict[str, str] = {"mg_ecfm": "MG-ECFM (proposed)", "bp1": "BP1 EARN+RXT-J",
                               "bp2": "BP2 GNN+AE", "bp3": "BP3 CARE-GNN", "bp4": "BP4 ARGUS"}
# Validated categorical order (fixed slots, never cycled); every series also carries
# a distinct marker / dash so identity never relies on colour alone (print, CVD).
SLOT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MODEL_STYLE: Dict[str, Dict[str, Any]] = {
    "mg_ecfm": {"color": SLOT[0], "marker": "o", "ls": "-", "lw": 1.8, "z": 6},
    "bp1": {"color": SLOT[1], "marker": "s", "ls": "--", "lw": 1.1, "z": 4},
    "bp2": {"color": SLOT[2], "marker": "^", "ls": "-.", "lw": 1.1, "z": 4},
    "bp3": {"color": SLOT[3], "marker": "D", "ls": ":", "lw": 1.3, "z": 4},
    "bp4": {"color": SLOT[4], "marker": "v", "ls": (0, (5, 1, 1, 1)), "lw": 1.1, "z": 4},
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"
SHADE = "#efeeea"


def apply_ieee_style() -> None:
    """Serif fonts at IEEE caption sizes, Type-42 (TrueType) font embedding, recessive grid."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "grid.alpha": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 2.4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": DPI,
        "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=DPI)
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return load_json(CONFIG_DIR / "hyperparams.json"), load_json(CONFIG_DIR / "project_config.json")


# --------------------------------------------------------------------------- #
# Drift-window helpers                                                        #
# --------------------------------------------------------------------------- #
def drift_onset_window(results: pd.DataFrame, manifest: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Window index (fractional) where the synthetic camouflage drift starts, if any."""
    if not manifest:
        return None
    kw = manifest.get("splits", {}).get("synthetic_kwargs")
    if not kw or float(kw.get("drift_strength", 0.0)) <= 0.0:
        return None
    n_steps = int(kw.get("n_steps", 400))
    frac = float(kw.get("drift_start_fraction", 0.6))
    start_step = int(round(1 + frac * (n_steps - 1)))
    win = results.drop_duplicates("window").sort_values("window")
    if start_step <= int(win["start_step"].min()):
        return float(win["window"].min()) - 0.5
    hit = win[win["end_step"] >= start_step]
    return float(hit["window"].iloc[0]) - 0.5 if not hit.empty else None


def rolling_mean(values: np.ndarray, k: int) -> np.ndarray:
    s = pd.Series(values, dtype="float64")
    return s.rolling(k, center=True, min_periods=max(1, k // 2)).mean().to_numpy()


# --------------------------------------------------------------------------- #
# Figure 1 - PR-AUC trajectories                                              #
# --------------------------------------------------------------------------- #
def figure_pr_auc(results: pd.DataFrame, manifest: Optional[Mapping[str, Any]], out_dir: Path,
                  smooth: int = 5) -> Dict[str, str]:
    models = [m for m in MODEL_ORDER if m in set(results["model"])]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 2.55))
    onset = drift_onset_window(results, manifest)
    n_win = int(results["window"].max())
    first = float(results["window"].min()) - 0.5
    if onset is not None and onset > first:
        ax.axvspan(onset, n_win + 0.5, color=SHADE, zorder=0, lw=0)
        ax.axvline(onset, color=INK_2, lw=0.7, ls="--", zorder=1)
        ax.text(onset + 0.4, 0.985, "camouflage drift onset", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=6.8, color=INK_2)
    elif onset is not None:
        ax.text(0.01, 0.985, "camouflage drift active over the whole test period (strength ramps up linearly)",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color=INK_2)
    k = max(1, min(smooth, n_win))
    handles = []
    for m in models:
        sub = results[results["model"] == m].sort_values("window")
        x = sub["window"].to_numpy()
        y = sub["pr_auc"].to_numpy(dtype=np.float64)
        st = MODEL_STYLE[m]
        ax.plot(x, y, color=st["color"], lw=0.6, alpha=0.35, zorder=st["z"] - 2)
        ax.plot(x, rolling_mean(y, k), color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st["z"],
                marker=st["marker"], markersize=3.2, markevery=max(1, n_win // 10),
                markeredgecolor="white", markeredgewidth=0.4)
        mean = float(np.nanmean(y))
        handles.append(Line2D([0], [0], color=st["color"], lw=st["lw"], ls=st["ls"], marker=st["marker"],
                              markersize=3.2, markeredgecolor="white", markeredgewidth=0.4,
                              label=f"{MODEL_LABEL[m]} (mean {mean:.3f})"))
    ax.set_xlim(0.5, n_win + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Chronological test window")
    ax.set_ylabel("PR-AUC")
    ax.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
    ax.text(0.995, 0.03, f"thin: per window, bold: {k}-window rolling mean", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=INK_2)
    return save_figure(fig, out_dir, "fig1_pr_auc_drift_trajectories")


# --------------------------------------------------------------------------- #
# Figure 2 - martingale e-process wealth                                      #
# --------------------------------------------------------------------------- #
CATEGORY_STYLE: Dict[str, Dict[str, Any]] = {
    "genuine": {"color": SLOT[0], "ls": "-", "label": "genuine rule"},
    "poisoned": {"color": SLOT[1], "ls": "--", "label": "poisoned rule"},
    "sybil": {"color": SLOT[6], "ls": "-.", "label": "Sybil-endorsed rule"},
    "stale": {"color": SLOT[3], "ls": ":", "label": "stale rule"},
    "conflicting": {"color": SLOT[4], "ls": (0, (5, 1, 1, 1)), "label": "conflicting rule"},
}


def _trajectory(process, xs: Sequence[int]) -> Tuple[np.ndarray, Optional[int]]:
    """Wealth path ``M_0 = 1, M_1, ...`` until (and including) the Ville crossing."""
    path = [process.wealth]
    stop = None
    for i, x in enumerate(xs):
        crossed = process.update(int(x))
        path.append(process.wealth)
        if crossed:
            stop = i + 1
            break
    return np.asarray(path, dtype=np.float64), stop


def martingale_trajectories(hyperparams: Mapping[str, Any], project_config: Mapping[str, Any],
                            poison_rate: float = 0.15, horizon: int = 150, n_per_group: int = 6,
                            seed: int = SEED) -> pd.DataFrame:
    """Simulated firing streams of a contaminated feed through the gate's e-processes.

    For each selected rule the verified firing outcomes are Bernoulli draws of
    its *true* precision. ``rollback`` feeds the false-alarm indicator
    ``1 - y`` to ``M^roll`` (rule injected as ACTIVE), ``admission`` feeds the
    hit indicator ``y`` to ``M^admit`` (rule in quarantine). The processes are
    built exactly as :class:`MartingaleEProcessGate` builds them (betting
    strategy, monitor, plug-in prior from the Subjective-Logic opinion).
    """
    from src.data.perturbation_generator import PerturbationConfig, PerturbationGenerator
    from src.epistemic_gate.martingale_eprocess import GateConfig, admission_process, rollback_process
    from src.epistemic_gate.subjective_logic import EpistemicFusionEngine

    cfg = GateConfig.from_config(hyperparams)
    engine = EpistemicFusionEngine.from_config(hyperparams)
    gen = PerturbationGenerator(PerturbationConfig.from_configs(project_config, hyperparams), seed=seed)
    feed = gen.generate_feed(float(poison_rate))
    harmful = [r for r in feed.rules if r.is_harmful and r.category in ("poisoned", "sybil")]
    genuine = [r for r in feed.rules if not r.is_harmful and (1.0 - r.true_precision) <= cfg.q0]
    pick = sorted(harmful, key=lambda r: r.rule_id)[:n_per_group] + \
        sorted(genuine, key=lambda r: r.rule_id)[:n_per_group]
    rng = np.random.default_rng([seed, 8])
    rows: List[Dict[str, Any]] = []
    for r in pick:
        y = (rng.random(horizon) < r.true_precision).astype(int)
        group = "genuine" if not r.is_harmful else r.category
        roll_bet = "likelihood_ratio" if cfg.rollback_monitor == "e_detector" else cfg.betting
        roll = rollback_process(cfg.q0, cfg.q1, cfg.alpha_roll, betting=roll_bet, monitor=cfg.rollback_monitor,
                                prior_mean=cfg.q1, prior_strength=cfg.plugin_prior_strength)
        prior = engine.fuse_rule(r, feed.current_step).opinion.expected_probability
        adm = admission_process(cfg.p0, cfg.p1, cfg.alpha_admit, betting=cfg.betting, prior_mean=prior,
                                prior_strength=cfg.plugin_prior_strength)
        for test, proc, xs, alpha in (("rollback", roll, 1 - y, cfg.alpha_roll),
                                      ("admission", adm, y, cfg.alpha_admit)):
            path, stop = _trajectory(proc, xs)
            for t, w in enumerate(path):
                rows.append({"test": test, "rule_id": r.rule_id, "group": group,
                             "true_precision": r.true_precision, "firing": t, "wealth": w,
                             "threshold": 1.0 / alpha, "stopping_time": stop if stop is not None else -1})
    return pd.DataFrame(rows)


def figure_martingale(traj: pd.DataFrame, out_dir: Path, poison_rate: float) -> Dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.5), sharey=True)
    panels = (("rollback", "(a) Rollback monitor $M_t^{roll}$ (rules injected as active)",
               "$H_0^{good}$: FPR $\\leq q_0$"),
              ("admission", "(b) Admission test $M_t^{admit}$ (quarantined candidates)",
               "$H_0$: precision $\\leq p_0$"))
    used_groups: List[str] = []
    for ax, (test, title, h0) in zip(axes, panels):
        sub = traj[traj["test"] == test]
        if sub.empty:
            continue
        thr = float(sub["threshold"].iloc[0])
        horizon = int(sub["firing"].max())
        for rid, path in sub.groupby("rule_id", sort=True):
            g = str(path["group"].iloc[0])
            st = CATEGORY_STYLE.get(g, CATEGORY_STYLE["poisoned"])
            if g not in used_groups:
                used_groups.append(g)
            ax.plot(path["firing"], path["wealth"], color=st["color"], ls=st["ls"], lw=1.0, zorder=3)
            stop = int(path["stopping_time"].iloc[0])
            if stop > 0:
                ax.plot([stop], [path["wealth"].iloc[-1]], marker="X", color=st["color"], markersize=5,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        ax.axhline(thr, color=INK, lw=0.8, ls="-", zorder=4)
        ax.text(horizon, thr * 1.25, f"Ville threshold $1/\\alpha = {thr:g}$", ha="right", va="bottom",
                fontsize=6.6, color=INK)
        ax.axhline(1.0, color=INK_2, lw=0.5, ls=":", zorder=2)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, thr * 30.0)
        ax.set_xlim(0, horizon)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Verified firings of the rule")
        stops = sub.drop_duplicates("rule_id")
        harm = stops[stops["group"] != "genuine"]["stopping_time"]
        good = stops[stops["group"] == "genuine"]["stopping_time"]
        if test == "rollback":
            txt = (f"{h0}; harmful stopped: {int((harm > 0).sum())}/{len(harm)} "
                   f"(median $\\tau$ = {np.median(harm[harm > 0]) if (harm > 0).any() else float('nan'):.0f}), "
                   f"genuine stopped: {int((good > 0).sum())}/{len(good)}")
        else:
            txt = (f"{h0}; genuine admitted: {int((good > 0).sum())}/{len(good)} "
                   f"(median $\\tau$ = {np.median(good[good > 0]) if (good > 0).any() else float('nan'):.0f}), "
                   f"harmful admitted: {int((harm > 0).sum())}/{len(harm)}")
        ax.text(0.0, -0.30, txt, transform=ax.transAxes, fontsize=6.3, color=INK_2, ha="left", va="top")
    axes[0].set_ylabel("Wealth $M_t$ (log scale)")
    handles = [Line2D([0], [0], color=CATEGORY_STYLE[g]["color"], ls=CATEGORY_STYLE[g]["ls"], lw=1.0,
                      label=CATEGORY_STYLE[g]["label"]) for g in used_groups]
    handles.append(Line2D([0], [0], marker="X", color=INK_2, lw=0, markersize=5, label="stopping time"))
    fig.legend(handles=handles, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.text(0.995, 1.0, f"feed contamination $\\rho$ = {100 * poison_rate:.0f}%", ha="right", va="bottom",
             fontsize=6.5, color=INK_2)
    fig.subplots_adjust(wspace=0.08)
    return save_figure(fig, out_dir, "fig2_martingale_wealth_trajectories")


# --------------------------------------------------------------------------- #
# Figure 3 - CS-ACRC thresholds                                               #
# --------------------------------------------------------------------------- #
def figure_crc_thresholds(results: pd.DataFrame, manifest: Optional[Mapping[str, Any]], out_dir: Path,
                          target_alpha: float) -> Dict[str, str]:
    sub = results[results["model"] == "mg_ecfm"].sort_values("window")
    if sub.empty or sub[["lambda_low", "lambda_mid", "lambda_high"]].isna().all().all():
        raise ValueError("drift results contain no MG-ECFM CS-ACRC thresholds")
    x = sub["window"].to_numpy()
    n_win = int(x.max())
    onset = drift_onset_window(results, manifest)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL_IN, 3.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.12})
    if onset is not None and onset <= float(x.min()) - 0.5:
        onset = None
    for ax in (ax1, ax2):
        if onset is not None:
            ax.axvspan(onset, n_win + 0.5, color=SHADE, zorder=0, lw=0)
            ax.axvline(onset, color=INK_2, lw=0.7, ls="--", zorder=1)
    lam = [("lambda_high", "$\\lambda^{high}$ (Hold | Block)", SLOT[1], "^", "-"),
           ("lambda_mid", "$\\lambda^{mid}$ (Step-Up | Hold)", SLOT[2], "s", "--"),
           ("lambda_low", "$\\lambda^{low}$ (Approve | Step-Up)", SLOT[0], "o", "-.")]
    for col, label, color, marker, ls in lam:
        y = sub[col].to_numpy(dtype=np.float64)
        ax1.step(x, y, where="mid", color=color, lw=1.3, ls=ls, label=label, zorder=3)
        ax1.plot(x, y, lw=0, marker=marker, markersize=2.6, color=color, markevery=max(1, n_win // 10),
                 zorder=4)
    ax1.fill_between(x, sub["lambda_low"], 0, step="mid", color=SLOT[0], alpha=0.07, lw=0)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_ylabel("Threshold $\\lambda_t$ (risk score)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.33), ncol=2, fontsize=6.4, columnspacing=0.8)
    if onset is not None:
        ax1.text(onset + 0.4, 0.02, "drift", fontsize=6.4, color=INK_2, ha="left", va="bottom")

    fraud = 100.0 * sub["fraud_rate"].to_numpy(dtype=np.float64)
    inter = 100.0 * sub["intervention_rate"].to_numpy(dtype=np.float64)
    ax2.plot(x, fraud, color=INK_2, lw=1.0, ls="-", marker="o", markersize=2.2, label="realised fraud rate")
    ax2.plot(x, inter, color=SLOT[0], lw=1.2, ls="--", marker="s", markersize=2.2,
             label="MG-ECFM intervention rate")
    ax2.set_ylabel("Share of window (%)")
    ax2.set_xlabel("Chronological test window")
    ax2.set_xlim(0.5, n_win + 0.5)
    ax2.set_ylim(0.0, max(1.0, float(np.nanmax(np.concatenate([fraud, inter]))) * 1.25))
    ax2.legend(loc="upper left", fontsize=6.4, ncol=1)
    eml = sub["expected_monetary_loss"].to_numpy(dtype=np.float64)
    ax2.text(0.99, 0.95, f"target risk $\\alpha$ = {target_alpha:g}\nmean EML = {np.nanmean(eml):.0f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=6.2, color=INK_2)
    return save_figure(fig, out_dir, "fig3_cs_acrc_threshold_dynamics")


# --------------------------------------------------------------------------- #
# Figure 4 - TD-TPF recovery vs detection delay                               #
# --------------------------------------------------------------------------- #
DEFAULT_DELAYS_HOURS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0,
                                           22.0, 23.0, 24.0, 25.0, 28.0, 32.0, 36.0)


def evaluated_canonical_stream(manifest: Optional[Mapping[str, Any]]) -> Tuple[pd.DataFrame, str]:
    """Rebuild the canonical (src, dst, amount, time) stream the drift evaluation used.

    The synthetic stream is regenerated deterministically from the manifest's
    ``synthetic_kwargs``; for a real PaySim run the raw file is read through
    the Phase-1 loader (``max_rows`` from the manifest).
    """
    from experiments.run_drift_eval import (inject_camouflage_drift, raw_paysim_to_canonical,
                                            raw_dataset_available, load_configs as drift_configs)
    from src.data.dataset_loaders import build_loader, make_synthetic_paysim_frame

    splits = (manifest or {}).get("splits", {})
    source = str(splits.get("source", "synthetic"))
    if source.startswith("synthetic"):
        kw = dict(splits.get("synthetic_kwargs") or {})
        n_steps = int(kw.get("n_steps", 400))
        tx = tuple(kw.get("tx_per_step", (50, 90)))
        raw = make_synthetic_paysim_frame(n_steps=n_steps, tx_per_step=(int(tx[0]), int(tx[1])),
                                          n_customers=int(kw.get("n_customers", 1500)),
                                          n_merchants=int(kw.get("n_merchants", 150)),
                                          fraud_rate=float(kw.get("fraud_rate", 0.03)),
                                          seed=int(kw.get("seed", SEED)))
        start = int(round(1 + float(kw.get("drift_start_fraction", 0.6)) * (n_steps - 1)))
        raw = inject_camouflage_drift(raw, start, float(kw.get("drift_strength", 0.5)),
                                      seed=int(kw.get("seed", SEED)))
        return raw_paysim_to_canonical(raw), source
    _, pc = drift_configs()
    if source == "paysim" and raw_dataset_available("paysim", pc):
        loader = build_loader("paysim", pc, repo_root=ROOT, max_rows=splits.get("max_rows") or 2_000_000)
        return loader.read_all(), source
    raise FileNotFoundError(f"cannot rebuild the canonical stream for source '{source}'")


def recovery_curves(canonical: pd.DataFrame, hyperparams: Mapping[str, Any], project_config: Mapping[str, Any],
                    delays: Sequence[float] = DEFAULT_DELAYS_HOURS, kappas: Optional[Sequence[float]] = None,
                    max_frauds: int = 250, seed: int = SEED) -> pd.DataFrame:
    """Portfolio recovery / prevented-loss ratio per detection delay and decay rate ``kappa``."""
    from src.recovery.action_simulator import RecoverySimulator
    from src.recovery.tainted_flow import EDGE_INTERNAL, EdgeSchema, TaintedFlowTracker

    base = TaintedFlowTracker.config_kwargs(hyperparams, project_config)
    kappa0 = float(base["kappa"])
    kappas = list(kappas) if kappas is not None else sorted({round(kappa0 / 2, 4), kappa0, round(kappa0 * 2, 4)})
    rows: List[Dict[str, Any]] = []
    edges = TaintedFlowTracker.edges_from_canonical(canonical)
    schema = EdgeSchema.paysim()
    candidates = [e for e in edges if e.label == 1 and e.amount > 0
                  and schema.classify(e) == EDGE_INTERNAL]
    if not candidates:
        raise ValueError("no traceable (internal-transfer) frauds in the stream")
    rng = np.random.default_rng([seed, 4])
    if len(candidates) > max_frauds:
        idx = np.sort(rng.choice(len(candidates), size=max_frauds, replace=False))
        candidates = [candidates[i] for i in idx]
    for kappa in kappas:
        kw = dict(base)
        kw["kappa"] = float(kappa)
        tracker = TaintedFlowTracker(edges, schema, **kw)
        sim = RecoverySimulator.from_config(tracker, hyperparams, project_config)
        for d in delays:
            frame = sim.simulate_many(candidates, float(d))
            agg = RecoverySimulator.aggregate(frame)
            rows.append({"kappa": float(kappa), "delay_hours": float(d), "n_frauds": int(len(candidates)),
                         "fraud_amount": agg["fraud_amount"], "recovered_amount": agg["recovered_amount"],
                         "recovery_rate_pct": agg["recovery_rate_pct"],
                         "prevented_loss_ratio_pct": agg["prevented_loss_ratio_pct"],
                         "mean_accounts_frozen": float(frame["n_accounts_frozen"].mean()),
                         "share_within_window": agg["share_within_window"],
                         "window_hours": float(kw["window_hours"]),
                         "is_default_kappa": bool(abs(kappa - kappa0) < 1e-12)})
    return pd.DataFrame(rows)


def figure_recovery(curves: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    window = float(curves["window_hours"].iloc[0])
    kappas = sorted(curves["kappa"].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.45))
    styles = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "D")]
    slots = [SLOT[2], SLOT[0], SLOT[1], SLOT[6]]
    for ax in (ax1, ax2):
        ax.axvspan(window, float(curves["delay_hours"].max()) + 0.5, color=SHADE, lw=0, zorder=0)
        ax.axvline(window, color=INK_2, lw=0.7, ls="--", zorder=1)
        ax.text(window + 0.4, 0.97, f"{window:g} h settlement\nwindow closes", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=6.2, color=INK_2)
        ax.axvline(1.0, color=SLOT[0], lw=0.6, ls=":", zorder=1)
    for i, kappa in enumerate(kappas):
        sub = curves[curves["kappa"] == kappa].sort_values("delay_hours")
        ls, mk = styles[i % len(styles)]
        default = bool(sub["is_default_kappa"].iloc[0])
        ax1.plot(sub["delay_hours"], sub["recovery_rate_pct"], color=slots[i % len(slots)], ls=ls, marker=mk,
                 markersize=2.8, lw=1.6 if default else 1.0, zorder=4 if default else 3,
                 label=f"$\\kappa$ = {kappa:g} h$^{{-1}}$" + (" (config)" if default else ""))
    ax1.set_xlabel("Time-to-detection after the fraudulent transfer (h)")
    ax1.set_ylabel("Fund-recovery rate (%)")
    ax1.set_title("(a) TD-TPF recovered share of the stolen amount", loc="left")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.06, 0.10), fontsize=6.4)
    ax1.text(1.15, 0.03, "step-up bypass alert (1 h)", transform=ax1.get_xaxis_transform(), rotation=90,
             fontsize=6.0, color=SLOT[0], va="bottom", ha="left")

    dflt = curves[curves["is_default_kappa"]].sort_values("delay_hours")
    ax2.plot(dflt["delay_hours"], dflt["prevented_loss_ratio_pct"], color=SLOT[0], lw=1.6, marker="o",
             markersize=2.8, label="prevented-loss ratio (freeze + block + chargeback)")
    ax2.plot(dflt["delay_hours"], dflt["recovery_rate_pct"], color=SLOT[2], lw=1.0, ls="--", marker="s",
             markersize=2.6, label="recovery rate (freeze + chargeback)")
    ax2.set_xlabel("Time-to-detection after the fraudulent transfer (h)")
    ax2.set_ylabel("Share of exposure (%)")
    ax2.set_title(f"(b) Loss prevention at the configured $\\kappa$ = {dflt['kappa'].iloc[0]:g}", loc="left")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.06, 0.10), fontsize=6.4)
    n = int(curves["n_frauds"].iloc[0])
    fig.text(0.995, 1.0, f"{n} traceable fraudulent transfers", ha="right", va="bottom", fontsize=6.5,
             color=INK_2)
    fig.subplots_adjust(wspace=0.22)
    return save_figure(fig, out_dir, "fig4_tdtpf_recovery_vs_detection_delay")


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def generate_figures(reports_dir: Path = REPORTS_DIR, out_dir: Optional[Path] = None,
                     poison_rate: float = 0.15, horizon: int = 150, max_frauds: int = 250,
                     seed: int = SEED, verbose: bool = True) -> Dict[str, Dict[str, str]]:
    """Render Figures 1-4 (PDF + 300-DPI PNG); returns ``{figure: {pdf, png}}``."""
    reports_dir = Path(reports_dir)
    out_dir = Path(out_dir) if out_dir is not None else (
        DEFAULT_OUT_DIR if reports_dir.resolve() == REPORTS_DIR.resolve() else reports_dir / "figures")
    data_dir = out_dir / "figure_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_path = reports_dir / "drift_evaluation_results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"{results_path} not found - run the drift evaluation first")
    results = pd.read_csv(results_path)
    manifest_path = reports_dir / "drift_evaluation_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else None
    hp, pc = load_configs()
    apply_ieee_style()
    out: Dict[str, Dict[str, str]] = {}

    out["fig1"] = figure_pr_auc(results, manifest, out_dir)
    if verbose:
        print(f"[figure] fig1 PR-AUC trajectories -> {out['fig1']['pdf']}", flush=True)

    traj = martingale_trajectories(hp, pc, poison_rate=poison_rate, horizon=horizon, seed=seed)
    traj.to_csv(data_dir / "fig2_martingale_trajectories.csv", index=False)
    out["fig2"] = figure_martingale(traj, out_dir, poison_rate)
    if verbose:
        print(f"[figure] fig2 martingale wealth     -> {out['fig2']['pdf']}", flush=True)

    alpha = float(hp.get("conformal_crc", {}).get("target_risk_alpha", 0.02))
    out["fig3"] = figure_crc_thresholds(results, manifest, out_dir, alpha)
    if verbose:
        print(f"[figure] fig3 CS-ACRC thresholds    -> {out['fig3']['pdf']}", flush=True)

    canonical, _ = evaluated_canonical_stream(manifest)
    curves = recovery_curves(canonical, hp, pc, max_frauds=max_frauds, seed=seed)
    curves.to_csv(data_dir / "fig4_tdtpf_recovery_curves.csv", index=False)
    out["fig4"] = figure_recovery(curves, out_dir)
    if verbose:
        print(f"[figure] fig4 TD-TPF recovery       -> {out['fig4']['pdf']}", flush=True)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate 300-DPI MG-ECFM publication figures (PDF + PNG).")
    ap.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--poison-rate", type=float, default=0.15, help="feed contamination for Fig. 2")
    ap.add_argument("--horizon", type=int, default=150, help="firings per rule in Fig. 2")
    ap.add_argument("--max-frauds", type=int, default=250, help="traceable frauds sampled for Fig. 4")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    generate_figures(args.reports_dir, args.out_dir, args.poison_rate, args.horizon, args.max_frauds, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
