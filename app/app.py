"""
app/app.py - MG-ECFM interactive dashboard (Phase 8)
====================================================

Streamlit console for examiners and analysts::

    streamlit run app/app.py          (or: python main.py --ui)

Tabs
----
1. Live Scoring       single-transaction authorisation (< 5 ms path) with action badge,
                      risk dial over the live CS-ACRC thresholds, TreeSHAP reason attribution,
                      fired knowledge-base rules and the expected loss of every action.
2. Mule Network       two-hop neighbourhood of the transaction and the camouflage filter's
                      keep / prune decision for every neighbour edge.
3. Batch Scorer       upload a transaction CSV (PaySim or canonical schema), download decisions.
4. Knowledge Base     every candidate rule with its gate status, Dirichlet opinion (b, d, u, a),
                      live e-process wealth and the versioned snapshot history.
5. Recovery Trace     TD-TPF fund tracing and freeze / block / chargeback protocol for a fraud
                      detected after a chosen delay.

The service is loaded from ``$MG_ECFM_SERVICE``, ``models/mg_ecfm_service.pkl``
(``python main.py --mode train``) or ``models/quick_demo_service.pkl``
(``python main.py --mode quick-demo``); if none exists a demo service is built once
and cached.
"""
from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from src.orchestrator.service import ACTION_LABELS, MGECFMService, ServiceConfig  # noqa: E402

LATENCY_BUDGET_MS = 5.0
ACTION_STYLE: Dict[str, Dict[str, str]] = {
    "Approve": {"bg": "#e3f4e6", "fg": "#0f5c1e", "bar": "#2e9e46", "icon": "✔"},
    "Step-Up": {"bg": "#fdf3d3", "fg": "#6b4e00", "bar": "#e0a800", "icon": "⚠"},
    "Hold": {"bg": "#fde5d2", "fg": "#7a3300", "bar": "#e26a12", "icon": "⏸"},
    "Block": {"bg": "#fbdcdc", "fg": "#7d1010", "bar": "#c62828", "icon": "⛔"},
}
INK, INK_2, GRID = "#1b1b1b", "#5b5a56", "#e2e1dc"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
TX_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]

st.set_page_config(page_title="MG-ECFM Console", page_icon="\U0001f6e1", layout="wide")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                     "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
st.markdown("""
<style>
.badge {display:inline-block;padding:0.55rem 1.1rem;border-radius:0.6rem;font-size:1.35rem;font-weight:700;
        letter-spacing:0.02em;border:1px solid rgba(0,0,0,0.08);}
.small-note {color:#5b5a56;font-size:0.85rem;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Service                                                                     #
# --------------------------------------------------------------------------- #
def candidate_paths() -> List[Path]:
    env = os.environ.get("MG_ECFM_SERVICE")
    paths = [Path(env)] if env else []
    return paths + [ROOT / "models" / "mg_ecfm_service.pkl", ROOT / "models" / "quick_demo_service.pkl"]


@st.cache_resource(show_spinner="Loading the MG-ECFM service ...")
def get_service() -> Tuple[MGECFMService, str]:
    for path in candidate_paths():
        if path.is_file():
            return MGECFMService.load(path), str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    service = MGECFMService.build(ServiceConfig.for_profile("demo"), verbose=False)
    out = ROOT / "models" / "quick_demo_service.pkl"
    service.save(out)
    return service, "built on first launch (demo profile) -> models/quick_demo_service.pkl"


@st.cache_data(show_spinner=False)
def cached_latency(_service: MGECFMService, key: str, n: int) -> Dict[str, float]:
    return _service.latency_benchmark(n=n)


@st.cache_data(show_spinner="Running TD-TPF ...")
def cached_trace(_service: MGECFMService, key: str, tx_id: int, delay: float) -> Dict[str, Any]:
    return _service.recovery_trace(tx_id, delay)


def fmt_money(x: float) -> str:
    return f"{x:,.2f}"


def action_badge(label: str, p: float) -> str:
    s = ACTION_STYLE[label]
    return (f"<span class='badge' style='background:{s['bg']};color:{s['fg']}'>{s['icon']} {label}</span>"
            f"&nbsp;&nbsp;<span class='small-note'>calibrated fraud risk p&#770; = {p:.4f}</span>")


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def risk_dial(score: float, thresholds: Tuple[float, float, float], binary_tau: float) -> plt.Figure:
    """Semicircular gauge: action bands from the live CS-ACRC thresholds, needle at the risk score."""
    lo, mid, hi = thresholds
    fig, ax = plt.subplots(figsize=(4.2, 2.45))
    bands = [(0.0, lo, "Approve"), (lo, mid, "Step-Up"), (mid, hi, "Hold"), (hi, 1.0, "Block")]
    for a, b, label in bands:
        if b <= a:
            continue
        ax.add_patch(Wedge((0, 0), 1.0, 180 - 180 * b, 180 - 180 * a, width=0.28,
                           facecolor=ACTION_STYLE[label]["bar"], edgecolor="white", linewidth=2))
        th = math.radians(180 - 90 * (a + b))
        ax.text(1.17 * math.cos(th), 1.17 * math.sin(th), label, ha="center", va="center", fontsize=7.5,
                color=INK_2)
    th = math.radians(180 - 180 * min(max(score, 0.0), 1.0))
    ax.plot([0, 0.80 * math.cos(th)], [0, 0.80 * math.sin(th)], color=INK, lw=2.4, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.06, color=INK))
    tt = math.radians(180 - 180 * binary_tau)
    ax.plot([0.70 * math.cos(tt), 1.02 * math.cos(tt)], [0.70 * math.sin(tt), 1.02 * math.sin(tt)],
            color=INK_2, lw=1.0, ls=":")
    ax.text(0, -0.18, f"risk score {score:.3f}", ha="center", va="top", fontsize=10, color=INK, weight="bold")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.32, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    return fig


def attribution_plot(attrib: pd.DataFrame, top: int = 10) -> plt.Figure:
    d = attrib.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(d) + 0.8))
    colors = [ORANGE if v > 0 else BLUE for v in d["shap_margin"]]
    ax.barh(d["component"], d["shap_margin"], color=colors, height=0.6)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("TreeSHAP contribution to the XGBoost margin (log-odds)")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(d["shap_margin"]):
        ax.text(v, i, f" {v:+.2f} ", va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK)
    fig.tight_layout()
    return fig


def graph_plot(nodes: pd.DataFrame, edges: pd.DataFrame) -> plt.Figure:
    """Radial layout: sender left, receiver right, hop-1 around its anchor, hop-2 one ring further."""
    pos: Dict[str, Tuple[float, float]] = {}
    sender = nodes.loc[nodes["role"] == "sender", "account"].iloc[0]
    receiver = nodes.loc[nodes["role"] == "receiver", "account"].iloc[0]
    centres = {sender: (-1.6, 0.0), receiver: (1.6, 0.0)}
    pos.update(centres)
    for hop, radius in ((1, 1.05), (2, 1.85)):
        layer = nodes[(nodes["hop"] == hop) & (~nodes["account"].isin(list(pos)))]
        for anchor, grp in layer.groupby("attached_to", sort=False):
            cx, cy = centres.get(anchor, pos.get(anchor, (0.0, 0.0)))
            side = -1.0 if cx < 0 else 1.0
            k = len(grp)
            span = math.radians(250 if hop == 1 else 200)
            base = math.pi if side < 0 else 0.0
            for i, acc in enumerate(grp["account"]):
                ang = base + (-span / 2 + span * (i + 0.5) / k) * (1 if side > 0 else -1)
                pos[acc] = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for _, e in edges.iterrows():
        a, c = pos.get(e["neighbour"]), pos.get(e["centre"])
        if a is None or c is None:
            continue
        if e["kind"] == "transaction":
            ax.annotate("", xy=pos[receiver], xytext=pos[sender],
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2, shrinkA=14, shrinkB=14))
            continue
        if e["kept"]:
            ax.plot([a[0], c[0]], [a[1], c[1]], color="#9b9a95", lw=1.2, zorder=1)
        else:
            ax.plot([a[0], c[0]], [a[1], c[1]], color=ACTION_STYLE["Block"]["bar"], lw=1.2, ls=(0, (4, 2)),
                    zorder=1)
            ax.plot([(a[0] + c[0]) / 2], [(a[1] + c[1]) / 2], marker="x", color=ACTION_STYLE["Block"]["bar"],
                    markersize=7, mew=2, zorder=2)
    for _, n in nodes.iterrows():
        x, y = pos.get(n["account"], (0.0, 0.0))
        role = n["role"]
        face = BLUE if role == "sender" else (VIOLET if role == "receiver" else ("#ffffff" if n["hop"] == 1
                                                                                  else "#f1f0ec"))
        edge = ACTION_STYLE["Block"]["bar"] if n["verified_flag"] else INK_2
        marker = "s" if n["is_merchant"] else "o"
        size = 420 if role in ("sender", "receiver") else 150
        ax.scatter([x], [y], s=size, c=face, marker=marker, edgecolors=edge,
                   linewidths=2.6 if n["verified_flag"] else 1.0, zorder=3)
        label = n["account"] if len(str(n["account"])) <= 12 else str(n["account"])[:4] + "…" + str(n["account"])[-5:]
        ax.text(x, y - (0.30 if role in ("sender", "receiver") else 0.22), label, ha="center", va="top",
                fontsize=7 if role not in ("sender", "receiver") else 8.5, color=INK, zorder=4)
    ax.text(-1.6, 0.42, "sender", ha="center", fontsize=9, color=BLUE, weight="bold")
    ax.text(1.6, 0.42, "receiver", ha="center", fontsize=9, color=VIOLET, weight="bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#9b9a95", lw=1.4, label="kept neighbour edge"),
               Line2D([0], [0], color=ACTION_STYLE["Block"]["bar"], lw=1.4, ls=(0, (4, 2)), marker="x",
                      label="pruned by camouflage filter"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                      markeredgecolor=ACTION_STYLE["Block"]["bar"], markeredgewidth=2.2, markersize=9,
                      label="verified fraud counterparty"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor=INK_2,
                      markersize=8, label="merchant")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def recovery_curve_plot(curve: pd.DataFrame, delay: float, window: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.axvspan(window, max(curve["delay_hours"].max(), delay) + 0.5, color="#efeeea", lw=0)
    ax.axvline(window, color=INK_2, ls="--", lw=0.8)
    ax.plot(curve["delay_hours"], curve["recovery_rate_pct"], color=AQUA, marker="o", ms=3, lw=1.6,
            label="recovery rate")
    ax.plot(curve["delay_hours"], curve["prevented_loss_ratio_pct"], color=BLUE, marker="s", ms=3, lw=1.2,
            ls="--", label="prevented-loss ratio")
    ax.axvline(delay, color=ORANGE, lw=1.2)
    ax.text(delay, 1.02, f" detected after {delay:g} h", color=ORANGE, fontsize=8,
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("time-to-detection (h)")
    ax.set_ylabel("% of the stolen amount")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def kb_plot(kb: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    colors = {"active": AQUA, "quarantined": "#e0a800", "rejected": "#9b9a95", "retired": VIOLET}
    for status, grp in kb.groupby("status"):
        ax.scatter(grp["belief"], grp["uncertainty"], s=26, alpha=0.85, label=f"{status} ({len(grp)})",
                   color=colors.get(status, INK_2), edgecolors="white", linewidths=0.5)
    ax.set_xlabel("belief b")
    ax.set_ylabel("uncertainty u")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.plot([0, 1], [1, 0], color=INK_2, lw=0.6, ls=":")
    ax.text(0.52, 0.52, "d = 0", rotation=-38, fontsize=7, color=INK_2)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Layout                                                                      #
# --------------------------------------------------------------------------- #
service, origin = get_service()
info = service.describe()
lam = info["crc_thresholds"]

st.title("MG-ECFM — Martingale-Gated Epistemic-Conformal Fraud Mitigation")
st.caption("Risk-aware prevention · camouflage-resistant relational detection · "
           "statistically certified knowledge evolution · post-incident fund recovery")

with st.sidebar:
    st.header("Live service")
    st.markdown(f"**Model:** `{origin}`")
    st.markdown(f"**Stream:** {info['source']} · profile `{info['profile']}`")
    st.markdown(f"**History:** {info['n_history']:,} scored test tx · {info['n_accounts']:,} accounts · "
                f"last step {info['last_step']}")
    st.markdown("**CS-ACRC thresholds** (live)")
    st.markdown(f"λ<sub>low</sub> = {lam[0]:.3f} · λ<sub>mid</sub> = {lam[1]:.3f} · "
                f"λ<sub>high</sub> = {lam[2]:.3f}", unsafe_allow_html=True)
    st.markdown(f"**Binary operating threshold** τ = {info['threshold']:.3f}")
    st.markdown(f"**Active KB rules:** {info['n_active_rules']}")
    tm = info["build_report"].get("test_metrics", {})
    if tm:
        st.markdown(f"**Test period:** PR-AUC {tm.get('pr_auc', float('nan')):.3f} · "
                    f"F1 {tm.get('f1', float('nan')):.3f} · Brier {tm.get('brier', float('nan')):.4f}")
    st.divider()
    if st.button("Profile authorisation latency (200 tx)"):
        st.session_state["latency"] = cached_latency(service, origin, 200)
    if "latency" in st.session_state:
        lat = st.session_state["latency"]
        st.metric("p95 authorisation latency", f"{lat['p95_ms']:.2f} ms",
                  delta=f"{LATENCY_BUDGET_MS - lat['p95_ms']:+.2f} ms vs 5 ms budget")
        st.caption(f"p50 {lat['p50_ms']:.2f} · p99 {lat['p99_ms']:.2f} · max {lat['max_ms']:.2f} ms; "
                   f"causal featurisation p50 {lat['featurize_p50_ms']:.2f} ms (timed separately)")

tabs = st.tabs(["⚡ Live Scoring", "\U0001f578 Mule Network", "\U0001f4c4 Batch Scorer",
                "\U0001f4da Knowledge Base", "\U0001f9ed Recovery Trace"])

# ---- shared transaction inputs ------------------------------------------------
presets = service.example_transactions()
preset_names = list(presets)

with tabs[0]:
    st.subheader("Single-transaction authorisation")
    c0, c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.4, 1.1, 1.1])
    preset = c0.selectbox("Scenario", preset_names, index=0, key="preset")
    ex = presets[preset]
    src = c1.text_input("Sender account", value=str(ex["src"]), key=f"src_{preset}")
    dst = c2.text_input("Receiver account", value=str(ex["dst"]), key=f"dst_{preset}")
    amount = c3.number_input("Amount", min_value=0.0, value=float(ex["amount"]), step=100.0, key=f"amt_{preset}")
    tx_type = c4.selectbox("Type", TX_TYPES, index=TX_TYPES.index(ex["tx_type"]) if ex["tx_type"] in TX_TYPES else 0,
                           key=f"type_{preset}")
    st.session_state["tx"] = {"src": src.strip(), "dst": dst.strip(), "amount": float(amount), "tx_type": tx_type}
    if not src.strip() or not dst.strip():
        st.warning("Enter both a sender and a receiver account.")
    else:
        res = service.score_transaction(src.strip(), dst.strip(), float(amount), tx_type, explain=True)
        st.markdown(action_badge(res["action_label"], res["p_hat"]), unsafe_allow_html=True)
        reason_txt = {"threshold": "risk score fell in this band of the CS-ACRC thresholds",
                      "kb_rule_floor": "an admitted knowledge-base rule fired (rule floor: at least Step-Up)",
                      "amount_escalation": "amount-aware escalation of a low score",
                      "uncertain_block_to_hold": "Block downgraded to Hold because the calibration interval is wide"}
        st.caption(f"Routing reason: {reason_txt.get(res['route_reason'], res['route_reason'])} · "
                   f"step {res['step']} (what-if: the live state is not modified)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Authorisation latency", f"{res['latency_ms']:.2f} ms",
                  delta=f"{LATENCY_BUDGET_MS - res['latency_ms']:+.2f} ms vs budget",
                  delta_color="normal" if res["latency_ms"] <= LATENCY_BUDGET_MS else "inverse")
        m2.metric("Calibrated risk p̂", f"{res['p_hat']:.4f}")
        m3.metric("GNN relational probability", f"{res['gnn_proba']:.3f}" if res["gnn_proba"] is not None else "n/a")
        m4.metric("AE residual percentile", f"{res['ae_residual_percentile']:.0f}%"
                  if np.isfinite(res["ae_residual_percentile"]) else "n/a")
        g1, g2 = st.columns([1, 1.25])
        with g1:
            st.markdown("**Risk dial** (bands = live CS-ACRC thresholds, dotted = binary τ)")
            fig = risk_dial(res["risk_score"], res["thresholds"], res["binary_threshold"])
            st.pyplot(fig, clear_figure=True)
            el = pd.DataFrame({"action": list(res["expected_loss"]),
                               "expected loss": [fmt_money(v) for v in res["expected_loss"].values()]})
            st.markdown("**Expected loss of each action** at this p̂ and amount")
            st.dataframe(el, hide_index=True, use_container_width=True)
        with g2:
            st.markdown("**Reason attribution** (TreeSHAP on the fusion vector, orange raises risk)")
            st.pyplot(attribution_plot(res["attribution"]), clear_figure=True)
            parts = [f"{k.replace('_ms', '')} {v:.2f} ms" for k, v in res["component_ms"].items()]
            st.caption("Component timings: " + " · ".join(parts) + f" · routing {res['route_ms']:.3f} ms"
                       f" · causal features {res['featurize_ms']:.2f} ms (before the authorisation path)")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Knowledge-base rules firing**")
            if res["fired_rules"]:
                st.dataframe(pd.DataFrame(res["fired_rules"]).rename(columns={"weight": "E[p] (opinion)"}),
                             hide_index=True, use_container_width=True)
            else:
                st.caption("No admitted rule fires on this transaction.")
            flags = [f"sender: {res['src_flag']}" if res["src_flag"] else "",
                     f"receiver: {res['dst_flag']}" if res["dst_flag"] else ""]
            flags = [f for f in flags if f]
            if flags:
                st.info("Verified history: " + "; ".join(flags))
        with r2:
            st.markdown("**Most unusual causal features** (signed-log z-score vs training split)")
            dev = res["feature_deviation"].head(8).copy()
            dev["value"] = dev["value"].map(lambda v: f"{v:,.3f}")
            dev["z_vs_train"] = dev["z_vs_train"].map(lambda v: f"{v:+.2f}")
            st.dataframe(dev, hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("Camouflage mule-network visualiser")
    tx = st.session_state.get("tx", next(iter(presets.values())))
    st.caption(f"Transaction from the Live Scoring tab: {tx['src']} → {tx['dst']} · "
               f"{tx['tx_type']} · {tx['amount']:,.2f}")
    if tx["src"] and tx["dst"]:
        g = service.explain_graph(tx["src"], tx["dst"], tx["amount"], tx["tx_type"])
        stt = g["stats"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Neighbourhood nodes", stt["n_nodes"])
        k2.metric("Neighbour edges", stt["n_neighbour_edges"])
        k3.metric("Pruned (camouflage)", stt["n_pruned"], delta=f"{100 * (1 - stt['kept_fraction']):.0f}% of edges",
                  delta_color="off")
        k4.metric("GNN probability", f"{stt['gnn_proba']:.3f}")
        if stt["n_nodes"] <= 2:
            st.info("Both accounts have no recorded counterparties yet (cold start): the relational branch "
                    "sees only the transaction itself.")
        st.pyplot(graph_plot(g["nodes"], g["edges"]), clear_figure=True)
        st.caption(f"Label-aware similarity filter: edges with S(u, w) below {stt['similarity_threshold']:.2f} "
                   f"({stt['filter_mode']} mode) are pruned before GraphSAGE aggregation, so camouflaged mule "
                   "links cannot dilute the receiver's embedding.")
        e1, e2 = st.columns([1.2, 1])
        with e1:
            st.markdown("**Neighbour edges**")
            show = g["edges"][g["edges"]["kind"] != "transaction"][["neighbour", "centre", "kind", "similarity",
                                                                     "kept"]]
            st.dataframe(show.sort_values("similarity"), hide_index=True, use_container_width=True)
        with e2:
            st.markdown("**Accounts**")
            st.dataframe(g["nodes"][["account", "role", "hop", "attached_to", "verified_flag"]], hide_index=True,
                         use_container_width=True)

with tabs[2]:
    st.subheader("Batch CSV scorer")
    st.markdown("Upload transactions in the **PaySim** schema (`step, type, amount, nameOrig, nameDest[, isFraud]`) "
                "or the **canonical** schema (`step, tx_type, amount, src, dst[, label]`). Post-event balance "
                "columns are stripped before featurisation (causal-leakage contract); rows are scored after the "
                "live history without modifying it.")
    from src.data.dataset_loaders import make_synthetic_paysim_frame
    sample = make_synthetic_paysim_frame(n_steps=4, tx_per_step=(20, 30), n_customers=900, n_merchants=90,
                                         fraud_rate=0.03, seed=11)
    st.download_button("Download a sample CSV", sample.to_csv(index=False).encode("utf-8"),
                       file_name="mg_ecfm_sample_transactions.csv", mime="text/csv")
    up = st.file_uploader("Transaction CSV", type=["csv"])
    use_sample = st.checkbox("Score the sample instead", value=up is None)
    frame: Optional[pd.DataFrame] = None
    if up is not None and not use_sample:
        try:
            frame = pd.read_csv(up)
        except Exception as exc:  # malformed upload
            st.error(f"Could not read the CSV: {exc}")
    elif use_sample:
        frame = sample
    if frame is not None:
        try:
            scored, summ = service.score_frame(frame)
        except ValueError as exc:
            st.error(str(exc))
        else:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Transactions", f"{summ['n']:,}")
            b2.metric("Scoring time", f"{summ['ms_per_txn']:.3f} ms/tx")
            b3.metric("Interventions", f"{(scored['action'] >= 1).mean() * 100:.1f}%")
            if "metrics" in summ:
                b4.metric("PR-AUC (labels in file)", f"{summ['metrics']['pr_auc']:.3f}")
            else:
                b4.metric("Labels", "none in file")
            if summ["dropped_leakage_columns"]:
                st.caption("Stripped leakage columns: " + ", ".join(summ["dropped_leakage_columns"]))
            dist = scored["action_label"].value_counts().reindex(list(ACTION_STYLE)).fillna(0).astype(int)
            fig, ax = plt.subplots(figsize=(5.2, 1.9))
            ax.barh(dist.index[::-1], dist.values[::-1], color=[ACTION_STYLE[k]["bar"] for k in dist.index[::-1]],
                    height=0.6)
            for i, v in enumerate(dist.values[::-1]):
                ax.text(v, i, f" {v}", va="center", fontsize=8)
            ax.set_xlabel("transactions")
            ax.grid(axis="y", visible=False)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.dataframe(scored.head(500), use_container_width=True, hide_index=True)
            buf = io.StringIO()
            scored.to_csv(buf, index=False)
            st.download_button("Download scored CSV", buf.getvalue().encode("utf-8"),
                               file_name="mg_ecfm_scored.csv", mime="text/csv")

with tabs[3]:
    st.subheader("Live knowledge-base inspector")
    kb = service.kb_table()
    ks = service.kb_summary()
    counts = ks.get("counts", {})
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Active", counts.get("active", info["n_active_rules"]))
    a2.metric("Quarantined", counts.get("quarantined", 0))
    a3.metric("Rejected", counts.get("rejected", 0))
    a4.metric("Promotions / rollbacks", f"{ks.get('n_promotions', 0)} / {ks.get('n_rollbacks', 0)}")
    integ = ks.get("integrity", {})
    a5.metric("Store integrity", "OK" if integ.get("ok", True) else "FAILED",
              delta=f"snapshot {ks.get('head_snapshot_id', '-')}", delta_color="off")
    if "alpha_admit" in ks:
        st.caption(f"Admission: H₀ precision ≤ p₀ = {ks['p0']:.2f}, promoted when Mᵃᵈᵐ "
                   f"≥ 1/α = {1 / ks['alpha_admit']:.0f}. Rollback: H₀ FPR ≤ q₀ = "
                   f"{ks['q0']:.2f}, rolled back when Mʳᵒˡˡ ≥ {1 / ks['alpha_roll']:.0f} "
                   "(Ville's inequality).")
    f1, f2 = st.columns([1, 1.6])
    with f1:
        st.pyplot(kb_plot(kb), clear_figure=True)
        st.caption("Dirichlet/Beta opinions ω = (b, d, u, a): points on the dotted line have d = 0.")
    with f2:
        statuses = sorted(kb["status"].unique().tolist())
        default = [s_ for s_ in ("active", "quarantined") if s_ in statuses] or statuses
        sel = st.multiselect("Status filter", statuses, default=default)
        view = kb[kb["status"].isin(sel)].copy()
        for c in ("belief", "disbelief", "uncertainty", "base_rate", "expected_probability"):
            view[c] = view[c].round(3)
        for c in ("admission_wealth", "rollback_wealth"):
            view[c] = view[c].map(lambda v: f"{v:.3g}" if np.isfinite(v) else "")
        st.dataframe(view, hide_index=True, use_container_width=True, height=360)
        st.caption("`simulated_harmful` / `true_precision` are the ground truth of the synthetic contaminated feed, "
                   "shown for evaluation only - the gate never sees them.")
    with st.expander("Versioned snapshot history"):
        st.dataframe(service.kb_snapshots(), hide_index=True, use_container_width=True)

with tabs[4]:
    st.subheader("Post-incident recovery trace (TD-TPF)")
    cands = service.recovery_candidates()
    if cands.empty:
        st.info("No fraud in the scored history.")
    else:
        st.caption("Frauds of the test period. Those MG-ECFM did not intercept (Approve / Step-Up) settle and must be "
                   "recovered; internal transfers can be traced through the mule network, cash-outs cannot.")
        only_tr = st.checkbox("Only traceable internal transfers", value=True)
        pool = cands[cands["traceable"]] if only_tr and cands["traceable"].any() else cands
        labels = [f"tx {int(r.tx_id)} · {r.src}→{r.dst} · {r.amount:,.0f} · {r.action_label}"
                  for r in pool.itertuples()]
        pick = st.selectbox("Fraudulent transaction", range(len(pool)), format_func=lambda i: labels[i])
        delay = st.slider("Time-to-detection after the transfer (hours)", 0.25, 36.0, 2.0, 0.25)
        row = pool.iloc[int(pick)]
        if not bool(row["traceable"]):
            st.warning(f"This fraud is a {row['edge_kind']} edge: funds left the reachable ledger on the first hop, "
                       "TD-TPF has nothing to freeze.")
        rec = cached_trace(service, origin, int(row["tx_id"]), float(delay))
        mt = rec["metrics"]
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Recovered", f"{mt['recovery_rate_pct']:.1f}%", delta=fmt_money(mt["recovered_amount"]),
                  delta_color="off")
        t2.metric("Prevented-loss ratio", f"{mt['prevented_loss_ratio_pct']:.1f}%")
        t3.metric("Accounts frozen", int(mt["n_accounts_frozen"]))
        t4.metric("Within 24 h window", "yes" if mt["within_window"] else "no",
                  delta=f"net savings {fmt_money(mt['net_savings'])}", delta_color="off")
        c_left, c_right = st.columns([1.1, 1])
        with c_left:
            st.pyplot(recovery_curve_plot(rec["curve"], delay, rec["window_hours"]), clear_figure=True)
            tm_ = rec["trace_metrics"]
            if tm_:
                st.caption(f"Taint decay κ = {rec['kappa']:g}/h · mule wallets {int(tm_['n_mule_wallets'])} "
                           f"· max hop {int(tm_['max_hop'])} · exited {tm_['exit_rate_pct']:.1f}% · "
                           f"conservation residual {tm_['conservation_residual']:.1e}")
        with c_right:
            st.markdown("**Recovery protocol actions**")
            acts = rec["actions"]
            if not acts.empty:
                st.dataframe(acts[[c for c in ("kind", "target", "tau", "amount_targeted", "amount_recovered",
                                               "cost", "executed", "succeeded") if c in acts.columns]],
                             hide_index=True, use_container_width=True)
            else:
                st.caption("No action executed (window expired).")
        if not rec["nodes"].empty:
            n1, n2 = st.columns(2)
            with n1:
                st.markdown("**Tainted accounts**")
                st.dataframe(rec["nodes"].sort_values("taint", ascending=False), hide_index=True,
                             use_container_width=True)
            with n2:
                st.markdown("**Tainted flow edges**")
                st.dataframe(rec["edges"], hide_index=True, use_container_width=True)
