"""
reports/generate_paper_tables.py
================================

Phase 8 - camera-ready tables for the MG-ECFM journal manuscript
(IEEE Access / IEEE TDSC style).

The script only *reads* the experiment outputs written by Phase 7; it never
re-runs a model, so every number in the manuscript is traceable to a CSV/JSON
file in ``reports/``:

=======  ===============================================  ============================================
Table    Content                                          Source files
=======  ===============================================  ============================================
1        Dataset summary and causal-ledger contract       drift_evaluation_manifest.json,
                                                          configs/project_config.json
2        Benchmark: MG-ECFM vs BP1-BP4 (PR-AUC, F1,       drift_evaluation_results.csv,
         Brier, expected monetary loss, latency)          statistical_significance_results.json
3        Five-point ablation matrix (B1 - B5)             ablation_study_results.csv
4        Adversarial stress test and Ville-bound          adversarial_robustness_results.csv
         verification (5 % - 25 % poisoning)
=======  ===============================================  ============================================

Outputs (default ``reports/``)::

    paper_tables.tex   LaTeX (IEEEtran-compatible: \\hline rules, caption above,
                       \\footnotesize, table* for two-column floats)
    paper_tables.md    GitHub-flavoured Markdown with the same numbers

Usage::

    python reports/generate_paper_tables.py
    python reports/generate_paper_tables.py --reports-dir reports/quick_demo --out-dir reports/quick_demo
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "configs"

MODEL_ORDER: Tuple[str, ...] = ("mg_ecfm", "bp1", "bp2", "bp3", "bp4")
MODEL_SHORT: Dict[str, str] = {
    "mg_ecfm": "MG-ECFM (proposed)",
    "bp1": "BP1: EARN+RXT-J [Almazroi \\& Ayub, 2023]",
    "bp2": "BP2: GNN+AE [Alarfaj \\& Shahzadi, 2025]",
    "bp3": "BP3: CARE-GNN [Dou et al., 2020]",
    "bp4": "BP4: ARGUS [Nayak \\& Bushara, 2026]",
}
ARM_ORDER: Tuple[str, ...] = ("B1_xgboost_only", "B2_graph_gate_off", "B3_static_crc", "B3_adaptive_crc",
                              "B4_full_mg_ecfm", "B5_assist_off", "B5_assist_on")
ARM_SHORT: Dict[str, str] = {
    "B1_xgboost_only": "B1 XGBoost only",
    "B2_graph_gate_off": "B2 Graph, gate OFF",
    "B3_static_crc": "B3 Static CRC",
    "B3_adaptive_crc": "B3 Adaptive CS-ACRC",
    "B4_full_mg_ecfm": "B4 Full MG-ECFM",
    "B5_assist_off": "B5 Assist OFF",
    "B5_assist_on": "B5 Assist ON",
}
POLICY_ORDER: Tuple[str, ...] = ("mg_ecfm_gate", "sl_screen_only", "auto_admit")

REQUIRED_INPUTS: Tuple[str, ...] = (
    "drift_evaluation_manifest.json", "drift_evaluation_results.csv",
    "ablation_study_results.csv", "adversarial_robustness_results.csv",
)


# --------------------------------------------------------------------------- #
# Generic table model                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class PaperTable:
    """Format-agnostic table: header, rows of cell strings, notes.

    Cells may contain the inline markers ``**...**`` (bold) which are rendered
    as ``\\textbf{...}`` in LaTeX and kept as-is in Markdown. ``panels`` maps a
    row index to a panel title printed as a spanning row before that row.
    """

    key: str
    caption: str
    header: List[str]
    rows: List[List[str]]
    align: Sequence[str]
    notes: List[str] = field(default_factory=list)
    panels: Dict[int, str] = field(default_factory=dict)
    wide: bool = False
    font: str = "footnotesize"

    def __post_init__(self) -> None:
        n = len(self.header)
        if isinstance(self.align, str):
            self.align = list(self.align)
        if len(self.align) != n:
            raise ValueError(f"{self.key}: align '{self.align}' does not match {n} columns")
        for i, row in enumerate(self.rows):
            if len(row) != n:
                raise ValueError(f"{self.key}: row {i} has {len(row)} cells, expected {n}")


LATEX_SPECIALS = {"%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$", "{": r"\{", "}": r"\}",
                  "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def latex_escape(text: str) -> str:
    """Escape LaTeX specials, keeping already-escaped ``\\&`` and math segments ``$...$``.

    Text between a pair of ``$`` signs written as ``$$...$$`` in the source is
    passed through as inline math ``$...$``.
    """
    out: List[str] = []
    parts = str(text).split("$$")
    for k, part in enumerate(parts):
        if k % 2 == 1:
            out.append(f"${part}$")
            continue
        buf = []
        i = 0
        while i < len(part):
            ch = part[i]
            if ch == "\\" and i + 1 < len(part) and part[i + 1] in "&%_":
                buf.append(part[i:i + 2])
                i += 2
                continue
            if ch == "&":
                buf.append(r"\&")
            elif ch in LATEX_SPECIALS:
                buf.append(LATEX_SPECIALS[ch])
            elif ch in ("✓", "✗"):
                buf.append(r"\checkmark" if ch == "✓" else r"$\times$")
            elif ch == "±":
                buf.append(r"$\pm$")
            elif ch == "≤":
                buf.append(r"$\leq$")
            elif ch == "≥":
                buf.append(r"$\geq$")
            elif ch == "α":
                buf.append(r"$\alpha$")
            elif ch == "−":
                buf.append(r"$-$")
            elif ch == "∞":
                buf.append(r"$\infty$")
            elif ch in "<>":
                buf.append(f"${ch}$")
            elif ch == "—":
                buf.append("---")
            elif ch == "–":
                buf.append("--")
            else:
                buf.append(ch)
            i += 1
        out.append("".join(buf))
    return "".join(out)


def _latex_cell(cell: str) -> str:
    cell = str(cell)
    pieces = cell.split("**")
    rendered = []
    for k, piece in enumerate(pieces):
        esc = latex_escape(piece)
        rendered.append(f"\\textbf{{{esc}}}" if k % 2 == 1 and esc else esc)
    return "".join(rendered)


def _md_cell(cell: str) -> str:
    text = str(cell).replace("\\&", "&").replace("$$", "$").replace("|", "\\|")
    return text


def render_latex(table: PaperTable) -> str:
    env = "table*" if table.wide else "table"
    n = len(table.header)
    lines = [f"% ---- {table.key} " + "-" * max(0, 60 - len(table.key)),
             f"\\begin{{{env}}}[!t]",
             f"\\caption{{{latex_escape(table.caption)}}}",
             f"\\label{{tab:{table.key}}}",
             "\\centering",
             f"\\{table.font}",
             "\\renewcommand{\\arraystretch}{1.15}",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\resizebox{\\linewidth}{!}{%" if table.wide else "{%",
             f"\\begin{{tabular}}{{{''.join(table.align)}}}",
             "\\hline",
             " & ".join(f"\\textbf{{{_latex_cell(h)}}}" for h in table.header) + " \\\\",
             "\\hline"]
    for i, row in enumerate(table.rows):
        if i in table.panels:
            if i > 0:
                lines.append("\\hline")
            lines.append(f"\\multicolumn{{{n}}}{{l}}{{\\textit{{{latex_escape(table.panels[i])}}}}} \\\\")
            lines.append("\\hline")
        lines.append(" & ".join(_latex_cell(c) for c in row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}}")
    if table.notes:
        lines.append("\\vspace{2pt}")
        lines.append(f"\\begin{{minipage}}{{\\{'textwidth' if table.wide else 'columnwidth'}}}")
        lines.append("\\scriptsize")
        for note in table.notes:
            lines.append(_latex_cell(note) + "\\\\")
        lines.append("\\end{minipage}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines)


def render_markdown(table: PaperTable, number: int) -> str:
    md_align = {"l": ":---", "c": ":---:", "r": "---:"}
    lines = [f"### Table {number}. {_md_cell(table.caption)}", ""]
    lines.append("| " + " | ".join(_md_cell(h) for h in table.header) + " |")
    lines.append("|" + "|".join(md_align.get(a[0], ":---") for a in table.align) + "|")
    for i, row in enumerate(table.rows):
        if i in table.panels:
            lines.append("| *" + _md_cell(table.panels[i]) + "* |" + " |" * (len(table.header) - 1))
        lines.append("| " + " | ".join(_md_cell(c) for c in row) + " |")
    if table.notes:
        lines.append("")
        for note in table.notes:
            lines.append(f"> {_md_cell(note)}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Number formatting                                                           #
# --------------------------------------------------------------------------- #
def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fmt(x: Any, nd: int = 3, dash: str = "—") -> str:
    if not finite(x):
        if isinstance(x, (float, np.floating)) and math.isinf(float(x)):
            return "∞" if float(x) > 0 else "-∞"
        return dash
    return f"{float(x):.{nd}f}"


def fmt_int(x: Any, dash: str = "—") -> str:
    return f"{int(round(float(x))):,}" if finite(x) else dash


def fmt_money(x: Any, dash: str = "—") -> str:
    if not finite(x):
        return dash
    v = float(x)
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.1f}"


def fmt_pct(x: Any, nd: int = 1, scale: float = 1.0) -> str:
    return f"{float(x) * scale:.{nd}f}" if finite(x) else "—"


def fmt_p(p: Any) -> str:
    if not finite(p):
        return "—"
    p = float(p)
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3f}"


def mean_std(values: Sequence[float], nd: int = 3) -> Tuple[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "—", float("nan")
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return f"{m:.{nd}f} ± {s:.{nd}f}", m


def bold_best(cells: List[str], values: Sequence[float], higher_is_better: bool) -> List[str]:
    """Wrap the best finite entry (ties included) in ``**``."""
    vals = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(vals)
    if not ok.any():
        return cells
    best = np.nanmax(vals[ok]) if higher_is_better else np.nanmin(vals[ok])
    return [f"**{c}**" if ok[i] and np.isclose(vals[i], best, rtol=1e-9, atol=1e-12) else c
            for i, c in enumerate(cells)]


def check(flag: Any) -> str:
    if isinstance(flag, str):
        flag = flag.strip().lower() in ("true", "1", "yes")
    if not isinstance(flag, (bool, np.bool_)) and not finite(flag):
        return "—"
    return "✓" if bool(flag) else "✗"


# --------------------------------------------------------------------------- #
# Input loading                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentInputs:
    reports_dir: Path
    manifest: Dict[str, Any]
    drift: pd.DataFrame
    ablation: pd.DataFrame
    stress: pd.DataFrame
    significance: Optional[Dict[str, Any]]
    project_config: Dict[str, Any]
    hyperparams: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_inputs(reports_dir: Path = REPORTS_DIR) -> ExperimentInputs:
    """Read all Phase-7 outputs; raises ``FileNotFoundError`` listing missing files."""
    reports_dir = Path(reports_dir)
    missing = [name for name in REQUIRED_INPUTS if not (reports_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing experiment outputs in {reports_dir}: {missing}. "
                                "Run the Phase-7 experiments (python main.py --mode full-pipeline) first.")
    sig_path = reports_dir / "statistical_significance_results.json"
    return ExperimentInputs(
        reports_dir=reports_dir,
        manifest=_read_json(reports_dir / "drift_evaluation_manifest.json"),
        drift=pd.read_csv(reports_dir / "drift_evaluation_results.csv"),
        ablation=pd.read_csv(reports_dir / "ablation_study_results.csv"),
        stress=pd.read_csv(reports_dir / "adversarial_robustness_results.csv"),
        significance=_read_json(sig_path) if sig_path.is_file() else None,
        project_config=_read_json(CONFIG_DIR / "project_config.json"),
        hyperparams=_read_json(CONFIG_DIR / "hyperparams.json"),
    )


# --------------------------------------------------------------------------- #
# Table 1 - datasets and causal ledger contract                               #
# --------------------------------------------------------------------------- #
def table_dataset_summary(inp: ExperimentInputs) -> PaperTable:
    pc = inp.project_config
    splits = inp.manifest.get("splits", {})
    source = str(splits.get("source", "unknown"))
    n_feat = splits.get("n_features")
    rows: List[List[str]] = []
    panels: Dict[int, str] = {}

    panels[len(rows)] = "(a) Causal ledger contract per dataset"
    ds_names = {"paysim": "PaySim", "ieee_cis": "IEEE-CIS", "amlsim": "AMLSim"}
    for key, ds in pc.get("datasets", {}).items():
        stripped = list(ds.get("leakage_columns", [])) + list(ds.get("rule_artifact_columns", []))
        step = int(ds.get("step_seconds", 3600))
        unit = "1 h" if step == 3600 else ("1 day" if step == 86400 else f"{step} s")
        entities = ", ".join(ds.get("entity_columns", [])) or ("orig$$\\to$$dest" if ds.get("include_destination")
                                                                else "orig")
        rows.append([ds_names.get(key, key), unit,
                     ", ".join(stripped) if stripped else "none (no post-event fields)",
                     entities, ", ".join(ds.get("inflow_types", [])) or "—",
                     fmt_int(ds["expected_rows"]) if "expected_rows" in ds else
                     (fmt_int(ds["subset_rows"]) + " (subset)" if "subset_rows" in ds else "—")])

    panels[len(rows)] = f"(b) Chronological split of the evaluated stream: {source}"
    for name in ("train", "calibration", "operating", "test"):
        part = splits.get(name)
        if not part:
            continue
        n, k = part.get("transactions", float("nan")), part.get("frauds", float("nan"))
        rate = 100.0 * k / n if finite(n) and float(n) > 0 and finite(k) else float("nan")
        use = {"train": "fit XGBoost / GraphSAGE / DAE; gate replay",
               "calibration": "isotonic calibration; gate replay",
               "operating": "$$\\tau$$ selection; CS-ACRC warm-up",
               "test": f"{splits.get('n_windows', '?')} prequential drift windows"}[name]
        rows.append([name.capitalize(), f"{part.get('start_step')}–{part.get('end_step')}",
                     f"{fmt_int(n)} tx / {fmt_int(k)} fraud ({rate:.2f}%)" if finite(rate) else fmt_int(n),
                     use, "—", "—"])

    vel = pc.get("causal_features", {}).get("velocity_windows", [])
    delay = pc.get("verification", {})
    notes = [
        "Contract: every feature of transaction $$e_t$$ is computed only from events with step $$< \\tau_t$$ "
        "(intra-step permutation invariant); stripped columns are removed at ingestion and asserted absent "
        "before featurisation.",
        f"Causal feature vector: {n_feat if n_feat is not None else '?'} features (log-amount, hour, one-hot type, "
        f"velocity windows {vel} steps, per-entity running moments, pre-event ledger balance and deltas). "
        f"Verified labels are released with a {delay.get('label_delay_days_min', 1)}–"
        f"{delay.get('label_delay_days_max', 14)} day dispute/chargeback delay (one window in the drift protocol).",
    ]
    return PaperTable(
        key="dataset_summary",
        caption="Dataset summary and causal-ledger contract of the MG-ECFM evaluation.",
        header=["Dataset / split", "Step", "Stripped (leakage) columns / size", "Entity keys / role",
                "Inflow types", "Rows"],
        rows=rows, align=["l", "l", "p{4.2cm}", "p{3.8cm}", "l", "l"], notes=notes, panels=panels, wide=True)


# --------------------------------------------------------------------------- #
# Table 2 - benchmark                                                         #
# --------------------------------------------------------------------------- #
def _significance_marks(sig: Optional[Mapping[str, Any]], model: str) -> Tuple[str, str]:
    """``(Wilcoxon PR-AUC Holm p, McNemar Holm p)`` of MG-ECFM vs ``model``."""
    if not sig or model == "mg_ecfm":
        return "ref.", "ref."
    w = sig.get("wilcoxon", {}).get("pr_auc", {}).get(model, {})
    m = sig.get("mcnemar", {}).get(model, {})
    w_p = w.get("p_value_holm", w.get("p_value"))
    m_p = m.get("p_value_holm", m.get("p_value"))
    w_txt = fmt_p(w_p)
    if finite(w_p) and float(w_p) < 0.05:
        sign = "+" if finite(w.get("p_value_one_sided")) and float(w["p_value_one_sided"]) < 0.5 else "−"
        w_txt = f"{w_txt} ({sign})"
    return w_txt, fmt_p(m_p)


def table_benchmark(inp: ExperimentInputs) -> PaperTable:
    df = inp.drift
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    metrics = [("pr_auc", True, 3), ("f1", True, 3), ("brier", False, 4),
               ("expected_monetary_loss", False, 1), ("latency_ms_per_txn", False, 3)]
    cols: Dict[str, List[str]] = {k: [] for k, _, _ in metrics}
    vals: Dict[str, List[float]] = {k: [] for k, _, _ in metrics}
    for m in models:
        sub = df[df["model"] == m]
        for key, _, nd in metrics:
            txt, mean = mean_std(sub[key].to_numpy(dtype=np.float64), nd)
            cols[key].append(txt)
            vals[key].append(mean)
    for key, hib, _ in metrics:
        cols[key] = bold_best(cols[key], vals[key], hib)
    rows = []
    for i, m in enumerate(models):
        w_p, m_p = _significance_marks(inp.significance, m)
        rows.append([MODEL_SHORT.get(m, m)] + [cols[k][i] for k, _, _ in metrics] + [w_p, m_p])
    n_win = int(df["window"].nunique())
    source = str(df["data_source"].iloc[0]) if "data_source" in df.columns else "?"
    notes = [
        f"Mean ± std over {n_win} chronological test windows ({source}); best value per column in bold. "
        "EML: mean realised asymmetric monetary + friction loss per transaction under each model's own policy "
        "(CS-ACRC four-action routing for MG-ECFM, approve/block at the operating threshold for BP1–BP4).",
        "Latency: amortised batch-scoring time per transaction (ms) on CPU; the single-transaction "
        "authorisation path of MG-ECFM is profiled separately in the dashboard / quick demo.",
        "p (Wilcoxon): Holm-adjusted two-sided Wilcoxon signed-rank test on per-window PR-AUC versus MG-ECFM; "
        "(+) MG-ECFM higher, (−) MG-ECFM lower. p (McNemar): Holm-adjusted McNemar test on paired "
        "per-transaction correctness at the operating thresholds.",
    ]
    return PaperTable(
        key="benchmark",
        caption="Benchmark comparison of MG-ECFM against the four state-of-the-art baselines under "
                "chronological concept drift.",
        header=["Model", "PR-AUC", "F1", "Brier", "EML", "Latency (ms)", "p (Wilcoxon)", "p (McNemar)"],
        rows=rows, align="lccccccc", notes=notes, wide=True)


# --------------------------------------------------------------------------- #
# Table 3 - ablation                                                          #
# --------------------------------------------------------------------------- #
def table_ablation(inp: ExperimentInputs) -> PaperTable:
    df = inp.ablation.set_index("arm")
    arms = [a for a in ARM_ORDER if a in df.index]
    spec = [("pr_auc", True, 3, fmt), ("f1", True, 3, fmt), ("brier", False, 4, fmt),
            ("expected_monetary_loss", False, 1, fmt), ("net_monetary_loss", False, 0, None),
            ("prevented_loss_ratio_pct", True, 1, fmt)]
    cells: Dict[str, List[str]] = {}
    for key, hib, nd, f in spec:
        vals = [float(df.loc[a, key]) if key in df.columns else float("nan") for a in arms]
        txt = [fmt_money(v) if f is None else f(v, nd) for v in vals]
        cells[key] = bold_best(txt, vals, hib)
    rows = []
    for i, a in enumerate(arms):
        r = df.loc[a]
        gate = str(r.get("gate_mode", ""))
        gate_txt = {"martingale": "✓", "auto_admit": "auto", "none": "✗"}.get(gate, gate or "—")
        crc = "binary" if str(r.get("routing")) == "binary" else str(r.get("crc_mode", "—"))
        kb = fmt_int(r.get("kb_rules_active_end")) if finite(r.get("kb_rules_active_end")) else "—"
        rows.append([ARM_SHORT.get(a, a), check(r.get("use_graph")), gate_txt, crc,
                     check(r.get("recovery")), check(r.get("assist"))]
                    + [cells[k][i] for k, *_ in spec] + [kb])
    src = str(inp.ablation["data_source"].iloc[0]) if "data_source" in inp.ablation.columns else "?"
    notes = [
        f"Pooled over all test windows ({src}). Graph: camouflage-filtered temporal GraphSAGE + DAE residual; "
        "Gate: ✓ martingale e-process admission/rollback, auto = every candidate admitted, ✗ = no KB; "
        "CRC: static / adaptive CS-ACRC four-action routing or binary approve/block; Rec.: TD-TPF post-incident "
        "recovery; Assist: retrieval-based analyst assist (off the decision path).",
        "Net loss = realised policy loss − recovered value + recovery cost (monetary units). Prevented: share of "
        "fraud value intercepted pre-settlement or recovered within the 24 h window. B3 adaptive, B4 and B5 branch "
        "from one fitted pipeline, so they share ranking metrics by construction.",
    ]
    return PaperTable(
        key="ablation",
        caption="Five-point ablation study (B1–B5) of the MG-ECFM components.",
        header=["Arm", "Graph", "Gate", "CRC", "Rec.", "Assist", "PR-AUC", "F1", "Brier", "EML",
                "Net loss", "Prevented (%)", "Active rules"],
        rows=rows, align="lccccc" + "c" * 7, notes=notes, wide=True)


# --------------------------------------------------------------------------- #
# Table 4 - adversarial stress test                                           #
# --------------------------------------------------------------------------- #
def table_stress(inp: ExperimentInputs) -> PaperTable:
    df = inp.stress
    rates = sorted(df["poison_rate"].unique())
    rows = []
    for rate in rates:
        sub = df[df["poison_rate"] == rate].set_index("policy")
        g = sub.loc["mg_ecfm_gate"] if "mg_ecfm_gate" in sub.index else None
        far = {p: (fmt_pct(sub.loc[p, "false_admission_rate"], 1, 100.0) if p in sub.index else "—")
               for p in POLICY_ORDER}
        if g is None:
            continue
        mc = f"{fmt(g['ville_mc_rate'], 4)} [{fmt(g['ville_mc_ci_low'], 3)}, {fmt(g['ville_mc_ci_high'], 3)}]"
        rows.append([
            f"{100 * float(rate):.0f}%",
            fmt_pct(g.get("realized_poison_rate"), 1, 100.0),
            far["mg_ecfm_gate"], far["sl_screen_only"], far["auto_admit"],
            fmt_pct(g.get("rollback_rate_harmful"), 1, 100.0),
            f"{fmt(g.get('time_to_rollback_mean'), 2)} / {fmt(g.get('time_to_rollback_max'), 0)}",
            fmt(g.get("ville_exact_crossing_prob"), 4), mc, fmt(g.get("ville_ingate_rate"), 4),
            check(g.get("ville_bound_satisfied")),
        ])
    any_row = df[df["policy"] == "mg_ecfm_gate"].iloc[0] if (df["policy"] == "mg_ecfm_gate").any() else None
    alpha = float(any_row["ville_alpha_roll"]) if any_row is not None else float("nan")
    a_adm = float(any_row["alpha_admit"]) if any_row is not None else float("nan")
    notes = [
        f"FAR: share of harmful (poisoned, Sybil, stale, conflicting) candidates ever admitted to the active KB. "
        f"Admission test $$H_0$$: precision $$\\leq p_0$$ = {fmt(any_row['p0'], 2) if any_row is not None else '?'} "
        f"at $$\\alpha_{{admit}}$$ = {fmt(a_adm, 2)}; rollback test $$H_0^{{good}}$$: FPR $$\\leq q_0$$ = "
        f"{fmt(any_row['q0'], 2) if any_row is not None else '?'} at $$\\alpha_{{roll}}$$ = {fmt(alpha, 2)}.",
        "Rollback / TTR: harmful rules injected directly into the active KB; share rolled back and mean / max "
        f"update cycles to rollback ({int(any_row['firings_per_cycle']) if any_row is not None else '?'} "
        "verified firings per cycle).",
        f"Ville's inequality $$P(\\sup_t M_t \\geq 1/\\alpha) \\leq \\alpha$$ = {fmt(alpha, 2)} under $$H_0$$: exact "
        "crossing probability over the horizon, Monte-Carlo rate with 95% Clopper-Pearson interval, and the "
        "false-rollback rate of genuine rules inside the gate. ✓: bound satisfied by all three checks.",
    ]
    return PaperTable(
        key="stress_test",
        caption="Adversarial knowledge-poisoning stress test and Ville's-bound verification of the martingale "
                "e-process gate.",
        header=["$$\\rho$$", "Realised", "FAR gate (%)", "FAR SL-only (%)", "FAR auto (%)", "Rolled back (%)",
                "TTR mean/max", "Ville exact", "Ville MC [95% CI]", "In-gate", "Bound"],
        rows=rows, align="c" * 11, notes=notes, wide=True)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def build_tables(inp: ExperimentInputs) -> List[PaperTable]:
    return [table_dataset_summary(inp), table_benchmark(inp), table_ablation(inp), table_stress(inp)]


def latex_document(tables: Sequence[PaperTable], inp: ExperimentInputs) -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    head = [
        "% =====================================================================",
        "% MG-ECFM camera-ready tables (auto-generated by reports/generate_paper_tables.py)",
        f"% Generated: {stamp}   Source directory: {inp.reports_dir.as_posix()}",
        "% Requirements in the manuscript preamble (IEEEtran): \\usepackage{amsmath,amssymb,graphicx}",
        "% Paste each table environment where it should float, or \\input{paper_tables.tex}.",
        "% =====================================================================",
        "",
    ]
    body = []
    for t in tables:
        body.append(render_latex(t))
        body.append("")
    return "\n".join(head + body)


def markdown_document(tables: Sequence[PaperTable], inp: ExperimentInputs) -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = ["# MG-ECFM - Camera-Ready Result Tables", "",
             f"*Auto-generated by `reports/generate_paper_tables.py` on {stamp} from `{inp.reports_dir.as_posix()}`.*",
             ""]
    for i, t in enumerate(tables, start=1):
        lines.append(render_markdown(t, i))
    return "\n".join(lines)


def generate_tables(reports_dir: Path = REPORTS_DIR, out_dir: Optional[Path] = None,
                    verbose: bool = True) -> Dict[str, Any]:
    """Build all tables and write ``paper_tables.tex`` / ``paper_tables.md``; returns a summary."""
    reports_dir = Path(reports_dir)
    out_dir = Path(out_dir) if out_dir is not None else reports_dir
    inp = load_inputs(reports_dir)
    tables = build_tables(inp)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / "paper_tables.tex"
    md_path = out_dir / "paper_tables.md"
    tex_path.write_text(latex_document(tables, inp), encoding="utf-8")
    md_path.write_text(markdown_document(tables, inp), encoding="utf-8")
    summary = {"tex": str(tex_path), "markdown": str(md_path),
               "tables": {t.key: {"rows": len(t.rows), "columns": len(t.header)} for t in tables}}
    if verbose:
        for t in tables:
            print(f"[table] {t.key:<16s} {len(t.rows):2d} rows x {len(t.header):2d} cols")
        print(f"[saved] {tex_path}")
        print(f"[saved] {md_path}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate IEEE camera-ready MG-ECFM tables (Markdown + LaTeX).")
    ap.add_argument("--reports-dir", type=Path, default=REPORTS_DIR,
                    help="directory holding the Phase-7 CSV/JSON outputs")
    ap.add_argument("--out-dir", type=Path, default=None, help="output directory (default: --reports-dir)")
    args = ap.parse_args(argv)
    generate_tables(args.reports_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
