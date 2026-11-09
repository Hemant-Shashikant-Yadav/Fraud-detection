# Code Walkthrough: `reports/generate_paper_tables.py`

**Purpose:** Builds the four **camera-ready tables** of the manuscript from the experiment outputs. It never re-runs a model, so every number in the paper is traceable to a CSV/JSON file in `reports/`.

| Table | Content | Source files |
|---|---|---|
| 1 | Dataset summary and causal-ledger contract (split sizes, stripped columns) | `drift_evaluation_manifest.json`, `configs/project_config.json` |
| 2 | Benchmark MG-ECFM vs BP1–BP4: PR-AUC, F1, Brier, EML, latency, Wilcoxon and McNemar p-values | `drift_evaluation_results.csv`, `statistical_significance_results.json` |
| 3 | Five-point ablation matrix (B1–B5) | `ablation_study_results.csv` |
| 4 | Adversarial stress test and Ville-bound verification | `adversarial_robustness_results.csv` |

**Outputs:** `reports/paper_tables.tex` (IEEEtran-compatible: `\hline` rules, caption above, `\footnotesize`, `table*` for two-column floats) and `reports/paper_tables.md` (GitHub Markdown with the same numbers).

## Section-by-section
* **`PaperTable` dataclass (87–118):** title/caption/header/rows/notes plus a column alignment spec.
* **Rendering (120–240):** `latex_escape` (escapes `& % $ # _ { } ~ ^ \`), `_latex_cell` / `_md_cell`, `render_latex`, `render_markdown`.
* **Formatting helpers (242–316):** `finite`, `fmt` (fixed decimals, "—" for missing), `fmt_int`, `fmt_money`, `fmt_pct`, `fmt_p` (`<0.001` style), `mean_std` (mean ± std), `bold_best` (bold the best value in a column, direction-aware), `check` (✓/✗).
* **Inputs (318–355):** `ExperimentInputs` bundles all loaded files; `load_inputs(reports_dir)` reads what exists (missing inputs simply omit a table).
* **Table builders:**
  * `table_dataset_summary` (357–414): per-dataset ledger contract plus the chronological split with transactions/frauds and fraud rates for the evaluated stream.
  * `_significance_marks` / `table_benchmark` (416–471): mean ± std over windows, best value bolded, Holm-adjusted Wilcoxon (with direction marker `(+)`/`(−)`) and McNemar p-values.
  * `table_ablation` (473–513): arm matrix with ✓/✗ switches and pooled metrics.
  * `table_stress` (515–561): per-ρ FAR for gate / SL-only / auto-admit, rollback share, TTR mean/max, Ville exact and Monte-Carlo CI, in-gate rate, bound check.
* **Documents (563–615):** `build_tables`, `latex_document`, `markdown_document`, `generate_tables(reports_dir, out_dir)` (writes both files; stamps the generation time and source directory).
* **`main` (617):** CLI.

## Notes
* The generated header states the machine path it ran on (e.g. a Mac path in the committed tables) — harmless but visible; regenerate locally for your own provenance line.
* Table 3's footnote explains that B3-adaptive/B4/B5 share ranking metrics by construction.
