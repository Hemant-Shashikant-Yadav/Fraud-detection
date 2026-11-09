# Code Walkthrough: `reports/generate_paper_figures.py`

**Purpose:** Produces the four **publication figures** for the manuscript. Each figure is saved twice — a vector **PDF** (fonts embedded, for LaTeX) and a **300-DPI PNG** (for Word/slides) — sized for IEEE two-column layout (3.5 in single, 7.16 in double column). It only *reads* experiment outputs (and cheap computed curves); it never trains a model.

| Figure | Content | Data |
|---|---|---|
| Fig. 1 | PR-AUC trajectories over the drift windows (MG-ECFM vs BP1–BP4), per-window values and 5-window rolling mean | `drift_evaluation_results.csv` |
| Fig. 2 | Martingale e-process wealth under poisoned rule injection: (a) rollback monitor `M^roll`, (b) admission test `M^admit`, stopping at Ville's threshold `1/α` | computed from the gate and the contaminated feed |
| Fig. 3 | CS-ACRC thresholds (`λ_low, λ_mid, λ_high`) under drift, with realised fraud/intervention rates | `drift_evaluation_results.csv` |
| Fig. 4 | TD-TPF recovery rate vs time-to-detection (κ sensitivity) and prevented-loss ratio in the 24 h window | computed by TD-TPF over the evaluated stream |

The data behind the computed figures (2 and 4) is also saved as CSV in `reports/figures/figure_data/`, so every plotted point is auditable.

## Section-by-section
* **Style and I/O (83–121):** `apply_ieee_style()` (fonts, sizes, line widths), `save_figure(fig, out_dir, stem)` (writes PDF + PNG, returns paths), `load_json`, `load_configs`.
* **Helpers (123–146):** `drift_onset_window` (finds where synthetic drift begins from the manifest, if any), `rolling_mean`.
* **Fig. 1 `figure_pr_auc` (148–198):** one line per model, markers for per-window values and a thicker rolling mean; shaded drift region.
* **Fig. 2 (200–314):** `_trajectory(process, xs)` runs a betting e-process over a fixed outcome sequence and returns log-wealth and stopping time; `martingale_trajectories(...)` generates trajectories for good and harmful rules using the project's gate configuration and the contaminated feed; `figure_martingale(...)` plots wealth on a log axis with the `1/α` line.
* **Fig. 3 `figure_crc_thresholds` (316–368):** λ trajectories from the per-window results plus fraud and intervention rates.
* **Fig. 4 (370–488):** `evaluated_canonical_stream(manifest)` rebuilds the canonical stream that was evaluated (synthetic streams are regenerated from the manifest; for a PaySim run the raw file is re-read through the loader with the manifest's `max_rows`, default 2 000 000, so the raw CSV must be present); `recovery_curves(...)` runs TD-TPF at several detection delays and κ values; `figure_recovery` plots the curves and the settlement-window boundary.
* **Driver (490–547):** `generate_figures(reports_dir, out_dir)` calls the four figure functions (skipping a figure when its input is missing) and returns the file map; `main` is the CLI.

## Notes
* Fig. 4 is a **simulation** of recovery, not a measured result; label it as such in the paper.
* `reports/figures/**` is git-ignored except placeholders; regenerate with `python main.py --mode paper-artifacts`.
