# Code Walkthrough: `scripts/build_results_charts.py`

**Purpose:** Renders the ~20 **presentation charts** (matplotlib + seaborn) used in the results deck from the files in `reports/`. It reads existing results only — it never trains or evaluates a model. Added by Atharv with the PaySim-2M deck.

**Outputs:** `reports/deck_charts/*.png` (220 DPI, sized in inches to their slide slot) and `reports/deck_charts/deck_data.json` (headline numbers and chart sizes, consumed by `build_results_deck.js`).

```bash
python scripts/build_results_charts.py [--reports reports] [--out reports/deck_charts]
```

## Section-by-section
* **Style and utilities (57–121):** `setup_style` (palette, fonts), `ygrid`, `ChartWriter` (saves a figure at a given slot size and records its dimensions in `chart_sizes`), `fmt_money`, `fmt_p`, `load_predictions` (reads the large predictions CSV — **needed locally; it is git-ignored**).
* **Chart functions** (each takes the `ChartWriter` plus its data and may return headline numbers for `deck_data.json`):
  * data: `chart_splits` (split sizes and fraud counts);
  * performance: `chart_pooled_ci` (pooled metrics with bootstrap CIs), `chart_window_box`, `chart_drift_lines` (PR-AUC per window), `chart_eml_lines`, `chart_curves` (pooled PR/ROC curves), `chart_confusion`, `chart_score_dist`;
  * decisions: `chart_actions` (four-action mix), `chart_binary_actions`, `chart_crc` (λ thresholds);
  * statistics: `chart_significance` (Holm-adjusted p heat-map), `chart_paired_diff` (paired-difference CIs);
  * ablation and robustness: `chart_ablation`, `chart_ablation_switches`, `chart_adversarial` (FAR and time-to-rollback), `chart_martingale`, `chart_recovery`;
  * efficiency: `chart_latency_fit` (latency and fit time).
* **`_clean` (699):** makes values JSON-safe. **`main` (715):** loads the manifest, drift results, predictions, significance JSON, ablation and adversarial CSVs and figure data, calls each chart, and writes `deck_data.json`.

## Notes
* Because it needs `drift_evaluation_predictions.csv` (≈250 MB) the script can only be re-run on a machine that produced the PaySim-2M run.
* The deck's numbers come from here, so any change in the results requires re-running this script and then `build_results_deck.js`.
