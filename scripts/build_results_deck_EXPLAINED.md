# Code Walkthrough: `scripts/build_results_deck.js`

**Purpose:** Node.js script that assembles the **PowerPoint results deck** `reports/MG_ECFM_Results_PaySim_2M.pptx` from the PNG charts and `deck_data.json` produced by `scripts/build_results_charts.py`. Slide text is generated from the data, so numbers in the deck always match the result files.

```bash
NODE_PATH=<dir>/node_modules node scripts/build_results_deck.js [--charts reports/deck_charts] [--out reports/MG_ECFM_Results_PaySim_2M.pptx]
```
Requires the `pptxgenjs` package (installed outside the repo) and, optionally, `PPTX_SKILL_DIR` for a theme script.

## Section-by-section
* **Header and arguments (1–36):** requires `fs`, `path`, `pptxgenjs`; resolves `ROOT`, the charts folder and output path from `--charts` / `--out`; loads `deck_data.json` into `D`.
* **`RUN` constant (≈20–36):** facts from the run log that are *not* stored in the result files (run date 2026-11-04, the exact command `python main.py --mode full-pipeline --source paysim --max-rows 2000000 --device cpu`, runtime 8 774 s, peak RSS 8.0 GB, "Apple Silicon Mac (CPU)"). **Edit this block when the run changes.**
* **Formatting helpers (38–53):** `f` (fixed decimals), `pct`, `money` (1.23M style), `fmtP` (p-value style), `boot(k, m)` (bootstrap estimate), `bestBase`, `rankOf`, `holm` (Holm-adjusted p lookup), `img`/`size` (chart paths and slot sizes).
* **Slide builders (105–155):** `newSlide(master, title, notes, section)`, `chart(slide, name, x, y, maxW, maxH, alt)` (places a PNG keeping its aspect ratio and adds alt text), `card` (heading + bullets), `takeaway` (highlighted message bar), `tile` (big-number tile).
* **Derived text (157–160):** `excl` (baselines whose paired-difference CI excludes zero), `sigList` (baselines MG-ECFM beats/loses to at Holm p < 0.05).
* **Slides (161–end):** ~22 slides: title; run setup and data splits; **"Caveat: the decision threshold is not tuned"**; headline results; pooled metrics with 95 % bootstrap intervals; per-window variation; PR-AUC across the 50 windows; **"Expected monetary loss is highest for MG-ECFM"**; pooled PR/ROC curves; confusion matrices at the untuned threshold 0.5; score distributions; action mix; CS-ACRC thresholds; significance against each baseline; paired differences; ablation design and results; knowledge-poisoning false-admission and rollback; martingale trajectories; Ville's bound and the 24 h recovery stop; latency and fit time; conclusions and next steps. Each has speaker notes.
* The script ends by writing the `.pptx` (and applies the theme if `PPTX_SKILL_DIR` is set).

## Notes
* Wording on the slides is generated from the numbers, but **claims and limitations text is hand-written** — note that the deck already states two caveats honestly (untuned threshold, highest monetary loss); keep them when editing.
