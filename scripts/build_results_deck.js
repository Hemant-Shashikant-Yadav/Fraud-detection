#!/usr/bin/env node
/*
 * Build reports/MG_ECFM_Results_PaySim_2M.pptx from the PNG charts and deck_data.json
 * written by scripts/build_results_charts.py.
 *
 * Usage (pptxgenjs lives outside the repo):
 *   NODE_PATH=<dir>/node_modules node scripts/build_results_deck.js [--charts reports/deck_charts] [--out reports/MG_ECFM_Results_PaySim_2M.pptx]
 * Env: PPTX_SKILL_DIR = pptx skill folder (for scripts/apply_theme.js).
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const CHARTS = path.resolve(arg("--charts", path.join(ROOT, "reports", "deck_charts")));
const OUT = path.resolve(arg("--out", path.join(ROOT, "reports", "MG_ECFM_Results_PaySim_2M.pptx")));
const SKILL = process.env.PPTX_SKILL_DIR;
const D = JSON.parse(fs.readFileSync(path.join(CHARTS, "deck_data.json"), "utf8"));

// Facts from the run log, not stored in the result files. Edit when the run changes.
const RUN = {
  date: "2026-11-04",
  command: "python main.py --mode full-pipeline --source paysim --max-rows 2000000 --device cpu",
  totalRows: 6362620,
  maxRows: 2000000,
  runtimeS: 8774,
  peakRssGb: 8.0,
  machine: "Apple Silicon Mac (CPU)",
};

// ---------------------------------------------------------------- helpers
const MODELS = ["mg_ecfm", "bp1", "bp2", "bp3", "bp4"];
const BASE = ["bp1", "bp2", "bp3", "bp4"];
const TINY = { mg_ecfm: "MG-ECFM", bp1: "BP1", bp2: "BP2", bp3: "BP3", bp4: "BP4" };
const n0 = (v) => Math.round(v).toLocaleString("en-US");
const f = (v, d = 3) => Number(v).toFixed(d);
const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
const money = (v) => (Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(2) + "M" : n0(v));
const fmtP = (p) => (p < 1e-4 ? "< 0.0001" : p < 0.01 ? p.toFixed(4) : p.toFixed(3));
const boot = (k, m) => D.bootstrap[k][m].estimate;
const bestBase = (m, lowerBetter) => BASE.reduce((a, b) => ((lowerBetter ? boot(b, m) < boot(a, m) : boot(b, m) > boot(a, m)) ? b : a));
const rankOf = (m, lowerBetter) => {
  const s = [...MODELS].sort((a, b) => (lowerBetter ? boot(a, m) - boot(b, m) : boot(b, m) - boot(a, m)));
  return s.indexOf("mg_ecfm") + 1;
};
const sig = D.significance; // rows x BASE
const holm = (row, b) => sig.holm[row][BASE.indexOf(b)];
const ab = Object.fromEntries(D.ablation.map((r) => [r.arm, r]));
const cm = D.confusion;
const img = (name) => path.join(CHARTS, name + ".png");
const size = (name) => D.chart_sizes[name];

// ---------------------------------------------------------------- theme
const THEME = {
  name: "MG-ECFM Results",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B1B2E", lt1: "FFFFFF", dk2: "14284B", lt2: "EEF3F9",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "E87BA4", accent6: "C62E2E", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const MUTED_HEX = "5F6B7A";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "MG-ECFM project";
pres.title = "MG-ECFM results on PaySim (2M rows)";
pres.subject = "Smoke run of the full MG-ECFM pipeline on real PaySim data";
const C = pres.SchemeColor;

const FOOT = `MG-ECFM results  |  PaySim 2M smoke run  |  ${RUN.date}`;
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.5, fontSize: 44, bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.7, h: 1.6, fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.35, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.02, w: 9, h: 0.3, fontSize: 10, color: MUTED_HEX, margin: 0 } } },
  ],
  slideNumber: { x: 12.1, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: MUTED_HEX, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.1, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: "CADCFC", align: "right" },
});

let section = null;
function newSlide(master, title, notes, sectionTitle) {
  if (sectionTitle && sectionTitle !== section) { pres.addSection({ title: sectionTitle }); section = sectionTitle; }
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
// Chart image sized to its native inches (so chart text keeps its pt size), scaled only if needed.
function chart(s, name, x, y, maxW, maxH, alt) {
  const [w0, h0] = size(name);
  const k = Math.min(1, maxW / w0, maxH / h0);
  const w = w0 * k, h = h0 * k;
  s.addImage({ path: img(name), x, y, w, h, altText: alt, objectName: "chart_" + name });
  return { w, h };
}
let cardId = 0;
function card(s, x, y, w, h, heading, bullets, opts = {}) {
  const fill = opts.fill || C.background2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: `card_${++cardId}` });
  const runs = [];
  if (heading) runs.push({ text: heading, options: { fontSize: opts.headSize || 16, bold: true, color: opts.headColor || C.text2, breakLine: true, paraSpaceAfter: 6 } });
  bullets.forEach((b, i) => runs.push({ text: b, options: { fontSize: opts.size || 14, color: opts.color || C.text1, bullet: opts.numbered ? { type: "number", numberStartAt: i + 1 } : true, paraSpaceAfter: 6, breakLine: i < bullets.length - 1 } }));
  s.addText(runs, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: h - 0.3, valign: "top", isTextBox: true, margin: 0, objectName: `card_text_${cardId}` });
}
function takeaway(s, y, text, h = 0.75) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 12.1, h, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "takeaway_bg" });
  s.addText([{ text: "Finding  ", options: { bold: true, color: C.accent1 } }, { text, options: { color: C.text1 } }],
    { x: 0.8, y, w: 11.7, h, fontSize: 14, valign: "middle", isTextBox: true, margin: 0, objectName: "takeaway_text" });
}
function tile(s, x, y, w, h, big, label, sub, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: opts.fill || C.background2 }, line: { color: opts.fill || C.background2 }, rectRadius: 0.08, objectName: "tile_bg" });
  s.addText(big, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 1.0, fontSize: opts.bigSize || 40, bold: true, color: opts.bigColor || C.accent1, fontFace: THEME.headFontFace, valign: "middle", isTextBox: true, margin: 0, objectName: "tile_big" });
  s.addText(label, { x: x + 0.2, y: y + 1.2, w: w - 0.4, h: 0.45, fontSize: 15, bold: true, color: C.text2, valign: "top", isTextBox: true, margin: 0, objectName: "tile_label" });
  if (sub) s.addText(sub, { x: x + 0.2, y: y + 1.65, w: w - 0.4, h: h - 1.8, fontSize: 13, color: C.text1, valign: "top", isTextBox: true, margin: 0, objectName: "tile_sub" });
}

// ---------------------------------------------------------------- numbers
const sp = D.splits;
const W = D.windows;
const mgPR = D.bootstrap.mg_ecfm.pr_auc;
const prBest = bestBase("pr_auc", false), brBest = bestBase("brier", true), f2Best = bestBase("f2", false), emlBest = bestBase("expected_monetary_loss", true);
const emlRank = rankOf("expected_monetary_loss", true);
const holmPrBp4 = holm(1, "bp4");
const thr = D.thresholds.length === 1 ? f(D.thresholds[0], 1) : D.thresholds.join(", ");
const act = D.actions;
const fraudN = sp.frauds.test;
const nonApprove = act.counts["Step-Up"] + act.counts.Hold + act.counts.Block;
const hours = Math.floor(RUN.runtimeS / 3600), mins = Math.round((RUN.runtimeS % 3600) / 60);
const adv = D.adversarial;
const lat = D.perf.latency_ms, fit = D.perf.fit_seconds;
const wm = D.window_mean;
const pd = D.paired_diff;
const excl = (m) => BASE.filter((b) => pd[b][m] && pd[b][m].excludes_zero).map((b) => TINY[b]);
const sigList = (row, wantFavour) => BASE.filter((b) => holm(row, b) < 0.05 && sig.direction[row][BASE.indexOf(b)] === wantFavour).map((b) => TINY[b]);
const mt = D.martingale || {};

// ================================================================ slides
// 1 Title
{
  const s = newSlide("TITLE_DARK", "MG-ECFM results on PaySim",
    `This deck shows the results of one smoke run of the full MG-ECFM pipeline on real PaySim data. ` +
    `The run used the first ${n0(RUN.maxRows)} of ${n0(RUN.totalRows)} rows. ` +
    `The numbers are not final paper numbers. The decision threshold was not tuned in this run. Slide 3 gives the reason.`,
    "Introduction");
  s.addText([
    { text: `Full-pipeline smoke run on the first ${(RUN.maxRows / 1e6).toFixed(0)}M of ${(RUN.totalRows / 1e6).toFixed(2)}M real PaySim rows`, options: { breakLine: true } },
    { text: `Five models, ${W.n} chronological drift windows, CPU run  |  ${RUN.date}`, options: { fontSize: 16, color: "CADCFC" } },
  ], { placeholder: "body" });
}

// 2 Run setup
{
  const s = newSlide("CONTENT", "Run setup and data splits",
    `The run used the fast profile with seed ${D.run.seed}. The configured step splits do not fit 2M rows. ` +
    `So the pipeline used the fallback split by share of distinct steps. ` +
    `Train has ${n0(sp.tx.train)} rows, calibration ${n0(sp.tx.calibration)}, operating ${n0(sp.tx.operating)} and test ${n0(sp.tx.test)}. ` +
    `The test split has ${fraudN} frauds. The total runtime was ${n0(RUN.runtimeS)} seconds. Peak memory (RSS) was ${RUN.peakRssGb.toFixed(1)} GB. ` +
    `The runtime and peak memory come from the run log. All other numbers come from reports/drift_evaluation_manifest.json.`,
    "Introduction");
  chart(s, "split_sizes", 0.6, 1.35, 8.3, 4.0, "Donut of rows per split and bar chart of fraud rate per split");
  s.addText([
    { text: "Command", options: { bold: true, color: C.text2, breakLine: true } },
    { text: RUN.command, options: { fontFace: "Courier New", fontSize: 11, color: C.text1 } },
  ], { x: 0.6, y: 5.55, w: 8.3, h: 0.9, fontSize: 13, valign: "top", isTextBox: true, margin: 0 });
  card(s, 9.25, 1.35, 3.45, 5.1, "Run facts", [
    `Real PaySim log: first ${n0(RUN.maxRows)} of ${n0(RUN.totalRows)} rows.`,
    `Fast profile, seed ${D.run.seed}, ${D.run.n_features} causal features.`,
    `${W.n} chronological test windows (steps 118 to 179).`,
    `Runtime ${n0(RUN.runtimeS)} s (${hours} h ${mins} min) on an ${RUN.machine}.`,
    `Peak RSS ${RUN.peakRssGb.toFixed(1)} GB.`,
    `Models: MG-ECFM (proposed) and baselines BP1 to BP4.`,
  ], { size: 13 });
}

// 3 Caveat
{
  const s = newSlide("CONTENT", "Caveat: the decision threshold is not tuned",
    `This slide is important for every threshold metric in the deck. ` +
    `The operating split has ${sp.tx.operating} rows and all of them are frauds (one class only). ` +
    `So select_operating_threshold could not tune the threshold, and it stays at the default ${thr} for every model. ` +
    `F1, F2, precision, recall, the confusion matrices, the McNemar test and the binary actions use this untuned threshold. ` +
    `PR-AUC, ROC-AUC, Brier and ECE use the scores only, so the threshold does not change them. ` +
    `Also, ${W.all_fraud} of the ${W.n} test windows contain only frauds. PR-AUC is not defined in those windows. ` +
    `This is a smoke run of the full pipeline on real data. The paper protocol uses all 6.36M rows with the configured splits.`,
    "Introduction");
  const tw = 3.8, gap = 0.35;
  tile(s, 0.6, 1.4, tw, 2.35, `${sp.frauds.operating} of ${sp.tx.operating}`, "Operating rows are fraud",
    "The split has one class only, so the threshold search cannot run.", { bigColor: C.accent6 });
  tile(s, 0.6 + tw + gap, 1.4, tw, 2.35, thr, "Threshold for all five models",
    "This is the default value. It is not tuned.", { bigColor: C.accent6 });
  tile(s, 0.6 + 2 * (tw + gap), 1.4, tw, 2.35, `${W.all_fraud} of ${W.n}`, "Test windows are 100% fraud",
    "PR-AUC is not defined there. These windows dominate window means.", { bigColor: C.accent6 });
  card(s, 0.6, 4.05, 5.9, 2.0, "Affected by the untuned threshold", [
    "F1, F2, precision and recall",
    "Confusion matrices and the McNemar test",
    "Binary approve or block actions of BP1 to BP4",
  ]);
  card(s, 6.8, 4.05, 5.9, 2.0, "Not affected (score-based)", [
    "PR-AUC and ROC-AUC",
    "Brier score and ECE",
    "PR and ROC curves",
  ]);
  s.addText("This is a smoke run of the full pipeline on real data. It does not give final paper numbers.",
    { x: 0.6, y: 6.25, w: 12.1, h: 0.45, fontSize: 15, bold: true, color: C.accent6, isTextBox: true, margin: 0 });
}

// 4 Headline
{
  const s = newSlide("CONTENT", "Headline results on the pooled test set",
    `All four numbers are pooled over the ${n0(sp.tx.test)} test transactions. They come from bootstrap_ci in reports/statistical_significance_results.json. ` +
    `MG-ECFM has the highest PR-AUC, the lowest Brier score and the highest F2. ` +
    `But MG-ECFM has the highest expected monetary loss of the five models. ` +
    `The PR-AUC lead over ${TINY[prBest]} is not significant in the per-window Wilcoxon test (Holm p = ${fmtP(holmPrBp4)}). ` +
    `F2 uses the untuned threshold ${thr}.`,
    "Detection results");
  const tw = 2.8, gap = 0.3;
  tile(s, 0.6, 1.4, tw, 2.7, f(mgPR.estimate), "MG-ECFM PR-AUC (pooled)",
    `95% CI ${f(mgPR.ci_low)} to ${f(mgPR.ci_high)}. Best baseline: ${TINY[prBest]} ${f(boot(prBest, "pr_auc"))}.`);
  tile(s, 0.6 + (tw + gap), 1.4, tw, 2.7, f(boot("mg_ecfm", "brier"), 5), "MG-ECFM Brier (pooled)",
    `Lower is better. Best baseline: ${TINY[brBest]} ${f(boot(brBest, "brier"), 4)}.`, { bigSize: 36 });
  tile(s, 0.6 + 2 * (tw + gap), 1.4, tw, 2.7, f(boot("mg_ecfm", "f2")), "MG-ECFM F2 (thr 0.5)",
    `Best baseline: ${TINY[f2Best]} ${f(boot(f2Best, "f2"))}. The threshold is untuned.`);
  tile(s, 0.6 + 3 * (tw + gap), 1.4, tw, 2.7, n0(boot("mg_ecfm", "expected_monetary_loss")), "MG-ECFM loss per txn",
    `Rank ${emlRank} of 5 (lower is better). Lowest: ${TINY[emlBest]} ${n0(boot(emlBest, "expected_monetary_loss"))}.`, { bigColor: C.accent6 });
  card(s, 0.6, 4.4, 12.1, 1.9, "What this means", [
    `MG-ECFM ranks frauds best and gives the best calibration of the five models.`,
    `Its routing approves ${pct(act.fraud.Approve / fraudN)} of frauds, so its monetary loss is the highest.`,
    `The PR-AUC lead over ${TINY[prBest]} is not significant (Wilcoxon, Holm p = ${fmtP(holmPrBp4)}).`,
  ]);
}

// 5 Pooled CI bars
{
  const s = newSlide("CONTENT", "Pooled metrics with 95% bootstrap intervals",
    `Each bar is the pooled value on the full test set. The error bar is the 95% block-bootstrap interval with the monitoring window as the unit (1,000 resamples). ` +
    `MG-ECFM leads on PR-AUC, F1, F2 and Brier. BP2 has the highest ROC-AUC (${f(boot("bp2", "roc_auc"))}) and MG-ECFM has the lowest (${f(boot("mg_ecfm", "roc_auc"))}). ` +
    `The PR-AUC intervals of MG-ECFM and BP4 overlap a lot. MG-ECFM has the highest expected monetary loss. F1 and F2 use the untuned threshold 0.5.`,
    "Detection results");
  chart(s, "pooled_metrics_ci", 0.6, 1.3, 12.1, 5.45, "Six bar charts of pooled metrics per model with bootstrap intervals");
}

// 6 Window distributions
{
  const s = newSlide("CONTENT", "Per-window results vary a lot",
    `Each dot is one test window. The diamond is the mean over windows. PR-AUC is defined in ${W.pr_auc_defined} of ${W.n} windows. ` +
    `On the window mean, BP4 has a higher PR-AUC (${f(wm.bp4.pr_auc)}) than MG-ECFM (${f(wm.mg_ecfm.pr_auc)}). ` +
    `On the pooled test set, MG-ECFM is higher. The two views do not agree because many windows have few frauds.`,
    "Detection results");
  chart(s, "window_boxplots", 0.6, 1.3, 8.4, 5.45, "Box and strip plots of per-window PR-AUC and F2 per model");
  card(s, 9.25, 1.35, 3.45, 4.85, "Per-window view", [
    `Mean PR-AUC over windows: BP4 ${f(wm.bp4.pr_auc)}, MG-ECFM ${f(wm.mg_ecfm.pr_auc)}.`,
    `Mean F2: MG-ECFM ${f(wm.mg_ecfm.f2)}, BP4 ${f(wm.bp4.f2)}.`,
    `Wilcoxon on PR-AUC: MG-ECFM is higher than ${sigList(1, "favours MG-ECFM").join(", ")} (Holm p < 0.05).`,
    `The difference to BP4 is not significant (Holm p = ${fmtP(holmPrBp4)}).`,
  ], { size: 13 });
}

// 7 Drift PR-AUC
{
  const s = newSlide("CONTENT", "PR-AUC across the 50 drift windows",
    `The top panel shows PR-AUC per window for each model. The bottom panel shows the number of frauds per window. ` +
    `The grey bands mark the ${W.all_fraud} windows that contain only frauds. PR-AUC is not defined there. ` +
    `Window sizes go from ${n0(W.rows_min)} to ${n0(W.rows_max)} transactions. Most windows have fewer than 20 frauds, so per-window PR-AUC is noisy. ` +
    `MG-ECFM and BP4 follow a similar path. BP3 is lowest in most windows.`,
    "Detection results");
  chart(s, "drift_pr_auc_windows", 0.6, 1.3, 12.1, 5.45, "Line chart of PR-AUC per window per model, with fraud counts below");
}

// 8 EML windows
{
  const s = newSlide("CONTENT", "Expected monetary loss is highest for MG-ECFM",
    `Expected monetary loss (EML) is the realised asymmetric loss per transaction under each model's own policy. ` +
    `The y axis uses a log scale. The loss of MG-ECFM rises sharply in the all-fraud windows, because MG-ECFM approves most of these transactions. ` +
    `BP1 to BP3 block ${pct(D.intervention_rate.bp1)} to ${pct(D.intervention_rate.bp3)} of all transactions. So they stop more fraud value, but they also stop many genuine payments. ` +
    `The Wilcoxon test on per-window EML favours every baseline after the Holm correction.`,
    "Detection results");
  chart(s, "drift_eml_windows", 0.6, 1.3, 8.4, 5.45, "Line chart of expected monetary loss per window per model, log scale");
  card(s, 9.25, 1.35, 3.45, 4.85, "Why the loss is high", [
    `MG-ECFM approves most transactions in the ${W.all_fraud} all-fraud windows.`,
    `Pooled EML per txn: MG-ECFM ${n0(boot("mg_ecfm", "expected_monetary_loss"))}, BP4 ${n0(boot("bp4", "expected_monetary_loss"))}, BP2 ${n0(boot("bp2", "expected_monetary_loss"))}.`,
    `BP1 to BP3 flag ${pct(D.intervention_rate.bp1)} to ${pct(D.intervention_rate.bp3)} of all txns.`,
    `Wilcoxon on EML favours ${sigList(3, "favours baseline").join(", ")} (Holm p < 0.05).`,
  ], { size: 13 });
}

// 9 PR / ROC
{
  const c = D.curves;
  const s = newSlide("CONTENT", "Pooled precision-recall and ROC curves",
    `The curves use all ${n0(sp.tx.test)} test transactions. They use the scores only, so the threshold does not change them. ` +
    `MG-ECFM has the highest average precision (${f(c.mg_ecfm.pooled_pr_auc)}). Its precision stays higher than the baselines up to a recall of about 0.8. ` +
    `On ROC-AUC all models are between ${f(Math.min(...MODELS.map((k) => c[k].pooled_roc_auc)))} and ${f(Math.max(...MODELS.map((k) => c[k].pooled_roc_auc)))}. ` +
    `ROC-AUC hides the differences because the fraud base rate is only ${pct(fraudN / sp.tx.test, 3)}.`,
    "Detection results");
  chart(s, "pooled_pr_roc_curves", 0.6, 1.3, 12.1, 5.45, "Precision-recall and ROC curves per model on the pooled test set");
}

// 10 Confusion
{
  const s = newSlide("CONTENT", "Confusion matrices at the untuned threshold 0.5",
    `Each matrix shows the count and the row-normalised rate. All five models use the untuned threshold ${thr}. ` +
    `MG-ECFM and BP4 are strict: they raise few false alarms but miss about two thirds of frauds. ` +
    `BP1 to BP3 catch more than 80% of frauds, but they flag 81,000 to 181,000 genuine transactions. ` +
    `A tuned threshold will move every model along its PR curve. So do not read these matrices as final results.`,
    "Detection results");
  chart(s, "confusion_matrices", 0.6, 1.3, 12.1, 3.6, "Five confusion matrix heatmaps, one per model");
  const ccard = (k) => [`${n0(cm[k].tp)} of ${fraudN} frauds caught (${pct(cm[k].tp / fraudN)}).`, `${n0(cm[k].fp)} false alarms (${pct(cm[k].fp / (cm[k].fp + cm[k].tn), 2)} of legit).`];
  card(s, 0.6, 5.15, 3.85, 1.5, "MG-ECFM", ccard("mg_ecfm"), { size: 13 });
  card(s, 4.72, 5.15, 3.85, 1.5, "BP4 ARGUS", ccard("bp4"), { size: 13 });
  card(s, 8.85, 5.15, 3.85, 1.5, "BP1 to BP3", [
    `${pct(Math.min(cm.bp1.tp, cm.bp2.tp, cm.bp3.tp) / fraudN)} to ${pct(Math.max(cm.bp1.tp, cm.bp2.tp, cm.bp3.tp) / fraudN)} of frauds caught.`,
    `${n0(Math.min(cm.bp1.fp, cm.bp2.fp, cm.bp3.fp))} to ${n0(Math.max(cm.bp1.fp, cm.bp2.fp, cm.bp3.fp))} false alarms.`,
  ], { size: 13 });
}

// 11 Score distributions
{
  const s = newSlide("CONTENT", "Score distributions by true class",
    `Each panel shows the share of each class per score bin on a log scale. The dashed line is the threshold 0.5. ` +
    `For MG-ECFM, a large share of fraud scores is between 0.001 and 0.1. These frauds fall below 0.5 and the model approves them. ` +
    `BP1 to BP3 give scores near 1 to most frauds, but also to many genuine transactions. ` +
    `This explains the confusion matrices. A tuned threshold can change the F1 and F2 results a lot.`,
    "Detection results");
  chart(s, "score_distributions", 0.6, 1.3, 12.1, 3.6, "Histograms of model scores for legit and fraud transactions, per model");
  card(s, 0.6, 5.1, 5.9, 1.6, "MG-ECFM", [
    "Many fraud scores are far below 0.5, so the 0.5 threshold misses them.",
    "Genuine scores are near zero, so false alarms are rare.",
  ], { size: 13 });
  card(s, 6.8, 5.1, 5.9, 1.6, "BP1 to BP3", [
    "Most fraud scores are near 1.",
    "Many genuine scores are also above 0.5, so false alarms are frequent.",
  ], { size: 13 });
}

// 12 Action mix
{
  const s = newSlide("CONTENT", "MG-ECFM action mix: Approve, Step-Up, Hold, Block",
    `CS-ACRC routes each transaction to one of four actions: Approve, Step-Up, Hold or Block. ` +
    `MG-ECFM sent ${nonApprove} of ${n0(sp.tx.test)} test transactions away from Approve. ` +
    `Of the ${fraudN} frauds, ${act.fraud.Approve} were approved, ${act.fraud["Step-Up"]} got Step-Up, ${act.fraud.Hold} got Hold and ${act.fraud.Block} were blocked. ` +
    `Of the genuine transactions, only ${act.legit["Step-Up"] + act.legit.Hold + act.legit.Block} got an intervention. ` +
    `The Step-Up and Hold paths were almost not used. The next slide shows the reason: the routing thresholds rose close to 1.`,
    "Routing");
  chart(s, "action_mix", 0.6, 1.3, 12.1, 4.6, "Donut of non-approve actions and stacked bars of action share by true class");
  takeaway(s, 6.0, `MG-ECFM approved ${act.fraud.Approve} of ${fraudN} frauds (${pct(act.fraud.Approve / fraudN)}). Only ${nonApprove} transactions got Step-Up, Hold or Block.`);
}

// 13 CS-ACRC thresholds
{
  const s = newSlide("CONTENT", "CS-ACRC thresholds rise close to 1",
    `The left chart shows the three CS-ACRC thresholds per window. After the operating split they are ${W.lambda_start.map((v) => f(v, 2)).join(", ")}. ` +
    `From window 3 the thresholds rise to a median of ${W.lambda_median.map((v) => f(v, 2)).join(", ")}. ` +
    `They drop a little only in the all-fraud windows. With thresholds this high, CS-ACRC approves almost everything. ` +
    `The small chart compares the pooled non-approve rate of all models on a log scale.`,
    "Routing");
  chart(s, "crc_thresholds", 0.6, 1.3, 8.4, 5.45, "Step lines of the three routing thresholds and stacked bars of intervention rate per window");
  chart(s, "intervention_rate_models", 9.25, 1.35, 3.5, 2.6, "Bar chart of pooled non-approve rate per model, log scale");
  card(s, 9.25, 4.1, 3.45, 2.6, null, [
    `Start: ${W.lambda_start.map((v) => f(v, 2)).join(" / ")}.`,
    `Median: ${W.lambda_median.map((v) => f(v, 2)).join(" / ")}.`,
    `Non-approve rate: MG-ECFM ${pct(D.intervention_rate.mg_ecfm, 3)}, BP3 ${pct(D.intervention_rate.bp3)}.`,
  ], { size: 13 });
}

// 14 Significance heatmap
{
  const s = newSlide("CONTENT", "Statistical significance against each baseline",
    `Each cell gives the Holm-adjusted p-value of MG-ECFM against one baseline. A star means significant at 0.05. The text gives the direction. ` +
    `McNemar compares per-transaction correctness at the untuned threshold 0.5. MG-ECFM accuracy is ${pct(D.accuracy.mg_ecfm, 2)}; BP4 is ${pct(D.accuracy.bp4, 2)}. ` +
    `The Wilcoxon tests compare per-window values. PR-AUC uses the ${W.pr_auc_defined} windows where it is defined. ` +
    `F2 is not significant after the Holm correction. EML is significant for all four baselines, in favour of the baselines.`,
    "Significance");
  chart(s, "significance_heatmap", 0.6, 1.3, 8.4, 5.45, "Heatmap of Holm-adjusted p-values per test and baseline");
  card(s, 9.25, 1.35, 3.45, 4.35, "Reading the tests", [
    `McNemar: MG-ECFM is more often correct than all baselines (untuned threshold).`,
    `PR-AUC: significant against ${sigList(1, "favours MG-ECFM").join(", ")}. Not against BP4.`,
    `F2: no test is significant after Holm.`,
    `EML: all baselines have lower loss (Holm p < 0.05).`,
  ], { size: 13 });
}

// 15 Paired differences
{
  const s = newSlide("CONTENT", "Pooled differences with 95% bootstrap intervals",
    `Each point is the pooled MG-ECFM value minus the baseline value. The bar is the 95% block-bootstrap interval. ` +
    `If the interval does not cross zero, the difference is clear. ` +
    `PR-AUC: the interval excludes zero for ${excl("pr_auc").join(", ") || "no baseline"}. ` +
    `F2: it excludes zero for ${excl("f2").join(", ") || "no baseline"}. Brier: for ${excl("brier").join(", ")}. EML: for ${excl("expected_monetary_loss").join(", ")}, and MG-ECFM has the higher loss.`,
    "Significance");
  chart(s, "paired_difference_ci", 0.6, 1.3, 12.1, 3.4, "Forest plots of pooled metric differences, MG-ECFM minus each baseline");
  const tw = 2.8, gap = 0.3;
  const mk = (i, head, lines) => card(s, 0.6 + i * (tw + gap), 4.95, tw, 1.75, head, lines, { size: 13 });
  mk(0, "PR-AUC", [`Clear gain over ${excl("pr_auc").join(", ") || "none"}.`, `Interval crosses zero for ${BASE.filter((b) => !pd[b].pr_auc.excludes_zero).map((b) => TINY[b]).join(", ")}.`]);
  mk(1, "F2 (threshold 0.5)", [`Clear gain over ${excl("f2").join(", ")}.`, `Interval crosses zero for ${BASE.filter((b) => !pd[b].f2.excludes_zero).map((b) => TINY[b]).join(", ") || "none"}.`]);
  mk(2, "Brier", [`MG-ECFM is lower than ${excl("brier").join(", ")}.`]);
  mk(3, "EML", [`MG-ECFM loss is higher than ${excl("expected_monetary_loss").join(", ")}.`]);
}

// 16 Ablation switches
{
  const s = newSlide("CONTENT", "Ablation design: seven arms, B1 to B5",
    `The matrix shows which component each arm switches on. Blue means on. ` +
    `B3 adaptive, B4 and B5 branch from one fitted pipeline. So they share the same ranking metrics by construction. ` +
    `In these arms the martingale knowledge gate admitted 0 rules on the real stream. So the knowledge base, recovery and assist had no rules to act on. ` +
    `The result shows that the gate admitted nothing in this run. It does not show that the components have no effect.`,
    "Ablation");
  chart(s, "ablation_switches", 0.6, 1.3, 6.0, 4.0, "Matrix of component switches per ablation arm");
  card(s, 6.9, 1.35, 5.8, 3.95, "Read this before the results", [
    `B1 is XGBoost only, with no graph, gate or recovery.`,
    `B2 admits every candidate rule (${n0(ab.B2_graph_gate_off.kb_rules_ever_admitted)} rules).`,
    `B3 adaptive, B4 and B5 share one fitted pipeline, so their ranking metrics are the same by construction.`,
    `The martingale gate admitted ${n0(ab.B4_full_mg_ecfm.kb_rules_ever_admitted)} rules in B3 to B5. The KB, recovery and assist had no rules to use.`,
    `So this run cannot measure the effect of these components.`,
  ], { size: 14 });
}

// 17 Ablation results
{
  const s = newSlide("CONTENT", "Ablation results per arm",
    `All values are pooled over the test windows. B1 XGBoost only has the highest PR-AUC (${f(ab.B1_xgboost_only.pr_auc)}). ` +
    `B2 with the gate off has the lowest EML (${n0(ab.B2_graph_gate_off.expected_monetary_loss)}) and the highest prevented loss (${f(ab.B2_graph_gate_off.prevented_loss_ratio_pct, 1)}%). ` +
    `Static CRC gives a lower EML (${n0(ab.B3_static_crc.expected_monetary_loss)}) than adaptive CS-ACRC (${n0(ab.B3_adaptive_crc.expected_monetary_loss)}). ` +
    `Recovery in B4 changes net loss only from ${money(ab.B3_adaptive_crc.net_monetary_loss)} to ${money(ab.B4_full_mg_ecfm.net_monetary_loss)}. ` +
    `F2 uses the untuned threshold 0.5.`,
    "Ablation");
  chart(s, "ablation_metrics", 0.6, 1.3, 12.1, 3.9, "Bar charts of PR-AUC, F2, EML and prevented loss per ablation arm");
  card(s, 0.6, 5.4, 5.9, 1.3, null, [
    `B1 XGBoost only has the highest PR-AUC (${f(ab.B1_xgboost_only.pr_auc)}).`,
    `B2 (gate off) has the lowest EML (${n0(ab.B2_graph_gate_off.expected_monetary_loss)}).`,
  ], { size: 13 });
  card(s, 6.8, 5.4, 5.9, 1.3, null, [
    `Static CRC EML ${n0(ab.B3_static_crc.expected_monetary_loss)} vs adaptive ${n0(ab.B3_adaptive_crc.expected_monetary_loss)}.`,
    `Recovery rate in B4: ${f(ab.B4_full_mg_ecfm.recovery_rate_pct, 2)}%.`,
  ], { size: 13 });
}

// 18 Adversarial FAR
{
  const s = newSlide("CONTENT", "Knowledge poisoning: false admission rate",
    `This stress test injects harmful candidate rules at poison rates from 5% to 25%. ` +
    `The martingale gate admits at most ${f(adv.far_gate_max_pct, 1)}% of harmful rules. Auto-admit admits all of them. ` +
    `The SL screen alone admits at most ${f(adv.far_sl_max_pct, 1)}% of harmful rules, which is close to the gate. ` +
    `But the SL screen admits at most ${f(adv.genuine_sl_max_pct, 1)}% of genuine rules. The gate admits ${f(adv.genuine_gate_min_pct, 1)}% to ${f(adv.genuine_gate_max_pct, 1)}%.`,
    "Knowledge gate");
  chart(s, "adversarial_far", 0.6, 1.3, 12.1, 4.3, "Line chart of false admission rate and bar chart of genuine admission rate per poison rate and policy");
  takeaway(s, 5.85, `The gate keeps harmful admissions at or below ${f(adv.far_gate_max_pct, 1)}% and still admits ${f(adv.genuine_gate_min_pct, 0)}% or more of genuine rules. The SL screen loses about one third of genuine rules.`, 0.85);
}

// 19 Rollback
{
  const s = newSlide("CONTENT", "Injected harmful rules are rolled back fast",
    `In this test, ${adv.n_injected} harmful rules go directly into the active knowledge base. ` +
    `The left chart shows the rule categories. The right chart shows the cycles to rollback per poison rate. One cycle has 8 verified firings. ` +
    `The gate rolled back ${f(adv.rollback_rate_gate_pct, 0)}% of the injected rules. ${f(adv.ttr_one_cycle_pct, 1)}% were rolled back in one cycle. The maximum was ${f(adv.ttr_max_cycles, 0)} cycles.`,
    "Knowledge gate");
  chart(s, "adversarial_rollback", 0.6, 1.3, 12.1, 4.1, "Donut of injected rule categories and stacked bars of cycles to rollback per poison rate");
  takeaway(s, 5.75, `${f(adv.rollback_rate_gate_pct, 0)}% of ${adv.n_injected} injected harmful rules were rolled back. ${f(adv.ttr_one_cycle_pct, 1)}% took one cycle (8 firings). The maximum was ${f(adv.ttr_max_cycles, 0)} cycles.`, 0.85);
}

// 20 Martingale wealth
{
  const rp = mt.rollback_poisoned || [], rs = mt.rollback_sybil || [], ag = mt.admission_genuine || [];
  const s = newSlide("CONTENT", "Martingale wealth trajectories",
    `Each line is the e-process wealth of one rule. A test stops when the wealth reaches 1/alpha = 20. ` +
    `In the rollback test, poisoned and Sybil rules reach 20 after ${Math.min(rp[0], rs[0])} to ${Math.max(rp[1], rs[1])} firings. The wealth of genuine rules falls toward zero, so they stay active. ` +
    `In the admission test, genuine rules reach 20 after ${ag[0]} to ${ag[1]} firings. Poisoned and Sybil rules never reach 20, so the gate never admits them. ` +
    `Curves are clipped at 1e-8 for display.`,
    "Knowledge gate");
  chart(s, "martingale_wealth", 0.6, 1.3, 8.4, 5.45, "Log-scale martingale wealth per rule for the rollback and admission tests");
  card(s, 9.25, 1.35, 3.45, 4.35, "What the curves show", [
    `Rollback test: poisoned and Sybil rules cross 20 after ${Math.min(rp[0], rs[0])} to ${Math.max(rp[1], rs[1])} firings.`,
    `Genuine rules never cross, so they stay active.`,
    `Admission test: genuine rules cross after ${ag[0]} to ${ag[1]} firings.`,
    `Poisoned and Sybil rules never cross, so they are never admitted.`,
  ], { size: 13 });
}

// 21 Ville + TD-TPF
{
  const r = D.recovery || {};
  const s = newSlide("CONTENT", "Ville's bound holds; recovery stops at 24 hours",
    `Left: under the null hypothesis, Ville's inequality says the crossing probability is at most alpha = 0.05. ` +
    `The exact crossing probability is ${f(adv.ville_exact, 4)}. The highest Monte-Carlo rate is ${f(adv.ville_mc_max, 4)}. All checks are below alpha. ` +
    `Right: TD-TPF recovery prevents 100% of the loss for detection delays up to ${r.full_until_h} hours. From ${r.zero_from_h} hours it prevents nothing, because the ${r.window_hours}-hour recovery window has closed. ` +
    `The three kappa values give the same curve, so the lines overlap. The recovery curves use ${r.n_frauds} frauds from fig4_tdtpf_recovery_curves.csv.`,
    "Knowledge gate");
  chart(s, "ville_bound", 0.6, 1.3, 5.9, 4.3, "Crossing probabilities per poison rate against alpha");
  chart(s, "tdtpf_recovery", 6.8, 1.3, 5.9, 4.3, "Prevented loss against detection delay for three kappa values");
  card(s, 0.6, 5.8, 5.9, 0.7, null, [`All checks are below alpha = 0.05 (max MC rate ${f(adv.ville_mc_max, 4)}).`], { size: 13 });
  card(s, 6.8, 5.8, 5.9, 0.7, null, [`100% prevented up to ${r.full_until_h} h; 0% from ${r.zero_from_h} h.`], { size: 13 });
}

// 22 Latency
{
  const s = newSlide("CONTENT", "Scoring latency and fit time",
    `Latency is the amortised batch-scoring time per transaction on CPU, as the mean over windows. ` +
    `MG-ECFM needs ${f(lat.mg_ecfm)} ms per transaction. Only BP1 is faster (${f(lat.bp1)} ms). ` +
    `MG-ECFM has the shortest fit time (${n0(fit.mg_ecfm)} s). BP3 CARE-GNN needs ${n0(fit.bp3)} s. ` +
    `Fit times come from fit_reports in the manifest. They use the fast profile.`,
    "Performance");
  chart(s, "latency_fit", 0.6, 1.3, 12.1, 4.5, "Bar charts of scoring latency and fit time per model");
  takeaway(s, 5.95, `MG-ECFM scores a transaction in ${f(lat.mg_ecfm)} ms and fits in ${n0(fit.mg_ecfm)} s, the shortest fit time of the five models. Only BP1 scores faster.`, 0.85);
}

// 23 Conclusions
{
  const s = newSlide("CLOSING_DARK", "Conclusions and next steps",
    `MG-ECFM gives the best pooled PR-AUC, Brier, F1 and F2 in this run. But the lead over BP4 on PR-AUC is not significant. ` +
    `MG-ECFM has the highest monetary loss because its routing thresholds rose close to 1 and it approved most frauds. ` +
    `The knowledge gate works well in the synthetic stress test, but it admitted no rules on the real stream. ` +
    `The main next step is a full run on all 6.36M rows with the configured splits, so that the operating split has both classes and the threshold is tuned.`,
    "Conclusions");
  const colW = 5.9;
  const findings = [
    `Best pooled PR-AUC (${f(mgPR.estimate)}), Brier, F1 and F2 of the five models.`,
    `The PR-AUC lead over BP4 is not significant (Holm p = ${fmtP(holmPrBp4)}).`,
    `Highest expected monetary loss. Every baseline has a lower loss.`,
    `Knowledge gate: false admission at most ${f(adv.far_gate_max_pct, 1)}%, and Ville's bound holds.`,
    `The gate admitted 0 rules on PaySim, so the KB had no effect here.`,
  ];
  const steps = [
    "Rerun on all 6.36M rows with the configured step splits.",
    "Make sure the operating split has both classes, so the threshold is tuned.",
    "Check why CS-ACRC thresholds rise close to 1.",
    "Find why the gate admits no rules on PaySim.",
    "Fix float64 on MPS so the Apple GPU can be used.",
  ];
  const col = (x, head, items, numbered) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.4, w: colW, h: 5.3, fill: { color: "1E3A66" }, line: { color: "1E3A66" }, rectRadius: 0.08, objectName: "closing_card" });
    s.addText([
      { text: head, options: { fontSize: 20, bold: true, color: "CADCFC", breakLine: true, paraSpaceAfter: 8 } },
      ...items.map((t, i) => ({ text: t, options: { fontSize: 17, color: C.background1, bullet: numbered ? { type: "number", numberStartAt: i + 1 } : true, paraSpaceAfter: 14, breakLine: i < items.length - 1 } })),
    ], { x: x + 0.3, y: 1.6, w: colW - 0.6, h: 4.9, valign: "top", isTextBox: true, margin: 0 });
  };
  col(0.6, "Findings", findings, false);
  col(6.8, "Next steps", steps, true);
}

// ---------------------------------------------------------------- write
(async () => {
  await pres.writeFile({ fileName: OUT });
  const candidates = [SKILL && path.join(SKILL, "scripts", "apply_theme.js")].filter(Boolean);
  const themeJs = candidates.find((p) => fs.existsSync(p));
  if (!themeJs) throw new Error("Set PPTX_SKILL_DIR to the pptx skill folder (needs scripts/apply_theme.js).");
  const { applyTheme } = require(themeJs);
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
