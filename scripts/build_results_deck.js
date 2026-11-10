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
  date: "2026-11-10",
  command: "python -m experiments.run_drift_eval --source paysim --sample-rows 2000000 --save-service models/mg_ecfm_paysim_2m.pkl",
  totalRows: 6362620,
  sampleRows: 2000000,
  runtimeS: 9682,
  machine: "Apple Silicon Mac (CPU)",
  modelFile: "models/mg_ecfm_paysim_2m.pkl",
  modelGb: 1.1,
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
pres.subject = "Updated MG-ECFM pipeline results on real PaySim data (tuned thresholds)";
const C = pres.SchemeColor;

const FOOT = `MG-ECFM results  |  PaySim 2M, tuned thresholds  |  ${RUN.date}`;
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
const thrList = MODELS.map((k, i) => `${TINY[k]} ${f(D.thresholds[i], 3)}`).join(", ");
const binTp = cm.mg_ecfm.tp;
// 1 Title
{
  const s = newSlide("TITLE_DARK", "MG-ECFM results on PaySim",
    `This deck shows the updated results of the full MG-ECFM pipeline on real PaySim data. ` +
    `The run used a seeded random sample of ${n0(RUN.sampleRows)} of ${n0(RUN.totalRows)} rows from all 743 steps. ` +
    `The decision threshold of every model is now tuned on the operating split. Slide 3 lists what changed from the first run.`,
    "Introduction");
  s.addText([
    { text: `Updated run: ${(RUN.sampleRows / 1e6).toFixed(0)}M rows sampled from all ${(RUN.totalRows / 1e6).toFixed(2)}M real PaySim rows`, options: { breakLine: true } },
    { text: `Five models, ${W.n} chronological drift windows, tuned thresholds, CPU run  |  ${RUN.date}`, options: { fontSize: 16, color: "CADCFC" } },
  ], { placeholder: "body" });
}

// 2 Run setup
{
  const s = newSlide("CONTENT", "Run setup and data splits",
    `The run used the fast profile with seed ${D.run.seed}. The rows are a uniform random sample of the full log, in time order. ` +
    `So the configured step splits of the project now apply. ` +
    `Train has ${n0(sp.tx.train)} rows, calibration ${n0(sp.tx.calibration)}, operating ${n0(sp.tx.operating)} and test ${n0(sp.tx.test)}. ` +
    `The test split has ${fraudN} frauds. The drift evaluation took ${n0(RUN.runtimeS)} seconds. ` +
    `All numbers come from reports/drift_evaluation_manifest.json.`,
    "Introduction");
  chart(s, "split_sizes", 0.6, 1.35, 8.3, 4.0, "Donut of rows per split and bar chart of fraud rate per split");
  s.addText([
    { text: "Command", options: { bold: true, color: C.text2, breakLine: true } },
    { text: RUN.command, options: { fontFace: "Courier New", fontSize: 11, color: C.text1 } },
  ], { x: 0.6, y: 5.55, w: 8.3, h: 0.9, fontSize: 13, valign: "top", isTextBox: true, margin: 0 });
  card(s, 9.25, 1.35, 3.45, 5.1, "Run facts", [
    `PaySim: ${n0(RUN.sampleRows)} of ${n0(RUN.totalRows)} rows, sampled from steps 1 to 743.`,
    `Fast profile, seed ${D.run.seed}, ${D.run.n_features} causal features.`,
    `${W.n} chronological test windows (steps 632 to 743).`,
    `Drift evaluation: ${hours} h ${mins} min on an ${RUN.machine}.`,
    `Saved model: ${RUN.modelFile} (${RUN.modelGb} GB).`,
  ], { size: 13 });
}

// 3 What changed
{
  const s = newSlide("CONTENT", "What changed from the first run",
    `In the first run, the operating split had 186 rows and all of them were frauds. So the threshold stayed at 0.5 for every model. ` +
    `Also, the first run used only the first 2M rows, which cover steps 1 to 179 of 743. ` +
    `This run samples 2M rows from all steps and requires both classes in the calibration and operating splits. ` +
    `The operating split now has ${n0(sp.tx.operating)} rows and ${sp.frauds.operating} frauds. The tuned thresholds are: ${thrList}. ` +
    `The test period changed, so do not compare the numbers of the two runs directly.`,
    "Introduction");
  const tw = 3.8, gap = 0.35;
  tile(s, 0.6, 1.4, tw, 2.35, `${sp.frauds.operating} of ${n0(sp.tx.operating)}`, "Operating rows are fraud",
    "Both classes are present, so the threshold search runs.");
  tile(s, 0.6 + tw + gap, 1.4, tw, 2.35, f(D.thresholds[0], 3), "MG-ECFM threshold",
    `Tuned on the operating split. It was 0.5 in the first run.`);
  tile(s, 0.6 + 2 * (tw + gap), 1.4, tw, 2.35, "1 to 743", "Steps in the sample",
    "The first run used steps 1 to 179 only.");
  card(s, 0.6, 4.05, 5.9, 2.35, "Code changes", [
    "Calibration and operating splits must contain both classes.",
    "New option --sample-rows: a seeded sample over the whole stream.",
    "New option --save-service: saves the evaluated MG-ECFM model.",
  ], { size: 13 });
  card(s, 6.8, 4.05, 5.9, 2.35, "Read with care", [
    `The test split has ${n0(sp.tx.test)} rows and a fraud rate of ${pct(fraudN / sp.tx.test, 2)}.`,
    `${W.all_fraud} of ${W.n} test windows contain only frauds. PR-AUC is not defined there.`,
    "The first run had a different test period, so its numbers are not comparable.",
  ], { size: 13 });
}

// 4 Headline
{
  const s = newSlide("CONTENT", "Headline results on the pooled test set",
    `All four numbers are pooled over the ${n0(sp.tx.test)} test transactions. They come from bootstrap_ci in reports/statistical_significance_results.json. ` +
    `MG-ECFM has the highest PR-AUC, F1 and F2 of the five models. Its Brier score is equal to BP4. ` +
    `The PR-AUC lead over ${TINY[prBest]} is small and not significant (Wilcoxon, Holm p = ${fmtP(holmPrBp4)}). ` +
    `MG-ECFM has rank ${emlRank} of 5 on expected monetary loss.`,
    "Detection results");
  const tw = 2.8, gap = 0.3;
  tile(s, 0.6, 1.4, tw, 2.7, f(mgPR.estimate), "MG-ECFM PR-AUC (pooled)",
    `95% CI ${f(mgPR.ci_low)} to ${f(mgPR.ci_high)}. Best baseline: ${TINY[prBest]} ${f(boot(prBest, "pr_auc"))}.`);
  tile(s, 0.6 + (tw + gap), 1.4, tw, 2.7, f(boot("mg_ecfm", "f1")), "MG-ECFM F1 (pooled)",
    `Tuned threshold ${f(D.thresholds[0], 3)}. Best baseline: ${TINY[bestBase("f1", false)]} ${f(boot(bestBase("f1", false), "f1"))}.`);
  tile(s, 0.6 + 2 * (tw + gap), 1.4, tw, 2.7, f(boot("mg_ecfm", "f2")), "MG-ECFM F2 (pooled)",
    `Best baseline: ${TINY[f2Best]} ${f(boot(f2Best, "f2"))}.`);
  tile(s, 0.6 + 3 * (tw + gap), 1.4, tw, 2.7, n0(boot("mg_ecfm", "expected_monetary_loss")), "MG-ECFM loss per txn",
    `Rank ${emlRank} of 5 (lower is better). Lowest: ${TINY[emlBest]} ${n0(boot(emlBest, "expected_monetary_loss"))}.`, { bigColor: C.accent6 });
  card(s, 0.6, 4.4, 12.1, 1.9, "What this means", [
    `MG-ECFM detects frauds best: it catches ${binTp} of ${fraudN} frauds at its threshold, more than any baseline.`,
    `The PR-AUC lead over ${TINY[prBest]} is not significant (Wilcoxon, Holm p = ${fmtP(holmPrBp4)}).`,
    `The monetary loss is high because CS-ACRC routing approves most frauds (slides 12 and 13).`,
  ]);
}

// 5 Pooled CI bars
{
  const s = newSlide("CONTENT", "Pooled metrics with 95% bootstrap intervals",
    `Each bar is the pooled value on the full test set. The error bar is the 95% block-bootstrap interval with the monitoring window as the unit (1,000 resamples). ` +
    `MG-ECFM leads on PR-AUC (${f(boot("mg_ecfm", "pr_auc"))}), ROC-AUC (${f(boot("mg_ecfm", "roc_auc"))}), F1 and F2. ` +
    `The PR-AUC intervals of all models overlap a lot, because the test split has only ${fraudN} frauds. ` +
    `BP2 has the lowest expected monetary loss.`,
    "Detection results");
  chart(s, "pooled_metrics_ci", 0.6, 1.3, 12.1, 5.45, "Six bar charts of pooled metrics per model with bootstrap intervals");
}

// 6 Window distributions
{
  const s = newSlide("CONTENT", "Per-window results vary a lot",
    `Each dot is one test window. The diamond is the mean over windows. PR-AUC is defined in ${W.pr_auc_defined} of ${W.n} windows. ` +
    `MG-ECFM has the highest window mean on PR-AUC (${f(wm.mg_ecfm.pr_auc)}) and F2 (${f(wm.mg_ecfm.f2)}). ` +
    `The spread inside each model is much larger than the gap between models. So the per-window tests do not find a significant PR-AUC difference.`,
    "Detection results");
  chart(s, "window_boxplots", 0.6, 1.3, 8.4, 5.45, "Box and strip plots of per-window PR-AUC and F2 per model");
  card(s, 9.25, 1.35, 3.45, 4.85, "Per-window view", [
    `Mean PR-AUC: MG-ECFM ${f(wm.mg_ecfm.pr_auc)}, BP1 ${f(wm.bp1.pr_auc)}, BP4 ${f(wm.bp4.pr_auc)}.`,
    `Mean F2: MG-ECFM ${f(wm.mg_ecfm.f2)}, BP2 ${f(wm.bp2.f2)}.`,
    `Wilcoxon on PR-AUC: no baseline is significant after Holm.`,
  ], { size: 13 });
}

// 7 Drift PR-AUC
{
  const s = newSlide("CONTENT", "PR-AUC across the 50 drift windows",
    `The top panel shows PR-AUC per window for each model. The bottom panel shows the number of frauds per window. ` +
    `The grey bands mark the ${W.all_fraud} windows that contain only frauds. PR-AUC is not defined there. ` +
    `Window sizes go from ${n0(W.rows_min)} to ${n0(W.rows_max)} transactions. The late PaySim steps have few genuine transactions, so many windows are small. ` +
    `All five models follow a similar path.`,
    "Detection results");
  chart(s, "drift_pr_auc_windows", 0.6, 1.3, 12.1, 5.45, "Line chart of PR-AUC per window per model, with fraud counts below");
}

// 8 EML windows
{
  const s = newSlide("CONTENT", "Expected monetary loss per window",
    `Expected monetary loss (EML) is the realised asymmetric loss per transaction under each model's own policy. ` +
    `The y axis uses a log scale. The loss of MG-ECFM is high in the all-fraud windows, because CS-ACRC approves most frauds there. ` +
    `BP1 to BP4 use a binary approve or block policy at their tuned threshold.`,
    "Detection results");
  chart(s, "drift_eml_windows", 0.6, 1.3, 8.4, 5.45, "Line chart of expected monetary loss per window per model, log scale");
  card(s, 9.25, 1.35, 3.45, 4.85, "Why the loss is high", [
    `CS-ACRC approved ${act.fraud.Approve} of ${fraudN} frauds.`,
    `At its threshold, MG-ECFM flags ${binTp} frauds, but the routing does not use that threshold.`,
    `Pooled EML per txn: MG-ECFM ${n0(boot("mg_ecfm", "expected_monetary_loss"))}, BP2 ${n0(boot("bp2", "expected_monetary_loss"))}, BP3 ${n0(boot("bp3", "expected_monetary_loss"))}.`,
  ], { size: 13 });
}

// 9 PR / ROC
{
  const c = D.curves;
  const s = newSlide("CONTENT", "Pooled precision-recall and ROC curves",
    `The curves use all ${n0(sp.tx.test)} test transactions. They use the scores only, so the threshold does not change them. ` +
    `MG-ECFM has the highest average precision (${f(c.mg_ecfm.pooled_pr_auc)}) and the highest ROC-AUC (${f(c.mg_ecfm.pooled_roc_auc)}). ` +
    `On ROC-AUC all models are between ${f(Math.min(...MODELS.map((k) => c[k].pooled_roc_auc)))} and ${f(Math.max(...MODELS.map((k) => c[k].pooled_roc_auc)))}. ` +
    `The fraud base rate in the test split is ${pct(fraudN / sp.tx.test, 2)}.`,
    "Detection results");
  chart(s, "pooled_pr_roc_curves", 0.6, 1.3, 12.1, 5.45, "Precision-recall and ROC curves per model on the pooled test set");
}

// 10 Confusion
{
  const s = newSlide("CONTENT", "Confusion matrices at the tuned thresholds",
    `Each matrix shows the count and the row-normalised rate. Each model uses its own threshold, tuned on the operating split. ` +
    `MG-ECFM catches the most frauds (${cm.mg_ecfm.tp} of ${fraudN}) with ${cm.mg_ecfm.fp} false alarms. ` +
    `BP4 is the most strict: it raises only ${cm.bp4.fp} false alarms but catches ${cm.bp4.tp} frauds. ` +
    `In the first run, BP1 to BP3 flagged most genuine transactions. With tuned thresholds they do not.`,
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
    `Each panel shows the share of each class per score bin on a log scale. The dashed line is the tuned threshold of each model. ` +
    `MG-ECFM gives low scores to genuine transactions and spreads fraud scores over a wide range. So its tuned threshold (${f(D.thresholds[0], 3)}) is low. ` +
    `BP1 to BP3 give high scores to most transactions, so their tuned thresholds are close to 1.`,
    "Detection results");
  chart(s, "score_distributions", 0.6, 1.3, 12.1, 3.6, "Histograms of model scores for legit and fraud transactions, per model");
  card(s, 0.6, 5.1, 5.9, 1.6, "MG-ECFM", [
    `Threshold ${f(D.thresholds[0], 3)}: low, because genuine scores are near zero.`,
    "Many fraud scores are between 0.01 and 0.5.",
  ], { size: 13 });
  card(s, 6.8, 5.1, 5.9, 1.6, "BP1 to BP3", [
    `Thresholds ${f(D.thresholds[1], 2)} to ${f(D.thresholds[2], 2)}: close to 1.`,
    "Scores are not calibrated (Brier 0.07 to 0.10).",
  ], { size: 13 });
}

// 12 Action mix
{
  const s = newSlide("CONTENT", "MG-ECFM action mix: Approve, Step-Up, Hold, Block",
    `CS-ACRC routes each transaction to one of four actions: Approve, Step-Up, Hold or Block. ` +
    `MG-ECFM sent ${nonApprove} of ${n0(sp.tx.test)} test transactions away from Approve. ` +
    `Of the ${fraudN} frauds, ${act.fraud.Approve} were approved, ${act.fraud["Step-Up"]} got Step-Up, ${act.fraud.Hold} got Hold and ${act.fraud.Block} were blocked. ` +
    `No genuine transaction got an intervention. The routing does not use the tuned threshold. The next slide shows the reason.`,
    "Routing");
  chart(s, "action_mix", 0.6, 1.3, 12.1, 4.6, "Donut of non-approve actions and stacked bars of action share by true class");
  takeaway(s, 6.0, `CS-ACRC approved ${act.fraud.Approve} of ${fraudN} frauds (${pct(act.fraud.Approve / fraudN)}). At the tuned threshold, the detector flags ${binTp} frauds. The routing loses this gain.`);
}

// 13 CS-ACRC thresholds
{
  const s = newSlide("CONTENT", "CS-ACRC thresholds are close to 1",
    `The left chart shows the three CS-ACRC thresholds per window. After the operating split they are ${W.lambda_start.map((v) => f(v, 2)).join(", ")}. ` +
    `The median over windows is ${W.lambda_median.map((v) => f(v, 2)).join(", ")}. The tuned detector threshold is only ${f(D.thresholds[0], 3)}. ` +
    `With routing thresholds this high, CS-ACRC approves almost everything. This is the main cause of the high monetary loss. ` +
    `The small chart compares the pooled non-approve rate of all models on a log scale.`,
    "Routing");
  chart(s, "crc_thresholds", 0.6, 1.3, 8.4, 5.45, "Step lines of the three routing thresholds and stacked bars of intervention rate per window");
  chart(s, "intervention_rate_models", 9.25, 1.35, 3.5, 2.6, "Bar chart of pooled non-approve rate per model, log scale");
  card(s, 9.25, 4.1, 3.45, 2.6, null, [
    `Start: ${W.lambda_start.map((v) => f(v, 2)).join(" / ")}.`,
    `Median: ${W.lambda_median.map((v) => f(v, 2)).join(" / ")}.`,
    `Detector threshold: ${f(D.thresholds[0], 3)}.`,
  ], { size: 13 });
}

// 14 Significance heatmap
{
  const s = newSlide("CONTENT", "Statistical significance against each baseline",
    `Each cell gives the Holm-adjusted p-value of MG-ECFM against one baseline. A star means significant at 0.05. The text gives the direction. ` +
    `McNemar compares per-transaction correctness at the tuned thresholds. Only BP4 differs (Holm p = ${fmtP(holm(0, "bp4"))}): BP4 is more often correct (${pct(D.accuracy.bp4, 2)} vs ${pct(D.accuracy.mg_ecfm, 2)}), because it raises fewer false alarms. ` +
    `The Wilcoxon test on per-window PR-AUC is not significant for any baseline. ` +
    `The Wilcoxon test on F2 is significant for all four baselines.`,
    "Significance");
  chart(s, "significance_heatmap", 0.6, 1.3, 8.4, 5.45, "Heatmap of Holm-adjusted p-values per test and baseline");
  card(s, 9.25, 1.35, 3.45, 4.35, "Reading the tests", [
    `McNemar: only BP4 differs, in favour of BP4.`,
    `PR-AUC: no significant difference (Holm p = 1.0).`,
    `F2: Holm p < 0.05 against all four baselines.`,
    `EML: BP2 has a lower loss (Holm p < 0.001).`,
  ], { size: 13 });
}

// 15 Paired differences
{
  const s = newSlide("CONTENT", "Pooled differences with 95% bootstrap intervals",
    `Each point is the pooled MG-ECFM value minus the baseline value. The bar is the 95% block-bootstrap interval. ` +
    `If the interval does not cross zero, the difference is clear. ` +
    `PR-AUC: the interval excludes zero for ${excl("pr_auc").join(", ") || "no baseline"}. ` +
    `F2: it excludes zero for ${excl("f2").join(", ") || "no baseline"}. Brier: for ${excl("brier").join(", ")}. EML: for ${excl("expected_monetary_loss").join(", ")}.`,
    "Significance");
  chart(s, "paired_difference_ci", 0.6, 1.3, 12.1, 3.4, "Forest plots of pooled metric differences, MG-ECFM minus each baseline");
  const tw = 2.8, gap = 0.3;
  const mk = (i, head, lines) => card(s, 0.6 + i * (tw + gap), 4.95, tw, 1.75, head, lines, { size: 13 });
  mk(0, "PR-AUC", [`Clear gain over ${excl("pr_auc").join(", ") || "none"}.`, `Interval crosses zero for ${BASE.filter((b) => !pd[b].pr_auc.excludes_zero).map((b) => TINY[b]).join(", ")}.`]);
  mk(1, "F2", [`Clear gain over ${excl("f2").join(", ")}.`]);
  mk(2, "Brier", [`MG-ECFM is lower than ${excl("brier").join(", ")}.`, `Equal to BP4.`]);
  mk(3, "EML", [`Clear difference for ${excl("expected_monetary_loss").join(", ")}.`, `Only BP3 has a higher loss.`]);
}

// 16 Ablation switches
{
  const s = newSlide("CONTENT", "Ablation design: seven arms, B1 to B5",
    `The matrix shows which component each arm switches on. Blue means on. ` +
    `B3 adaptive, B4 and B5 branch from one fitted pipeline. So they share the same ranking metrics by construction. ` +
    `The martingale knowledge gate again admitted 0 rules on the real stream. So the knowledge base, recovery and assist had no rules to act on.`,
    "Ablation");
  chart(s, "ablation_switches", 0.6, 1.3, 6.0, 4.0, "Matrix of component switches per ablation arm");
  card(s, 6.9, 1.35, 5.8, 3.95, "Read this before the results", [
    `B1 is XGBoost only, with no graph, gate or recovery.`,
    `B2 admits every candidate rule (${n0(ab.B2_graph_gate_off.kb_rules_ever_admitted)} rules).`,
    `B3 adaptive, B4 and B5 share one fitted pipeline, so their ranking metrics are the same by construction.`,
    `The gate admitted ${n0(ab.B4_full_mg_ecfm.kb_rules_ever_admitted)} rules in B3 to B5.`,
    `So this run cannot measure the effect of the KB, recovery and assist.`,
  ], { size: 14 });
}

// 17 Ablation results
{
  const s = newSlide("CONTENT", "Ablation results per arm",
    `All values are pooled over the test windows. B1 XGBoost only has the highest PR-AUC (${f(ab.B1_xgboost_only.pr_auc)}). The full MG-ECFM has ${f(ab.B4_full_mg_ecfm.pr_auc)}. ` +
    `B2 with the gate off has the lowest EML (${n0(ab.B2_graph_gate_off.expected_monetary_loss)}) and the highest prevented loss (${f(ab.B2_graph_gate_off.prevented_loss_ratio_pct, 1)}%). ` +
    `Static CRC (${n0(ab.B3_static_crc.expected_monetary_loss)}) and adaptive CS-ACRC (${n0(ab.B3_adaptive_crc.expected_monetary_loss)}) give almost the same EML. ` +
    `Recovery in B4 does not change the net loss (${money(ab.B4_full_mg_ecfm.net_monetary_loss)}).`,
    "Ablation");
  chart(s, "ablation_metrics", 0.6, 1.3, 12.1, 3.9, "Bar charts of PR-AUC, F2, EML and prevented loss per ablation arm");
  card(s, 0.6, 5.4, 5.9, 1.3, null, [
    `B1 XGBoost only: PR-AUC ${f(ab.B1_xgboost_only.pr_auc)}, higher than full MG-ECFM.`,
    `B2 (gate off): lowest EML (${n0(ab.B2_graph_gate_off.expected_monetary_loss)}).`,
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
  const faster = BASE.filter((b) => lat[b] < lat.mg_ecfm).map((b) => `${TINY[b]} (${f(lat[b])} ms)`);
  const s = newSlide("CONTENT", "Scoring latency and fit time",
    `Latency is the amortised batch-scoring time per transaction on CPU, as the mean over windows. ` +
    `MG-ECFM needs ${f(lat.mg_ecfm)} ms per transaction. ${faster.length ? "Faster: " + faster.join(", ") + "." : "No baseline is faster."} ` +
    `MG-ECFM has the shortest fit time (${n0(fit.mg_ecfm)} s). BP3 CARE-GNN needs ${n0(fit.bp3)} s. ` +
    `Fit times come from fit_reports in the manifest. They use the fast profile on ${n0(sp.tx.train)} training rows.`,
    "Performance");
  chart(s, "latency_fit", 0.6, 1.3, 12.1, 4.5, "Bar charts of scoring latency and fit time per model");
  takeaway(s, 5.95, `MG-ECFM scores a transaction in ${f(lat.mg_ecfm)} ms and fits in ${n0(fit.mg_ecfm)} s, the shortest fit time of the five models.`, 0.85);
}

// 23 Conclusions
{
  const s = newSlide("CLOSING_DARK", "Conclusions and next steps",
    `With tuned thresholds, MG-ECFM gives the best pooled PR-AUC, ROC-AUC, F1 and F2 of the five models. ` +
    `The PR-AUC lead is small and not significant. The F2 lead is significant against all four baselines. ` +
    `The monetary loss is still high, because the CS-ACRC routing thresholds are close to 1 and approve most frauds. ` +
    `The knowledge gate still admits no rules on the real stream.`,
    "Conclusions");
  const colW = 5.9;
  const findings = [
    `Best pooled PR-AUC (${f(mgPR.estimate)}), F1 (${f(boot("mg_ecfm", "f1"))}) and F2 (${f(boot("mg_ecfm", "f2"))}).`,
    `PR-AUC lead is not significant. F2 lead is significant (Holm p < 0.05).`,
    `CS-ACRC approves ${act.fraud.Approve} of ${fraudN} frauds, so the loss is rank ${emlRank} of 5.`,
    `XGBoost alone (B1) has a higher PR-AUC than the full model.`,
    `The gate admitted 0 rules, so the KB had no effect.`,
  ];
  const steps = [
    "Align the CS-ACRC thresholds with the tuned detector threshold.",
    "Find why the graph and fusion layers lower PR-AUC versus XGBoost.",
    "Find why the gate admits no rules on PaySim.",
    "Use more test data: the test split has only 404 frauds.",
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
