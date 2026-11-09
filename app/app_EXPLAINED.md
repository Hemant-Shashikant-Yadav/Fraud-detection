# Code Walkthrough: `app/app.py`

**Purpose:** The **Streamlit dashboard** — an interactive front end for the live MG-ECFM service (`src/orchestrator/service.py`). It is the demo tool for the presentation: score one transaction, see why, inspect the mule network, upload a CSV, look at the knowledge base, and run a recovery trace.

```bash
python main.py --ui          # or: streamlit run app/app.py
```

---

## 1. Section-by-section

### 1.1 Setup and helpers (lines 1–113)
Imports, page configuration and service loading.
* `candidate_paths()` — where to look for a saved service (`MG_ECFM_SERVICE` env var, `models/mg_ecfm_service.pkl`, `models/quick_demo_service.pkl`).
* `get_service()` — cached loader (`st.cache_resource`): loads a saved model if found, otherwise builds a small demo service; returns the service and a note on its origin.
* `cached_latency`, `cached_trace` — cached latency benchmark and recovery trace (so repeated clicks don't recompute).
* `fmt_money`, `action_badge` — formatting helpers (coloured Approve / Step-Up / Hold / Block badges).

### 1.2 Plot functions (lines 115–260)
Matplotlib figures:
* `risk_dial` — the risk score against the three CS-ACRC thresholds and the binary threshold.
* `attribution_plot` — top TreeSHAP contributions (grouped into graph embedding, autoencoder residual, KB triggers, tabular features).
* `graph_plot` — the camouflage neighbourhood: nodes by role/hop, kept vs pruned edges with similarity.
* `recovery_curve_plot` — recovery rate vs detection delay with the 24 h settlement window marked.
* `kb_plot` — rule status and e-process wealth.

### 1.3 Page layout and sidebar (lines 261–299)
Sidebar "Live service": data source, profile, last step, active rules, current CS-ACRC thresholds, latency benchmark button. Then `st.tabs` with five tabs:

### 1.4 ⚡ Live Scoring (lines ~300–368)
Inputs for sender, receiver, amount and type (with example presets: mule-network fraud, routine payment, large transfer to a never-seen account). Calls `service.score_transaction(...)` and shows probability, risk score, **action** with the route reason (threshold, amount escalation, KB floor, uncertain-block→hold), expected loss per action, fired rules, per-component latency (graph / AE / KB / tabular / routing), and the explanation (attribution plot, feature deviations vs training, autoencoder-residual percentile).

### 1.5 🕸 Mule Network (lines ~370–400)
`service.explain_graph(...)`: the two-hop neighbourhood and how the camouflage filter keeps or prunes each edge, with statistics (kept fraction, pruned edges, GNN probability, whether accounts are known).

### 1.6 📄 Batch Scorer (lines ~402–453)
Upload a CSV (or use the built-in sample); leakage columns are stripped automatically; `service.score_frame` returns scored rows, an action distribution, ms per transaction and — if labels are present — metrics and the share of frauds intercepted. Download button for the decisions.

### 1.7 Knowledge-base inspector (lines ~455–490)
`service.kb_table()` / `kb_summary()` / `kb_snapshots()`: every candidate rule with status, Subjective-Logic belief/disbelief/uncertainty, expected probability, admission and rollback e-process wealth, store integrity and snapshot history.

### 1.8 Recovery trace (lines ~492–543)
Pick an undetected fraud (`recovery_candidates`) and a time-to-detection; `recovery_trace` shows the TD-TPF taint path through mule wallets, the freeze/block/chargeback actions, and the recovery-vs-delay curve.

## 2. Things to remember
* The dashboard shows whatever model it loads: results from a **synthetic demo model** must not be presented as real-data results.
* Single-transaction scoring is a what-if: it never changes the live state.
