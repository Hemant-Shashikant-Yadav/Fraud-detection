# Code Walkthrough: `src/recovery/tainted_flow.py`

**Purpose:** **TD-TPF — Time-Decaying Tainted Proportional Flow.** After a fraud slips through and is detected, this module follows the stolen money forward through the transaction ledger (through mule wallets) so it can be frozen or charged back inside the 24-hour settlement window.

**Leakage note:** it uses ledger balances, but only in this *post-detection forensic* stage (the issuer can read its own ledger once fraud is confirmed). Balances are never fed to the pre-authorisation detector, so the causal-leakage guarantee is unaffected.

---

## 1. The model (docstring, lines 7–58)
A confirmed fraud `e0 = (u0, v0, A0, τ0)` is detected at `τ_detect ≤ τ0 + W_rec` (`W_rec = 24 h`). Taint is a **mass in money units**, seeded as `T(v0, τ0) = A0`. When node `a` holding taint received at `τ` sends `A'` at `τ'`, the taint delivered to `b` is

`T(b, τ') = T(a, τ) · [A' / (Balance(a) + A')] · exp(−κ(τ' − τ))`

* The bracket is the **pro-rata ("haircut") share** of `a`'s holdings that left.
* `exp(−κ·dt)` (κ = 0.10 per hour) is the **attribution-confidence decay**; the difference is booked as `attribution_loss`.
* A node that receives taint several times keeps one **lot** per arrival, each decaying from its own arrival time.
* **Termination:** tracking stops on edges that leave the reachable ledger — PaySim `CASH_OUT` (cash leaves), merchant payments (`nameDest` starting with `M`: out of freeze reach but eligible for chargeback), AMLSim `CASH_OUT` and external wires. Inflow types (`CASH_IN`, `DEPOSIT`) never carry taint.
* **Conservation:** `A0 = held + exited + attribution_loss + pruned + beyond_horizon` (checked by `conservation_residual`). Freezable value at a node is `min(T(v), Balance(v))`.

## 2. Section-by-section

### 2.1 Constants (lines 76–82)
Edge kinds: `internal`, `cash_out`, `external_wire`, `merchant`, `inflow`; `EXIT_KINDS` are the three that leave the ledger.

### 2.2 `FlowEdge` (lines 88–104)
Frozen directed edge `(edge_id, src, dst, amount, tau [hours], tx_type, label)`; validates finite, non-negative amount and finite time.

### 2.3 `EdgeSchema` (lines 107–157)
Maps a dataset's transaction types onto edge kinds: exit types, wire types, inflow types, merchant prefixes, hours per step. Presets `paysim()` (1 h/step, merchants `M*`), `amlsim()` (24 h/step, wires external), `generic()`; `by_name`. `classify(edge, external_accounts)` returns the kind (inflow → cash-out → external wire → merchant → internal).

### 2.4 Trace containers (lines 163–327)
* `TaintLot` (amount, arrival time, hop, path), `TaintedEdge` (all TD-TPF factors of one transfer: share, decay, moved, delivered, balance after debit), `ExitFlow`.
* `TaintTrace`: result of one traversal. Views: `node_taint`, `taint_concentration` (`T/Balance`), `min_hop`, `liquid_recoverable` (freezable value per mule wallet, victim excluded), `returned_to_victim`, aggregates `held`, `exited`, `conservation_residual`, and `metrics()` (recovered amount, **recovery rate %**, exits by kind, attribution loss, pruned, max hop, detection delay). `edges_frame` / `nodes_frame` give DataFrames for plots and the dashboard.

### 2.5 `TaintedFlowTracker.__init__` (lines 348–374)
Parameters: `kappa=0.10`, `window_hours=24`, `min_taint=0.01` (lots below `1%·A0` are pruned), `max_hops=6`, optional opening balances and external accounts. Edges are sorted by `(tau, edge_id)` (ids must be unique) and indexed for fast window queries.

### 2.6 Constructors (lines 377–472)
`config_kwargs` / `from_config` (read `hyperparams.json['tdtpf']`), edge builders `edges_from_paysim`, `edges_from_amlsim` (integer days × 24 h, or datetimes → hours), `edges_from_canonical` (event_time seconds → hours), `from_frame(frame, dataset)`, and `paysim_opening_balances` (forensic use of the raw PaySim balance columns before a given step).

### 2.7 Queries (lines 475–491)
`edges`, `edge(id)`, `window_edges(start, end)` (binary search), `decay(dt)`.

### 2.8 `trace(fraud, tau_detect)` (lines 494–598) — the traversal
1. Validate: `tau_detect` within `[τ0, τ0 + W_rec]`, fraud amount > 0, the fraud edge must credit an in-ledger beneficiary.
2. Find the edge range by binary search; keep a ledger (`bal(v)`, opening balance default 0).
3. For each edge in time order: inflows only credit the originator; edges before `e0` (same timestamp) update the ledger only; `e0` seeds a lot on the beneficiary; later edges debit the source (an unknown opening balance is treated as fully funded and counted as `deficit`).
4. If the source holds taint lots: `share = A'/(post + A')`; for each lot `moved = taint·share`, `delivered = moved·decay`, `attribution_loss += moved − delivered`; the lot shrinks; delivered taint is either recorded as an **exit flow** (exit kinds), counted as **beyond horizon** (`hop > max_hops`), **pruned** (`< min_taint·A0`), or becomes a **new lot** on the destination.
5. Build balances and return a `TaintTrace`.

## 3. Things to remember
* PaySim's ledger-balance columns are used here, and only here.
* Dust handling (`min_taint`) and `max_hops` are part of the conservation equation, so nothing disappears silently.
* The "≥ 80 % recovery" objective needs this tracer run on a graph with realistic multi-hop mule structure (AMLSim); PaySim's graph is too simple.
