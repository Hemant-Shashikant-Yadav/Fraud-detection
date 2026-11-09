# Code Walkthrough: `src/epistemic_gate/knowledge_store.py`

**Purpose:** A **versioned, transactional database** of fraud rules (SQLite). It is the "memory" of the knowledge-evolution gate: every admission or rollback creates a new *immutable snapshot* `V_j`, so the knowledge base can always be audited and rolled back without losing history.

---

## 1. Data model (docstring, lines 7–24)
| Table | Contents |
|---|---|
| `snapshots` | KB versions `V_0, V_1, …` with parent pointer, operation, reason, logical step, restored-from id |
| `rule_versions` | append-only rule history `(rule_id, version_id, snapshot_id, rule_logic, opinion_tuple, status, timestamp)` |
| `snapshot_members` | the exact `rule_id → version_id` map of each snapshot |
| `kb_head` | one row: the current snapshot |
| `certificates` | e-value certificates (`admission` / `rollback`) justifying each change |
| `opinion_log` | append-only Subjective-Logic opinion telemetry |

Rules: every write runs in one `BEGIN IMMEDIATE` transaction under a re-entrant lock and produces **exactly one** snapshot. **Rollback never deletes history** — it appends a new snapshot whose membership equals the target snapshot. JSON is stored as SQLite JSONB if available (≥ 3.45), otherwise validated JSON text.

## 2. Section-by-section

### 2.1 Constants and errors (lines 52–62)
`STATUSES = quarantined | active | rejected | retired`; `KnowledgeStoreError` for invalid transitions.

### 2.2 (De)serialisation helpers (lines 68–147)
* `_json_default`: converts NumPy scalars for JSON.
* `rule_to_logic(rule)`: rule → JSON `{conditions, predicate, meta}` (conditions are `feature op value`).
* `opinion_to_json` / `opinion_from_json`.
* `StoredRule`: a rule version loaded from the store. `weight` = Subjective-Logic expected probability (or 1.0); `evaluate(frame)` (vectorised) and `fires(event)` (scalar) make it **duck-compatible** with the fusion encoder and router.
* `_conditions_from_logic`: JSON → `Condition` objects.

### 2.3 `KnowledgeStore.__init__` and plumbing (lines 156–206)
Opens SQLite (`:memory:` by default; file mode uses WAL + `synchronous=FULL`), enables foreign keys and a busy timeout, detects JSONB. `_transaction()` is a context manager: begin immediate → yield → optional **fault injector** hook (used by tests to prove atomicity) → commit; any exception triggers rollback.

### 2.4 `_init_schema` (lines 208–271)
Creates the six tables with `CHECK` constraints (valid JSON, valid status) and indexes, and inserts the **genesis snapshot `V_0`** ("empty knowledge base") and the head pointer.

### 2.5 Internal helpers (lines 274–352)
`_head`, `_members`, `_snapshot_exists`, `_version_row`, `_current`, `_open_snapshot` (new id = max+1, parent = head), `_write_members`, `_insert_version` (version = max+1 per rule).
`_transition(operation, changes, …)` — the single mutation primitive: opens a snapshot, copies the base membership, applies each change (inserting a new rule version, inheriting logic/opinion from the previous version unless overridden by the `_KEEP` sentinel), writes membership, moves the head — **all in one transaction**.

### 2.6 Lifecycle API (lines 355–453)
* `head_snapshot_id`, `status_of`.
* `quarantine_candidate` (new or demoted rule; new rules need logic).
* `admit_candidate`: only from `quarantined` → `active`.
* `reject_candidate`, `retire_rule`.
* `revert_rule`: **targeted rollback** — one active rule back to quarantine, all others untouched.
* `rollback_to_snapshot(snapshot_id, quarantine_rule_ids, preserve_non_active)`: atomic restore to `V_j`; listed rules are quarantined; with `preserve_non_active` pending candidates added after `V_j` are kept so their e-processes are not orphaned.
* `pre_admission_snapshot(rule_id)`: the parent of the snapshot that last admitted the rule (`V_j` for rollback).

### 2.7 Telemetry (lines 456–486)
`record_certificate` / `certificates` (e-value payloads), `record_opinions` / `latest_opinion`.

### 2.8 Queries (lines 489–557)
`_materialise` (row → `StoredRule`, using the latest logged opinion), `get_rules`, `get_active_rules`, `active_weights`, `snapshot_members`, `list_snapshots`, `rule_history`, `table_columns`.
`verify_integrity()`: audits a **linear parent chain**, no dangling members, no foreign-key violations, `PRAGMA integrity_check`, head is the latest snapshot.

### 2.9 Bridge to other phases (lines 560–570)
* `sync_encoder(encoder)`: installs active rules into `KnowledgeBaseTriggerEncoder` (feature block `φ_KB`).
* `firing_rule_ids(event)`: ids of active rules that fire on a transaction → the `active_rules` argument of `PreTransactionRouter.route`.

### 2.10 Lifetime (lines 573–599)
`close`, `__del__`, context-manager support, and the `_KEEP` sentinel.

## 3. Things to remember
* This gives the paper a concrete **audit trail**: any rollback can be traced to its certificate and snapshot.
* The store itself makes no statistical decisions; the gate (`martingale_eprocess.py`) does.
