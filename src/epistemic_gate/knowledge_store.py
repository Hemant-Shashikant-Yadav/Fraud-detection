"""
src/epistemic_gate/knowledge_store.py
=====================================

Versioned, transactional fraud-knowledge store (SQLite + JSON/JSONB).

Data model
----------
``rule_versions``   append-only rule history with the required schema
                    ``(rule_id, version_id, snapshot_id, rule_logic,
                    opinion_tuple, status, timestamp)``.
``snapshots``       immutable knowledge-base versions ``V_0, V_1, ...`` with
                    parent pointer, operation, reason, logical step and (for
                    restores) the snapshot they were restored from.
``snapshot_members`` the exact ``rule_id -> version_id`` map of every snapshot.
``kb_head``         the current snapshot (single row).
``certificates``    e-value certificates justifying each admission / rollback.
``opinion_log``     append-only Subjective-Logic opinion telemetry.

Every mutating call runs in one ``BEGIN IMMEDIATE`` transaction under a
re-entrant lock and creates exactly one new snapshot, so the knowledge base is
always at a committed, auditable version. Rollback never deletes history: it
creates a *new* snapshot whose membership equals the target snapshot (plus the
quarantine of the offending rule), and moves the head.

JSON payloads are stored as SQLite JSONB blobs when the linked SQLite supports
it (>= 3.45) and as validated JSON text otherwise; they are always read back
through ``json(...)``.

Integration: :meth:`KnowledgeStore.get_active_rules` returns rule objects that
plug into Phase 2's ``KnowledgeBaseTriggerEncoder.set_rules`` (fusion feature
``phi_KB``), and :meth:`KnowledgeStore.firing_rule_ids` produces the
``active_rules`` argument of Phase 3's ``PreTransactionRouter.route``.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.data.perturbation_generator import Condition
from src.epistemic_gate.subjective_logic import SubjectiveOpinion

STATUSES: Tuple[str, ...] = ("quarantined", "active", "rejected", "retired")
RULE_VERSION_COLUMNS: Tuple[str, ...] = ("rule_id", "version_id", "snapshot_id", "rule_logic",
                                         "opinion_tuple", "status", "timestamp")
_SCALAR_OPS = {
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


class KnowledgeStoreError(RuntimeError):
    """Invalid lifecycle transition or unknown snapshot / rule."""


# --------------------------------------------------------------------------- #
# Rule (de)serialisation                                                      #
# --------------------------------------------------------------------------- #
def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def rule_to_logic(rule) -> Dict[str, object]:
    """Executable rule logic as JSON: conjunction of ``feature op value`` predicates."""
    conds = [{"feature": c.feature, "op": c.op, "value": c.value} for c in rule.conditions]
    meta = {}
    for key in ("category", "emitted_at_step", "base_rate", "conflict_target"):
        if hasattr(rule, key) and getattr(rule, key) is not None:
            meta[key] = getattr(rule, key)
    describe = getattr(rule, "describe", None)
    predicate = describe() if callable(describe) else " AND ".join(
        f"{c['feature']} {c['op']} {c['value']}" for c in conds)
    return {"conditions": conds, "predicate": predicate, "meta": meta}


def opinion_to_json(opinion: Optional[SubjectiveOpinion]) -> Optional[Dict[str, float]]:
    return None if opinion is None else opinion.to_dict()


def opinion_from_json(payload) -> Optional[SubjectiveOpinion]:
    if payload is None:
        return None
    return SubjectiveOpinion.from_dict(payload)


@dataclass(frozen=True)
class StoredRule:
    """A rule version materialised from the store; duck-compatible with Phase 2/3.

    ``weight`` (used by ``KnowledgeBaseTriggerEncoder``) is the Subjective-Logic
    expected probability of the latest opinion, or ``1.0`` without an opinion.
    """

    rule_id: str
    version_id: int
    snapshot_id: int
    status: str
    conditions: Tuple[Condition, ...]
    rule_logic: Dict[str, object]
    opinion: Optional[SubjectiveOpinion]
    timestamp: str

    @property
    def weight(self) -> float:
        return 1.0 if self.opinion is None else float(self.opinion.expected_probability)

    def describe(self) -> str:
        return str(self.rule_logic.get("predicate", ""))

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(frame), dtype=bool)
        for cond in self.conditions:
            mask &= cond.evaluate(frame)
        return mask

    def fires(self, event: Mapping[str, object]) -> bool:
        for c in self.conditions:
            raw = event[c.feature]
            op = _SCALAR_OPS[c.op]
            ok = op(str(raw), c.value) if isinstance(c.value, str) else op(float(raw), float(c.value))
            if not ok:
                return False
        return True


def _conditions_from_logic(logic: Mapping[str, object]) -> Tuple[Condition, ...]:
    out = []
    for c in logic.get("conditions", []):
        value = c["value"]
        out.append(Condition(str(c["feature"]), str(c["op"]),
                             value if isinstance(value, str) else float(value)))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Store                                                                       #
# --------------------------------------------------------------------------- #
class KnowledgeStore:
    """Thread-safe versioned knowledge base with snapshot / rollback API."""

    def __init__(self, path: Union[str, Path] = ":memory:",
                 clock: Optional[Callable[[], str]] = None) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat(timespec="microseconds"))
        self._fault_injector: Optional[Callable[[str], None]] = None
        self._conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None,
                                     check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 30000")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = FULL")
        self.use_jsonb = self._detect_jsonb()
        self._init_schema()

    # ------------------------------------------------------------- plumbing
    def _detect_jsonb(self) -> bool:
        try:
            self._conn.execute("SELECT json(jsonb('{\"a\":1}'))").fetchone()
            return True
        except sqlite3.OperationalError:
            return False

    @property
    def storage_format(self) -> str:
        return "jsonb" if self.use_jsonb else "json"

    def _enc(self) -> str:
        """SQL expression wrapping a bound JSON-text parameter for storage."""
        return "jsonb(?)" if self.use_jsonb else "json(?)"

    @staticmethod
    def _dumps(obj) -> Optional[str]:
        return None if obj is None else json.dumps(obj, sort_keys=True, default=_json_default)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                yield cur
                if self._fault_injector is not None:
                    self._fault_injector("before_commit")
                cur.execute("COMMIT")
            except BaseException:
                cur.execute("ROLLBACK")
                raise
            finally:
                cur.close()

    def _init_schema(self) -> None:
        json_check = "json_valid(json({col}))"
        with self._transaction() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id         INTEGER PRIMARY KEY,
                    parent_snapshot_id  INTEGER REFERENCES snapshots(snapshot_id),
                    operation           TEXT NOT NULL,
                    reason              TEXT NOT NULL DEFAULT '',
                    logical_step        INTEGER,
                    restored_from       INTEGER REFERENCES snapshots(snapshot_id),
                    created_at          TEXT NOT NULL
                )""")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS rule_versions (
                    rule_id        TEXT    NOT NULL,
                    version_id     INTEGER NOT NULL,
                    snapshot_id    INTEGER NOT NULL REFERENCES snapshots(snapshot_id),
                    rule_logic     BLOB    NOT NULL CHECK ({json_check.format(col='rule_logic')}),
                    opinion_tuple  BLOB    CHECK (opinion_tuple IS NULL OR
                                                  {json_check.format(col='opinion_tuple')}),
                    status         TEXT    NOT NULL CHECK (status IN ({", ".join(repr(s) for s in STATUSES)})),
                    timestamp      TEXT    NOT NULL,
                    PRIMARY KEY (rule_id, version_id)
                )""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS snapshot_members (
                    snapshot_id  INTEGER NOT NULL REFERENCES snapshots(snapshot_id),
                    rule_id      TEXT    NOT NULL,
                    version_id   INTEGER NOT NULL,
                    PRIMARY KEY (snapshot_id, rule_id),
                    FOREIGN KEY (rule_id, version_id) REFERENCES rule_versions(rule_id, version_id)
                )""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kb_head (
                    id           INTEGER PRIMARY KEY CHECK (id = 1),
                    snapshot_id  INTEGER NOT NULL REFERENCES snapshots(snapshot_id)
                )""")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS certificates (
                    cert_id      INTEGER PRIMARY KEY,
                    rule_id      TEXT NOT NULL,
                    snapshot_id  INTEGER NOT NULL REFERENCES snapshots(snapshot_id),
                    test         TEXT NOT NULL CHECK (test IN ('admission', 'rollback')),
                    payload      BLOB NOT NULL CHECK ({json_check.format(col='payload')}),
                    logical_step INTEGER,
                    created_at   TEXT NOT NULL
                )""")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS opinion_log (
                    log_id        INTEGER PRIMARY KEY,
                    rule_id       TEXT NOT NULL,
                    logical_step  INTEGER,
                    opinion_tuple BLOB NOT NULL CHECK ({json_check.format(col='opinion_tuple')}),
                    created_at    TEXT NOT NULL
                )""")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_rv_status ON rule_versions(rule_id, status)")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_members_rule ON snapshot_members(rule_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_oplog_rule ON opinion_log(rule_id, log_id)")
            if cur.execute("SELECT COUNT(*) FROM kb_head").fetchone()[0] == 0:
                cur.execute("INSERT INTO snapshots(snapshot_id, parent_snapshot_id, operation, reason, "
                            "logical_step, created_at) VALUES (0, NULL, 'genesis', 'empty knowledge base', 0, ?)",
                            (self._clock(),))
                cur.execute("INSERT INTO kb_head(id, snapshot_id) VALUES (1, 0)")

    # ------------------------------------------------------------- internals
    def _head(self, cur: sqlite3.Cursor) -> int:
        return int(cur.execute("SELECT snapshot_id FROM kb_head WHERE id = 1").fetchone()[0])

    def _members(self, cur: sqlite3.Cursor, snapshot_id: int) -> Dict[str, int]:
        rows = cur.execute("SELECT rule_id, version_id FROM snapshot_members WHERE snapshot_id = ?",
                           (snapshot_id,)).fetchall()
        return {str(r): int(v) for r, v in rows}

    def _snapshot_exists(self, cur: sqlite3.Cursor, snapshot_id: int) -> bool:
        return cur.execute("SELECT 1 FROM snapshots WHERE snapshot_id = ?",
                           (int(snapshot_id),)).fetchone() is not None

    def _version_row(self, cur: sqlite3.Cursor, rule_id: str, version_id: int):
        return cur.execute(
            "SELECT rule_id, version_id, snapshot_id, json(rule_logic), json(opinion_tuple), status, timestamp "
            "FROM rule_versions WHERE rule_id = ? AND version_id = ?", (rule_id, version_id)).fetchone()

    def _current(self, cur: sqlite3.Cursor, rule_id: str, snapshot_id: Optional[int] = None):
        sid = self._head(cur) if snapshot_id is None else int(snapshot_id)
        row = cur.execute("SELECT version_id FROM snapshot_members WHERE snapshot_id = ? AND rule_id = ?",
                          (sid, rule_id)).fetchone()
        return None if row is None else self._version_row(cur, rule_id, int(row[0]))

    def _open_snapshot(self, cur: sqlite3.Cursor, operation: str, reason: str, step: Optional[int],
                       restored_from: Optional[int] = None) -> Tuple[int, int]:
        parent = self._head(cur)
        new_id = int(cur.execute("SELECT COALESCE(MAX(snapshot_id), -1) + 1 FROM snapshots").fetchone()[0])
        cur.execute("INSERT INTO snapshots(snapshot_id, parent_snapshot_id, operation, reason, logical_step, "
                    "restored_from, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (new_id, parent, operation, reason, step, restored_from, self._clock()))
        return new_id, parent

    def _write_members(self, cur: sqlite3.Cursor, snapshot_id: int, members: Mapping[str, int]) -> None:
        cur.executemany("INSERT INTO snapshot_members(snapshot_id, rule_id, version_id) VALUES (?, ?, ?)",
                        [(snapshot_id, rid, int(v)) for rid, v in sorted(members.items())])

    def _insert_version(self, cur: sqlite3.Cursor, rule_id: str, snapshot_id: int,
                        logic: Mapping[str, object], opinion_json: Optional[str], status: str) -> int:
        if status not in STATUSES:
            raise KnowledgeStoreError(f"unknown status '{status}'")
        version = int(cur.execute("SELECT COALESCE(MAX(version_id), 0) + 1 FROM rule_versions WHERE rule_id = ?",
                                  (rule_id,)).fetchone()[0])
        op_expr = self._enc() if opinion_json is not None else "?"
        cur.execute(f"INSERT INTO rule_versions(rule_id, version_id, snapshot_id, rule_logic, opinion_tuple, "
                    f"status, timestamp) VALUES (?, ?, ?, {self._enc()}, {op_expr}, ?, ?)",
                    (rule_id, version, snapshot_id, self._dumps(logic), opinion_json, status, self._clock()))
        return version

    def _transition(self, operation: str, changes: Sequence[Tuple[str, Optional[Mapping], object, str]],
                    reason: str, step: Optional[int], base_members: Optional[Mapping[str, int]] = None,
                    restored_from: Optional[int] = None) -> int:
        """Atomically create a snapshot applying ``(rule_id, logic|None, opinion|_KEEP, status)`` changes."""
        with self._transaction() as cur:
            sid, parent = self._open_snapshot(cur, operation, reason, step, restored_from)
            members = dict(self._members(cur, parent) if base_members is None else base_members)
            for rule_id, logic, opinion, status in changes:
                rule_id = str(rule_id)
                prev = None
                if rule_id in members:
                    prev = self._version_row(cur, rule_id, members[rule_id])
                if prev is None:
                    latest = cur.execute("SELECT MAX(version_id) FROM rule_versions WHERE rule_id = ?",
                                         (rule_id,)).fetchone()[0]
                    if latest is not None:
                        prev = self._version_row(cur, rule_id, int(latest))
                if logic is None:
                    if prev is None:
                        raise KnowledgeStoreError(f"rule '{rule_id}' has no stored logic")
                    logic = json.loads(prev[3])
                if opinion is _KEEP:
                    opinion_json = prev[4] if prev is not None else None
                else:
                    opinion_json = self._dumps(opinion_to_json(opinion))
                members[rule_id] = self._insert_version(cur, rule_id, sid, logic, opinion_json, status)
                if self._fault_injector is not None:
                    self._fault_injector("after_version")
            self._write_members(cur, sid, members)
            cur.execute("UPDATE kb_head SET snapshot_id = ? WHERE id = 1", (sid,))
            return sid

    # ------------------------------------------------------------ lifecycle
    @property
    def head_snapshot_id(self) -> int:
        with self._lock:
            return self._head(self._conn.cursor())

    def status_of(self, rule_id: str, snapshot_id: Optional[int] = None) -> Optional[str]:
        with self._lock:
            row = self._current(self._conn.cursor(), str(rule_id), snapshot_id)
            return None if row is None else str(row[5])

    def quarantine_candidate(self, rule_id: str, rule_logic: Optional[Mapping] = None,
                             opinion: Optional[SubjectiveOpinion] = None, reason: str = "",
                             step: Optional[int] = None) -> int:
        """Place a candidate (new, or demoted from active) into the quarantine buffer."""
        rule_id = str(rule_id)
        with self._lock:
            if rule_logic is None and not self.rule_history(rule_id):
                raise KnowledgeStoreError(f"new candidate '{rule_id}' requires rule_logic")
            return self._transition("quarantine", [(rule_id, rule_logic,
                                                    opinion if opinion is not None else _KEEP,
                                                    "quarantined")], reason, step)

    def admit_candidate(self, rule_id: str, opinion: Optional[SubjectiveOpinion] = None,
                        reason: str = "", step: Optional[int] = None) -> int:
        """Promote a quarantined candidate to the active KB (new snapshot ``V_{j+1}``)."""
        rule_id = str(rule_id)
        with self._lock:
            status = self.status_of(rule_id)
            if status != "quarantined":
                raise KnowledgeStoreError(f"cannot admit '{rule_id}' from status {status!r}")
            return self._transition("admit", [(rule_id, None, opinion if opinion is not None else _KEEP,
                                               "active")], reason, step)

    def reject_candidate(self, rule_id: str, rule_logic: Optional[Mapping] = None,
                         opinion: Optional[SubjectiveOpinion] = None, reason: str = "",
                         step: Optional[int] = None) -> int:
        rule_id = str(rule_id)
        with self._lock:
            if rule_logic is None and not self.rule_history(rule_id):
                raise KnowledgeStoreError(f"unknown rule '{rule_id}'")
            return self._transition("reject", [(rule_id, rule_logic,
                                                opinion if opinion is not None else _KEEP, "rejected")],
                                    reason, step)

    def retire_rule(self, rule_id: str, reason: str = "", step: Optional[int] = None) -> int:
        with self._lock:
            if self.status_of(rule_id) is None:
                raise KnowledgeStoreError(f"rule '{rule_id}' is not in the head snapshot")
            return self._transition("retire", [(str(rule_id), None, _KEEP, "retired")], reason, step)

    def revert_rule(self, rule_id: str, reason: str = "", step: Optional[int] = None,
                    restored_from: Optional[int] = None) -> int:
        """Targeted rollback: restore ``rule_id`` to quarantine, all other rules unchanged."""
        rule_id = str(rule_id)
        with self._lock:
            if self.status_of(rule_id) != "active":
                raise KnowledgeStoreError(f"cannot revert '{rule_id}': not active")
            return self._transition("rollback_rule", [(rule_id, None, _KEEP, "quarantined")], reason, step,
                                    restored_from=restored_from)

    def rollback_to_snapshot(self, snapshot_id: int, quarantine_rule_ids: Iterable[str] = (),
                             reason: str = "", step: Optional[int] = None,
                             preserve_non_active: bool = False) -> int:
        """Atomic restore of the KB to snapshot ``V_j``; listed rules are quarantined.

        History is preserved: a new snapshot with ``V_j``'s exact membership is
        appended and becomes the head. Rules added after ``V_j`` disappear from
        the head (they remain in ``rule_versions`` and older snapshots).

        ``preserve_non_active=True`` keeps rules that entered the store after
        ``V_j`` and are *not* active (quarantined / rejected candidates) in the
        restored head, so pending e-processes are not orphaned. Rules active in
        the head but not in ``V_j`` should then be listed in
        ``quarantine_rule_ids``; the active set of the new head equals ``V_j``'s
        minus the quarantined rules.
        """
        snapshot_id = int(snapshot_id)
        quarantine_rule_ids = [str(r) for r in quarantine_rule_ids]
        with self._lock:
            cur = self._conn.cursor()
            if not self._snapshot_exists(cur, snapshot_id):
                raise KnowledgeStoreError(f"unknown snapshot {snapshot_id}")
            target = self._members(cur, snapshot_id)
            if preserve_non_active:
                for rid, ver in self._members(cur, self._head(cur)).items():
                    if rid not in target and self._version_row(cur, rid, ver)[5] != "active":
                        target[rid] = ver
            changes = [(rid, None, _KEEP, "quarantined") for rid in quarantine_rule_ids]
            return self._transition("rollback", changes, reason, step, base_members=target,
                                    restored_from=snapshot_id)

    def pre_admission_snapshot(self, rule_id: str) -> Optional[int]:
        """``V_j``: the parent of the snapshot that last admitted ``rule_id``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT s.parent_snapshot_id FROM rule_versions v JOIN snapshots s ON s.snapshot_id = v.snapshot_id "
                "WHERE v.rule_id = ? AND v.status = 'active' AND s.operation = 'admit' "
                "ORDER BY v.version_id DESC LIMIT 1", (str(rule_id),)).fetchone()
            return None if row is None else int(row[0])

    # ---------------------------------------------------------- telemetry
    def record_certificate(self, rule_id: str, snapshot_id: int, test: str, payload: Mapping,
                           step: Optional[int] = None) -> int:
        with self._transaction() as cur:
            cur.execute(f"INSERT INTO certificates(rule_id, snapshot_id, test, payload, logical_step, created_at) "
                        f"VALUES (?, ?, ?, {self._enc()}, ?, ?)",
                        (str(rule_id), int(snapshot_id), test, self._dumps(dict(payload)), step, self._clock()))
            return int(cur.lastrowid)

    def certificates(self, rule_id: Optional[str] = None) -> List[Dict[str, object]]:
        sql = "SELECT cert_id, rule_id, snapshot_id, test, json(payload), logical_step FROM certificates"
        args: Tuple = ()
        if rule_id is not None:
            sql += " WHERE rule_id = ?"
            args = (str(rule_id),)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY cert_id", args).fetchall()
        return [{"cert_id": r[0], "rule_id": r[1], "snapshot_id": r[2], "test": r[3],
                 "payload": json.loads(r[4]), "logical_step": r[5]} for r in rows]

    def record_opinions(self, opinions: Mapping[str, SubjectiveOpinion], step: Optional[int] = None) -> None:
        with self._transaction() as cur:
            cur.executemany(f"INSERT INTO opinion_log(rule_id, logical_step, opinion_tuple, created_at) "
                            f"VALUES (?, ?, {self._enc()}, ?)",
                            [(str(rid), step, self._dumps(op.to_dict()), self._clock())
                             for rid, op in sorted(opinions.items())])

    def latest_opinion(self, rule_id: str) -> Optional[SubjectiveOpinion]:
        with self._lock:
            row = self._conn.execute("SELECT json(opinion_tuple) FROM opinion_log WHERE rule_id = ? "
                                     "ORDER BY log_id DESC LIMIT 1", (str(rule_id),)).fetchone()
        return None if row is None else opinion_from_json(json.loads(row[0]))

    # -------------------------------------------------------------- queries
    def _materialise(self, row, use_latest_opinion: bool = True) -> StoredRule:
        logic = json.loads(row[3])
        opinion = opinion_from_json(json.loads(row[4])) if row[4] is not None else None
        if use_latest_opinion:
            opinion = self.latest_opinion(row[0]) or opinion
        return StoredRule(str(row[0]), int(row[1]), int(row[2]), str(row[5]),
                          _conditions_from_logic(logic), logic, opinion, str(row[6]))

    def get_rules(self, status: Optional[str] = None, snapshot_id: Optional[int] = None) -> List[StoredRule]:
        with self._lock:
            cur = self._conn.cursor()
            sid = self._head(cur) if snapshot_id is None else int(snapshot_id)
            if not self._snapshot_exists(cur, sid):
                raise KnowledgeStoreError(f"unknown snapshot {sid}")
            rows = cur.execute(
                "SELECT v.rule_id, v.version_id, v.snapshot_id, json(v.rule_logic), json(v.opinion_tuple), "
                "v.status, v.timestamp FROM snapshot_members m JOIN rule_versions v "
                "ON v.rule_id = m.rule_id AND v.version_id = m.version_id WHERE m.snapshot_id = ? "
                + ("AND v.status = ? " if status else "") + "ORDER BY v.rule_id",
                (sid, status) if status else (sid,)).fetchall()
            return [self._materialise(r) for r in rows]

    def get_active_rules(self, snapshot_id: Optional[int] = None) -> List[StoredRule]:
        """Active rules of the head (or given) snapshot, sorted by ``rule_id``."""
        return self.get_rules("active", snapshot_id)

    def active_weights(self, snapshot_id: Optional[int] = None) -> Dict[str, float]:
        return {r.rule_id: r.weight for r in self.get_active_rules(snapshot_id)}

    def snapshot_members(self, snapshot_id: Optional[int] = None) -> Dict[str, Tuple[int, str]]:
        return {r.rule_id: (r.version_id, r.status) for r in self.get_rules(None, snapshot_id)}

    def list_snapshots(self) -> List[Dict[str, object]]:
        with self._lock:
            rows = self._conn.execute("SELECT snapshot_id, parent_snapshot_id, operation, reason, logical_step, "
                                      "restored_from, created_at FROM snapshots ORDER BY snapshot_id").fetchall()
        keys = ("snapshot_id", "parent_snapshot_id", "operation", "reason", "logical_step",
                "restored_from", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    def rule_history(self, rule_id: str) -> List[Dict[str, object]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT rule_id, version_id, snapshot_id, json(rule_logic), json(opinion_tuple), status, timestamp "
                "FROM rule_versions WHERE rule_id = ? ORDER BY version_id", (str(rule_id),)).fetchall()
        return [{"rule_id": r[0], "version_id": r[1], "snapshot_id": r[2], "rule_logic": json.loads(r[3]),
                 "opinion_tuple": json.loads(r[4]) if r[4] is not None else None,
                 "status": r[5], "timestamp": r[6]} for r in rows]

    def table_columns(self, table: str = "rule_versions") -> Tuple[str, ...]:
        with self._lock:
            return tuple(r[1] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall())

    def verify_integrity(self) -> Dict[str, object]:
        """Structural audit: linear parent chain, members reference stored versions."""
        with self._lock:
            cur = self._conn.cursor()
            snaps = cur.execute("SELECT snapshot_id, parent_snapshot_id FROM snapshots ORDER BY snapshot_id").fetchall()
            chain_ok = all((p is None and s == 0) or (p == s - 1) for s, p in snaps)
            dangling = cur.execute(
                "SELECT COUNT(*) FROM snapshot_members m LEFT JOIN rule_versions v "
                "ON v.rule_id = m.rule_id AND v.version_id = m.version_id WHERE v.rule_id IS NULL").fetchone()[0]
            fk = cur.execute("PRAGMA foreign_key_check").fetchall()
            sqlite_ok = cur.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            head = self._head(cur)
        return {"n_snapshots": len(snaps), "head": head, "chain_ok": bool(chain_ok),
                "head_is_latest": head == snaps[-1][0], "dangling_members": int(dangling),
                "foreign_key_violations": len(fk), "sqlite_integrity_ok": bool(sqlite_ok),
                "ok": bool(chain_ok and dangling == 0 and not fk and sqlite_ok and head == snaps[-1][0])}

    # ----------------------------------------------------- Phase 2/3 bridge
    def sync_encoder(self, encoder, snapshot_id: Optional[int] = None) -> int:
        """Install the active snapshot into a ``KnowledgeBaseTriggerEncoder`` (``phi_KB``)."""
        rules = self.get_active_rules(snapshot_id)
        encoder.set_rules(rules, {r.rule_id: r.weight for r in rules})
        return len(rules)

    def firing_rule_ids(self, event: Mapping[str, object],
                        rules: Optional[Sequence[StoredRule]] = None) -> Tuple[str, ...]:
        """Admitted rules firing on ``event`` -> ``active_rules`` for ``PreTransactionRouter.route``."""
        rules = self.get_active_rules() if rules is None else rules
        return tuple(r.rule_id for r in rules if r.fires(event))

    # ------------------------------------------------------------- lifetime
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __del__(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                return

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _Keep:
    """Sentinel: keep the opinion of the previous version."""

    def __repr__(self) -> str:
        return "<keep>"


_KEEP = _Keep()
