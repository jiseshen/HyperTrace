"""SQLite-backed hypothesis memory. All writes use short transactions."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def default_database() -> Path:
    override = os.environ.get("HYPERTRACE_DB")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        root = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return root / "hypertrace" / "memory.sqlite3"


def _clean(value: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected text")
    value = " ".join(value.split())
    if not value or len(value) > limit:
        raise ValueError(f"Text must contain 1 to {limit} characters")
    if re.search(r"(?i)(?:sk-[a-z0-9_-]{12,}|gh[pousr]_[a-z0-9]{12,}|AKIA[A-Z0-9]{16}|-----BEGIN .*PRIVATE KEY|(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*\S+)", value):
        raise ValueError("Credential-like text cannot be stored")
    return value


def _label(value: str) -> tuple[str, str]:
    display = _clean(value, 60)
    key = re.sub(r"[^\w]+", " ", display.casefold()).strip()
    if not key:
        raise ValueError("Empty category label")
    return key, display


def _words(value: str) -> set[str]:
    return {x for x in re.findall(r"[\w]+", value.casefold()) if len(x) > 2}


def _score(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "description": row["description"], "weight": round(row["weight"], 3),
            "project_kind": row["project_kind"], "preference_axis": row["preference_axis"],
            "status": row["status"]}


class MemoryStore:
    def __init__(self, path: Path | None = None):
        self.path = path or default_database()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS categories (
                kind TEXT NOT NULL, key TEXT NOT NULL, name TEXT NOT NULL,
                canonical_key TEXT NOT NULL, PRIMARY KEY(kind, key));
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY, description TEXT NOT NULL, detail TEXT NOT NULL,
                project_kind TEXT NOT NULL, preference_axis TEXT NOT NULL,
                weight REAL NOT NULL, status TEXT NOT NULL DEFAULT 'active',
                merged_to TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, excerpt TEXT NOT NULL, project TEXT NOT NULL,
                scoring_mode TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence (
                event_id TEXT NOT NULL, hypothesis_id TEXT NOT NULL,
                likelihood REAL NOT NULL, baseline REAL NOT NULL,
                PRIMARY KEY(event_id, hypothesis_id));
            CREATE TABLE IF NOT EXISTS consolidation_queue (
                id TEXT PRIMARY KEY, left_id TEXT NOT NULL, right_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', lease_until REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS merges (
                id TEXT PRIMARY KEY, target_id TEXT NOT NULL, source_ids TEXT NOT NULL,
                created_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_hypotheses_active ON hypotheses(status, weight DESC);
            CREATE INDEX IF NOT EXISTS idx_hypotheses_categories
                ON hypotheses(project_kind, preference_axis, status, weight DESC);
            CREATE INDEX IF NOT EXISTS idx_hypotheses_merged_to ON hypotheses(merged_to);
            CREATE INDEX IF NOT EXISTS idx_evidence_hypothesis ON evidence(hypothesis_id, event_id);
            CREATE INDEX IF NOT EXISTS idx_queue_status ON consolidation_queue(status, lease_until, created_at);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=10000")
        try:
            yield db
        finally:
            db.close()

    def _category(self, db: sqlite3.Connection, kind: str, value: str) -> str:
        key, display = _label(value)
        row = db.execute("SELECT canonical_key FROM categories WHERE kind=? AND key=?", (kind, key)).fetchone()
        if row:
            canonical = db.execute("SELECT name FROM categories WHERE kind=? AND key=?", (kind, row[0])).fetchone()
            return canonical[0]
        db.execute("INSERT INTO categories VALUES(?,?,?,?)", (kind, key, display, key))
        return display

    def browse(self, project_kinds: list[str] | None = None, preference_axes: list[str] | None = None,
               limit: int = 10) -> dict:
        if not 1 <= limit <= 20:
            raise ValueError("limit must be 1 to 20")
        with self._connect() as db:
            cats = db.execute("""SELECT 'project_kind' AS kind, project_kind AS name, COUNT(*) AS count
                FROM hypotheses WHERE status='active' GROUP BY project_kind
                UNION ALL
                SELECT 'preference_axis' AS kind, preference_axis AS name, COUNT(*) AS count
                FROM hypotheses WHERE status='active' GROUP BY preference_axis
                ORDER BY count DESC,name""").fetchall()
            conditions = ["status='active'"]
            params: list[str | int] = []
            for column, values in (("project_kind", project_kinds), ("preference_axis", preference_axes)):
                if values:
                    canonical = []
                    for value in values[:10]:
                        key, _ = _label(value)
                        row = db.execute("SELECT canonical_key FROM categories WHERE kind=? AND key=?",
                                         (column, key)).fetchone()
                        if row:
                            name = db.execute("SELECT name FROM categories WHERE kind=? AND key=?",
                                              (column, row[0])).fetchone()
                            canonical.append(name[0])
                    if not canonical:
                        return {"categories": [dict(x) for x in cats], "hypotheses": []}
                    conditions.append(f"{column} IN ({','.join('?' for _ in canonical)})")
                    params.extend(canonical)
            rows = db.execute("SELECT * FROM hypotheses WHERE " + " AND ".join(conditions) +
                              " ORDER BY weight DESC, updated_at DESC LIMIT ?", (*params, limit)).fetchall()
            return {"categories": [dict(x) for x in cats], "hypotheses": [_score(x) for x in rows]}

    @staticmethod
    def _lineage(db: sqlite3.Connection, hid: str) -> list[str]:
        # UNION also terminates safely if a damaged database contains a cycle.
        return [r[0] for r in db.execute("""WITH RECURSIVE sources(id) AS (
            SELECT id FROM hypotheses WHERE id=?
            UNION SELECT h.id FROM hypotheses h JOIN sources s ON h.merged_to=s.id
        ) SELECT id FROM sources""", (hid,))]

    def details(self, ids: list[str]) -> list[dict]:
        if not ids or len(ids) > 10:
            raise ValueError("Provide 1 to 10 hypothesis IDs")
        with self._connect() as db:
            db.execute("BEGIN")
            return self._details(db, ids)

    def _details(self, db: sqlite3.Connection, ids: list[str]) -> list[dict]:
        result = []
        for hid in ids:
            row = db.execute("SELECT * FROM hypotheses WHERE id=?", (hid,)).fetchone()
            if not row:
                continue
            sources = self._lineage(db, hid) if row["status"] == "active" else [hid]
            marks = ",".join("?" for _ in sources)
            ev = db.execute(f"""SELECT DISTINCT e.id,e.excerpt,e.project,e.created_at FROM evidence v
                JOIN events e ON e.id=v.event_id WHERE v.hypothesis_id IN ({marks})
                ORDER BY e.created_at DESC,e.id LIMIT 5""", sources).fetchall()
            result.append({**_score(row), "detail": row["detail"],
                           "evidence": [{k: x[k] for k in ("excerpt", "project", "created_at")} for x in ev],
                           "merged_to": row["merged_to"]})
        return result

    def save_feedback(self, event_id: str, excerpt: str, project: str, proposals: list[dict],
                      scoring_mode: str = "subagents") -> dict:
        event_id = _clean(event_id, 120)
        excerpt = _clean(excerpt, 300)
        project = _clean(project, 100)
        if scoring_mode not in {"subagents", "main_agent"}:
            raise ValueError("Invalid scoring mode")
        if not 1 <= len(proposals) <= 5:
            raise ValueError("Provide 1 to 5 hypotheses")
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                if db.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone():
                    return {"duplicate": True, "event_id": event_id}
                staged = []
                seen = set()
                for p in proposals:
                    action = p.get("action")
                    if action not in {"new", "reuse", "revise"}:
                        raise ValueError("Invalid hypothesis action")
                    hid = p.get("id") if action != "new" else "h_" + uuid.uuid4().hex[:16]
                    if hid in seen:
                        raise ValueError("Hypothesis repeated in event")
                    seen.add(hid)
                    old = db.execute("SELECT * FROM hypotheses WHERE id=? AND status='active'", (hid,)).fetchone() if action != "new" else None
                    if action != "new" and not old:
                        raise ValueError(f"Unknown active hypothesis: {hid}")
                    description = _clean(p.get("description", old["description"] if old else ""), 120)
                    detail = _clean(p.get("detail", old["detail"] if old else ""), 500)
                    category = self._category(db, "project_kind", p.get("project_kind", old["project_kind"] if old else ""))
                    axis = self._category(db, "preference_axis", p.get("preference_axis", old["preference_axis"] if old else ""))
                    likelihood, baseline = p.get("likelihood"), p.get("baseline")
                    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 5
                               for x in (likelihood, baseline)):
                        raise ValueError("Scores must be finite numbers in [0, 5]")
                    if action == "new" and likelihood - baseline < 2:
                        raise ValueError("A new hypothesis needs clear evidence beyond its baseline")
                    # Bounded evidence ratios keep an uncalibrated model score from dominating memory.
                    ratio = math.log((likelihood + 0.5) / (baseline + 0.5))
                    delta = max(-0.55, min(0.55, ratio)) * (0.5 if scoring_mode == "main_agent" else 1.0)
                    if action == "new":
                        weight = max(0.15, min(0.45, 0.28 + max(0, delta) * 0.3))
                    else:
                        prior = max(0.01, min(0.99, old["weight"]))
                        weight = 1 / (1 + math.exp(-(math.log(prior / (1 - prior)) + delta)))
                    staged.append((hid, action, description, detail, category, axis, weight, likelihood, baseline))
                db.execute("INSERT INTO events VALUES(?,?,?,?,?)", (event_id, excerpt, project, scoring_mode, now))
                cards = []
                for hid, action, description, detail, category, axis, weight, likelihood, baseline in staged:
                    if action == "new":
                        db.execute("INSERT INTO hypotheses VALUES(?,?,?,?,?,?,?,?,?,?)",
                                   (hid, description, detail, category, axis, weight, "active", None, now, now))
                    else:
                        db.execute("""UPDATE hypotheses SET description=?,detail=?,project_kind=?,preference_axis=?,weight=?,updated_at=?
                            WHERE id=?""", (description, detail, category, axis, weight, now, hid))
                    db.execute("INSERT INTO evidence VALUES(?,?,?,?)", (event_id, hid, likelihood, baseline))
                    cards.append({"id": hid, "description": description, "weight": round(weight, 3)})
                    self._invalidate_reviews(db, hid)
                    self._queue_overlap(db, hid, description, category, axis, now)
                db.execute("COMMIT")
                return {"event_id": event_id, "hypotheses": cards, "consolidation_pending":
                        bool(db.execute("SELECT 1 FROM consolidation_queue WHERE status='pending' LIMIT 1").fetchone())}
            except Exception:
                db.execute("ROLLBACK")
                raise

    @staticmethod
    def _invalidate_reviews(db: sqlite3.Connection, hid: str) -> None:
        db.execute("""UPDATE consolidation_queue SET status='pending',lease_until=0
            WHERE status='leased' AND (left_id=? OR right_id=?)""", (hid, hid))

    def _queue_overlap(self, db, hid, description, category, axis, now):
        words = _words(description)
        if not words:
            return
        others = db.execute("""SELECT id,description FROM hypotheses WHERE status='active'
            AND id<>? AND project_kind=? AND preference_axis=? ORDER BY weight DESC LIMIT 50""",
            (hid, category, axis)).fetchall()
        for other in others:
            other_words = _words(other["description"])
            if other_words and len(words & other_words) / len(words | other_words) >= 0.6:
                pair = sorted((hid, other["id"]))
                qid = "|".join(pair)
                db.execute("INSERT OR IGNORE INTO consolidation_queue(id,left_id,right_id,created_at) VALUES(?,?,?,?)",
                           (qid, pair[0], pair[1], now))

    def next_consolidation(self) -> dict:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            rows = db.execute("""SELECT q.* FROM consolidation_queue q
                JOIN hypotheses a ON a.id=q.left_id JOIN hypotheses b ON b.id=q.right_id
                WHERE (q.status='pending' OR (q.status='leased' AND q.lease_until<?))
                AND a.status='active' AND b.status='active' ORDER BY q.created_at LIMIT 3""", (now,)).fetchall()
            for row in rows:
                db.execute("UPDATE consolidation_queue SET status='leased',lease_until=? WHERE id=?", (now + 600, row["id"]))
            candidates = [{"queue_id": r["id"],
                           "hypotheses": self._details(db, [r["left_id"], r["right_id"]])} for r in rows]
            db.execute("COMMIT")
            return {"candidates": candidates}

    def finish_consolidation(self, queue_id: str, decision: str, description: str = "", detail: str = "") -> dict:
        if decision not in {"merge", "distinct"}:
            raise ValueError("decision must be merge or distinct")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                q = db.execute("SELECT * FROM consolidation_queue WHERE id=? AND status='leased' AND lease_until>=?",
                               (queue_id, time.time())).fetchone()
                if not q:
                    raise ValueError("No active consolidation lease")
                pair = [db.execute("SELECT * FROM hypotheses WHERE id=? AND status='active'", (q[k],)).fetchone()
                        for k in ("left_id", "right_id")]
                if any(x is None for x in pair):
                    raise ValueError("Consolidation source changed")
                if decision == "distinct":
                    db.execute("UPDATE consolidation_queue SET status='dismissed' WHERE id=?", (queue_id,))
                    db.execute("COMMIT")
                    return {"decision": "distinct"}
                description, detail = _clean(description, 120), _clean(detail, 500)
                if pair[0]["project_kind"] != pair[1]["project_kind"] or pair[0]["preference_axis"] != pair[1]["preference_axis"]:
                    raise ValueError("Merge requires matching categories")
                target = "h_" + uuid.uuid4().hex[:16]
                now = time.time()
                weight = max(pair[0]["weight"], pair[1]["weight"])
                db.execute("INSERT INTO hypotheses VALUES(?,?,?,?,?,?,?,?,?,?)",
                           (target, description, detail, pair[0]["project_kind"], pair[0]["preference_axis"],
                            weight, "active", None, now, now))
                db.execute("UPDATE hypotheses SET status='merged',merged_to=? WHERE id IN (?,?)",
                           (target, pair[0]["id"], pair[1]["id"]))
                db.execute("INSERT INTO merges VALUES(?,?,?,?)", (queue_id, target, json.dumps([x["id"] for x in pair]), now))
                db.execute("UPDATE consolidation_queue SET status='merged' WHERE id=?", (queue_id,))
                db.execute("COMMIT")
                return {"decision": "merge", "id": target, "source_ids": [x["id"] for x in pair]}
            except Exception:
                db.execute("ROLLBACK")
                raise

    def manage(self, hid: str, action: str, description: str = "", detail: str = "") -> dict:
        if action not in {"correct", "forget", "undo_merge"}:
            raise ValueError("Invalid action")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT * FROM hypotheses WHERE id=?", (hid,)).fetchone()
                if not row:
                    raise ValueError("Unknown hypothesis")
                if action == "correct":
                    if row["status"] != "active":
                        raise ValueError("Only active hypotheses can be corrected")
                    db.execute("UPDATE hypotheses SET description=?,detail=?,updated_at=? WHERE id=?",
                               (_clean(description, 120), _clean(detail, 500), time.time(), hid))
                elif action == "forget":
                    if row["status"] == "merged":
                        raise ValueError("Forget the active merged hypothesis to remove its inherited evidence")
                    ids = self._lineage(db, hid)
                    for source in ids:
                        db.execute("DELETE FROM evidence WHERE hypothesis_id=?", (source,))
                    db.execute("DELETE FROM events WHERE id NOT IN (SELECT event_id FROM evidence)")
                    db.execute("UPDATE hypotheses SET status='forgotten',detail='',description='[forgotten]',merged_to=NULL WHERE id IN ("+
                               ",".join("?" for _ in ids)+")", ids)
                else:
                    merge = db.execute("SELECT * FROM merges WHERE target_id=?", (hid,)).fetchone()
                    if not merge or row["status"] != "active":
                        raise ValueError("Not an active merge")
                    if db.execute("SELECT 1 FROM evidence WHERE hypothesis_id=?", (hid,)).fetchone():
                        raise ValueError("Cannot undo a merge after new feedback on it")
                    ids = json.loads(merge["source_ids"])
                    db.execute("UPDATE hypotheses SET status='active',merged_to=NULL WHERE id IN (?,?)", ids)
                    db.execute("UPDATE hypotheses SET status='forgotten',detail='',description='[undone]' WHERE id=?", (hid,))
                    db.execute("UPDATE consolidation_queue SET status='dismissed' WHERE id=?", (merge["id"],))
                self._invalidate_reviews(db, hid)
                db.execute("COMMIT")
                return {"id": hid, "action": action, "success": True}
            except Exception:
                db.execute("ROLLBACK")
                raise

    def alias_category(self, kind: str, alias: str, canonical: str) -> dict:
        if kind not in {"project_kind", "preference_axis"}:
            raise ValueError("Invalid category kind")
        alias_key, _ = _label(alias)
        canonical_key, _ = _label(canonical)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                target = db.execute("SELECT name FROM categories WHERE kind=? AND key=? AND canonical_key=key",
                                    (kind, canonical_key)).fetchone()
                if not target:
                    raise ValueError("Canonical category does not exist")
                current = db.execute("SELECT canonical_key FROM categories WHERE kind=? AND key=?", (kind, alias_key)).fetchone()
                if current and current[0] != alias_key:
                    raise ValueError("Alias already points elsewhere")
                alias_name = db.execute("SELECT name FROM categories WHERE kind=? AND key=?", (kind, alias_key)).fetchone()
                db.execute("INSERT OR REPLACE INTO categories VALUES(?,?,?,?)", (kind, alias_key, alias_name[0] if alias_name else alias,
                                                                            canonical_key))
                # Redirect existing aliases as well, keeping every lookup one hop.
                db.execute("UPDATE categories SET canonical_key=? WHERE kind=? AND canonical_key=?",
                           (canonical_key, kind, alias_key))
                if alias_name:
                    affected = db.execute(f"SELECT id FROM hypotheses WHERE {kind}=?", (alias_name[0],)).fetchall()
                    for item in affected:
                        self._invalidate_reviews(db, item[0])
                    db.execute(f"UPDATE hypotheses SET {kind}=? WHERE {kind}=?", (target[0], alias_name[0]))
                db.execute("COMMIT")
                return {"alias": alias, "canonical": target[0]}
            except Exception:
                db.execute("ROLLBACK")
                raise
