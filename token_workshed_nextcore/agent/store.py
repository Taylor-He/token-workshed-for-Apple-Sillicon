"""Local, versioned state for the NextCore Agent."""

from __future__ import annotations

import json
import ipaddress
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?key|client[_-]?secret|password|passwd|"
        r"credential|private[_-]?key|bearer|token)\s*[:=]\s*\S+"
    ),
    re.compile(r"\b(?:sk|hf|ghp|github_pat)[_-][A-Za-z0-9_-]{16,}\b", re.I),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/@:\s]+:[^/@\s]+@"),
    re.compile(r"(?i)\bBearer\s+\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
)
_TRANSFER_CATEGORIES = frozenset({"memories", "skills", "chat_history"})
_MAX_TRANSFER_BYTES = 16 * 1024 * 1024


def _redact_transfer_text(value: str) -> str:
    text = str(value or "")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted credential]", text)
    return text


def _transfer_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def contains_secret(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def _plain_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_plain_text(item) for item in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), (str, list)):
            return _plain_text(value["content"])
        return ""
    return ""


class AgentStateStore:
    """SQLite store for memories, Skills, local chat search and scheduled runs."""

    def __init__(self, app_data_dir: Path):
        self.root = Path(app_data_dir).expanduser() / "agent"
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass
        self.path = self.root / "agent-state.sqlite3"
        self._schema_lock = threading.Lock()
        self._ensure_schema()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=8.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=8000")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._schema_lock, self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    revision INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    deleted_at REAL
                );
                CREATE TABLE IF NOT EXISTS memory_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(memory_id, revision),
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    slug TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    deleted_at REAL
                );
                CREATE TABLE IF NOT EXISTS skill_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    body TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(skill_id, revision),
                    FOREIGN KEY(skill_id) REFERENCES skills(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    conversation_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    PRIMARY KEY(conversation_id, message_index),
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS conversation_search USING fts5(
                    conversation_id UNINDEXED,
                    message_index UNINDEXED,
                    role UNINDEXED,
                    title,
                    content,
                    tokenize='unicode61 remove_diacritics 2'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(
                    memory_id UNINDEXED,
                    title,
                    body,
                    tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    timezone TEXT NOT NULL DEFAULT 'UTC',
                    model_id TEXT NOT NULL,
                    tool_scope_json TEXT NOT NULL DEFAULT '[]',
                    authorization_scope_json TEXT NOT NULL DEFAULT '{}',
                    server_url TEXT NOT NULL DEFAULT 'http://127.0.0.1:30000',
                    tool_parser TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    state TEXT NOT NULL DEFAULT 'scheduled',
                    next_run_at REAL,
                    last_run_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scheduled_job_runs (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL,
                    summary TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(job_id) REFERENCES scheduled_jobs(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS mcp_servers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    transport TEXT NOT NULL,
                    command TEXT NOT NULL DEFAULT '',
                    args_json TEXT NOT NULL DEFAULT '[]',
                    url TEXT NOT NULL DEFAULT '',
                    env_names_json TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    timeout REAL NOT NULL DEFAULT 30,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS browser_profile_grants (
                    id TEXT PRIMARY KEY,
                    pid INTEGER NOT NULL,
                    window_id INTEGER NOT NULL,
                    app_name TEXT NOT NULL,
                    bundle_id TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    UNIQUE(pid, window_id)
                );
                CREATE INDEX IF NOT EXISTS idx_job_due
                    ON scheduled_jobs(enabled, next_run_at);
                CREATE INDEX IF NOT EXISTS idx_job_runs
                    ON scheduled_job_runs(job_id, started_at DESC);
                """
            )
            existing_columns = {row["name"] for row in db.execute("PRAGMA table_info(scheduled_jobs)").fetchall()}
            for name, definition in (
                ("authorization_scope_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("server_url", "TEXT NOT NULL DEFAULT 'http://127.0.0.1:30000'"),
                ("tool_parser", "TEXT NOT NULL DEFAULT ''"),
            ):
                if name not in existing_columns:
                    db.execute(f"ALTER TABLE scheduled_jobs ADD COLUMN {name} {definition}")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    @staticmethod
    def _revision_number(db: sqlite3.Connection, table: str, key: str, value: str) -> int:
        row = db.execute(
            f"SELECT COALESCE(MAX(revision), 0) AS revision FROM {table} WHERE {key}=?",
            (value,),
        ).fetchone()
        return int(row["revision"] or 0) + 1

    def save_memory(
        self,
        title: str,
        body: str,
        tags: list[str] | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        title = " ".join(str(title).split())[:180]
        body = str(body).strip()[:12_000]
        if not title or not body:
            raise ValueError("Memory title and content are required.")
        if contains_secret(title + "\n" + body):
            raise ValueError("This content looks like a credential and was not saved.")
        safe_tags = sorted({" ".join(str(tag).split())[:60] for tag in (tags or []) if str(tag).strip()})[:20]
        now = time.time()
        with self._connect() as db:
            if memory_id:
                row = db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            else:
                row = db.execute(
                    "SELECT * FROM memories WHERE lower(title)=lower(?) AND deleted_at IS NULL",
                    (title,),
                ).fetchone()
            if row:
                memory_id = row["id"]
                revision = int(row["revision"]) + 1
                db.execute(
                    "UPDATE memories SET title=?, body=?, tags_json=?, revision=?, updated_at=?, deleted_at=NULL WHERE id=?",
                    (title, body, json.dumps(safe_tags), revision, now, memory_id),
                )
                db.execute("DELETE FROM memory_search WHERE memory_id=?", (memory_id,))
                operation = "update"
            else:
                memory_id = memory_id or uuid.uuid4().hex
                revision = 1
                db.execute(
                    "INSERT INTO memories(id,title,body,tags_json,revision,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                    (memory_id, title, body, json.dumps(safe_tags), revision, now, now),
                )
                operation = "create"
            db.execute(
                "INSERT INTO memory_revisions(memory_id,revision,title,body,tags_json,operation,created_at) VALUES(?,?,?,?,?,?,?)",
                (memory_id, revision, title, body, json.dumps(safe_tags), operation, now),
            )
            db.execute(
                "INSERT INTO memory_search(memory_id,title,body) VALUES(?,?,?)",
                (memory_id, title, body),
            )
            saved = db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return dict(saved)

    def list_memories(self, query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 250))
        with self._connect() as db:
            if query.strip():
                terms = _fts_query(query)
                rows = db.execute(
                    "SELECT m.* FROM memory_search f JOIN memories m ON m.id=f.memory_id "
                    "WHERE memory_search MATCH ? AND m.deleted_at IS NULL ORDER BY rank LIMIT ?",
                    (terms, limit),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM memories WHERE deleted_at IS NULL ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return self._rows(rows)

    def memory_revisions(self, memory_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT revision,title,body,tags_json,operation,created_at FROM memory_revisions WHERE memory_id=? ORDER BY revision DESC",
                (memory_id,),
            ).fetchall()
        return self._rows(rows)

    def rollback_memory(self, memory_id: str, revision: int) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT title,body,tags_json FROM memory_revisions WHERE memory_id=? AND revision=?",
                (memory_id, int(revision)),
            ).fetchone()
        if not row:
            raise KeyError("Memory revision not found.")
        return self.save_memory(
            row["title"], row["body"], json.loads(row["tags_json"]), memory_id=memory_id
        )

    def delete_memory(self, memory_id: str) -> None:
        now = time.time()
        with self._connect() as db:
            row = db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row:
                return
            revision = int(row["revision"]) + 1
            db.execute(
                "INSERT INTO memory_revisions(memory_id,revision,title,body,tags_json,operation,created_at) VALUES(?,?,?,?,?,'delete',?)",
                (memory_id, revision, row["title"], row["body"], row["tags_json"], now),
            )
            db.execute("UPDATE memories SET revision=?,updated_at=?,deleted_at=? WHERE id=?", (revision, now, now, memory_id))
            db.execute("DELETE FROM memory_search WHERE memory_id=?", (memory_id,))

    def save_skill(
        self,
        name: str,
        description: str,
        body: str,
        skill_id: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        name = " ".join(str(name).split())[:120]
        description = " ".join(str(description).split())[:500]
        body = str(body).strip()[:40_000]
        if not name or not body:
            raise ValueError("Skill name and Markdown body are required.")
        if contains_secret(name + "\n" + description + "\n" + body):
            raise ValueError("This Skill looks like it contains a credential and was not saved.")
        slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:72] or uuid.uuid4().hex[:12]
        now = time.time()
        with self._connect() as db:
            row = db.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone() if skill_id else db.execute("SELECT * FROM skills WHERE slug=?", (slug,)).fetchone()
            if row:
                skill_id = row["id"]
                revision = int(row["revision"]) + 1
                db.execute(
                    "UPDATE skills SET slug=?,name=?,description=?,body=?,revision=?,enabled=?,updated_at=?,deleted_at=NULL WHERE id=?",
                    (slug, name, description, body, revision, int(enabled), now, skill_id),
                )
                operation = "update"
            else:
                skill_id = skill_id or uuid.uuid4().hex
                revision = 1
                db.execute(
                    "INSERT INTO skills(id,slug,name,description,body,revision,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (skill_id, slug, name, description, body, revision, int(enabled), now, now),
                )
                operation = "create"
            db.execute(
                "INSERT INTO skill_revisions(skill_id,revision,name,description,body,operation,created_at) VALUES(?,?,?,?,?,?,?)",
                (skill_id, revision, name, description, body, operation, now),
            )
            saved = db.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()
        return dict(saved)

    def list_skills(self, query: str = "", include_disabled: bool = True) -> list[dict[str, Any]]:
        clause = "deleted_at IS NULL"
        args: list[Any] = []
        if not include_disabled:
            clause += " AND enabled=1"
        if query.strip():
            clause += " AND (name LIKE ? OR description LIKE ? OR body LIKE ?)"
            needle = f"%{query.strip()[:120]}%"
            args.extend([needle, needle, needle])
        with self._connect() as db:
            rows = db.execute(
                f"SELECT id,slug,name,description,body,revision,enabled,created_at,updated_at FROM skills WHERE {clause} ORDER BY name LIMIT 200",
                args,
            ).fetchall()
        return self._rows(rows)

    def skill_revisions(self, skill_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT revision,name,description,body,operation,created_at FROM skill_revisions WHERE skill_id=? ORDER BY revision DESC",
                (skill_id,),
            ).fetchall()
        return self._rows(rows)

    def rollback_skill(self, skill_id: str, revision: int) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT name,description,body FROM skill_revisions WHERE skill_id=? AND revision=?",
                (skill_id, int(revision)),
            ).fetchone()
            current = db.execute("SELECT enabled FROM skills WHERE id=?", (skill_id,)).fetchone()
        if not row:
            raise KeyError("Skill revision not found.")
        return self.save_skill(
            row["name"], row["description"], row["body"], skill_id=skill_id,
            enabled=bool(current["enabled"]) if current else True,
        )

    def set_skill_enabled(self, skill_id: str, enabled: bool) -> dict[str, Any]:
        now = time.time()
        with self._connect() as db:
            row = db.execute("SELECT * FROM skills WHERE id=? AND deleted_at IS NULL", (skill_id,)).fetchone()
            if not row:
                raise KeyError("Skill not found.")
            revision = int(row["revision"]) + 1
            db.execute("UPDATE skills SET enabled=?,revision=?,updated_at=? WHERE id=?", (int(enabled), revision, now, skill_id))
            db.execute(
                "INSERT INTO skill_revisions(skill_id,revision,name,description,body,operation,created_at) VALUES(?,?,?,?,?,?,?)",
                (skill_id, revision, row["name"], row["description"], row["body"], "enable" if enabled else "disable", now),
            )
            saved = db.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()
        return dict(saved)

    def delete_skill(self, skill_id: str) -> None:
        now = time.time()
        with self._connect() as db:
            row = db.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()
            if not row:
                return
            revision = int(row["revision"]) + 1
            db.execute(
                "INSERT INTO skill_revisions(skill_id,revision,name,description,body,operation,created_at) VALUES(?,?,?,?,?,'delete',?)",
                (skill_id, revision, row["name"], row["description"], row["body"], now),
            )
            db.execute("UPDATE skills SET revision=?,updated_at=?,deleted_at=?,enabled=0 WHERE id=?", (revision, now, now, skill_id))

    def index_conversation(
        self, conversation_id: str, messages: list[dict[str, Any]], title: str = ""
    ) -> None:
        conversation_id = str(conversation_id or "").strip()[:120]
        if not conversation_id:
            return
        now = time.time()
        safe_messages: list[tuple[int, str, str]] = []
        for index, message in enumerate(messages[-500:]):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")[:20]
            content = _plain_text(message.get("content")).strip()[:120_000]
            if role not in {"user", "assistant", "tool"} or not content:
                continue
            safe_messages.append((index, role, content))
        with self._connect() as db:
            db.execute(
                "INSERT INTO conversations(id,title,updated_at) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,updated_at=excluded.updated_at",
                (conversation_id, str(title or "")[:240], now),
            )
            db.execute("DELETE FROM conversation_messages WHERE conversation_id=?", (conversation_id,))
            db.execute("DELETE FROM conversation_search WHERE conversation_id=?", (conversation_id,))
            for index, role, content in safe_messages:
                db.execute(
                    "INSERT INTO conversation_messages(conversation_id,message_index,role,content) VALUES(?,?,?,?)",
                    (conversation_id, index, role, content),
                )
                db.execute(
                    "INSERT INTO conversation_search(conversation_id,message_index,role,title,content) VALUES(?,?,?,?,?)",
                    (conversation_id, index, role, str(title or ""), content),
                )

    def search_conversations(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        query = query.strip()[:300]
        if not query:
            return []
        limit = max(1, min(int(limit), 20))
        with self._connect() as db:
            rows = db.execute(
                "SELECT f.conversation_id,f.message_index,f.role,f.title,f.content,bm25(conversation_search) AS score "
                "FROM conversation_search f WHERE conversation_search MATCH ? ORDER BY score LIMIT ?",
                (_fts_query(query), limit),
            ).fetchall()
        return self._rows(rows)

    def export_transfer(
        self, source_id: str = "server", categories: list[str] | None = None
    ) -> dict[str, Any]:
        """Return a bounded, credential-scrubbed transfer of user Agent data.

        MCP definitions, browser grants, credentials, schedules, run logs and
        all other machine-specific state are deliberately excluded.
        """
        source_id = re.sub(r"[^A-Za-z0-9_.@-]+", "_", str(source_id or "server"))[:120]
        selected = set(_TRANSFER_CATEGORIES) if categories is None else self._requested_transfer_categories(categories)
        memories: list[dict[str, Any]] = []
        excluded_sensitive = {"memories": 0, "skills": 0}
        for row in self.list_memories(limit=250) if "memories" in selected else []:
            if contains_secret(f"{row.get('title', '')}\n{row.get('body', '')}"):
                excluded_sensitive["memories"] += 1
                continue
            try:
                tags = json.loads(str(row.get("tags_json") or "[]"))
            except (TypeError, ValueError):
                tags = []
            if not isinstance(tags, list):
                tags = []
            memories.append({
                "title": str(row.get("title") or "")[:180],
                "body": str(row.get("body") or "")[:12_000],
                "tags": [str(tag)[:60] for tag in tags[:20] if isinstance(tag, str)],
            })

        skills: list[dict[str, Any]] = []
        for row in self.list_skills(include_disabled=True) if "skills" in selected else []:
            if contains_secret(
                f"{row.get('name', '')}\n{row.get('description', '')}\n{row.get('body', '')}"
            ):
                excluded_sensitive["skills"] += 1
                continue
            skills.append({
                "name": str(row.get("name") or "")[:120],
                "description": str(row.get("description") or "")[:500],
                "body": str(row.get("body") or "")[:40_000],
                "enabled": bool(row.get("enabled", True)),
            })

        payload: dict[str, Any] = {
            "format": "token-workshed-agent-transfer-v1",
            "source_id": source_id,
            "created_at": time.time(),
            "categories": {"memories": memories, "skills": skills, "chat_history": []},
            "counts": {"memories": len(memories), "skills": len(skills), "chat_history": 0},
            "excluded_sensitive": excluded_sensitive,
            "truncated": False,
        }
        conversations: list[sqlite3.Row] = []
        if "chat_history" in selected:
            with self._connect() as db:
                conversations = db.execute(
                    "SELECT id,title,updated_at FROM conversations ORDER BY updated_at DESC LIMIT 1000"
                ).fetchall()
                for conversation in conversations:
                    rows = db.execute(
                        "SELECT role,content FROM conversation_messages WHERE conversation_id=? ORDER BY message_index",
                        (conversation["id"],),
                    ).fetchall()
                    messages = [
                        {"role": str(row["role"]), "content": _redact_transfer_text(str(row["content"])[:20_000])}
                        for row in rows
                        if str(row["role"]) in {"user", "assistant", "tool"} and str(row["content"]).strip()
                    ][-200:]
                    if not messages:
                        continue
                    item = {
                        "id": str(conversation["id"])[:120],
                        "title": _redact_transfer_text(str(conversation["title"] or "")[:240]),
                        "messages": messages,
                    }
                    payload["categories"]["chat_history"].append(item)
                    if _transfer_size(payload) > _MAX_TRANSFER_BYTES:
                        payload["categories"]["chat_history"].pop()
                        payload["truncated"] = True
                        break
        payload["counts"]["chat_history"] = len(payload["categories"]["chat_history"])
        if len(conversations) >= 1000 and not payload["truncated"]:
            payload["truncated"] = True
        if _transfer_size(payload) > _MAX_TRANSFER_BYTES:
            # Memories and Skills have bounded record counts and individual
            # lengths. If those alone exceed the cap, fail closed.
            raise ValueError("Agent data export exceeds the 16 MiB transfer limit.")
        return payload

    @staticmethod
    def _validate_transfer(transfer: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(transfer, dict) or transfer.get("format") != "token-workshed-agent-transfer-v1":
            raise ValueError("Unsupported Agent data transfer format.")
        if _transfer_size(transfer) > _MAX_TRANSFER_BYTES:
            raise ValueError("Agent data transfer exceeds the 16 MiB limit.")
        categories = transfer.get("categories")
        if not isinstance(categories, dict):
            raise ValueError("Agent data transfer is missing categories.")
        return categories

    @staticmethod
    def _requested_transfer_categories(categories: list[str]) -> set[str]:
        selected = {str(item).strip() for item in categories}
        unknown = selected - _TRANSFER_CATEGORIES
        if unknown:
            raise ValueError(f"Unsupported Agent data categories: {', '.join(sorted(unknown))}")
        if not selected:
            raise ValueError("Select at least one Agent data category.")
        return selected

    def preview_transfer_import(
        self, transfer: dict[str, Any], categories: list[str]
    ) -> dict[str, Any]:
        """Count safe, new, duplicate and conflicting entries without mutation."""
        incoming = self._validate_transfer(transfer)
        selected = self._requested_transfer_categories(categories)
        source_id = re.sub(r"[^A-Za-z0-9_.@-]+", "_", str(transfer.get("source_id") or "server"))[:120]
        summary: dict[str, Any] = {"source_id": source_id, "categories": {}, "total_importable": 0}

        if "memories" in selected:
            existing = {item["title"].casefold(): item for item in self.list_memories(limit=250)}
            added = duplicates = conflicts = sensitive = invalid = 0
            for item in incoming.get("memories", [])[:250]:
                if not isinstance(item, dict):
                    invalid += 1
                    continue
                title, body = str(item.get("title") or "").strip(), str(item.get("body") or "").strip()
                if not title or not body:
                    invalid += 1
                elif contains_secret(title + "\n" + body):
                    sensitive += 1
                elif title.casefold() not in existing:
                    added += 1
                elif str(existing[title.casefold()].get("body") or "") == body:
                    duplicates += 1
                else:
                    conflicts += 1
            summary["categories"]["memories"] = {
                "importable": added, "duplicates": duplicates, "conflicts": conflicts,
                "sensitive_skipped": sensitive, "invalid": invalid,
            }
            summary["total_importable"] += added

        if "skills" in selected:
            existing = {item["slug"]: item for item in self.list_skills(include_disabled=True)}
            added = duplicates = conflicts = sensitive = invalid = 0
            for item in incoming.get("skills", [])[:200]:
                if not isinstance(item, dict):
                    invalid += 1
                    continue
                name, description, body = (
                    str(item.get("name") or "").strip(),
                    str(item.get("description") or "").strip(),
                    str(item.get("body") or "").strip(),
                )
                if not name or not body:
                    invalid += 1
                    continue
                if contains_secret(name + "\n" + description + "\n" + body):
                    sensitive += 1
                    continue
                slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:72]
                current = existing.get(slug)
                if current is None:
                    added += 1
                elif current["name"] == name and current["description"] == description and current["body"] == body:
                    duplicates += 1
                else:
                    conflicts += 1
            summary["categories"]["skills"] = {
                "importable": added, "duplicates": duplicates, "conflicts": conflicts,
                "sensitive_skipped": sensitive, "invalid": invalid,
            }
            summary["total_importable"] += added

        if "chat_history" in selected:
            source_prefix = f"transfer:{source_id}:"
            with self._connect() as db:
                existing_ids = {
                    str(row["id"]) for row in db.execute(
                        "SELECT id FROM conversations WHERE substr(id,1,?)=?",
                        (len(source_prefix), source_prefix),
                    ).fetchall()
                }
            added = duplicates = invalid = 0
            for item in incoming.get("chat_history", [])[:1000]:
                if not isinstance(item, dict) or not str(item.get("id") or "").strip() or not isinstance(item.get("messages"), list):
                    invalid += 1
                    continue
                imported_id = source_prefix + str(item["id"])[:120]
                if imported_id in existing_ids:
                    duplicates += 1
                else:
                    added += 1
            summary["categories"]["chat_history"] = {
                "importable": added, "duplicates": duplicates, "conflicts": 0,
                "sensitive_skipped": 0, "invalid": invalid,
            }
            summary["total_importable"] += added
        return summary

    def import_transfer(self, transfer: dict[str, Any], categories: list[str]) -> dict[str, Any]:
        incoming = self._validate_transfer(transfer)
        selected = self._requested_transfer_categories(categories)
        preview = self.preview_transfer_import(transfer, categories)
        source_id = preview["source_id"]
        imported = {key: 0 for key in selected}
        skipped = {key: {"duplicates": 0, "conflicts": 0, "sensitive": 0, "invalid": 0} for key in selected}

        if "memories" in selected:
            existing = {item["title"].casefold(): item for item in self.list_memories(limit=250)}
            for item in incoming.get("memories", [])[:250]:
                if not isinstance(item, dict):
                    skipped["memories"]["invalid"] += 1
                    continue
                title, body = str(item.get("title") or "").strip(), str(item.get("body") or "").strip()
                if not title or not body:
                    skipped["memories"]["invalid"] += 1
                elif contains_secret(title + "\n" + body):
                    skipped["memories"]["sensitive"] += 1
                elif title.casefold() in existing:
                    key = "duplicates" if existing[title.casefold()].get("body") == body else "conflicts"
                    skipped["memories"][key] += 1
                else:
                    self.save_memory(title, body, item.get("tags") if isinstance(item.get("tags"), list) else [])
                    imported["memories"] += 1

        if "skills" in selected:
            existing = {item["slug"]: item for item in self.list_skills(include_disabled=True)}
            for item in incoming.get("skills", [])[:200]:
                if not isinstance(item, dict):
                    skipped["skills"]["invalid"] += 1
                    continue
                name, description, body = str(item.get("name") or "").strip(), str(item.get("description") or "").strip(), str(item.get("body") or "").strip()
                if not name or not body:
                    skipped["skills"]["invalid"] += 1
                    continue
                if contains_secret(name + "\n" + description + "\n" + body):
                    skipped["skills"]["sensitive"] += 1
                    continue
                slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:72]
                current = existing.get(slug)
                if current:
                    key = "duplicates" if current["name"] == name and current["description"] == description and current["body"] == body else "conflicts"
                    skipped["skills"][key] += 1
                    continue
                self.save_skill(name, description, body, enabled=bool(item.get("enabled", True)))
                imported["skills"] += 1

        if "chat_history" in selected:
            source_prefix = f"transfer:{source_id}:"
            with self._connect() as db:
                existing_ids = {
                    str(row["id"]) for row in db.execute(
                        "SELECT id FROM conversations WHERE substr(id,1,?)=?",
                        (len(source_prefix), source_prefix),
                    ).fetchall()
                }
            for item in incoming.get("chat_history", [])[:1000]:
                if not isinstance(item, dict) or not str(item.get("id") or "").strip() or not isinstance(item.get("messages"), list):
                    skipped["chat_history"]["invalid"] += 1
                    continue
                imported_id = source_prefix + str(item["id"])[:120]
                if imported_id in existing_ids:
                    skipped["chat_history"]["duplicates"] += 1
                    continue
                messages = [
                    {"role": str(message.get("role") or ""), "content": _redact_transfer_text(_plain_text(message.get("content")))}
                    for message in item["messages"][-200:]
                    if isinstance(message, dict)
                ]
                self.index_conversation(imported_id[:120], messages, f"[Imported] {_redact_transfer_text(str(item.get('title') or '')[:200])}")
                imported["chat_history"] += 1

        return {"ok": True, "imported": imported, "skipped": skipped, "preview": preview}

    def get_memory_context(self, limit: int = 24) -> str:
        memories = self.list_memories(limit=limit)
        if not memories:
            return ""
        lines = ["Persistent notes saved in this NextCore installation:"]
        for item in memories:
            lines.append(f"- {item['title']}: {item['body']}")
        return "\n".join(lines)[:16_000]

    def get_skill_context(self, limit: int = 12) -> str:
        skills = self.list_skills(include_disabled=False)[:max(1, min(int(limit), 24))]
        if not skills:
            return ""
        lines = ["Enabled NextCore Skills available for this conversation:"]
        for item in skills:
            lines.append(f"\n## {item['name']}\n{item['description']}\n{item['body']}")
        return "\n".join(lines)[:24_000]

    def create_job(
        self,
        *,
        title: str,
        prompt: str,
        schedule: str,
        timezone: str,
        model_id: str,
        tool_scope: list[str],
        authorization_scope: dict[str, Any] | None = None,
        server_url: str = "http://127.0.0.1:30000",
        tool_parser: str = "",
        next_run_at: float | None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        title = " ".join(str(title).split())[:180]
        prompt = str(prompt).strip()[:20_000]
        schedule = str(schedule).strip()[:160]
        model_id = str(model_id).strip()[:500]
        server_url = str(server_url or "").strip()[:2000]
        if not title or not prompt or not schedule or not model_id:
            raise ValueError("A job needs a title, prompt, schedule, and SGLang model.")
        parsed_server = urlsplit(server_url)
        try:
            local_server = bool(parsed_server.hostname) and ipaddress.ip_address(parsed_server.hostname).is_loopback
        except ValueError:
            local_server = (parsed_server.hostname or "").casefold() == "localhost"
        if parsed_server.scheme != "http" or not local_server:
            raise ValueError("Scheduled Agent tasks can connect only to HTTP SGLang on this Mac.")
        allowed = {
            "web_search", "web_fetch", "terminal", "file", "computer_use", "browser",
            "code_execution", "memory", "skills", "session_search", "delegate",
        }
        scope = sorted({
            str(item) for item in tool_scope
            if str(item) in allowed or re.fullmatch(r"[A-Za-z0-9_-]{1,80}__[A-Za-z0-9_-]{1,120}", str(item))
        })
        auth_scope = authorization_scope if isinstance(authorization_scope, dict) else {}
        if {"computer_use", "browser"}.intersection(scope) and not bool(auth_scope.get("computer_control")):
            raise ValueError("Scheduled computer control requires explicit per-task authorization.")
        auth_scope = {
            "computer_control": bool(auth_scope.get("computer_control")),
            "allowed_paths": [str(item)[:1024] for item in auth_scope.get("allowed_paths", []) if isinstance(item, str)][:32],
            "allowed_domains": [str(item).lower()[:253] for item in auth_scope.get("allowed_domains", []) if isinstance(item, str)][:64],
            "browser_profile_ids": sorted({str(item)[:100] for item in auth_scope.get("browser_profile_ids", []) if isinstance(item, str)})[:16],
        }
        if "browser" in scope and not auth_scope["allowed_domains"]:
            raise ValueError("Scheduled browser access requires an explicit allowed_domains list.")
        if auth_scope["browser_profile_ids"] and "browser" not in scope:
            raise ValueError("A signed-in browser profile requires browser access in the scheduled task's tool scope.")
        approved_profile_ids = {item["id"] for item in self.list_browser_profiles() if item["enabled"]}
        if any(item not in approved_profile_ids for item in auth_scope["browser_profile_ids"]):
            raise ValueError("Scheduled tasks can use only browser profiles already enabled in NextCore.")
        now = time.time()
        job_id = job_id or uuid.uuid4().hex
        with self._connect() as db:
            db.execute(
                "INSERT INTO scheduled_jobs(id,title,prompt,schedule,timezone,model_id,tool_scope_json,authorization_scope_json,server_url,tool_parser,enabled,state,next_run_at,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,1,'scheduled',?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,prompt=excluded.prompt,schedule=excluded.schedule,timezone=excluded.timezone,model_id=excluded.model_id,tool_scope_json=excluded.tool_scope_json,authorization_scope_json=excluded.authorization_scope_json,server_url=excluded.server_url,tool_parser=excluded.tool_parser,enabled=1,state='scheduled',next_run_at=excluded.next_run_at,updated_at=excluded.updated_at",
                (job_id, title, prompt, schedule, timezone or "UTC", model_id, json.dumps(scope), json.dumps(auth_scope), server_url, tool_parser, next_run_at, now, now),
            )
            saved = db.execute("SELECT * FROM scheduled_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(saved)

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM scheduled_jobs ORDER BY enabled DESC, next_run_at LIMIT ?", (max(1, min(int(limit), 250)),)).fetchall()
        result = self._rows(rows)
        for row in result:
            row["tool_scope"] = json.loads(row.pop("tool_scope_json", "[]"))
            row["authorization_scope"] = json.loads(row.pop("authorization_scope_json", "{}"))
        return result

    def run_job_once(self, job_id: str) -> dict[str, Any]:
        """Queue a separately recorded immediate one-time copy of a task."""
        with self._connect() as db:
            row = db.execute("SELECT * FROM scheduled_jobs WHERE id=?", (str(job_id),)).fetchone()
        if not row:
            raise KeyError("Scheduled job not found.")
        if row["state"] == "running":
            raise ValueError("Stop the scheduled task before queuing a one-time run.")
        tool_scope = json.loads(row["tool_scope_json"] or "[]")
        authorization_scope = json.loads(row["authorization_scope_json"] or "{}")
        queued_at = time.time() + 2
        return self.create_job(
            title=f"{row['title']} (one-time run)",
            prompt=row["prompt"],
            schedule=f"once:{queued_at}",
            timezone=row["timezone"],
            model_id=row["model_id"],
            tool_scope=tool_scope,
            authorization_scope=authorization_scope,
            server_url=row["server_url"],
            tool_parser=row["tool_parser"],
            next_run_at=queued_at,
        )

    def set_job_enabled(self, job_id: str, enabled: bool) -> dict[str, Any]:
        now = time.time()
        with self._connect() as db:
            db.execute(
                "UPDATE scheduled_jobs SET enabled=?,state=CASE WHEN state='running' THEN state ELSE ? END,updated_at=? WHERE id=?",
                (int(enabled), "scheduled" if enabled else "paused", now, job_id),
            )
            row = db.execute("SELECT * FROM scheduled_jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError("Scheduled job not found.")
        result = dict(row)
        result["tool_scope"] = json.loads(result.pop("tool_scope_json", "[]"))
        result["authorization_scope"] = json.loads(result.pop("authorization_scope_json", "{}"))
        return result

    def delete_job(self, job_id: str) -> None:
        with self._connect() as db:
            row = db.execute("SELECT state FROM scheduled_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return
            if row["state"] == "running":
                raise ValueError("Stop the scheduled task before deleting it.")
            db.execute("DELETE FROM scheduled_jobs WHERE id=?", (job_id,))

    def due_jobs(self, now: float | None = None) -> list[dict[str, Any]]:
        now = now if now is not None else time.time()
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM scheduled_jobs WHERE enabled=1 AND next_run_at IS NOT NULL AND next_run_at<=? AND state!='running' ORDER BY next_run_at LIMIT 25",
                (now,),
            ).fetchall()
        result = self._rows(rows)
        for row in result:
            row["tool_scope"] = json.loads(row.pop("tool_scope_json", "[]"))
            row["authorization_scope"] = json.loads(row.pop("authorization_scope_json", "{}"))
        return result

    def start_job_run(self, job_id: str, *, next_run_at: float | None = None) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as db:
            changed = db.execute(
                "UPDATE scheduled_jobs SET state='running',last_run_at=?,next_run_at=?,enabled=CASE WHEN ? IS NULL THEN 0 ELSE enabled END,updated_at=? WHERE id=? AND enabled=1 AND state!='running'",
                (now, next_run_at, next_run_at, now, job_id),
            ).rowcount
            if not changed:
                raise RuntimeError("Scheduled job is already running or disabled.")
            db.execute(
                "INSERT INTO scheduled_job_runs(id,job_id,state,started_at) VALUES(?,?,'running',?)",
                (run_id, job_id, now),
            )
            job = db.execute("SELECT * FROM scheduled_jobs WHERE id=?", (job_id,)).fetchone()
        result = dict(job)
        result["tool_scope"] = json.loads(result.pop("tool_scope_json", "[]"))
        result["authorization_scope"] = json.loads(result.pop("authorization_scope_json", "{}"))
        result["run_id"] = run_id
        return result

    def finish_job_run(
        self, run_id: str, job_id: str, *, state: str, summary: str = "", error: str = "", next_run_at: float | None = None
    ) -> None:
        now = time.time()
        state = state if state in {"succeeded", "failed", "cancelled", "interrupted"} else "failed"
        with self._connect() as db:
            db.execute(
                "UPDATE scheduled_job_runs SET state=?,finished_at=?,summary=?,error=? WHERE id=?",
                (state, now, summary[:4_000], error[:1_000], run_id),
            )
            db.execute(
                "UPDATE scheduled_jobs SET state=CASE WHEN enabled=0 THEN ? ELSE ? END,next_run_at=CASE WHEN enabled=0 THEN NULL ELSE ? END,updated_at=? WHERE id=?",
                (state, "scheduled" if next_run_at is not None else state, next_run_at, now, job_id),
            )

    def recover_interrupted_runs(self) -> int:
        """Record a previous scheduler exit without replaying side effects."""
        now = time.time()
        with self._connect() as db:
            rows = db.execute("SELECT DISTINCT job_id FROM scheduled_job_runs WHERE state='running'").fetchall()
            changed = db.execute(
                "UPDATE scheduled_job_runs SET state='interrupted',finished_at=?,error=? WHERE state='running'",
                (now, "Scheduler stopped before this run completed; it was not replayed."),
            ).rowcount
            for row in rows:
                db.execute(
                    "UPDATE scheduled_jobs SET state=CASE WHEN enabled=1 THEN 'scheduled' ELSE 'paused' END,updated_at=? WHERE id=?",
                    (now, row["job_id"]),
                )
        return changed

    def list_job_runs(self, job_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            if job_id:
                rows = db.execute(
                    "SELECT * FROM scheduled_job_runs WHERE job_id=? ORDER BY started_at DESC LIMIT ?",
                    (job_id, max(1, min(int(limit), 250))),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM scheduled_job_runs ORDER BY started_at DESC LIMIT ?",
                    (max(1, min(int(limit), 250)),),
                ).fetchall()
        return self._rows(rows)

    def save_mcp_server(self, config: dict[str, Any], secret_env_names: list[str]) -> dict[str, Any]:
        name = " ".join(str(config.get("name") or "").split())[:100]
        transport = str(config.get("transport") or "stdio").strip().lower()
        if not name or transport not in {"stdio", "sse"}:
            raise ValueError("MCP server name and a supported transport are required.")
        command = str(config.get("command") or "").strip()[:1000]
        args = [str(arg)[:4000] for arg in config.get("args", []) if isinstance(arg, (str, int, float))][:100]
        url = str(config.get("url") or "").strip()[:2000]
        if transport == "stdio" and not command:
            raise ValueError("An MCP stdio server needs a command.")
        if transport == "sse" and not url:
            raise ValueError("An MCP SSE server needs a URL.")
        if url and (urlparse_scheme(url) not in {"http", "https"}):
            raise ValueError("MCP SSE URLs must use HTTP or HTTPS.")
        if contains_secret("\n".join([command, url, *args])):
            raise ValueError("MCP credentials belong in the macOS Keychain-backed secrets field, not command arguments or URLs.")
        parsed_url = urlsplit(url) if url else None
        if parsed_url and (parsed_url.username or parsed_url.password):
            raise ValueError("MCP URLs cannot embed credentials; use a Keychain-backed environment secret.")
        env_names = sorted({str(key) for key in secret_env_names if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", str(key))})
        now = time.time()
        server_id = str(config.get("id") or uuid.uuid4().hex)
        timeout = max(1.0, min(float(config.get("timeout", 30.0)), 300.0))
        enabled = bool(config.get("enabled", True))
        with self._connect() as db:
            db.execute(
                "INSERT INTO mcp_servers(id,name,transport,command,args_json,url,env_names_json,enabled,timeout,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET transport=excluded.transport,command=excluded.command,args_json=excluded.args_json,url=excluded.url,env_names_json=excluded.env_names_json,enabled=excluded.enabled,timeout=excluded.timeout,updated_at=excluded.updated_at",
                (server_id, name, transport, command, json.dumps(args), url, json.dumps(env_names), int(enabled), timeout, now),
            )
            row = db.execute("SELECT * FROM mcp_servers WHERE name=?", (name,)).fetchone()
        return self._mcp_public_row(row)

    @staticmethod
    def _mcp_public_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["args"] = json.loads(result.pop("args_json"))
        result["env_names"] = json.loads(result.pop("env_names_json"))
        result["has_secrets"] = bool(result["env_names"])
        return result

    def list_mcp_servers(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM mcp_servers ORDER BY name").fetchall()
        return [self._mcp_public_row(row) for row in rows]

    def delete_mcp_server(self, server_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM mcp_servers WHERE id=? OR name=?", (server_id, server_id)).fetchone()
            if row:
                db.execute("DELETE FROM mcp_servers WHERE id=?", (row["id"],))
        return self._mcp_public_row(row) if row else None

    def save_browser_profile(self, pid: int, window_id: int, app_name: str, bundle_id: str = "") -> dict[str, Any]:
        pid = int(pid)
        window_id = int(window_id)
        app_name = " ".join(str(app_name or "").split())[:120]
        bundle_id = str(bundle_id or "").strip()[:200]
        if pid <= 0 or window_id <= 0:
            raise ValueError("Choose a Chromium window with a valid process and window ID.")
        if not app_name or not any(item in (app_name + " " + bundle_id).casefold() for item in ("chrome", "edge", "chromium")):
            raise ValueError("Existing signed-in profile access is supported only for Chromium browsers.")
        profile_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as db:
            db.execute(
                "INSERT INTO browser_profile_grants(id,pid,window_id,app_name,bundle_id,enabled,created_at) VALUES(?,?,?,?,?,1,?) "
                "ON CONFLICT(pid,window_id) DO UPDATE SET app_name=excluded.app_name,bundle_id=excluded.bundle_id,enabled=1",
                (profile_id, pid, window_id, app_name, bundle_id, now),
            )
            row = db.execute("SELECT * FROM browser_profile_grants WHERE pid=? AND window_id=?", (pid, window_id)).fetchone()
        return dict(row)

    def list_browser_profiles(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM browser_profile_grants ORDER BY created_at DESC").fetchall()
        return self._rows(rows)

    def get_browser_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM browser_profile_grants WHERE id=? AND enabled=1", (str(profile_id),)).fetchone()
        return dict(row) if row else None

    def has_browser_profile_grants(self) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT 1 FROM browser_profile_grants WHERE enabled=1 LIMIT 1").fetchone()
        return row is not None

    def revoke_browser_profile(self, profile_id: str) -> dict[str, Any]:
        profile_id = str(profile_id)
        paused_jobs: list[str] = []
        with self._connect() as db:
            changed = db.execute("DELETE FROM browser_profile_grants WHERE id=?", (profile_id,)).rowcount
            if changed:
                jobs = db.execute("SELECT id,authorization_scope_json,state FROM scheduled_jobs WHERE enabled=1").fetchall()
                now = time.time()
                for job in jobs:
                    try:
                        scope = json.loads(job["authorization_scope_json"] or "{}")
                    except (TypeError, ValueError):
                        scope = {}
                    if profile_id in scope.get("browser_profile_ids", []):
                        db.execute(
                            "UPDATE scheduled_jobs SET enabled=0,state=CASE WHEN state='running' THEN state ELSE 'paused' END,updated_at=? WHERE id=?",
                            (now, job["id"]),
                        )
                        paused_jobs.append(job["id"])
        return {"removed": bool(changed), "paused_jobs": paused_jobs}

    def browser_profile_context(self) -> str:
        profiles = [item for item in self.list_browser_profiles() if item["enabled"]]
        if not profiles:
            return ""
        lines = ["Existing Chromium profiles explicitly enabled by the user in NextCore (use only these exact process/window pairs):"]
        for item in profiles[:8]:
            lines.append(f"- profile_id={item['id']} app={item['app_name']} pid={item['pid']} window_id={item['window_id']}")
        return "\n".join(lines)


def urlparse_scheme(value: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(value).scheme.lower()


def _fts_query(value: str) -> str:
    terms = re.findall(r"[\w\u4e00-\u9fff-]+", value, flags=re.UNICODE)[:12]
    return " AND ".join('"' + term.replace('"', '""') + '"' for term in terms) or '""'
