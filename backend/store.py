"""Persistent users, sessions, conversations and messages (SQLite, standard library only).

Privacy design (mental-health conversations are sensitive):

* Every conversation and message query is scoped by the authenticated ``user_id``; another user's conversation is
  indistinguishable from a missing one (the API answers 404).
* Session cookies hold a random token; only its SHA-256 is stored, so a copy of the database cannot be replayed as a
  session.  Passwords (optional accounts) are scrypt hashes.
* Conversations are stored only after the user consents.  A user who declines gets an in-memory store with the
  same schema: nothing of their chat reaches the disk and it disappears on restart or after inactivity.
* Retention: each user chooses how long inactive conversations are kept (``RETENTION_OPTIONS``; 0 = until they
  delete them).  ``purge()`` deletes expired conversations, expired sessions and abandoned anonymous users.
* Message metadata holds the model, generation statistics, context trimming and steering settings.

The database file is created with owner-only permissions (0600).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 2          # 2: steering modes always / policy / off (1 had auto / off)
CONSENT_VERSION = "2026-11-21"
RETENTION_OPTIONS = (7, 30, 90, 365, 0)     # days of inactivity before a conversation is deleted; 0 = never
DEFAULT_RETENTION_DAYS = 30
SESSION_SECONDS = 30 * 24 * 3600
ANONYMOUS_IDLE_SECONDS = 90 * 24 * 3600     # anonymous users with nothing stored are removed after this
EPHEMERAL_IDLE_SECONDS = 6 * 3600           # unsaved (in-memory) conversations are dropped after this
MAX_TITLE = 120
USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "chatbot_feature" / "data" / "mindmate_chat.sqlite"

_CONVERSATION_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL {user_fk},
    title TEXT NOT NULL DEFAULT 'New chat',
    title_source TEXT NOT NULL DEFAULT 'pending' CHECK (title_source IN ('pending', 'auto', 'fallback', 'user')),
    model_key TEXT,
    steering_mode TEXT NOT NULL DEFAULT 'always' CHECK (steering_mode IN ('always', 'policy', 'off')),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS conversations_by_user ON conversations (user_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    user_id TEXT NOT NULL {user_fk},
    seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('complete', 'cancelled', 'error')),
    model_key TEXT,
    model_id TEXT,
    turn_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{{}}',
    created_at REAL NOT NULL,
    UNIQUE (conversation_id, seq)
);
CREATE INDEX IF NOT EXISTS messages_by_turn ON messages (turn_id);
CREATE INDEX IF NOT EXISTS messages_by_user ON messages (user_id);
"""

_USER_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE COLLATE NOCASE,
    password_hash TEXT,
    created_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    consent_version TEXT,
    consent_at REAL,
    store_history INTEGER NOT NULL DEFAULT 0,
    retention_days INTEGER NOT NULL DEFAULT 30
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    last_used_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_by_user ON sessions (user_id);
"""


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt$16384$8$1${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    try:
        scheme, n, r, p, salt, digest = (stored or "").split("$")
        if scheme != "scrypt":
            return False
        candidate = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                                   dklen=len(bytes.fromhex(digest)))
        return hmac.compare_digest(candidate.hex(), digest)
    except (ValueError, TypeError):
        return False


def clean_title(title: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(title))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TITLE].strip()


def _now() -> float:
    return time.time()


class ConversationStore:
    """Conversations and messages.  ``path=None`` keeps them in memory only (same schema, nothing on disk)."""

    persistent: bool

    def __init__(self, path: str | Path | None = None) -> None:
        self.persistent = path is not None
        if path is None:
            self._uri = f"file:mindmate-ephemeral-{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._anchor = sqlite3.connect(self._uri, uri=True, check_same_thread=False)   # keeps the memory DB alive
            with self._connect() as db:
                db.executescript(_CONVERSATION_DDL.format(user_fk=""))
        else:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.touch(mode=0o600)
            os.chmod(self.path, 0o600)
            with self._connect() as db:
                db.execute("PRAGMA journal_mode = WAL")
                db.executescript(_USER_DDL + _CONVERSATION_DDL.format(user_fk="REFERENCES users (id) ON DELETE CASCADE"))
                version = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
                if version is None:
                    db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
                elif version == 1:
                    self._migrate_v1(db)

    @staticmethod
    def _migrate_v1(db: sqlite3.Connection) -> None:
        """Schema 1 -> 2: steering modes ``always`` / ``policy`` / ``off``.  Conversations left on the old default
        ("auto") follow the new default, every reply steered; "off" stays the unsteered baseline."""
        db.execute("PRAGMA foreign_keys = OFF")
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute(_CONVERSATION_DDL.split(";")[0].format(user_fk="REFERENCES users (id) ON DELETE CASCADE")
                       .replace("CREATE TABLE IF NOT EXISTS conversations", "CREATE TABLE conversations_v2"))
            db.execute("INSERT INTO conversations_v2 SELECT id, user_id, title, title_source, model_key, "
                       "CASE steering_mode WHEN 'off' THEN 'off' ELSE 'always' END, created_at, updated_at "
                       "FROM conversations")
            db.execute("DROP TABLE conversations")
            db.execute("ALTER TABLE conversations_v2 RENAME TO conversations")
            db.execute("CREATE INDEX IF NOT EXISTS conversations_by_user ON conversations (user_id, updated_at DESC)")
            db.execute("UPDATE schema_version SET version = 2")
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        finally:
            db.execute("PRAGMA foreign_keys = ON")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self.persistent:
            db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        else:
            db = sqlite3.connect(self._uri, uri=True, timeout=30, isolation_level=None, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------ conversations

    def create_conversation(self, user_id: str, *, model_key: str | None = None, title: str | None = None,
                            steering_mode: str = "always") -> dict[str, Any]:
        now = _now()
        cleaned = clean_title(title) if title else ""
        item = {"id": uuid.uuid4().hex, "user_id": user_id, "title": cleaned or "New chat",
                "title_source": "user" if cleaned else "pending", "model_key": model_key,
                "steering_mode": steering_mode, "created_at": now, "updated_at": now}
        with self._transaction() as db:
            db.execute("INSERT INTO conversations (id, user_id, title, title_source, model_key, steering_mode, created_at, "
                       "updated_at) VALUES (:id, :user_id, :title, :title_source, :model_key, :steering_mode, :created_at, "
                       ":updated_at)", item)
        return {**item, "message_count": 0}

    def list_conversations(self, user_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
                "FROM conversations c WHERE c.user_id = ? ORDER BY c.updated_at DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
                "FROM conversations c WHERE c.id = ? AND c.user_id = ?", (conversation_id, user_id)).fetchone()
        return dict(row) if row else None

    def rename_conversation(self, user_id: str, conversation_id: str, title: str) -> dict[str, Any] | None:
        cleaned = clean_title(title)
        if not cleaned:
            raise ValueError("A title needs at least one visible character.")
        with self._transaction() as db:
            changed = db.execute("UPDATE conversations SET title = ?, title_source = 'user' WHERE id = ? AND user_id = ?",
                                 (cleaned, conversation_id, user_id)).rowcount
        return self.get_conversation(user_id, conversation_id) if changed else None

    def set_auto_title(self, user_id: str, conversation_id: str, title: str, source: str = "auto") -> bool:
        """Set a generated title unless the user named the conversation in the meantime."""
        cleaned = clean_title(title)
        if not cleaned:
            return False
        with self._transaction() as db:
            return db.execute("UPDATE conversations SET title = ?, title_source = ? WHERE id = ? AND user_id = ? "
                              "AND title_source = 'pending'", (cleaned, source, conversation_id, user_id)).rowcount > 0

    def update_conversation(self, user_id: str, conversation_id: str, *, model_key: str | None = None,
                            steering_mode: str | None = None) -> dict[str, Any] | None:
        fields, values = [], []
        if model_key is not None:
            fields.append("model_key = ?")
            values.append(model_key)
        if steering_mode is not None:
            if steering_mode not in ("always", "policy", "off"):
                raise ValueError("steering must be always, policy or off.")
            fields.append("steering_mode = ?")
            values.append(steering_mode)
        if fields:
            with self._transaction() as db:
                db.execute(f"UPDATE conversations SET {', '.join(fields)} WHERE id = ? AND user_id = ?",
                           (*values, conversation_id, user_id))
        return self.get_conversation(user_id, conversation_id)

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        with self._transaction() as db:
            return db.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?",
                              (conversation_id, user_id)).rowcount > 0

    def delete_user_conversations(self, user_id: str) -> int:
        with self._transaction() as db:
            return db.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,)).rowcount

    # ------------------------------------------------------------------ messages

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.get("metadata") or "{}")
        except ValueError:
            item["metadata"] = {}
        return item

    def add_message(self, user_id: str, conversation_id: str, role: str, content: str, *, status: str = "complete",
                    model_key: str | None = None, model_id: str | None = None, turn_id: str | None = None,
                    metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Append a message to the user's conversation (``None`` if the conversation is not theirs)."""
        now = _now()
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                          (conversation_id, user_id)).fetchone() is None:
                return None
            seq = db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE conversation_id = ?",
                             (conversation_id,)).fetchone()[0]
            item = {"id": uuid.uuid4().hex, "conversation_id": conversation_id, "user_id": user_id, "seq": seq,
                    "role": role, "content": content, "status": status, "model_key": model_key, "model_id": model_id,
                    "turn_id": turn_id, "metadata": json.dumps(metadata or {}, ensure_ascii=False), "created_at": now}
            db.execute("INSERT INTO messages (id, conversation_id, user_id, seq, role, content, status, model_key, model_id, "
                       "turn_id, metadata, created_at) VALUES (:id, :conversation_id, :user_id, :seq, :role, :content, "
                       ":status, :model_key, :model_id, :turn_id, :metadata, :created_at)", item)
            db.execute("UPDATE conversations SET updated_at = ?" + (", model_key = ?" if model_key else "")
                       + " WHERE id = ?", (now, model_key, conversation_id) if model_key else (now, conversation_id))
        return {**item, "metadata": metadata or {}}

    def messages(self, user_id: str, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM messages WHERE conversation_id = ? AND user_id = ? ORDER BY seq",
                              (conversation_id, user_id)).fetchall()
        return [self._message(row) for row in rows]

    def owns_turn(self, user_id: str, turn_id: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM messages WHERE turn_id = ? AND user_id = ? LIMIT 1",
                              (turn_id, user_id)).fetchone() is not None

    def export(self, user_id: str) -> list[dict[str, Any]]:
        out = []
        for conversation in self.list_conversations(user_id, limit=100_000):
            out.append({**{key: conversation[key] for key in ("id", "title", "model_key", "created_at", "updated_at")},
                        "messages": [{key: message[key] for key in ("seq", "role", "content", "status", "model_key",
                                                                     "model_id", "created_at", "metadata")}
                                     for message in self.messages(user_id, conversation["id"])]})
        return out

    def purge_inactive(self, older_than: float) -> int:
        """Delete conversations not updated since ``older_than`` (the in-memory store's expiry)."""
        with self._transaction() as db:
            return db.execute("DELETE FROM conversations WHERE updated_at < ?", (older_than,)).rowcount


class Store(ConversationStore):
    """The persistent database: users and sessions as well as conversations."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        super().__init__(path)

    # ------------------------------------------------------------------ users

    @staticmethod
    def _user(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        item.pop("password_hash", None)
        item["store_history"] = bool(item["store_history"])
        item["anonymous"] = item["username"] is None
        return item

    def create_user(self) -> dict[str, Any]:
        now = _now()
        user_id = uuid.uuid4().hex
        with self._transaction() as db:
            db.execute("INSERT INTO users (id, created_at, last_seen_at, retention_days) VALUES (?, ?, ?, ?)",
                       (user_id, now, now, DEFAULT_RETENTION_DAYS))
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            return self._user(db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())

    def set_consent(self, user_id: str, *, store_history: bool, retention_days: int) -> dict[str, Any] | None:
        if retention_days not in RETENTION_OPTIONS:
            raise ValueError("retention_days must be one of " + ", ".join(map(str, RETENTION_OPTIONS)) + ".")
        with self._transaction() as db:
            db.execute("UPDATE users SET consent_version = ?, consent_at = ?, store_history = ?, retention_days = ? "
                       "WHERE id = ?", (CONSENT_VERSION, _now(), int(bool(store_history)), retention_days, user_id))
        return self.get_user(user_id)

    def update_settings(self, user_id: str, *, store_history: bool | None = None,
                        retention_days: int | None = None) -> dict[str, Any] | None:
        if retention_days is not None and retention_days not in RETENTION_OPTIONS:
            raise ValueError("retention_days must be one of " + ", ".join(map(str, RETENTION_OPTIONS)) + ".")
        with self._transaction() as db:
            if store_history is not None:
                db.execute("UPDATE users SET store_history = ? WHERE id = ?", (int(bool(store_history)), user_id))
            if retention_days is not None:
                db.execute("UPDATE users SET retention_days = ? WHERE id = ?", (retention_days, user_id))
        return self.get_user(user_id)

    def register(self, user_id: str, username: str, password: str) -> dict[str, Any]:
        """Give the current (anonymous) user a username and password, keeping their conversations."""
        if not USERNAME.fullmatch(username or ""):
            raise ValueError("Usernames have 3 to 32 letters, digits, dots, dashes or underscores.")
        if not isinstance(password, str) or len(password) < 8 or len(password) > 256:
            raise ValueError("Passwords need at least 8 characters.")
        hashed = hash_password(password)
        try:
            with self._transaction() as db:
                row = db.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
                if row is None:
                    raise ValueError("Unknown user.")
                if row["username"] is not None:
                    raise ValueError("This account already has a username.")
                db.execute("UPDATE users SET username = ?, password_hash = ? WHERE id = ?", (username, hashed, user_id))
        except sqlite3.IntegrityError as exc:
            raise ValueError("This username is taken.") from exc
        return self.get_user(user_id)

    def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", (username or "",)).fetchone()
        if row is None:
            verify_password(password or "", hash_password("timing-equaliser"))     # similar cost for unknown names
            return None
        return self._user(row) if verify_password(password or "", row["password_hash"]) else None

    def delete_user(self, user_id: str) -> None:
        with self._transaction() as db:
            db.execute("DELETE FROM users WHERE id = ?", (user_id,))         # cascades: sessions, conversations, messages

    # ------------------------------------------------------------------ sessions

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._transaction() as db:
            db.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at, last_used_at) "
                       "VALUES (?, ?, ?, ?, ?)", (_hash_token(token), user_id, now, now + SESSION_SECONDS, now))
        return token

    def session_user(self, token: str | None) -> dict[str, Any] | None:
        if not token or len(token) > 128:
            return None
        now = _now()
        digest = _hash_token(token)
        with self._connect() as db:
            row = db.execute("SELECT s.expires_at, s.last_used_at, u.* FROM sessions s JOIN users u ON u.id = s.user_id "
                             "WHERE s.token_hash = ?", (digest,)).fetchone()
            if row is None or row["expires_at"] < now:
                return None
            if now - row["last_used_at"] > 3600:        # rolling expiry, written at most hourly
                db.execute("UPDATE sessions SET last_used_at = ?, expires_at = ? WHERE token_hash = ?",
                           (now, now + SESSION_SECONDS, digest))
                db.execute("UPDATE users SET last_seen_at = ? WHERE id = ?", (now, row["id"]))
        user = dict(row)
        user.pop("expires_at")
        user.pop("last_used_at")
        return self._user(user)          # type: ignore[arg-type]

    def delete_session(self, token: str | None) -> None:
        if token:
            with self._transaction() as db:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))

    def delete_user_sessions(self, user_id: str) -> None:
        with self._transaction() as db:
            db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    # ------------------------------------------------------------------ retention

    def purge(self, now: float | None = None) -> dict[str, int]:
        """Apply every user's retention period; drop expired sessions and abandoned anonymous users."""
        now = _now() if now is None else now
        with self._transaction() as db:
            conversations = db.execute(
                "DELETE FROM conversations WHERE id IN (SELECT c.id FROM conversations c JOIN users u ON u.id = c.user_id "
                "WHERE u.retention_days > 0 AND c.updated_at < ? - u.retention_days * 86400)", (now,)).rowcount
            sessions = db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,)).rowcount
            users = db.execute(
                "DELETE FROM users WHERE username IS NULL AND last_seen_at < ? AND NOT EXISTS "
                "(SELECT 1 FROM conversations c WHERE c.user_id = users.id) AND NOT EXISTS "
                "(SELECT 1 FROM sessions s WHERE s.user_id = users.id)", (now - ANONYMOUS_IDLE_SECONDS,)).rowcount
        return {"conversations": conversations, "sessions": sessions, "users": users}
