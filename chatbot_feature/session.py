"""Bounded, expiring in-memory state. Keys are trusted server identities, never client user IDs."""
import hashlib
import json
import threading
import time
from collections import OrderedDict


def session_key(user_id, session_id):
    return hashlib.sha256(json.dumps([user_id, session_id], ensure_ascii=False).encode()).hexdigest()


class SessionStore:
    def __init__(self, ttl_seconds=1800, max_sessions=256):
        if ttl_seconds <= 0 or max_sessions < 1:
            raise ValueError("Retention and capacity must be positive")
        self.ttl_seconds, self.max_sessions = ttl_seconds, max_sessions
        self._items, self._lock = OrderedDict(), threading.RLock()

    def get(self, key):
        with self._lock:
            self._purge()
            item = self._items.get(key)
            return item[1] if item else None

    def put(self, key, value):
        with self._lock:
            self._purge()
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_sessions:
                self._items.popitem(last=False)

    def delete(self, key):
        with self._lock:
            self._items.pop(key, None)

    def _purge(self):
        now = time.monotonic()
        for key in list(self._items):
            if now - self._items[key][0] >= self.ttl_seconds:
                del self._items[key]
