"""Users, sessions, conversations and messages: isolation, retention, deletion and the in-memory mode."""
import os
import sqlite3
import stat
import tempfile
import time
import unittest
from pathlib import Path

from backend.store import (CONSENT_VERSION, ConversationStore, Store, clean_title, hash_password, verify_password)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "chat.sqlite"
        self.store = Store(self.path)

    def test_database_file_is_private(self):
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_sessions_store_only_a_hash_and_survive_reopening(self):
        user = self.store.create_user()
        token = self.store.create_session(user["id"])
        with sqlite3.connect(self.path) as db:
            stored = [row[0] for row in db.execute("SELECT token_hash FROM sessions")]
        self.assertNotIn(token, stored)
        self.assertEqual(Store(self.path).session_user(token)["id"], user["id"])      # a restarted server
        self.store.delete_session(token)
        self.assertIsNone(self.store.session_user(token))
        self.assertIsNone(self.store.session_user("not-a-token"))

    def test_conversations_are_isolated_between_users(self):
        alice, bob = self.store.create_user(), self.store.create_user()
        conversation = self.store.create_conversation(alice["id"], model_key="llama")
        self.store.add_message(alice["id"], conversation["id"], "user", "hello")
        self.assertIsNone(self.store.get_conversation(bob["id"], conversation["id"]))
        self.assertEqual(self.store.list_conversations(bob["id"]), [])
        self.assertEqual(self.store.messages(bob["id"], conversation["id"]), [])
        self.assertIsNone(self.store.add_message(bob["id"], conversation["id"], "user", "intrusion"))
        self.assertIsNone(self.store.rename_conversation(bob["id"], conversation["id"], "mine now"))
        self.assertFalse(self.store.delete_conversation(bob["id"], conversation["id"]))
        self.assertEqual(len(self.store.messages(alice["id"], conversation["id"])), 1)

    def test_messages_are_ordered_and_carry_model_and_metadata(self):
        user = self.store.create_user()
        conversation = self.store.create_conversation(user["id"])
        self.store.add_message(user["id"], conversation["id"], "user", "hi")
        self.store.add_message(user["id"], conversation["id"], "assistant", "hello", model_key="sarvam",
                               model_id="sarvamai/sarvam-30b", turn_id="a" * 32,
                               metadata={"steering": {"applied": False}})
        rows = self.store.messages(user["id"], conversation["id"])
        self.assertEqual([row["seq"] for row in rows], [0, 1])
        self.assertEqual(rows[1]["model_key"], "sarvam")
        self.assertEqual(rows[1]["metadata"], {"steering": {"applied": False}})
        self.assertTrue(self.store.owns_turn(user["id"], "a" * 32))
        self.assertEqual(self.store.get_conversation(user["id"], conversation["id"])["model_key"], "sarvam")

    def test_titles_auto_only_until_the_user_renames(self):
        user = self.store.create_user()
        conversation = self.store.create_conversation(user["id"])
        self.assertEqual((conversation["title"], conversation["title_source"]), ("New chat", "pending"))
        self.assertTrue(self.store.set_auto_title(user["id"], conversation["id"], "Work stress"))
        self.assertFalse(self.store.set_auto_title(user["id"], conversation["id"], "Again"))     # only once
        renamed = self.store.rename_conversation(user["id"], conversation["id"], "  My\n chat \t")
        self.assertEqual((renamed["title"], renamed["title_source"]), ("My chat", "user"))
        with self.assertRaises(ValueError):
            self.store.rename_conversation(user["id"], conversation["id"], "\n\t ")
        self.assertEqual(clean_title("x" * 500), "x" * 120)

    def test_deleting_a_conversation_or_a_user_cascades(self):
        user = self.store.create_user()
        token = self.store.create_session(user["id"])
        first = self.store.create_conversation(user["id"])
        second = self.store.create_conversation(user["id"])
        for conversation in (first, second):
            self.store.add_message(user["id"], conversation["id"], "user", "text")
        self.assertTrue(self.store.delete_conversation(user["id"], first["id"]))
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 1)
        self.store.delete_user(user["id"])
        with sqlite3.connect(self.path) as db:
            for table in ("users", "sessions", "conversations", "messages"):
                self.assertEqual(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0, table)
        self.assertIsNone(self.store.session_user(token))

    def test_retention_purges_only_expired_conversations(self):
        keep_forever, monthly = self.store.create_user(), self.store.create_user()
        self.store.set_consent(keep_forever["id"], store_history=True, retention_days=0)
        self.store.set_consent(monthly["id"], store_history=True, retention_days=30)
        old_a = self.store.create_conversation(keep_forever["id"])
        old_b = self.store.create_conversation(monthly["id"])
        fresh = self.store.create_conversation(monthly["id"])
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE conversations SET updated_at = ? WHERE id IN (?, ?)",
                       (time.time() - 31 * 86400, old_a["id"], old_b["id"]))
        removed = self.store.purge()
        self.assertEqual(removed["conversations"], 1)
        self.assertIsNotNone(self.store.get_conversation(keep_forever["id"], old_a["id"]))
        self.assertIsNone(self.store.get_conversation(monthly["id"], old_b["id"]))
        self.assertIsNotNone(self.store.get_conversation(monthly["id"], fresh["id"]))

    def test_consent_and_settings_validation(self):
        user = self.store.create_user()
        self.assertIsNone(user["consent_version"])
        user = self.store.set_consent(user["id"], store_history=True, retention_days=90)
        self.assertEqual((user["consent_version"], user["store_history"], user["retention_days"]),
                         (CONSENT_VERSION, True, 90))
        with self.assertRaises(ValueError):
            self.store.set_consent(user["id"], store_history=True, retention_days=12)
        self.assertFalse(self.store.update_settings(user["id"], store_history=False)["store_history"])

    def test_accounts(self):
        user = self.store.create_user()
        self.store.register(user["id"], "asha_k", "correct horse")
        self.assertEqual(self.store.authenticate("ASHA_K", "correct horse")["id"], user["id"])    # case-insensitive
        self.assertIsNone(self.store.authenticate("asha_k", "wrong password"))
        self.assertIsNone(self.store.authenticate("nobody", "correct horse"))
        other = self.store.create_user()
        for name, password in (("asha_k", "long enough"), ("x", "long enough"), ("valid_name", "short")):
            with self.assertRaises(ValueError):
                self.store.register(other["id"], name, password)
        with self.assertRaises(ValueError):
            self.store.register(user["id"], "second_name", "long enough")      # one username per account
        self.assertTrue(verify_password("pw-123456", hash_password("pw-123456")))
        self.assertFalse(verify_password("pw-123456", "plain-text"))


class EphemeralStoreTests(unittest.TestCase):
    def test_in_memory_store_writes_nothing_to_disk_and_expires(self):
        with tempfile.TemporaryDirectory() as folder:
            before = set(os.listdir(folder))
            store = ConversationStore(None)
            conversation = store.create_conversation("user-1")
            store.add_message("user-1", conversation["id"], "user", "private")
            self.assertFalse(store.persistent)
            self.assertEqual(set(os.listdir(folder)), before)
            self.assertEqual(len(store.messages("user-1", conversation["id"])), 1)
            self.assertEqual(store.messages("user-2", conversation["id"]), [])
            self.assertEqual(store.purge_inactive(time.time() + 1), 1)
            self.assertIsNone(store.get_conversation("user-1", conversation["id"]))

    def test_two_memory_stores_are_separate(self):
        first, second = ConversationStore(None), ConversationStore(None)
        conversation = first.create_conversation("u")
        self.assertIsNone(second.get_conversation("u", conversation["id"]))


if __name__ == "__main__":
    unittest.main()
