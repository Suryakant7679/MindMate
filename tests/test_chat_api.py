"""Persistent multi-user chat over real HTTP: a real AppServer, store and SSE stream; scripted model runtimes.

Covers user isolation, persistence across a server restart, multi-turn context, switching models inside one
conversation, token streaming, cancellation, consent / unsaved mode, accounts and deletion.
"""
import http.client
import http.cookiejar
import json
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from backend.chat_api import ServerState
from backend.config import MODEL_REGISTRY
from backend.generation_control import current
from backend.models import ModelManager
from backend.server import AppServer, RequestHandler, attach_runtime, model_loaded, model_unloading
from backend.store import Store


class ScriptedRuntime:
    """Streams a reply naming the model (word by word, honouring the request's Stop) and writes short titles."""

    block = False                   # set to make chat generations wait for cancellation

    def __init__(self, key):
        self.key, self.model_id = key, MODEL_REGISTRY[key].config.model_id
        self.lock = threading.Lock()
        self.model = object()
        self._emotion_panels = {}
        self.context_tokens = 4096
        self.seen = []
        self.generating = threading.Event()

    def generate(self, messages, max_new_tokens=1024, deterministic=True, temperature=0.7, system_prompt=None, **_):
        if messages[-1]["content"].startswith("Write a short, neutral title"):
            return {"response": "Title: Feeling stressed at work."}
        return {"response": f"reply from {self.key} after {len(messages)} messages"}

    def stream(self, messages, max_new_tokens=1024, deterministic=True, temperature=0.7, system_prompt=None, **_):
        self.seen.append([dict(message) for message in messages])
        control = current()
        if ScriptedRuntime.block:
            # A long reply: a few words, then more only until the request is stopped (like a stopping criterion).
            self.generating.set()
            deadline = time.monotonic() + 10
            for word in ("this", " is", " a", " long", " answer"):
                if control:
                    control.step()
                yield word
            while not (control and control.cancelled) and time.monotonic() < deadline:
                if control:
                    control.step()
                time.sleep(0.01)
            return
        for word in f"reply from {self.key} after {len(messages)} messages".split(" "):
            if control:
                control.step()
            yield word + " "


class Client:
    """A browser: its own cookie jar (identity)."""

    def __init__(self, base):
        self.base = base
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))

    def call(self, method, path, payload=None, headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={**({"Content-Type": "application/json"} if data else {}),
                                                  **(headers or {})})
        try:
            with self.opener.open(request, timeout=30) as response:
                body = response.read()
                kind = response.headers.get("Content-Type", "")
                return response.status, (parse_sse(body.decode()) if "event-stream" in kind else json.loads(body or b"{}"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def send(self, conversation_id, content, **options):
        return self.call("POST", f"/api/conversations/{conversation_id}/messages", {"content": content, **options})

    def consent(self, store_history=True):
        return self.call("POST", "/api/me/consent", {"store_history": store_history, "retention_days": 30})


def parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        name, data = None, []
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        if name:
            events.append((name, json.loads("\n".join(data))))
    return events


def event(events, name):
    return next((data for kind, data in events if kind == name), None)


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.db = Path(self.folder.name) / "chat.sqlite"
        self.runtimes = {}
        patcher = patch("backend.models.is_downloaded", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        ScriptedRuntime.block = False
        self.start()

    def start(self, max_loaded=1):
        server = AppServer(("127.0.0.1", 0), RequestHandler)
        server.runtime, server.emotion_services = None, {}
        server.research_compare = server.allow_overrides = server.telemetry = False
        ServerState.install(server, Store(self.db), default_model="llama")

        def factory(key, allow_download=False, **options):
            self.runtimes[key] = ScriptedRuntime(key)
            return self.runtimes[key]
        server.models = ModelManager(factory=factory, max_loaded=max_loaded, free_memory=lambda: None,
                                     on_load=lambda k, r: model_loaded(server, k, r),
                                     on_unload=lambda k, r: model_unloading(server, k, r),
                                     on_change=lambda k, r: attach_runtime(server, k, r) if r is not None else None)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.server, self.thread = server, thread
        self.base = f"http://127.0.0.1:{server.server_address[1]}"
        self.addCleanup(self.stop, server, thread)

    @staticmethod
    def stop(server, thread):
        server.shutdown()
        server.server_close()
        thread.join(5)

    def conversation(self, client, **options):
        status, payload = client.call("POST", "/api/conversations", options)
        self.assertEqual(status, 201, payload)
        return payload["conversation"]["id"]

    # ------------------------------------------------------------------ tests

    def test_consent_is_required_before_anything_is_stored(self):
        client = Client(self.base)
        status, me = client.call("GET", "/api/me")
        self.assertEqual(status, 200)
        self.assertTrue(me["consent"]["required"])
        status, payload = client.call("POST", "/api/conversations", {})
        self.assertEqual((status, payload.get("consent_required")), (403, True))
        client.consent()
        self.assertFalse(client.call("GET", "/api/me")[1]["consent"]["required"])

    def test_full_turn_stream_title_and_stored_metadata(self):
        client = Client(self.base)
        client.consent()
        conversation = self.conversation(client, model="llama")
        status, events = client.send(conversation, "Work has been really stressful lately.")
        self.assertEqual(status, 200)
        names = [name for name, _ in events]
        self.assertEqual(names[0], "start")
        self.assertIn("done", names)
        phases = [data["phase"] for name, data in events if name == "status"]
        self.assertEqual(phases, ["loading_model", "generating"])          # first use loads lazily
        pieces = [data["text"] for name, data in events if name == "delta"]
        self.assertGreater(len(pieces), 3)                                 # streamed as generated, word by word
        done = event(events, "done")
        self.assertEqual("".join(pieces).strip(), done["message"]["content"])
        self.assertEqual(done["message"]["model_key"], "llama")
        self.assertNotIn("safety", done)
        self.assertEqual(event(events, "title")["title"], "Feeling stressed at work")
        status, payload = client.call("GET", f"/api/conversations/{conversation}")
        assistant = payload["messages"][1]
        self.assertEqual((assistant["role"], assistant["model_key"]), ("assistant", "llama"))
        self.assertEqual(assistant["metadata"]["steering"]["status"], "not_run")
        self.assertIn("context", assistant["metadata"])
        self.assertEqual(assistant["metadata"]["generation"]["generated_tokens"], len(pieces))

    def test_multi_turn_context_and_model_switch_in_one_conversation(self):
        client = Client(self.base)
        client.consent()
        conversation = self.conversation(client, model="llama")
        client.send(conversation, "My name is Asha and I failed my exam.")
        client.send(conversation, "What should I do now?")
        self.assertEqual(len(self.runtimes["llama"].seen[-1]), 3)        # user, assistant, user
        status, events = client.send(conversation, "Do you remember my name?", model="emollama")
        self.assertEqual(status, 200)
        history = self.runtimes["emollama"].seen[-1]
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "user", "assistant", "user"])
        self.assertIn("Asha", history[0]["content"])
        self.assertIn("reply from llama", history[1]["content"])        # the other model's reply is context
        messages = client.call("GET", f"/api/conversations/{conversation}")[1]["messages"]
        self.assertEqual([m["model_key"] for m in messages if m["role"] == "assistant"], ["llama", "llama", "emollama"])
        self.assertEqual(self.server.models.loads, {"llama": 1, "emollama": 1})   # each model loaded once
        listed = client.call("GET", "/api/conversations")[1]["conversations"]
        self.assertEqual(listed[0]["model_key"], "emollama")

    def test_users_cannot_see_or_touch_each_others_conversations(self):
        alice, bob = Client(self.base), Client(self.base)
        alice.consent()
        bob.consent()
        conversation = self.conversation(alice)
        alice.send(conversation, "private message")
        self.assertEqual(bob.call("GET", "/api/conversations")[1]["conversations"], [])
        for method, suffix, payload in (("GET", "", None), ("PATCH", "", {"title": "x"}), ("DELETE", "", None),
                                        ("POST", "/messages", {"content": "hi"}), ("POST", "/cancel", None)):
            status, _ = bob.call(method, f"/api/conversations/{conversation}{suffix}", payload)
            self.assertEqual(status, 404, (method, suffix))
        self.assertEqual(len(alice.call("GET", f"/api/conversations/{conversation}")[1]["messages"]), 2)

    def test_history_and_identity_survive_a_restart(self):
        client = Client(self.base)
        client.consent()
        conversation = self.conversation(client)
        client.send(conversation, "Remember this after a restart.")
        self.stop(self.server, self.thread)
        self.start()                                        # new process state, same database
        client.base = self.base
        listed = client.call("GET", "/api/conversations")[1]["conversations"]
        self.assertEqual([item["id"] for item in listed], [conversation])
        status, events = client.send(conversation, "Still there?")
        self.assertEqual(status, 200)
        self.assertEqual(len(self.runtimes["llama"].seen[-1]), 3)        # the restored history reached the model
        self.assertEqual(event(events, "status")["phase"], "loading_model")

    def test_rename_delete_and_cross_site_protection(self):
        client = Client(self.base)
        client.consent()
        conversation = self.conversation(client)
        status, payload = client.call("PATCH", f"/api/conversations/{conversation}", {"title": "Exam worries"})
        self.assertEqual((status, payload["conversation"]["title"]), (200, "Exam worries"))
        status, _ = client.call("PATCH", f"/api/conversations/{conversation}", {"title": "x"},
                                headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(client.call("DELETE", f"/api/conversations/{conversation}")[0], 200)
        self.assertEqual(client.call("GET", f"/api/conversations/{conversation}")[0], 404)

    def test_stop_generation_keeps_only_what_was_shown(self):
        client = Client(self.base)
        client.consent()
        conversation = self.conversation(client)
        client.send(conversation, "warm up the model")
        ScriptedRuntime.block = True
        result = {}
        worker = threading.Thread(target=lambda: result.update(zip(("status", "events"), client.send(conversation, "long one"))))
        worker.start()
        self.assertTrue(self.runtimes["llama"].generating.wait(5))
        other = client.send(conversation, "second message at the same time")
        self.assertEqual(other[0], 409)
        self.assertEqual(client.call("POST", f"/api/conversations/{conversation}/cancel")[1], {"cancelled": True})
        worker.join(10)
        names = [name for name, _ in result["events"]]
        self.assertIn("cancelled", names)
        self.assertNotIn("done", names)
        shown = "".join(data["text"] for name, data in result["events"] if name == "delta")
        self.assertEqual(shown, "this is a long answer")
        cancelled = event(result["events"], "cancelled")["message"]
        self.assertEqual((cancelled["status"], cancelled["content"]), ("cancelled", shown))
        messages = client.call("GET", f"/api/conversations/{conversation}")[1]["messages"]
        self.assertEqual([(m["role"], m["status"]) for m in messages],
                         [("user", "complete"), ("assistant", "complete"), ("user", "complete"), ("assistant", "cancelled")])
        ScriptedRuntime.block = False
        status, events = client.call("POST", f"/api/conversations/{conversation}/messages", {})      # regenerate
        self.assertEqual(status, 200)
        self.assertIn("long one", self.runtimes["llama"].seen[-1][-1]["content"])
        self.assertNotIn(conversation, self.server.generations)

    def test_unsaved_mode_keeps_conversations_off_disk(self):
        client = Client(self.base)
        client.consent(store_history=False)
        conversation = self.conversation(client)
        client.send(conversation, "do not store this")
        listed = client.call("GET", "/api/conversations")[1]["conversations"]
        self.assertEqual((listed[0]["id"], listed[0]["saved"]), (conversation, False))
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0], 0)

    def test_accounts_reach_history_from_another_browser_and_delete_everything(self):
        laptop = Client(self.base)
        laptop.consent()
        conversation = self.conversation(laptop)
        laptop.send(conversation, "hello")
        status, me = laptop.call("POST", "/api/auth/register", {"username": "asha", "password": "a long password"})
        self.assertEqual((status, me["user"]["username"]), (200, "asha"))
        phone = Client(self.base)
        self.assertEqual(phone.call("POST", "/api/auth/login", {"username": "asha", "password": "nope-nope"})[0], 401)
        status, _ = phone.call("POST", "/api/auth/login", {"username": "asha", "password": "a long password"})
        self.assertEqual(status, 200)
        self.assertEqual([c["id"] for c in phone.call("GET", "/api/conversations")[1]["conversations"]], [conversation])
        status, export = phone.call("GET", "/api/me/export")
        self.assertEqual(export["saved_conversations"][0]["messages"][0]["content"], "hello")
        self.assertEqual(phone.call("DELETE", "/api/me")[0], 200)
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(laptop.call("GET", "/api/conversations")[1]["conversations"], [])   # old session is gone

    def test_new_conversations_steer_every_reply_by_default(self):
        client = Client(self.base)
        client.consent()
        status, created = client.call("POST", "/api/conversations", {})
        self.assertEqual(created["conversation"]["steering_mode"], "always")
        conversation = created["conversation"]["id"]
        for mode in ("policy", "off", "always"):
            status, payload = client.call("PATCH", f"/api/conversations/{conversation}", {"steering": mode})
            self.assertEqual((status, payload["conversation"]["steering_mode"]), (200, mode))
        self.assertEqual(client.call("PATCH", f"/api/conversations/{conversation}", {"steering": "auto"})[0], 400)

    def test_traces_of_an_unloaded_model_stay_readable(self):
        from backend.server import model_unloading

        class Service:
            def __init__(self, runtime):
                self.runtime = runtime

            def has_turn_id(self, turn_id):
                return turn_id == "b" * 32
        runtime = ScriptedRuntime("llama")
        self.server.emotion_services["llama"] = service = Service(runtime)
        model_unloading(self.server, "llama", runtime)
        self.assertNotIn("llama", self.server.emotion_services)
        self.assertEqual(self.server.retired_services, [service])
        handler = RequestHandler.__new__(RequestHandler)
        handler.server = self.server
        self.assertIs(handler._service_for_turn("b" * 32), service)

    def test_models_endpoint_lists_capabilities_and_memory(self):
        status, payload = Client(self.base).call("GET", "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["models"]), 4)
        self.assertEqual(payload["memory"]["max_loaded"], 1)
        self.assertIn("steering_status", payload["models"][0])


if __name__ == "__main__":
    unittest.main()
