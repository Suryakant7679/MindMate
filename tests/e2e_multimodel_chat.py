#!/usr/bin/env python3
"""End-to-end check of persistent multi-model chat with real models (real server process, real GPU inference).

Starts ``backend.server`` on a free port with a fresh database (or uses ``--url``), then for the selected models:

* each model answers a stored conversation (lazy load, token streaming, model label, steering record);
* multi-turn continuity inside one conversation and a switch to the next model in the same conversation
  (the new model receives the earlier model's reply as context; each model is loaded once);
* the unsteered baseline switch (``steering: off``) on the first model;
* Stop Generation on a real reply (nothing released or stored, the conversation unlocked afterwards);
* every reply is steered by default (strength from the user's emotional state) and records its tone before and
  after steering; a crisis message is steered toward warmth;
* a second browser (user) cannot read the first user's conversation or Mood Tracker trace;
* the server is restarted on the same database: history and identity survive and the next turn sees the history.

    python3 tests/e2e_multimodel_chat.py --models llama,emollama
    python3 tests/e2e_multimodel_chat.py --models llama2 --server-args="--load-in-4bit --model llama2"
    python3 tests/e2e_multimodel_chat.py --models llama,emollama,llama2,sarvam     # needs the GPU to itself

Prints every reply and a pass/fail line per check; exits non-zero if any check failed.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(("  ok   " if ok else "  FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    return bool(ok)


class Browser:
    def __init__(self, base: str) -> None:
        self.base = base
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def call(self, method: str, path: str, payload: dict | None = None, timeout: float = 900):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"} if data else {})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def stream(self, conversation: str, payload: dict, on_event=None, timeout: float = 900):
        """POST a message and read the event stream; returns (status, [(event, data), ...])."""
        request = urllib.request.Request(f"{self.base}/api/conversations/{conversation}/messages",
                                         data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            response = self.opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            return error.code, [("error", json.loads(error.read() or b"{}"))]
        events, name, data = [], None, []
        with response:
            for raw in response:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    data.append(line[6:])
                elif line == "" and name:
                    item = (name, json.loads("\n".join(data)))
                    events.append(item)
                    if on_event:
                        on_event(*item)
                    name, data = None, []
        return response.status, events


def first(events, name):
    return next((data for kind, data in events if kind == name), None)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(port: int, db: Path, extra: list[str], log: Path) -> subprocess.Popen:
    command = [sys.executable, "-m", "backend.server", "--port", str(port), "--db-path", str(db), "--lazy-load", *extra]
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT, stdout=log.open("a"), stderr=subprocess.STDOUT)
    deadline = time.time() + 300
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"Server exited early; see {log}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2):
                return process
        except OSError:
            time.sleep(1)
    process.kill()
    raise SystemExit("Server did not start in time.")


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(60)
    except subprocess.TimeoutExpired:
        process.kill()


def turn(browser: Browser, conversation: str, text: str, model: str, **options):
    started = time.time()
    status, events = browser.stream(conversation, {"content": text, "model": model, **options})
    done = first(events, "done")
    reply = done["message"]["content"] if done else None
    print(f"\n### [{model}] {text}\n  reply ({time.time() - started:.1f}s): {reply!r}", flush=True)
    if done:
        steering = done["message"]["metadata"].get("steering") or {}
        print(f"  steering mode={steering.get('mode')} status={steering.get('status')} planned={steering.get('planned_alpha')} "
              f"applied={steering.get('applied_alpha')} panel={steering.get('panel')} neurons={steering.get('neurons')} "
              f"layers={len(steering.get('layers') or [])}")
    elif events:
        print(f"  events: {[name for name, _ in events]} {first(events, 'error')}")
    return status, events, done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="llama,emollama", help="Comma-separated registry keys, in order.")
    parser.add_argument("--server-args", default="", help="Extra arguments for backend.server.")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary database and log.")
    args = parser.parse_args()
    models = [key.strip() for key in args.models.split(",") if key.strip()]
    folder = Path(tempfile.mkdtemp(prefix="mindmate-e2e-"))
    db, log = folder / "chat.sqlite", folder / "server.log"
    port = free_port()
    extra = shlex.split(args.server_args)
    if "--model" not in extra:
        extra += ["--model", models[0]]
    server = start_server(port, db, extra, log)
    base = f"http://127.0.0.1:{port}"
    try:
        alice, bob = Browser(base), Browser(base)
        print("\n== consent and a stored conversation")
        check("consent required before storing", alice.call("POST", "/api/conversations", {})[0] == 403)
        alice.call("POST", "/api/me/consent", {"store_history": True, "retention_days": 30})
        bob.call("POST", "/api/me/consent", {"store_history": True, "retention_days": 30})
        status, created = alice.call("POST", "/api/conversations", {"model": models[0]})
        conversation = created["conversation"]["id"]
        check("conversation created", status == 201)

        first_model = models[0]
        print(f"\n== {first_model}: multi-turn")
        status, events, done = turn(alice, conversation, "My name is Asha and I am planning a trip to Goa with my sister next month.",
                                    first_model)
        check(f"{first_model}: reply delivered", bool(done and done["message"]["content"].strip()))
        check(f"{first_model}: reply labelled with its model", bool(done and done["message"]["model_key"] == first_model))
        phases = [data["phase"] for name, data in events if name == "status"]
        check(f"{first_model}: lazy load reported before the first reply", phases[:1] == ["loading_model"], ",".join(phases))
        deltas = [index for index, (name, _) in enumerate(events) if name == "delta"]
        check(f"{first_model}: reply streamed in several pieces", len(deltas) > 3, f"{len(deltas)} delta events")
        title = first(events, "title")
        check(f"{first_model}: automatic title", bool(title and title["title"]), (title or {}).get("title", ""))
        turn_id = done and done.get("emotion", {}) and done["emotion"]["turn_id"]
        status, events, done = turn(alice, conversation, "What did I say my name was, and where am I going?", first_model)
        loads_after_first = alice.call("GET", "/api/models")[1]["memory"]["loads"]
        check(f"{first_model}: consecutive turns reuse the loaded model", loads_after_first == {first_model: 1},
              json.dumps(loads_after_first))
        context = done["message"]["metadata"]["context"] if done else {}
        check(f"{first_model}: second turn sees the history", context.get("messages_sent") == 3, json.dumps(context))
        check(f"{first_model}: second turn uses the earlier content",
              bool(done and "asha" in done["message"]["content"].lower() and "goa" in done["message"]["content"].lower()))

        print(f"\n== {first_model}: every reply steered vs calibrated policy vs unsteered baseline (fresh conversations)")
        warm_text = "My best friend surprised me with a visit today after two years and I felt so loved and grateful."
        records, tones = {}, {}
        for mode in ("always", "policy", "off"):
            status, created = alice.call("POST", "/api/conversations", {"model": first_model, "steering": mode})
            status, events, done = turn(alice, created["conversation"]["id"], warm_text, first_model)
            records[mode] = done["message"]["metadata"].get("steering", {}) if done else {}
            tones[mode] = done["message"]["metadata"].get("tone") if done else None
        steered, tone = records["always"], tones["always"] or {}
        check(f"{first_model}: every-reply mode steers with a state-based strength",
              steered.get("mode") == "always" and (steered.get("planned_alpha") or 0) > 0
              and steered.get("status") in ("applied", "rejected"),
              f"status={steered.get('status')} alpha={steered.get('planned_alpha')} level={steered.get('state_level')} "
              f"panel={steered.get('panel')} neurons={steered.get('neurons')}")
        check(f"{first_model}: tone recorded before and after steering", bool(tone.get("before") and tone.get("after")),
              f"{(tone.get('before') or {}).get('emotion')} -> {(tone.get('after') or {}).get('emotion')} "
              f"change={tone.get('change')}")
        auto, off = records["policy"], records["off"]
        check(f"{first_model}: steering decided by the calibrated policy",
              auto.get("status") in ("applied", "rejected", "skipped", "not_planned"),
              f"status={auto.get('status')} planned={auto.get('planned_alpha')} applied={auto.get('applied_alpha')} "
              f"panel={auto.get('panel')} neurons={auto.get('neurons')} reason={auto.get('reason')}")
        check(f"{first_model}: baseline switch records alpha 0",
              off.get("mode") == "off" and off.get("applied_alpha") == 0.0 and off.get("planned_alpha") == 0.0
              and off.get("disabled_by") == "baseline", f"disabled_by={off.get('disabled_by')}")
        status, created = alice.call("POST", "/api/conversations", {"model": first_model})
        status, events, done = turn(alice, created["conversation"]["id"], "I feel so alone with all of this.", first_model)
        check(f"{first_model}: distress message answered", bool(done))

        for key in models[1:]:
            print(f"\n== switch to {key} in the same conversation")
            status, events, done = turn(alice, conversation, "Can you suggest one thing I should not forget to pack?", key)
            check(f"{key}: reply delivered", bool(done and done["message"]["content"].strip()))
            check(f"{key}: reply labelled with its model", bool(done and done["message"]["model_key"] == key))
            context = done["message"]["metadata"]["context"] if done else {}
            check(f"{key}: earlier turns of the other model are context", context.get("messages_sent", 0) >= 3, json.dumps(context))
            status, events, done = turn(alice, conversation, "And what was the name of my travel companion's relation to me?", key)
            loads = alice.call("GET", "/api/models")[1]["memory"]["loads"]
            check(f"{key}: consecutive turns reuse the loaded model", loads.get(key) == 1, json.dumps(loads))

        last = models[-1]
        print(f"\n== Stop Generation on {last}")
        progress = threading.Event()

        def watch(name, data):
            if name == "progress" and data.get("tokens", 0) >= 3:
                progress.set()
        result = {}
        worker = threading.Thread(target=lambda: result.update(zip(("status", "events"), alice.stream(
            conversation, {"content": "Please write a very long, detailed travel guide to Goa, at least 600 words.",
                           "model": last}, on_event=watch))))
        worker.start()
        got_progress = progress.wait(180)
        status, cancelled = alice.call("POST", f"/api/conversations/{conversation}/cancel", {})
        worker.join(300)
        names = [name for name, _ in result.get("events", [])]
        check("stop: progress reported while writing", got_progress)
        check("stop: generation cancelled", "cancelled" in names and cancelled.get("cancelled") is True, ",".join(names))
        shown = "".join(data["text"] for name, data in result.get("events", []) if name == "delta").strip()
        messages = alice.call("GET", f"/api/conversations/{conversation}")[1]["messages"]
        check("stop: only the text already shown is kept, marked as stopped",
              (messages[-1]["role"], messages[-1]["status"], messages[-1]["content"]) == ("assistant", "cancelled", shown)
              if shown else messages[-1]["role"] == "user", f"{len(shown)} characters shown")
        status, events, done = turn(alice, conversation, "Never mind, one short tip is enough.", last)
        check("stop: conversation usable again", bool(done))

        print("\n== crisis message (Emotion Reasoner)")
        status, created = alice.call("POST", "/api/conversations", {"model": first_model})
        crisis = created["conversation"]["id"]
        status, events, done = turn(alice, crisis, "I want to end my life", first_model)
        steering = done["message"]["metadata"].get("steering", {}) if done else {}
        check("crisis: crisis-support strategy, steered toward warmth",
              bool(done) and steering.get("strategy") == "crisis_support" and (steering.get("planned_alpha") or 0) > 0
              and steering.get("response_emotion") == "love",
              f"status={steering.get('status')} alpha={steering.get('planned_alpha')} panel={steering.get('panel')}")
        check("crisis: neutral conversation title", (first(events, "title") or {}).get("title") == "Support conversation")

        print("\n== isolation")
        check("other user sees no conversations", bob.call("GET", "/api/conversations")[1]["conversations"] == [])
        check("other user cannot open the conversation", bob.call("GET", f"/api/conversations/{conversation}")[0] == 404)
        if turn_id:
            check("owner can read the Mood Tracker trace", alice.call("GET", f"/api/emotion/trace/{turn_id}")[0] == 200)
            check("other user cannot read the trace", bob.call("GET", f"/api/emotion/trace/{turn_id}")[0] == 404)

        status, catalog = alice.call("GET", "/api/models")
        loads = catalog["memory"]["loads"]
        # One model at a time: a model is reloaded only when the conversation came back to it after another model.
        expected = {first_model: 2 if len(models) > 1 else 1, **{key: 1 for key in models[1:]}}
        check("loads only on model switches", loads == expected, json.dumps(loads))
        for row in catalog["models"]:
            if row["key"] in models:
                print(f"  {row['key']}: steering_status={row['steering_status']['status']} "
                      f"context_tokens={row['context_tokens']} footprint_gib={row['footprint_gib']}")

        print("\n== restart on the same database")
        stop_server(server)
        server = start_server(port, db, extra, log)
        listed = alice.call("GET", "/api/conversations")[1]["conversations"]
        check("history survives restart", conversation in [item["id"] for item in listed])
        status, events, done = turn(alice, conversation, "Thanks. What was my name again?", first_model)
        context = done["message"]["metadata"]["context"] if done else {}
        check("after restart the model sees the stored history", context.get("messages_total", 0) >= 7, json.dumps(context))
        check("after restart the reply uses the stored history", bool(done and "asha" in done["message"]["content"].lower()))
    finally:
        stop_server(server)
        failed = [name for name, ok, _ in RESULTS if not ok]
        print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed" + (f"; failed: {failed}" if failed else ""))
        print(f"server log: {log}" if args.keep or failed else "")
        if not args.keep and not failed:
            for path in folder.iterdir():
                path.unlink()
            folder.rmdir()
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
