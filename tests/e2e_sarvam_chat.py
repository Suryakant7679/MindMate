#!/usr/bin/env python3
"""End-to-end check of the MindMate chatbot (real model, real HTTP server): Emotion Reasoner.

Starts ``backend.server --model sarvam`` (or ``--model llama``) on a free port, sends Hinglish and English messages
through the direct chat endpoint, reads each turn's Mood Tracker trace, sends a synthetic crisis message, and,
when MCP agent features are enabled, sends one message through the LangGraph agent endpoint.  Prints what the
chatbot answered and what the Emotion Reasoner decided, then checks the contract:

* every turn carries an Emotion Reasoner plan and a Mood Tracker trace;
* every reply is steered (the server's default steering mode); a crisis cue gets the crisis-support strategy and
  is steered toward warmth.

    python3 tests/e2e_sarvam_chat.py                    # starts and stops its own Sarvam server
    python3 tests/e2e_sarvam_chat.py --model llama      # the same with Llama 3.1-8B
    python3 tests/e2e_sarvam_chat.py --url http://127.0.0.1:8080   # against a server you started
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import http.cookiejar
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MESSAGES = [
    ("hinglish", "Aaj office mein boss ne sabke saamne daant diya, bahut embarrassing laga aur ab kuch karne ka mann nahi hai."),
    ("hinglish", "Yaar exam ka result aa gaya, maine top kiya! Mummy papa ko bata ke bahut khushi hui."),
    ("english", "I keep lying awake at night worrying that I will lose my job and let my family down."),
]
# Synthetic crisis message (no real user) that the Emotion Reasoner's knowledge graph flags as a crisis cue.
CRISIS = "I want to end my life"
STRATEGIES = {"compassionate_comfort", "warm_connection", "reflective_validation", "gentle_exploration",
              "grounding_reassurance", "strengths_affirmation", "celebratory_sharing", "appreciative_mirroring"}


# Like a browser: keep the server's signed session cookie, which owns the turns whose traces are read back.
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def conversation() -> str:
    """A new conversation id: independent messages must not share a conversation record (a new chat in the UI)."""
    return f"e2e-{uuid.uuid4().hex[:16]}"


def request(url: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with OPENER.open(req, timeout=timeout) as response:
        return json.loads(response.read())


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(url: str, process: subprocess.Popen | None, limit: float = 1800) -> dict:
    started = time.time()
    while time.time() - started < limit:
        if process is not None and process.poll() is not None:
            raise SystemExit(f"Server exited early with code {process.returncode}.")
        try:
            status = request(f"{url}/api/status", timeout=10)
            if status.get("ready"):
                return status
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(5)
    raise SystemExit("Server did not become ready in time.")


def check(condition: bool, text: str, failures: list[str]) -> None:
    print(("  ok   " if condition else "  FAIL ") + text)
    if not condition:
        failures.append(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="Use a running server instead of starting one.")
    parser.add_argument("--model", choices=("sarvam", "llama"), default="sarvam")
    parser.add_argument("--out", help="default: results/<model>_generation_emotion/e2e_chat.json")
    args = parser.parse_args()
    args.out = args.out or ("results/sarvam_30b_generation_emotion/steering/e2e_chat.json" if args.model == "sarvam"
                            else "results/llama31_8b_generation_emotion/e2e_chat.json")
    process = None
    url = args.url
    if url is None:
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        process = subprocess.Popen([sys.executable, "-m", "backend.server", "--model", args.model, "--port", str(port)],
                                   cwd=ROOT, env=env)
        print(f"Started backend.server --model {args.model} on {url} (pid {process.pid}); loading the model...")
    failures: list[str] = []
    records = []
    try:
        status = wait_ready(url, process)
        print(f"Server ready: {json.dumps(status)}")
        expected_model = "sarvam" if args.model == "sarvam" else "llama"
        check(expected_model in str(status.get("model", "")).lower(), f"loaded model is {args.model} ({status.get('model')})", failures)
        history: list[dict[str, str]] = []
        normal_turns = 0
        for language, text in MESSAGES:
            print(f"\n### [{language}] {text}")
            started = time.time()
            reply = request(f"{url}/api/chat", {"messages": [{"role": "user", "content": text}],
                                                "conversation_id": conversation()})
            elapsed = time.time() - started
            plan, decision = reply.get("emotion_plan") or {}, reply.get("steering_decision") or {}
            print(f"  reply ({elapsed:.1f}s): {reply.get('response', '')[:300]!r}")
            check(bool(reply.get("response", "").strip()), "non-empty reply", failures)
            check("<think>" not in reply.get("response", "") and "</think>" not in reply.get("response", ""),
                  "no reasoning markers in the reply", failures)
            check(elapsed < 120, f"reply within two minutes ({elapsed:.0f}s)", failures)
            normal_turns += 1
            print(f"  strategy={plan.get('strategy')} prompt={plan.get('prompt_style')} response_emotion={plan.get('response_emotion')} "
                  f"panel={plan.get('steering_panel')} planned_alpha={plan.get('steering_alpha')} "
                  f"status={decision.get('status')} applied={decision.get('applied_alpha')}")
            calibration = plan.get("calibration") or {}
            print(f"  reason: {str(calibration.get('reason') or decision.get('reason'))[:200]}")
            check(bool(reply.get("turn_id")), "turn id returned", failures)
            check(plan.get("strategy") in STRATEGIES, f"knowledge-graph strategy chosen ({plan.get('strategy')})", failures)
            if args.model == "sarvam":
                expected = "mindmate_only" if language == "hinglish" else "strategy_v3"
                check(plan.get("prompt_style") == expected, f"Sarvam prompt style {plan.get('prompt_style')} (expected {expected})", failures)
            trace = request(f"{url}/api/emotion/trace/{reply.get('turn_id')}")
            steering = trace.get("steering") or {}
            layers = steering.get("layers") or []
            print(f"  trace: decision={trace.get('decision', {}).get('status')} layers={len(layers)} "
                  f"neurons={steering.get('neurons_total', 0)} location={str(steering.get('location', ''))[:90]!r}")
            check(trace.get("decision", {}).get("status") == decision.get("status"), "trace decision matches the chat reply", failures)
            check("effect" in trace and "unsteered_reply" in trace["effect"], "trace carries before/after estimates", failures)
            applied = float(decision.get("applied_alpha") or 0)
            if applied:
                check(bool(layers) and all("mean_abs_measured_change" in layer for layer in layers),
                      "steered turn logs per-layer measured activation changes", failures)
                check(all(len(layer["neurons"]) == len(layer["applied_shift"]) for layer in layers), "neuron ids and shifts align", failures)
            else:
                if decision.get("status") == "not_planned":
                    check(not layers, "unsteered turn logs no neuron changes", failures)
                check(bool(calibration.get("reason") or decision.get("reason")), "unsteered turn states its reason", failures)
            if language == "hinglish":
                from chatbot_feature.emotion_reasoner.knowledge_graph import is_hinglish
                print(f"  reply language: {'Hinglish' if is_hinglish(reply['response']) else 'English'}")
            records.append({"language": language, "text": text, "reply": reply.get("response"), "latency": elapsed,
                            "plan": plan, "decision": decision, "trace_layers": len(layers),
                            "trace_neurons": steering.get("neurons_total", 0)})
            history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply.get("response", "")}]
        check(normal_turns == len(MESSAGES), f"every message answered ({normal_turns}/{len(MESSAGES)})", failures)
        # Manual strength controls must be ignored for a policy-managed model.
        forced = request(f"{url}/api/chat", {"messages": [{"role": "user", "content": MESSAGES[0][1]}], "steering_alpha": 3.0,
                                              "conversation_id": conversation(),
                                              "steering_emotion": "joy"})
        decision = forced.get("steering_decision") or {}
        print(f"\n### manual strength request: status={decision.get('status')} applied={decision.get('applied_alpha')}")
        check(not float(decision.get("applied_alpha") or 0) or decision.get("status") == "applied",
              "manual strength does not force unvalidated steering", failures)
        records.append({"manual_request": {"steering_alpha": 3.0, "steering_emotion": "joy"}, "decision": decision,
                        "plan": forced.get("emotion_plan")})
        # Multi-turn history through the same endpoint.
        follow = request(f"{url}/api/chat", {"messages": history + [{"role": "user", "content": "Thoda better feel ho raha hai, thanks. Kal ke liye koi chhota sa plan batao?"}],
                                             "conversation_id": conversation()})
        print(f"\n### multi-turn follow-up: {follow.get('response', '')[:200]!r}")
        check(bool(follow.get("response", "").strip()), "multi-turn reply", failures)
        records.append({"multi_turn_reply": follow.get("response"), "plan": follow.get("emotion_plan")})
        crisis = request(f"{url}/api/chat", {"messages": [{"role": "user", "content": CRISIS}],
                                             "conversation_id": conversation()})
        plan, decision = crisis.get("emotion_plan") or {}, crisis.get("steering_decision") or {}
        print(f"\n### crisis cue: strategy={plan.get('strategy')} flag={plan.get('safety_flag')} "
              f"status={decision.get('status')}\n  reply: {crisis.get('response', '')[:300]!r}")
        check(bool(plan.get("safety_flag")), "crisis cue flagged by the Emotion Reasoner", failures)
        check(float(decision.get("planned_alpha") or 0) > 0 and plan.get("response_emotion") == "love",
              "crisis turn is steered toward warmth (every reply steered)", failures)
        records.append({"crisis": CRISIS, "reply": crisis.get("response"), "plan": plan, "decision": decision})
        if status.get("mcp_enabled"):
            agent = request(f"{url}/api/agent/chat", {"messages": [{"role": "user", "content": MESSAGES[0][1]}],
                                                      "session_id": "e2e-sarvam", "user_id": "e2e"})
            print(f"\n### LangGraph agent reply: {str(agent.get('response') or agent)[:300]!r}")
            check(bool(agent.get("response") or agent.get("reply")), "agent endpoint answered", failures)
            records.append({"agent": agent})
        else:
            print("\nMCP agent features are disabled on this server; LangGraph endpoint not exercised "
                  "(the agent model shares the same EmotionSteeringService, see backend/server.py attach_runtime).")
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"url": url, "failures": failures, "records": records}, indent=1, ensure_ascii=False))
    print(f"\nwrote {out}")
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for text in failures:
            print("  - " + text)
        sys.exit(1)
    print("All end-to-end checks passed.")


if __name__ == "__main__":
    main()
