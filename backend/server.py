#!/usr/bin/env python3
"""Serve the local Llama or Sarvam API for the MindMate chatbot."""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import re
import hashlib
import hmac
import secrets
from http.cookies import SimpleCookie, CookieError
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .config import DEFAULT_MODEL, MODEL_REGISTRY
from .llama import LlamaRuntime
from .models import ModelManager, ModelUnavailable
from .sarvam import SarvamRuntime
from .runtime import HuggingFaceRuntime
from chatbot_feature.safety import SafetyAgent
from chatbot_feature.safety.runtime import SafetyRuntimeModel
from chatbot_feature.safety.session import SessionStore, session_key

from .validation import validate_chat_request, validate_steering_alpha, validate_steering_emotion


IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class AppServer(ThreadingHTTPServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.safety_sessions = SessionStore(float(os.getenv("MINDMATE_SAFETY_RETENTION_SECONDS", "1800")))
        self.resource_owners = SessionStore(float(os.getenv("MINDMATE_SAFETY_RETENTION_SECONDS", "1800")), max_sessions=4096)
        self.session_secret = secrets.token_bytes(32)

    runtime: HuggingFaceRuntime | None
    agent_service: Any | None = None
    emotion_service: Any | None = None
    # Research options, kept across model swaps (see main()).
    research_compare: bool = False
    allow_overrides: bool = False
    telemetry: bool = False
    # Loads and swaps chat models on request (None with --no-model).
    models: Any | None = None


class RequestHandler(BaseHTTPRequestHandler):
    server: AppServer
    frontend_root = Path(__file__).resolve().parent.parent / "frontend" / "dist"

    def _owner(self) -> str:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            value = cookie["mindmate_session"].value if "mindmate_session" in cookie else ""
            token, signature = value.split(".", 1)
            expected = hmac.new(self.server.session_secret, token.encode(), hashlib.sha256).hexdigest()
            if not re.fullmatch(r"[0-9a-f]{64}", token) or not hmac.compare_digest(signature, expected):
                raise ValueError("Invalid session")
        except (ValueError, KeyError, CookieError):
            token = secrets.token_hex(32)
            signature = hmac.new(self.server.session_secret, token.encode(), hashlib.sha256).hexdigest()
            self._session_cookie = f"mindmate_session={token}.{signature}; HttpOnly; SameSite=Strict; Path=/"
        return token

    def end_headers(self):
        if getattr(self, "_session_cookie", None):
            self.send_header("Set-Cookie", self._session_cookie)
        super().end_headers()

    def _approve(self, draft: str) -> str:
        reply, validation = asyncio.run(self._safety_agent.finalize(
            self._safety_message, draft, self._safety_result, country=self._safety_country))
        self._safety_validation = validation.model_dump()
        return reply

    def _send_safety(self, response: str, streaming: bool) -> None:
        if streaming:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Steering-Alpha", "0")
            self.send_header("X-Safety-Route", self._safety_result.policy.route)
            self.end_headers()
            self.wfile.write(response.encode("utf-8"))
        else:
            self.send_json(HTTPStatus.OK, {"response": response, "model": getattr(self.server.runtime, "model_id", None),
                "safety": {"risk_level": self._safety_result.policy.risk_level,
                           "route": self._safety_result.policy.route, "validation": self._safety_validation}})

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _emotion_service(self) -> Any | None:
        """The Emotion Reasoner pipeline of the loaded model, if it has one."""
        # Built per loaded model, and only for a model whose own steering analysis exists.
        if not isinstance(self.server.runtime, HuggingFaceRuntime):
            return None
        return getattr(self.server, "emotion_service", None)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            health = self.server.agent_service.manager.health() if self.server.agent_service else {}
            self.send_json(
                HTTPStatus.OK,
                {
                    "ready": self.server.runtime is not None,
                    "model": self.server.runtime.model_id if self.server.runtime else None,
                    "model_key": self.server.models.active if isinstance(self.server.models, ModelManager) else None,
                    "mcp_enabled": self.server.agent_service is not None,
                    "mcp_servers": {name: item.status for name, item in health.items()},
                },
            )
            return
        if self.path == "/api/models":
            manager = self.server.models if isinstance(self.server.models, ModelManager) else None
            self.send_json(HTTPStatus.OK, {"models": manager.catalog() if manager else [],
                                           "active": manager.active if manager else None})
            return
        path = unquote(urlsplit(self.path).path)
        if path.startswith("/api/emotion/") and not path.startswith("/api/emotion/trace/"):
            self._emotion_get(path)
            return
        if path.startswith("/api/emotion/trace/"):
            service = self._emotion_service()
            turn_id = path.rsplit("/", 1)[1]
            if self.server.resource_owners.get(turn_id) != self._owner():
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "No finished turn with this id."})
                return
            try:
                analysis = service.trace(turn_id) if service and re.fullmatch(r"[0-9a-f]{32}", turn_id) else None
            except Exception as exc:
                print(f"Emotion analysis error: {type(exc).__name__}")
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Emotion analysis failed."})
                return
            if analysis is None:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "No finished turn with this id."})
            else:
                self.send_json(HTTPStatus.OK, analysis)
            return
        if path == "/" or path == "/chatbot" or path.startswith("/assets/"):
            root = self.frontend_root.resolve()
            target = (root / ("index.html" if path in ("/", "/chatbot") else path.lstrip("/"))).resolve()
            if target.is_relative_to(root) and target.is_file():
                body = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(target)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-cache" if target.name == "index.html" else "public, max-age=31536000, immutable")
                self.end_headers()
                self.wfile.write(body)
                return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found. Build the MindMate frontend with npm run build inside frontend/."})

    def _emotion_get(self, path: str) -> None:
        """Research read endpoints of the Mood Tracker (conversation trajectory, neuron records, evidence)."""
        service = self._emotion_service()
        if path == "/api/emotion/models":          # saved results of every model; needs no loaded model
            from .research import model_evaluation
            try:
                self.send_json(HTTPStatus.OK, model_evaluation(getattr(service, "tag", None)))
            except Exception as exc:
                print(f"Model evaluation error: {type(exc).__name__}")
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Saved results could not be read."})
            return
        if service is None:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "The loaded model has no Emotion Reasoner pipeline."})
            return
        try:
            if path == "/api/emotion/research":
                self.send_json(HTTPStatus.OK, service.overview())
                return
            kind, _, identifier = path[len("/api/emotion/"):].partition("/")
            if kind in ("conversation", "neurons") and self.server.resource_owners.get(identifier) != self._owner():
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Nothing recorded under this id."})
                return
            if kind == "conversation" and IDENTIFIER.fullmatch(identifier):
                view = service.conversation_view(identifier)
                if view["trajectory"]["turns"]:
                    self.send_json(HTTPStatus.OK, view)
                    return
            if kind == "panel" and re.fullmatch(r"[A-Za-z0-9_:+-]{1,64}", identifier):
                panel = service.panel_view(identifier)
                if panel is not None:
                    self.send_json(HTTPStatus.OK, panel)
                    return
            if kind == "neurons" and re.fullmatch(r"[0-9a-f]{32}", identifier):
                record = service.neuron_record(identifier)
                if record is not None:
                    self.send_json(HTTPStatus.OK, record)
                    return
        except Exception as exc:
            print(f"Emotion research error: {type(exc).__name__}")
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Emotion research data could not be read."})
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Nothing recorded under this id."})

    def _emotion_post(self) -> None:
        """Voluntary feedback and the research-compare switch."""
        service = self._emotion_service()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("Request is empty or too large.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object.")
            if service is None:
                self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "The loaded model has no Emotion Reasoner pipeline."})
                return
            if self.path == "/api/emotion/settings":
                enabled = payload.get("research_compare")
                if not isinstance(enabled, bool):
                    raise ValueError("research_compare must be true or false.")
                self.server.research_compare = service.research_compare = enabled
                self.send_json(HTTPStatus.OK, service.overview()["settings"])
                return
            conversation, index = payload.get("conversation_id"), payload.get("turn_index")
            if not isinstance(conversation, str) or not IDENTIFIER.fullmatch(conversation):
                raise ValueError("A valid conversation_id is required.")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 200:
                raise ValueError("turn_index must be the position of the user turn in the conversation.")
            if self.server.resource_owners.get(conversation) != self._owner():
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "No recorded conversation."})
                return
            ratings = {}
            for name in ("mood_rating", "helpfulness"):
                value = payload.get(name)
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                        raise ValueError(f"{name} must be a whole number from 1 to 5.")
                    ratings[name] = value
            if not ratings:
                raise ValueError("Send mood_rating and/or helpfulness (1 to 5).")
            if not service.research.has_turn(conversation, index):
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "No recorded turn at this position."})
                return
            self.send_json(HTTPStatus.OK, service.add_feedback(conversation, index, **ratings))
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc) or "Invalid request data."})
        except Exception as exc:
            print(f"Emotion feedback error: {type(exc).__name__}")
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "The feedback could not be stored."})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/safety/clear":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request length")
                data = json.loads(self.rfile.read(length))
                owner = self._owner()
                for name in ("conversation_id", "session_id"):
                    identifier = data.get(name)
                    if isinstance(identifier, str) and IDENTIFIER.fullmatch(identifier):
                        key = session_key(owner, identifier)
                        self.server.safety_sessions.delete(key)
                        if self.server.agent_service:
                            self.server.agent_service.sessions.delete(key)
                self.send_json(HTTPStatus.OK, {"cleared": True})
            except (ValueError, TypeError, AttributeError):
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid context identifier"})
            return
        if self.path in ("/api/emotion/feedback", "/api/emotion/settings"):
            self._emotion_post()
            return
        if self.path not in ("/api/chat", "/api/chat/stream", "/api/workspace/chat", "/api/agent/chat"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        streaming = self.path == "/api/chat/stream"
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid content length."})
            return
        if length <= 0 or length > 262_144:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Request is empty or too large."})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            messages = validate_chat_request(payload)
            owner = self._owner()
            country = payload.get("country")
            if country is not None and (not isinstance(country, str) or not re.fullmatch(r"[A-Z]{2}", country)):
                raise ValueError("country must be an explicitly selected ISO country code.")
            if self.path == "/api/workspace/chat":
                prompt = payload.get("system_prompt")
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
                    raise ValueError("Workspace system_prompt must contain 1 to 16000 characters.")
            if self.path == "/api/agent/chat":
                if not isinstance(payload.get("session_id"), str) or not IDENTIFIER.fullmatch(payload["session_id"]):
                    raise ValueError("A valid session_id is required for agent chat.")
            self._safety_country = country
            self._safety_message = messages[-1]["content"]
            conversation = payload.get("conversation_id", "direct")
            if not isinstance(conversation, str) or not IDENTIFIER.fullmatch(conversation):
                raise ValueError("Invalid conversation_id.")
            key = session_key(owner, conversation)
            if payload.get("clear_safety_context") is True:
                self.server.safety_sessions.delete(key)
            self._safety_agent = SafetyAgent(SafetyRuntimeModel(self.server.runtime))
            self._safety_result = asyncio.run(self._safety_agent.assess(
                self._safety_message, messages[:-1], self.server.safety_sessions.get(key)))
            self.server.safety_sessions.put(key, self._safety_result.policy)
            if self._safety_result.policy.route != "normal":
                reply, validation = asyncio.run(self._safety_agent.response(
                    self._safety_message, self._safety_result, country))
                self._safety_validation = validation.model_dump()
                self._send_safety(reply, streaming)
                return
            requested_model = payload.get("model")
            if requested_model is not None and isinstance(self.server.models, ModelManager):
                # The chat composer's model selector: load the selected model first (never downloads).
                if not isinstance(requested_model, str):
                    raise ValueError("model must be the key of a listed model.")
                self.server.models.activate(requested_model)
            manual_alpha = validate_steering_alpha(payload) if "steering_alpha" in payload else None
            requested_emotion = validate_steering_emotion(payload)
            conversation_id = payload.get("conversation_id")
            if conversation_id is not None and (not isinstance(conversation_id, str) or not IDENTIFIER.fullmatch(conversation_id)):
                raise ValueError("Invalid conversation_id.")
            generation_options: dict[str, Any] = {}
            steering_details = turn_id = plan = None
            service = self._emotion_service()
            if service is not None and self.path in ("/api/chat", "/api/chat/stream"):
                # The Emotion Reasoner Agent chooses strategy, response emotion and steering strength.
                turn_id, plan = service.begin_turn(
                    messages, alpha_override=manual_alpha,
                    emotion_override=None if requested_emotion in ("auto", "mirror") else requested_emotion,
                    conversation_id=session_key(owner, conversation_id or "direct"))
                self.server.resource_owners.put(turn_id, owner)
                self.server.resource_owners.put(session_key(owner, conversation_id or "direct"), owner)
            elif manual_alpha and self.path != "/api/agent/chat":
                if not getattr(self.server.runtime, "_emotion_panels", None):
                    raise ValueError("Emotion steering is not available for the selected model: "
                                     "its neuron analysis has not been run.")
                detected = self.server.runtime.detect_emotion(messages[-1]["content"])
                target = self.server.runtime.steering_target(detected["emotion"], requested_emotion)
                generation_options = {"steering_alpha": manual_alpha, "steering_emotion": target}
                steering_details = {**detected, "steering_emotion": target, "steering_alpha": manual_alpha,
                                    "panel_neurons": self.server.runtime.panel_size(target)}
            if self.path == "/api/workspace/chat":
                prompt = payload.get("system_prompt")
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
                    raise ValueError("Workspace system_prompt must contain 1 to 16000 characters.")
                generation_options["system_prompt"] = prompt.strip()
            if self.path == "/api/agent/chat":
                session_id = payload.get("session_id")
                user_id = owner  # Client user_id is not an authenticated identity.
                if not isinstance(session_id, str) or not IDENTIFIER.fullmatch(session_id):
                    raise ValueError("A valid session_id is required for agent chat.")
                if not isinstance(user_id, str) or not IDENTIFIER.fullmatch(user_id):
                    raise ValueError("Invalid user_id.")
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc) or "Invalid request data."})
            return
        if self.server.runtime is None:
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "The chat model is not loaded."})
            return
        if self.path == "/api/agent/chat":
            if self.server.agent_service is None:
                self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "MCP agent features are disabled."})
                return
            try:
                result = asyncio.run(self.server.agent_service.run(
                    messages[-1]["content"], session_id=session_id, user_id=user_id, country=country
                ))
                result["model"] = self.server.runtime.model_id
                service = self._emotion_service()
                if service is not None and result.get("safety", {}).get("route") == "normal":
                    # Same conversation record, trajectory and comparison as direct chat.
                    try:
                        analysis = service.track_agent_turn(
                            messages, result, session_key(owner, conversation_id or session_id))
                        self.server.resource_owners.put(analysis["turn_id"], owner)
                        self.server.resource_owners.put(analysis["conversation_id"], owner)
                        result.update({key: analysis[key] for key in ("turn_id", "conversation_id", "turn_index",
                                                                      "trajectory", "summary", "comparison")})
                    except Exception as exc:        # research records never block a reply
                        print(f"Agent turn tracking error: {type(exc).__name__}")
                self.send_json(HTTPStatus.OK, result)
            except Exception as exc:
                print(f"Agent generation error: {type(exc).__name__}")
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Agent generation failed. Check the server terminal for details."})
            return
        if turn_id:
            # Reasoner path.  Without planned steering the reply is streamed token by token.  With it,
            # the unsteered reply and the steered candidate are decoded together and the steered one
            # is sent only if it passes the safeguards, so the text follows the decision.
            def emotion_payload(decision: dict[str, Any]) -> dict[str, Any]:
                return {"emotion": plan["user_emotion"], "confidence": plan["confidence"],
                        "scores": plan["probe_scores"], "detector": plan["detector"],
                        "steering_emotion": plan["response_emotion"], "steering_alpha": decision["applied_alpha"]}
            if not streaming:
                try:
                    analysis = service.respond(turn_id, max_new_tokens=1024, safety_filter=self._approve)
                except Exception as exc:
                    print(f"Generation error: {type(exc).__name__}")
                    self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                                   {"error": "Generation failed. Check the server terminal for details."})
                    return
                self.send_json(HTTPStatus.OK, {
                    "response": analysis["reply"], "model": self.server.runtime.model_id, "turn_id": turn_id,
                    "conversation_id": analysis.get("conversation_id"),
                    "latency_seconds": analysis["latency_seconds"], "emotion_plan": plan,
                    "steering_decision": analysis["decision"], "comparison": analysis.get("comparison"),
                    "emotion_detection": emotion_payload(analysis["decision"]),
                })
                return
            started = False

            def send_headers(decision: dict[str, Any]) -> None:
                nonlocal started
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("X-Detected-Emotion", plan["user_emotion"])
                self.send_header("X-Emotion-Confidence", str(plan["confidence"]))
                if plan["response_emotion"]:
                    self.send_header("X-Steering-Emotion", plan["response_emotion"])
                self.send_header("X-Steering-Alpha", str(decision["applied_alpha"]))
                self.send_header("X-Steering-Status", decision["status"])
                self.send_header("X-Turn-Id", turn_id)
                self.end_headers()
                started = True

            def send_text(piece: str) -> None:
                self.wfile.write(piece.encode("utf-8"))
                self.wfile.flush()
            try:
                analysis = service.respond(turn_id, max_new_tokens=1024, safety_filter=self._approve)
                send_headers(analysis["decision"])
                send_text(analysis["reply"])
            except Exception as exc:
                print(f"Streaming error: {type(exc).__name__}")
                if not started:
                    self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                                   {"error": "Generation failed. Check the server terminal for details."})
            return
        if streaming:
            try:
                stream = self.server.runtime.stream(
                    messages, max_new_tokens=4096 if isinstance(self.server.runtime, SarvamRuntime) else 1024,
                    temperature=0.7, deterministic=True, **generation_options
                )
                approved = self._approve("".join(stream))
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                if steering_details:
                    self.send_header("X-Detected-Emotion", steering_details["emotion"])
                    self.send_header("X-Emotion-Confidence", str(steering_details["confidence"]))
                    if steering_details.get("steering_emotion"):
                        self.send_header("X-Steering-Emotion", steering_details["steering_emotion"])
                        self.send_header("X-Steering-Alpha", str(steering_details["steering_alpha"]))
                        self.send_header("X-Steering-Neuron-Count", str(steering_details["panel_neurons"]))
                self.end_headers()
                self.wfile.write(approved.encode("utf-8"))
                self.wfile.flush()
            except Exception as exc:
                print(f"Streaming error: {type(exc).__name__}")
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Generation failed before an approved reply was available."})
            return
        try:
            result = self.server.runtime.generate(
                messages, max_new_tokens=4096 if isinstance(self.server.runtime, SarvamRuntime) else 1024,
                temperature=0.7, deterministic=True,
                **generation_options
            )
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # Keep internal traces out of the browser.
            print(f"Generation error: {type(exc).__name__}")
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "Generation failed. Check the server terminal for details."},
            )
            return
        result["response"] = self._approve(result["response"])
        result["safety_validation"] = self._safety_validation
        if steering_details:
            result["emotion_detection"] = steering_details
        self.send_json(HTTPStatus.OK, result)


def attach_runtime(server: AppServer, key: str, runtime: Any) -> None:
    """Point the server at a newly loaded model (or at none while models are being swapped)."""
    server.runtime = runtime
    server.emotion_service = None
    if runtime is not None and (isinstance(runtime, LlamaRuntime) or getattr(runtime, "_emotion_panels", None)):
        # Emotion Reasoner + neuron steering.  Other models chat without it until their own
        # neuron analysis (panels, policy) has been produced by the experiment pipeline.
        from .emotion_service import EmotionSteeringService
        from .research import ResearchStore, telemetry_enabled

        tag = getattr(runtime, "results_tag", None) or "llama31_8b"
        server.emotion_service = EmotionSteeringService(
            runtime, research=ResearchStore(tag, persist=server.telemetry and telemetry_enabled()),
            allow_overrides=server.allow_overrides, research_compare=server.research_compare)
    if server.agent_service is not None and runtime is not None:
        from chatbot_feature.agents import EmotionReasonerNode, LocalRuntimeModel

        service = server.emotion_service
        # The agent model shares the emotion service, so agent replies use the same steering
        # selection and safeguards as direct chat.
        server.agent_service.model = LocalRuntimeModel(runtime, emotion_service=service)
        server.agent_service.emotion_reasoner = (
            EmotionReasonerNode(service.agent, service.probe_scores) if service is not None else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), default=DEFAULT_MODEL,
                        help="Model loaded at start (default: llama). Others can be selected in the chat.")
    parser.add_argument("--model-id", help="Override the selected preset with a Hugging Face ID or local checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Local inference API port (default: 8080)",
    )
    parser.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=None,
        help="Override preset quantization (Sarvam: 4-bit, others: full precision)",
    )
    parser.add_argument("--allow-download", action="store_true",
                        help="Let the server download a selected model that is not on disk. Without it, "
                             "only models already in the local Hugging Face cache can be used.")
    parser.add_argument("--no-model", action="store_true", help="Run the API without loading a model for connectivity checks")
    parser.add_argument("--research-compare", action="store_true",
                        help="Research mode: on turns the policy does not steer, also generate a steered candidate for "
                             "comparison in the Mood Tracker. It is never sent. Can be switched in the Mood Tracker too.")
    parser.add_argument("--allow-overrides", action="store_true",
                        help="Accept steering_alpha / steering_emotion in chat requests (experiments only). Never on "
                             "safety turns; the forced reply still has to pass the reply safeguards.")
    parser.add_argument("--research-telemetry", action="store_true",
                        help="Opt in to persistent raw research conversation records; obtain participant consent first.")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="Keep per-turn research records in memory only (the default).")
    args = parser.parse_args()

    try:
        server = AppServer((args.host, args.port), RequestHandler)
    except OSError as exc:
        if exc.errno == 98:
            raise SystemExit(
                f"Port {args.port} is already in use. Start with another port, for example: "
                f"python3 -m backend.server --port {args.port + 1}"
            ) from exc
        raise
    try:
        server.runtime = None
        server.research_compare = args.research_compare or os.getenv("MINDMATE_RESEARCH_COMPARE", "").lower() in {"1", "true", "yes"}
        server.allow_overrides = args.allow_overrides
        server.telemetry = args.research_telemetry and not args.no_telemetry
        if not args.no_model:
            if os.getenv("MCP_ENABLED", "false").lower() in {"1", "true", "yes"}:
                from chatbot_feature.graph import AgentChatService
                from chatbot_feature.mcp_connector import MCPManager

                server.agent_service = AgentChatService(None, MCPManager())      # model attached below
            server.models = ModelManager(allow_download=args.allow_download,
                                         on_change=lambda key, runtime: attach_runtime(server, key, runtime))
            options = {name: value for name, value in (("model_id", args.model_id), ("load_in_4bit", args.load_in_4bit))
                       if value is not None}
            try:
                server.models.activate(args.model, **options)
            except ModelUnavailable as exc:
                raise SystemExit(str(exc)) from exc
        print(f"MindMate API: http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        if server.agent_service is not None:
            asyncio.run(server.agent_service.close())
        server.server_close()


if __name__ == "__main__":
    main()
