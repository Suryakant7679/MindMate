#!/usr/bin/env python3
"""Serve the MindMate chatbot: multi-model local inference, persistent multi-user conversations and the research API."""

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
import threading
import time
from http.cookies import SimpleCookie, CookieError
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .chat_api import ChatApiMixin, ServerState
from .config import DEFAULT_MODEL, MODEL_REGISTRY
from .llama import LlamaRuntime
from .models import ModelBusy, ModelManager, ModelUnavailable
from .sarvam import SarvamRuntime
from .runtime import HuggingFaceRuntime
from chatbot_feature.session import SessionStore, session_key

from .validation import validate_chat_request, validate_steering_alpha, validate_steering_emotion


IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class AppServer(ThreadingHTTPServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Which session owns an Emotion Reasoner turn or research conversation (expiring, in memory).
        self.resource_owners = SessionStore(float(os.getenv("MINDMATE_RESOURCE_RETENTION_SECONDS", "1800")), max_sessions=4096)
        self.session_secret = secrets.token_bytes(32)

    runtime: HuggingFaceRuntime | None
    agent_service: Any | None = None
    emotion_service: Any | None = None
    # Emotion Reasoner service of every loaded model (built when the model loads, dropped when it unloads).
    emotion_services: dict[str, Any]
    # Research options, kept across model swaps (see main()).
    research_compare: bool = False
    allow_overrides: bool = False
    telemetry: bool = False
    # Loads and swaps chat models on request (None with --no-model).
    models: Any | None = None
    secure_cookies: bool = False
    # Default steering mode of new conversations and the direct chat endpoints (--steering-mode).
    steering_mode: str = "always"

    def mark_active(self, key: str, runtime: Any) -> None:
        """Make ``key`` the default model of the research endpoints and the agent mode (most recently used)."""
        if self.runtime is not runtime:
            attach_runtime(self, key, runtime)


class RequestHandler(ChatApiMixin, BaseHTTPRequestHandler):
    server: AppServer
    frontend_root = Path(__file__).resolve().parent.parent / "frontend" / "dist"

    def _owner(self) -> str:
        """The trusted identity of this browser: the stored session's user, or (without a database) a signed cookie."""
        if self._chat_api_enabled():
            return self._current_user()["id"]
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

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    @property
    def _runtime(self) -> Any:
        """The runtime this request generates with: the leased one, else the server's default (most recent) model."""
        return self.__dict__.get("_leased_runtime", self.server.runtime)

    def _lease_model(self, key: str | None) -> Any:
        """Hold ``key`` loaded for the rest of this request (released in ``do_POST``) and make it the default."""
        models = self.server.models
        if not isinstance(models, ModelManager) or key is None:
            return self.server.runtime
        runtime = models.acquire(key)
        self.__dict__.setdefault("_leases", []).append(key)
        self._leased_runtime = runtime
        if self.server.runtime is not runtime:
            attach_runtime(self.server, key, runtime)
        return runtime

    def _legacy_steering_mode(self, payload: dict[str, Any]) -> str:
        """``steering`` from the request, else the server's default (``policy`` for a server without one)."""
        if payload.get("steering") is not None:
            return self._steering(payload["steering"])
        mode = getattr(self.server, "steering_mode", None)
        return mode if mode in ("always", "policy", "off") else "policy"

    def _emotion_service(self, model: str | None = None) -> Any | None:
        """The Emotion Reasoner pipeline of this request's model (or of the loaded model ``model``), if it has one."""
        # Built per loaded model, and only for a model whose own steering analysis exists.
        services = getattr(self.server, "emotion_services", None)
        if model and isinstance(services, dict):
            return services.get(model)
        runtime = self._runtime
        if not isinstance(runtime, HuggingFaceRuntime):
            return None
        if isinstance(services, dict):
            for service in services.values():
                if getattr(service, "runtime", None) is runtime:
                    return service
        service = getattr(self.server, "emotion_service", None)
        return service if service is not None and getattr(service, "runtime", runtime) is runtime else None

    def _all_services(self) -> list[Any]:
        """Services of the loaded models, then those kept from unloaded models (newest first)."""
        services = getattr(self.server, "emotion_services", None)
        retired = getattr(self.server, "retired_services", None)
        return [*(services.values() if isinstance(services, dict) else ()), *(retired if isinstance(retired, list) else ())]

    def _service_for_turn(self, turn_id: str) -> Any | None:
        services = self._all_services()
        if services:
            for service in services:
                try:
                    if service.has_turn_id(turn_id):
                        return service
                except AttributeError:
                    pass
        return self._emotion_service()

    def _owns_turn(self, turn_id: str) -> bool:
        owner = self._owner()
        if self.server.resource_owners.get(turn_id) == owner:
            return True
        return self._chat_api_enabled() and self.server.store.owns_turn(owner, turn_id)

    def do_GET(self) -> None:  # noqa: N802
        split = urlsplit(self.path)
        path = unquote(split.path)
        query = parse_qs(split.query)
        if self.chat_api("GET", path):
            return
        if path == "/api/status":
            health = self.server.agent_service.manager.health() if self.server.agent_service else {}
            manager = self.server.models if isinstance(self.server.models, ModelManager) else None
            self.send_json(
                HTTPStatus.OK,
                {
                    # Ready to chat: a model is loaded, or the pool can load the selected one on demand.
                    "ready": self.server.runtime is not None or manager is not None,
                    "model": self.server.runtime.model_id if self.server.runtime else None,
                    "model_key": manager.active if manager else None,
                    "loaded_models": sorted(manager.loaded()) if manager else [],
                    "persistence": self._chat_api_enabled(),
                    "mcp_enabled": self.server.agent_service is not None,
                    "mcp_servers": {name: item.status for name, item in health.items()},
                },
            )
            return
        if path == "/api/models":
            manager = self.server.models if isinstance(self.server.models, ModelManager) else None
            self.send_json(HTTPStatus.OK, {"models": manager.catalog() if manager else [],
                                           "active": manager.active if manager else None,
                                           "default": getattr(self.server, "default_model", None) or DEFAULT_MODEL,
                                           "memory": manager.memory() if manager else None})
            return
        if path.startswith("/api/emotion/") and not path.startswith("/api/emotion/trace/"):
            self._emotion_get(path, (query.get("model") or [None])[0])
            return
        if path.startswith("/api/emotion/trace/"):
            turn_id = path.rsplit("/", 1)[1]
            if not re.fullmatch(r"[0-9a-f]{32}", turn_id) or not self._owns_turn(turn_id):
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "No finished turn with this id."})
                return
            service = self._service_for_turn(turn_id)
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

    def _emotion_get(self, path: str, model: str | None = None) -> None:
        """Research read endpoints of the Mood Tracker (conversation trajectory, neuron records, evidence).

        ``model`` (query parameter) selects a loaded model's service; the default is the most recently used model."""
        service = self._emotion_service(model) if model in MODEL_REGISTRY else self._emotion_service()
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
            if kind == "conversation" and self.server.resource_owners.get(identifier) != self._owner():
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Nothing recorded under this id."})
                return
            if kind == "neurons":
                if not re.fullmatch(r"[0-9a-f]{32}", identifier) or not self._owns_turn(identifier):
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "Nothing recorded under this id."})
                    return
                service = self._service_for_turn(identifier) or service
            if kind == "conversation" and IDENTIFIER.fullmatch(identifier):
                candidates = [service, *[item for item in self._all_services() if item is not service]]
                for candidate in candidates:
                    view = candidate.conversation_view(identifier)
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

    def do_PATCH(self) -> None:  # noqa: N802
        if not self.chat_api("PATCH", unquote(urlsplit(self.path).path)):
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_DELETE(self) -> None:  # noqa: N802
        if not self.chat_api("DELETE", unquote(urlsplit(self.path).path)):
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        if self.chat_api("POST", unquote(urlsplit(self.path).path)):
            return
        try:
            self._legacy_post()
        finally:
            for key in self.__dict__.pop("_leases", []):
                self.server.models.release(key)
            self.__dict__.pop("_leased_runtime", None)

    def _legacy_post(self) -> None:
        """Endpoints whose client sends the whole history (research tools, agent mode, conversations not stored)."""
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
            if self.path == "/api/workspace/chat":
                prompt = payload.get("system_prompt")
                if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
                    raise ValueError("Workspace system_prompt must contain 1 to 16000 characters.")
            if self.path == "/api/agent/chat":
                if not isinstance(payload.get("session_id"), str) or not IDENTIFIER.fullmatch(payload["session_id"]):
                    raise ValueError("A valid session_id is required for agent chat.")
            conversation_id = payload.get("conversation_id")
            if conversation_id is not None and (not isinstance(conversation_id, str) or not IDENTIFIER.fullmatch(conversation_id)):
                raise ValueError("Invalid conversation_id.")
            requested_model = payload.get("model")
            if isinstance(self.server.models, ModelManager):
                # The chat composer's model selector: lease the selected model (loading it if needed, never
                # downloading) for the whole request, so no other request can unload it mid-reply.
                if requested_model is not None and not isinstance(requested_model, str):
                    raise ValueError("model must be the key of a listed model.")
                default = getattr(self.server, "default_model", None)
                self._lease_model(requested_model or self.server.models.active
                                  or (default if isinstance(default, str) else DEFAULT_MODEL))
            manual_alpha = validate_steering_alpha(payload) if "steering_alpha" in payload else None
            requested_emotion = validate_steering_emotion(payload)
            generation_options: dict[str, Any] = {}
            steering_details = turn_id = plan = None
            service = self._emotion_service()
            if service is not None and self.path in ("/api/chat", "/api/chat/stream"):
                # The Emotion Reasoner Agent chooses strategy, response emotion and steering strength.
                turn_id, plan = service.begin_turn(
                    messages, alpha_override=manual_alpha,
                    emotion_override=None if requested_emotion in ("auto", "mirror") else requested_emotion,
                    conversation_id=session_key(owner, conversation_id or "direct"),
                    steering_mode=self._legacy_steering_mode(payload))
                self.server.resource_owners.put(turn_id, owner)
                self.server.resource_owners.put(session_key(owner, conversation_id or "direct"), owner)
            elif manual_alpha and self.path != "/api/agent/chat":
                if not getattr(self._runtime, "_emotion_panels", None):
                    raise ValueError("Emotion steering is not available for the selected model: "
                                     "its neuron analysis has not been run.")
                detected = self._runtime.detect_emotion(messages[-1]["content"])
                target = self._runtime.steering_target(detected["emotion"], requested_emotion)
                generation_options = {"steering_alpha": manual_alpha, "steering_emotion": target}
                steering_details = {**detected, "steering_emotion": target, "steering_alpha": manual_alpha,
                                    "panel_neurons": self._runtime.panel_size(target)}
            if self.path == "/api/workspace/chat":
                generation_options["system_prompt"] = payload["system_prompt"].strip()
            if self.path == "/api/agent/chat":
                session_id = payload["session_id"]
                user_id = owner  # Client user_id is not an authenticated identity.
                if not isinstance(user_id, str) or not IDENTIFIER.fullmatch(user_id):
                    raise ValueError("Invalid user_id.")
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc) or "Invalid request data."})
            return
        except ModelBusy as exc:
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            return
        if self._runtime is None:
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "The chat model is not loaded."})
            return
        if self.path == "/api/agent/chat":
            if self.server.agent_service is None:
                self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "MCP agent features are disabled."})
                return
            try:
                result = asyncio.run(self.server.agent_service.run(
                    messages[-1]["content"], session_id=session_id, user_id=user_id))
                result["model"] = self._runtime.model_id
                service = self._emotion_service()
                if service is not None:
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
                    analysis = service.respond(turn_id, max_new_tokens=1024)
                except Exception as exc:
                    print(f"Generation error: {type(exc).__name__}")
                    self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                                   {"error": "Generation failed. Check the server terminal for details."})
                    return
                self.send_json(HTTPStatus.OK, {
                    "response": analysis["reply"], "model": self._runtime.model_id, "turn_id": turn_id,
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
                service.respond(turn_id, max_new_tokens=1024, on_decision=send_headers, on_text=send_text)
            except Exception as exc:
                print(f"Streaming error: {type(exc).__name__}")
                if not started:
                    self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                                   {"error": "Generation failed. Check the server terminal for details."})
            return
        if streaming:
            started = False
            try:
                stream = self._runtime.stream(
                    messages, max_new_tokens=4096 if isinstance(self._runtime, SarvamRuntime) else 1024,
                    temperature=0.7, deterministic=True, **generation_options
                )
                for chunk in stream:
                    if not started:
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "text/plain; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("X-Accel-Buffering", "no")
                        if steering_details:
                            self.send_header("X-Detected-Emotion", steering_details["emotion"])
                            self.send_header("X-Emotion-Confidence", str(steering_details["confidence"]))
                            if steering_details.get("steering_emotion"):
                                self.send_header("X-Steering-Emotion", steering_details["steering_emotion"])
                                self.send_header("X-Steering-Alpha", str(steering_details["steering_alpha"]))
                                self.send_header("X-Steering-Neuron-Count", str(steering_details["panel_neurons"]))
                        self.end_headers()
                        started = True
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except Exception as exc:
                print(f"Streaming error: {type(exc).__name__}")
                if not started:
                    self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                                   {"error": "Generation failed. Check the server terminal for details."})
            return
        try:
            result = self._runtime.generate(
                messages, max_new_tokens=4096 if isinstance(self._runtime, SarvamRuntime) else 1024,
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
        if steering_details:
            result["emotion_detection"] = steering_details
        self.send_json(HTTPStatus.OK, result)


def _services(server: Any) -> dict[str, Any]:
    services = vars(server).get("emotion_services")
    if not isinstance(services, dict):
        services = {}
        server.emotion_services = services
    return services


def build_emotion_service(server: Any, runtime: Any) -> Any | None:
    """The Emotion Reasoner + neuron-steering service of one loaded runtime (None for a model without panels)."""
    if runtime is None or not (isinstance(runtime, LlamaRuntime) or getattr(runtime, "_emotion_panels", None)):
        # Other models chat without it until their own neuron analysis (panels, policy) has been produced.
        return None
    from .emotion_service import EmotionSteeringService
    from .research import ResearchStore, telemetry_enabled

    tag = getattr(runtime, "results_tag", None) or "llama31_8b"
    service = EmotionSteeringService(
        runtime, research=ResearchStore(tag, persist=server.telemetry and telemetry_enabled()),
        allow_overrides=server.allow_overrides, research_compare=server.research_compare)
    runtime.steering_enabled = service.steering_enabled
    return service


def model_loaded(server: Any, key: str, runtime: Any) -> None:
    """A model finished loading: build its own Emotion Reasoner service (its panels, policy and evidence)."""
    services = _services(server)
    service = build_emotion_service(server, runtime)
    if service is not None:
        services[key] = service
    else:
        services.pop(key, None)


RETIRED_SERVICES = 8


def retired_services(server: Any) -> list[Any]:
    retired = vars(server).get("retired_services")
    if not isinstance(retired, list):
        retired = []
        server.retired_services = retired
    return retired


def model_unloading(server: Any, key: str, runtime: Any) -> None:
    """A model is being unloaded.  Its service is kept (without weights) so Mood Tracker traces and timelines of its
    turns stay readable; reading them needs no inference.  The newest few unloaded services are kept."""
    services = _services(server)
    if getattr(services.get(key), "runtime", None) is runtime:
        retired = retired_services(server)
        retired.insert(0, services.pop(key))
        del retired[RETIRED_SERVICES:]
    if server.runtime is runtime:
        server.runtime, server.emotion_service = None, None


def attach_runtime(server: AppServer, key: str, runtime: Any) -> None:
    """Make ``runtime`` the server's default (most recently used) model, or none while models are being swapped.

    The default model serves the research endpoints that name no model and the LangGraph agent mode.  The
    Emotion Reasoner service of each loaded model is built once and reused (``emotion_services``)."""
    server.runtime = runtime
    services = _services(server)
    if runtime is None:
        server.emotion_service = None
    else:
        service = services.get(key)
        if service is None or getattr(service, "runtime", None) is not runtime:
            service = build_emotion_service(server, runtime)
            if service is not None:
                services[key] = service
            else:
                services.pop(key, None)
        server.emotion_service = service
    if server.agent_service is not None and runtime is not None:
        from chatbot_feature.agents import EmotionReasonerNode, LocalRuntimeModel

        service = server.emotion_service
        # The agent model shares the emotion service, so agent replies use the same steering
        # selection and safeguards as direct chat.
        server.agent_service.model = LocalRuntimeModel(runtime, emotion_service=service)
        server.agent_service.emotion_reasoner = (
            EmotionReasonerNode(service.agent, service.probe_scores) if service is not None else None)


def maintenance(server: AppServer, stop: threading.Event, interval: float = 60.0) -> None:
    """Background upkeep: retention purge (hourly), unsaved-conversation expiry, idle model unloading."""
    from .store import EPHEMERAL_IDLE_SECONDS

    last_purge = 0.0
    while not stop.wait(interval):
        try:
            if time.monotonic() - last_purge >= 3600:
                last_purge = time.monotonic()
                removed = server.store.purge()
                expired = server.ephemeral.purge_inactive(time.time() - EPHEMERAL_IDLE_SECONDS)
                if any(removed.values()) or expired:
                    print(f"Retention purge: {removed}, unsaved conversations expired: {expired}")
            if isinstance(server.models, ModelManager):
                server.models.unload_idle()
        except Exception as exc:          # noqa: BLE001 - upkeep never stops the server
            print(f"Maintenance error: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), default=DEFAULT_MODEL,
                        help=f"Model loaded at start (default: {DEFAULT_MODEL}). Others can be selected in the chat.")
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
                             "turns the Emotion Reasoner reads as a crisis; the forced reply still has to pass the reply "
                             "safeguards.")
    parser.add_argument("--steering-mode", choices=("always", "policy", "off"),
                        default=os.getenv("MINDMATE_STEERING_MODE", "always"),
                        help="Default for new conversations and the direct chat endpoints: always (every reply steered, "
                             "strength from the user's emotional state; the default), policy (the calibrated policy "
                             "decides) or off (unsteered baseline). Each conversation can change it.")
    parser.add_argument("--research-telemetry", action="store_true",
                        help="Opt in to persistent raw research conversation records; obtain participant consent first.")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="Keep per-turn research records in memory only (the default).")
    # Multi-model inference.
    parser.add_argument("--max-loaded-models", type=int, default=int(os.getenv("MINDMATE_MAX_LOADED_MODELS", "1")),
                        help="Models kept loaded at once (default 1: one model on the GPU at a time). Raise it only when "
                             "the GPU holds several; free GPU memory is checked before every load either way.")
    parser.add_argument("--gpu-headroom-gib", type=float, default=float(os.getenv("MINDMATE_GPU_HEADROOM_GIB", "2")),
                        help="GPU memory kept free for activations and the KV cache when deciding what fits (default 2).")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=os.getenv("MINDMATE_DEVICE", "auto"),
                        help="Model placement: auto (accelerate device_map, the default), cuda, or cpu (slow; testing).")
    parser.add_argument("--gpu-max-memory-gib", type=float, default=None,
                        help="Cap the GPU memory one model may take; layers beyond it are offloaded to the CPU.")
    parser.add_argument("--idle-unload-minutes", type=float, default=float(os.getenv("MINDMATE_IDLE_UNLOAD_MINUTES", "0")),
                        help="Unload a model unused for this long, freeing the GPU (default 0: keep it loaded).")
    parser.add_argument("--lazy-load", action="store_true",
                        help="Do not load --model at start; load each model when a message first needs it.")
    # Conversations.
    parser.add_argument("--db-path", default=os.getenv("MINDMATE_DB_PATH"),
                        help="SQLite file for users and stored conversations "
                             "(default chatbot_feature/data/mindmate_chat.sqlite, created with mode 0600).")
    parser.add_argument("--secure-cookies", action="store_true",
                        help="Mark the session cookie Secure (when MindMate is served over HTTPS).")
    args = parser.parse_args()
    if args.max_loaded_models < 1:
        parser.error("--max-loaded-models must be at least 1")
    if not args.allow_download:
        from .models import stay_offline
        stay_offline()                    # models, tokenizers and the reply classifier come from the local cache only

    try:
        server = AppServer((args.host, args.port), RequestHandler)
    except OSError as exc:
        if exc.errno == 98:
            raise SystemExit(
                f"Port {args.port} is already in use. Start with another port, for example: "
                f"python3 -m backend.server --port {args.port + 1}"
            ) from exc
        raise
    stop = threading.Event()
    try:
        server.runtime = None
        server.emotion_services = {}
        server.research_compare = args.research_compare or os.getenv("MINDMATE_RESEARCH_COMPARE", "").lower() in {"1", "true", "yes"}
        server.allow_overrides = args.allow_overrides
        server.telemetry = args.research_telemetry and not args.no_telemetry
        server.secure_cookies = args.secure_cookies
        server.steering_mode = args.steering_mode
        from .store import DEFAULT_PATH, Store
        db_path = Path(args.db_path) if args.db_path else DEFAULT_PATH
        ServerState.install(server, Store(db_path), default_model=args.model)
        print(f"Conversations: {db_path}")
        threading.Thread(target=maintenance, args=(server, stop), daemon=True, name="mindmate-maintenance").start()
        if not args.no_model:
            if os.getenv("MCP_ENABLED", "false").lower() in {"1", "true", "yes"}:
                from chatbot_feature.graph import AgentChatService
                from chatbot_feature.mcp_connector import MCPManager

                server.agent_service = AgentChatService(None, MCPManager())      # model attached below
            placement: dict[str, Any] = {}
            if args.device != "auto":
                placement["device_map"] = args.device
            if args.gpu_max_memory_gib:
                placement["max_memory"] = {0: f"{args.gpu_max_memory_gib}GiB", "cpu": "200GiB"}
            options = {name: value for name, value in (("model_id", args.model_id), ("load_in_4bit", args.load_in_4bit))
                       if value is not None}
            server.models = ModelManager(
                allow_download=args.allow_download, max_loaded=args.max_loaded_models,
                gpu_headroom_gib=args.gpu_headroom_gib, placement=placement,
                idle_unload_seconds=args.idle_unload_minutes * 60,
                overrides={args.model: options} if options else None,
                on_load=lambda key, runtime: model_loaded(server, key, runtime),
                on_unload=lambda key, runtime: model_unloading(server, key, runtime),
                on_change=lambda key, runtime: attach_runtime(server, key, runtime) if runtime is not None else None)
            print(f"Models: up to {args.max_loaded_models} loaded at once; placement {placement or {'device_map': 'auto'}}.")
            if not args.lazy_load:
                try:
                    server.models.activate(args.model)
                except ModelUnavailable as exc:
                    raise SystemExit(str(exc)) from exc
        print(f"MindMate API: http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        stop.set()
        if server.agent_service is not None:
            asyncio.run(server.agent_service.close())
        server.server_close()


if __name__ == "__main__":
    main()
