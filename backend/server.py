#!/usr/bin/env python3
"""Serve the local Llama or Sarvam API for the MindMate chatbot."""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import re
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
from .validation import validate_chat_request, validate_steering_alpha, validate_steering_emotion


class AppServer(ThreadingHTTPServer):
    runtime: HuggingFaceRuntime | None
    agent_service: Any | None = None
    emotion_service: Any | None = None
    # Loads and swaps chat models on request (None with --no-model).
    models: Any | None = None


class RequestHandler(BaseHTTPRequestHandler):
    server: AppServer
    frontend_root = Path(__file__).resolve().parent.parent / "frontend" / "dist"

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
        if path.startswith("/api/emotion/trace/"):
            service = self._emotion_service()
            turn_id = path.rsplit("/", 1)[1]
            try:
                analysis = service.analysis(turn_id) if service and re.fullmatch(r"[0-9a-f]{32}", turn_id) else None
            except Exception as exc:
                print(f"Emotion analysis error: {exc!r}")
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

    def do_POST(self) -> None:  # noqa: N802
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
            requested_model = payload.get("model")
            if requested_model is not None and isinstance(self.server.models, ModelManager):
                # The chat composer's model selector: load the selected model first (never downloads).
                if not isinstance(requested_model, str):
                    raise ValueError("model must be the key of a listed model.")
                self.server.models.activate(requested_model)
            manual_alpha = validate_steering_alpha(payload) if "steering_alpha" in payload else None
            requested_emotion = validate_steering_emotion(payload)
            generation_options: dict[str, Any] = {}
            steering_details = turn_id = plan = None
            service = self._emotion_service()
            if service is not None and self.path in ("/api/chat", "/api/chat/stream"):
                # The Emotion Reasoner Agent chooses strategy, response emotion and steering strength.
                turn_id, plan = service.begin_turn(
                    messages, alpha_override=manual_alpha,
                    emotion_override=None if requested_emotion in ("auto", "mirror") else requested_emotion)
            elif manual_alpha:
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
                user_id = payload.get("user_id", "anonymous")
                identifier = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
                if not isinstance(session_id, str) or not identifier.fullmatch(session_id):
                    raise ValueError("A valid session_id is required for agent chat.")
                if not isinstance(user_id, str) or not identifier.fullmatch(user_id):
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
                    messages[-1]["content"], session_id=session_id, user_id=user_id
                ))
                result["model"] = self.server.runtime.model_id
                self.send_json(HTTPStatus.OK, result)
            except Exception as exc:
                print(f"Agent generation error: {type(exc).__name__}: {exc}")
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
                    print(f"Generation error: {exc!r}")
                    self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                                   {"error": "Generation failed. Check the server terminal for details."})
                    return
                self.send_json(HTTPStatus.OK, {
                    "response": analysis["reply"], "model": self.server.runtime.model_id, "turn_id": turn_id,
                    "latency_seconds": analysis["latency_seconds"], "emotion_plan": plan,
                    "steering_decision": analysis["decision"],
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
                print(f"Streaming error: {exc!r}")
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
                for chunk in stream:
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except Exception as exc:
                print(f"Streaming error: {exc!r}")
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
            print(f"Generation error: {exc!r}")
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "Generation failed. Check the server terminal for details."},
            )
            return
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

        server.emotion_service = EmotionSteeringService(runtime)
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
