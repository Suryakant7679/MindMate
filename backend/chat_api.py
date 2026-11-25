"""HTTP endpoints of persistent, multi-user, multi-model chat (mixed into ``server.RequestHandler``).

Identity: every browser gets an anonymous user and a server-side session (HttpOnly, SameSite=Strict cookie holding a
random token; the database stores only its hash).  Users can add a username and password to reach their history
from another browser.  Every conversation endpoint authorises on the server: the conversation must belong to the
session's user, otherwise the answer is 404.

Consent: nothing is stored before the user has chosen.  "Save my conversations" keeps them in the database for the
chosen retention period; "Don't save" keeps them in server memory only (gone on restart or after inactivity).

    GET    /api/me                          user, consent state, retention options
    POST   /api/me/consent                  {store_history, retention_days}
    PATCH  /api/me                          {store_history?, retention_days?}
    GET    /api/me/export                   every stored conversation as JSON
    DELETE /api/me                          delete the user, their conversations and sessions
    POST   /api/auth/register               {username, password}  (keeps the current conversations)
    POST   /api/auth/login                  {username, password}
    POST   /api/auth/logout
    GET    /api/conversations               the user's conversations, most recent first
    POST   /api/conversations               {model?, title?, steering?}
    GET    /api/conversations/<id>          conversation and messages
    PATCH  /api/conversations/<id>          {title?, model?, steering?}
    DELETE /api/conversations/<id>
    POST   /api/conversations/<id>/messages {content?, model?, steering?}  -> text/event-stream
    POST   /api/conversations/<id>/cancel   stop the reply being generated
"""
from __future__ import annotations

import json
import re
import threading
import time
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from typing import Any
from urllib.parse import urlsplit

from chatbot_feature.session import session_key

from .chat_engine import EventStream, TurnEngine, make_title
from .config import DEFAULT_MODEL, MODEL_REGISTRY, SYSTEM_PROMPT
from .context import fit, normalize, prompt_budget, token_counter
from .generation_control import GenerationCancelled, GenerationControl
from .models import ModelBusy, ModelUnavailable
from .store import CONSENT_VERSION, RETENTION_OPTIONS, SESSION_SECONDS, ConversationStore, Store

COOKIE = "mindmate_session"
CONVERSATION_ID = re.compile(r"^[0-9a-f]{32}$")
MAX_BODY = 65_536
PROGRESS_INTERVAL = 0.25          # seconds between progress events
LOGIN_FAILURES = 5                # failed logins per username before a lockout
LOGIN_LOCKOUT = 60.0


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


class ChatApiMixin:
    """Request handling for the endpoints above.  Expects ``self.server`` to carry ``store`` (``Store``),
    ``ephemeral`` (in-memory ``ConversationStore``), ``models``, ``engine`` and the generation registry."""

    # ------------------------------------------------------------------ plumbing

    def _chat_api_enabled(self) -> bool:
        return isinstance(getattr(self.server, "store", None), Store)

    def _session_token(self) -> str | None:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except CookieError:
            return None
        return cookie[COOKIE].value if COOKIE in cookie else None

    def _set_session_cookie(self, token: str | None) -> None:
        secure = "; Secure" if getattr(self.server, "secure_cookies", False) else ""
        if token is None:
            self._session_cookie = f"{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0{secure}"
        else:
            self._session_cookie = (f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; "
                                    f"Max-Age={SESSION_SECONDS}{secure}")

    def _current_user(self) -> dict[str, Any]:
        """The session's user; a new anonymous user (and session cookie) when there is no valid session."""
        cached = getattr(self, "_user_cache", None)
        if cached is not None:
            return cached
        store: Store = self.server.store
        user = store.session_user(self._session_token())
        if user is None:
            user = store.create_user()
            self._set_session_cookie(store.create_session(user["id"]))
        self._user_cache = user
        return user

    def _json_body(self, required: bool = True) -> dict[str, Any]:
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid content length.") from exc
        if length <= 0:
            if required:
                raise ApiError(HTTPStatus.BAD_REQUEST, "Send a JSON object.")
            return {}
        if content_type != "application/json":
            raise ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send JSON (Content-Type: application/json).")
        if length > MAX_BODY:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request is too large.")
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Request must be a JSON object.")
        return payload

    def _check_origin(self) -> None:
        """Reject state-changing requests sent from another site (in addition to the SameSite=Strict cookie)."""
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != (self.headers.get("Host") or ""):
            raise ApiError(HTTPStatus.FORBIDDEN, "Cross-site requests are not allowed.")

    def _require_consent(self, user: dict[str, Any]) -> None:
        if not user.get("consent_version"):
            raise ApiError(HTTPStatus.FORBIDDEN, "Choose whether MindMate may save your conversations first.",
                           consent_required=True)

    def _stores(self) -> list[ConversationStore]:
        return [self.server.store, self.server.ephemeral]

    def _store_for_new(self, user: dict[str, Any]) -> ConversationStore:
        return self.server.store if user.get("store_history") else self.server.ephemeral

    def _find(self, user: dict[str, Any], conversation_id: str) -> tuple[ConversationStore, dict[str, Any]]:
        if CONVERSATION_ID.fullmatch(conversation_id or ""):
            for store in self._stores():
                found = store.get_conversation(user["id"], conversation_id)
                if found is not None:
                    return store, {**found, "saved": store.persistent}
        raise ApiError(HTTPStatus.NOT_FOUND, "No such conversation.")

    @staticmethod
    def _user_view(user: dict[str, Any]) -> dict[str, Any]:
        return {key: user.get(key) for key in ("id", "username", "anonymous", "store_history", "retention_days",
                                               "consent_version", "consent_at", "created_at")}

    def _me_payload(self, user: dict[str, Any]) -> dict[str, Any]:
        return {"user": self._user_view(user),
                "consent": {"required": not user.get("consent_version"), "version": CONSENT_VERSION,
                            "outdated": bool(user.get("consent_version")) and user.get("consent_version") != CONSENT_VERSION},
                "retention_options": list(RETENTION_OPTIONS)}

    def _api_send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self.send_json(status, payload)

    def _dispatch(self, handler, *args) -> None:
        try:
            handler(*args)
        except ApiError as exc:
            self._api_send(exc.status, {"error": exc.message, **exc.extra})
        except ValueError as exc:
            self._api_send(HTTPStatus.BAD_REQUEST, {"error": str(exc) or "Invalid request."})
        except Exception as exc:       # noqa: BLE001 - keep internals out of the browser
            print(f"Chat API error: {type(exc).__name__}: {exc}")
            self._api_send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "The request failed. Check the server terminal."})

    # ------------------------------------------------------------------ routing

    def chat_api(self, method: str, path: str) -> bool:
        """Handle ``method path`` if it belongs to this API; returns False otherwise."""
        if not (path.startswith("/api/me") or path.startswith("/api/auth/") or path.startswith("/api/conversations")):
            return False
        if not self._chat_api_enabled():
            self._api_send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Conversation storage is not configured."})
            return True
        if method != "GET":
            try:
                self._check_origin()
            except ApiError as exc:
                self._api_send(exc.status, {"error": exc.message})
                return True
        parts = [part for part in path.split("/") if part][1:]          # drop "api"
        routes = {
            ("GET", ("me",)): self._get_me, ("PATCH", ("me",)): self._patch_me, ("DELETE", ("me",)): self._delete_me,
            ("POST", ("me", "consent")): self._post_consent, ("GET", ("me", "export")): self._get_export,
            ("POST", ("auth", "register")): self._post_register, ("POST", ("auth", "login")): self._post_login,
            ("POST", ("auth", "logout")): self._post_logout,
            ("GET", ("conversations",)): self._list_conversations, ("POST", ("conversations",)): self._create_conversation,
        }
        handler = routes.get((method, tuple(parts)))
        if handler is not None:
            self._dispatch(handler)
            return True
        if len(parts) >= 2 and parts[0] == "conversations":
            conversation_id, rest = parts[1], tuple(parts[2:])
            handler = {("GET", ()): self._get_conversation, ("PATCH", ()): self._patch_conversation,
                       ("DELETE", ()): self._delete_conversation, ("POST", ("messages",)): self._post_message,
                       ("POST", ("cancel",)): self._cancel}.get((method, rest))
            if handler is not None:
                self._dispatch(handler, conversation_id)
                return True
        self._api_send(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        return True

    # ------------------------------------------------------------------ me / consent / accounts

    def _get_me(self) -> None:
        self._api_send(HTTPStatus.OK, self._me_payload(self._current_user()))

    def _post_consent(self) -> None:
        user = self._current_user()
        body = self._json_body()
        store_history, retention = body.get("store_history"), body.get("retention_days", 30)
        if not isinstance(store_history, bool):
            raise ValueError("store_history must be true or false.")
        if isinstance(retention, bool) or not isinstance(retention, int):
            raise ValueError("retention_days must be a whole number.")
        user = self.server.store.set_consent(user["id"], store_history=store_history, retention_days=retention)
        self._api_send(HTTPStatus.OK, self._me_payload(user))

    def _patch_me(self) -> None:
        user = self._current_user()
        body = self._json_body()
        store_history, retention = body.get("store_history"), body.get("retention_days")
        if store_history is not None and not isinstance(store_history, bool):
            raise ValueError("store_history must be true or false.")
        if retention is not None and (isinstance(retention, bool) or not isinstance(retention, int)):
            raise ValueError("retention_days must be a whole number.")
        if store_history is not None or retention is not None:
            self._require_consent(user)
        user = self.server.store.update_settings(user["id"], store_history=store_history, retention_days=retention)
        self._api_send(HTTPStatus.OK, self._me_payload(user))

    def _get_export(self) -> None:
        user = self._current_user()
        body = json.dumps({"exported_at": time.time(), "user": self._user_view(user),
                           "saved_conversations": self.server.store.export(user["id"]),
                           "unsaved_conversations": self.server.ephemeral.export(user["id"])},
                          ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="mindmate-conversations.json"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _forget_conversation(self, user_id: str, conversation_id: str) -> None:
        """Drop everything kept in memory about a conversation: active generation and Emotion Reasoner records."""
        control = self.server.generations.get(conversation_id)
        if control is not None:
            control.cancel()
        key = session_key(user_id, conversation_id)
        self.server.resource_owners.delete(key)
        for service in self._all_services():
            try:
                service.forget_conversation(key)
            except Exception as exc:       # noqa: BLE001
                print(f"Research records not cleared ({type(exc).__name__}).")

    def _delete_me(self) -> None:
        user = self._current_user()
        for store in self._stores():
            for conversation in store.list_conversations(user["id"], limit=100_000):
                self._forget_conversation(user["id"], conversation["id"])
            store.delete_user_conversations(user["id"])
        self.server.store.delete_user(user["id"])
        self._set_session_cookie(None)
        self._user_cache = None
        self._api_send(HTTPStatus.OK, {"deleted": True})

    def _credentials(self) -> tuple[str, str]:
        body = self._json_body()
        username, password = body.get("username"), body.get("password")
        if not isinstance(username, str) or not isinstance(password, str):
            raise ValueError("Send a username and a password.")
        return username.strip(), password

    def _post_register(self) -> None:
        user = self._current_user()
        username, password = self._credentials()
        user = self.server.store.register(user["id"], username, password)
        # A fresh session after a privilege change; other sessions of this user stay valid.
        self.server.store.delete_session(self._session_token())
        self._set_session_cookie(self.server.store.create_session(user["id"]))
        self._api_send(HTTPStatus.OK, self._me_payload(user))

    def _post_login(self) -> None:
        username, password = self._credentials()
        failures = self.server.login_failures
        key = username.lower()
        with self.server.login_lock:
            count, since = failures.get(key, (0, 0.0))
            if count >= LOGIN_FAILURES and time.monotonic() - since < LOGIN_LOCKOUT:
                raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, "Too many attempts. Wait a minute and try again.")
        user = self.server.store.authenticate(username, password)
        with self.server.login_lock:
            if user is None:
                count, since = failures.get(key, (0, time.monotonic()))
                failures[key] = (count + 1, since if count else time.monotonic())
                raise ApiError(HTTPStatus.UNAUTHORIZED, "Wrong username or password.")
            failures.pop(key, None)
        self.server.store.delete_session(self._session_token())
        self._set_session_cookie(self.server.store.create_session(user["id"]))
        self._user_cache = user
        self._api_send(HTTPStatus.OK, self._me_payload(user))

    def _post_logout(self) -> None:
        self.server.store.delete_session(self._session_token())
        self._set_session_cookie(None)
        self._api_send(HTTPStatus.OK, {"signed_out": True})

    # ------------------------------------------------------------------ conversations

    def _list_conversations(self) -> None:
        user = self._current_user()
        rows = []
        for store in self._stores():
            rows += [{**row, "saved": store.persistent} for row in store.list_conversations(user["id"])]
        rows.sort(key=lambda row: row["updated_at"], reverse=True)
        for row in rows:
            row.pop("user_id", None)
        self._api_send(HTTPStatus.OK, {"conversations": rows})

    def _model_key(self, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or value not in MODEL_REGISTRY:
            raise ValueError("model must be one of: " + ", ".join(MODEL_REGISTRY) + ".")
        return value

    @staticmethod
    def _steering(value: Any) -> str | None:
        if value is None:
            return None
        if value not in ("always", "policy", "off"):
            raise ValueError("steering must be always (every reply steered, strength from the user's emotional "
                             "state), policy (the calibrated policy decides) or off (unsteered baseline).")
        return value

    def _default_steering(self) -> str:
        mode = getattr(self.server, "steering_mode", None)
        return mode if mode in ("always", "policy", "off") else "always"

    def _create_conversation(self) -> None:
        user = self._current_user()
        self._require_consent(user)
        body = self._json_body(required=False)
        title = body.get("title")
        if title is not None and not isinstance(title, str):
            raise ValueError("title must be text.")
        store = self._store_for_new(user)
        conversation = store.create_conversation(user["id"], model_key=self._model_key(body.get("model")), title=title,
                                                 steering_mode=self._steering(body.get("steering")) or self._default_steering())
        conversation.pop("user_id", None)
        self._api_send(HTTPStatus.CREATED, {"conversation": {**conversation, "saved": store.persistent}})

    def _get_conversation(self, conversation_id: str) -> None:
        user = self._current_user()
        store, conversation = self._find(user, conversation_id)
        messages = store.messages(user["id"], conversation_id)
        for message in messages:
            message.pop("user_id", None)
        conversation.pop("user_id", None)
        conversation["generating"] = conversation_id in self.server.generations
        self._api_send(HTTPStatus.OK, {"conversation": conversation, "messages": messages})

    def _patch_conversation(self, conversation_id: str) -> None:
        user = self._current_user()
        store, _ = self._find(user, conversation_id)
        body = self._json_body()
        if "title" in body:
            if not isinstance(body["title"], str):
                raise ValueError("title must be text.")
            store.rename_conversation(user["id"], conversation_id, body["title"])
        store.update_conversation(user["id"], conversation_id, model_key=self._model_key(body.get("model")),
                                  steering_mode=self._steering(body.get("steering")))
        _, conversation = self._find(user, conversation_id)
        conversation.pop("user_id", None)
        self._api_send(HTTPStatus.OK, {"conversation": conversation})

    def _delete_conversation(self, conversation_id: str) -> None:
        user = self._current_user()
        store, _ = self._find(user, conversation_id)
        self._forget_conversation(user["id"], conversation_id)
        store.delete_conversation(user["id"], conversation_id)
        self._api_send(HTTPStatus.OK, {"deleted": True})

    def _cancel(self, conversation_id: str) -> None:
        user = self._current_user()
        self._find(user, conversation_id)
        control = self.server.generations.get(conversation_id)
        if control is not None:
            control.cancel()
        self._api_send(HTTPStatus.OK, {"cancelled": control is not None})

    # ------------------------------------------------------------------ one turn

    def _post_message(self, conversation_id: str) -> None:
        user = self._current_user()
        self._require_consent(user)
        store, conversation = self._find(user, conversation_id)
        body = self._json_body()
        content = body.get("content")
        if content is not None and (not isinstance(content, str) or not content.strip() or len(content) > 8000):
            raise ValueError("A message has 1 to 8000 characters.")
        manager = self.server.models
        if manager is None:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "The server was started without models.")
        key = (self._model_key(body.get("model")) or conversation.get("model_key") or manager.active
               or getattr(self.server, "default_model", None) or DEFAULT_MODEL)
        spec = manager.spec(key)
        from .models import is_downloaded
        if manager.get_loaded(key) is None and not manager.allow_download and not is_downloaded(spec.config.model_id):
            raise ValueError(f"{spec.label} is not downloaded on this server.")
        steering = self._steering(body.get("steering")) or conversation.get("steering_mode") or self._default_steering()

        control = GenerationControl()
        with self.server.generations_lock:
            if conversation_id in self.server.generations:
                raise ApiError(HTTPStatus.CONFLICT, "A reply is already being written in this conversation.")
            self.server.generations[conversation_id] = control
        stream = EventStream(self, on_disconnect=control.cancel)
        try:
            user_message = None
            if content is not None:
                user_message = store.add_message(user["id"], conversation_id, "user", content.strip())
            history = normalize(store.messages(user["id"], conversation_id))
            if not history:
                raise ValueError("There is no message to answer.")
            if key != conversation.get("model_key") or steering != conversation.get("steering_mode"):
                store.update_conversation(user["id"], conversation_id, model_key=key, steering_mode=steering)
            stream.start()
            stream.send("start", {"conversation_id": conversation_id, "model": key, "label": spec.label,
                                  "steering": steering,
                                  "user_message": self._public_message(user_message) if user_message else None})
            self._run_turn(user, store, conversation, history, key, spec, steering, control, stream)
        except GenerationCancelled:
            if stream.started:
                stream.send("cancelled", {"conversation_id": conversation_id})
        except (ApiError, ValueError) as exc:
            if not stream.started:
                raise
            message = exc.message if isinstance(exc, ApiError) else str(exc)
            if "context budget" in message:
                message = (f"This message is too long for {spec.label}'s context window. Shorten it, or choose a model "
                           "with a longer window (Llama 3.1-8B or Sarvam-30B).")
            stream.send("error", {"error": message})
        except (ModelUnavailable, ModelBusy) as exc:
            if not stream.started:
                raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, str(exc)) from exc
            stream.send("error", {"error": str(exc)})
        except Exception as exc:       # noqa: BLE001
            print(f"Conversation turn error: {type(exc).__name__}: {exc}")
            if not stream.started:
                raise
            stream.send("error", {"error": "Generation failed. Check the server terminal for details."})
        finally:
            with self.server.generations_lock:
                if self.server.generations.get(conversation_id) is control:
                    del self.server.generations[conversation_id]
            stream.close()

    @staticmethod
    def _public_message(message: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in message.items() if key != "user_id"}

    def _run_turn(self, user, store, conversation, history, key, spec, steering, control, stream) -> None:
        last = {"at": 0.0}

        def progress(steps: int) -> None:
            now = time.monotonic()
            if now - last["at"] >= PROGRESS_INTERVAL:
                last["at"] = now
                stream.send("progress", {"tokens": steps})

        def emit(event: str, data: dict[str, Any]) -> None:
            stream.send(event, data)

        control.on_progress = progress
        manager = self.server.models
        with manager.lease(key, on_loading=lambda name: emit("status", {"phase": "loading_model", "model": name,
                                                                        "label": MODEL_REGISTRY[name].label})) as runtime:
            control.check()
            self.server.mark_active(key, runtime)
            service = self.server.emotion_services.get(key)
            if service is not None and getattr(service, "runtime", None) is not runtime:
                service = None
            messages, context = fit(history, count=token_counter(runtime),
                                    budget=prompt_budget(runtime, spec.max_new_tokens),
                                    system_prompt=SYSTEM_PROMPT if spec.persona else None)
            owner = user["id"]
            result = self.server.engine.run(
                runtime=runtime, service=service, spec=spec, owner=owner,
                conversation_key=session_key(owner, conversation["id"]), messages=messages,
                steering_mode=steering, control=control, emit=emit)
            metadata: dict[str, Any] = {"context": context, "context_window": getattr(runtime, "context_tokens", None),
                                        "generation": {"latency_seconds": result["latency_seconds"],
                                                       "generated_tokens": result["generated_tokens"],
                                                       "max_new_tokens": spec.max_new_tokens,
                                                       "deterministic": spec.deterministic},
                                        "steering": result["steering"], "tone": result["tone"]}
            if result["cancelled"]:
                # Text already shown stays in the conversation (marked as stopped); nothing shown, nothing stored.
                message = None
                if result["reply"]:
                    message = store.add_message(owner, conversation["id"], "assistant", result["reply"],
                                                status="cancelled", model_key=key,
                                                model_id=getattr(runtime, "model_id", None), metadata=metadata)
                stream.send("cancelled", {"conversation_id": conversation["id"],
                                          "message": self._public_message(message) if message else None})
                return
            message = store.add_message(owner, conversation["id"], "assistant", result["reply"], model_key=key,
                                        model_id=getattr(runtime, "model_id", None), turn_id=result["turn_id"],
                                        metadata=metadata)
            if message is None:                         # deleted while the reply was written
                raise GenerationCancelled()
            plan = result.get("plan") or {}
            stream.send("done", {
                "message": self._public_message(message),
                "emotion": ({"turn_id": result["turn_id"], "user_emotion": plan.get("user_emotion"),
                             "confidence": plan.get("confidence"), "response_emotion": plan.get("response_emotion"),
                             "strategy": plan.get("strategy_title") or plan.get("strategy")}
                            if result["turn_id"] else None),
            })
            current = store.get_conversation(owner, conversation["id"])
            if current and current["title_source"] == "pending":
                title, source = make_title(runtime, messages[-1]["content"], sensitive=bool(plan.get("safety_flag")))
                if store.set_auto_title(owner, conversation["id"], title, source):
                    stream.send("title", {"conversation_id": conversation["id"], "title": title, "source": source})


class ServerState:
    """Attributes ``AppServer`` needs for this API (kept here so tests can build a server around a fake manager)."""

    @staticmethod
    def install(server: Any, store: Store, *, default_model: str | None = None) -> None:
        server.store = store
        server.ephemeral = ConversationStore(None)
        server.engine = TurnEngine(server.resource_owners)
        server.generations = {}
        server.generations_lock = threading.Lock()
        server.login_failures = {}
        server.login_lock = threading.Lock()
        server.default_model = default_model
