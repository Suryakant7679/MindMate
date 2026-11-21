import asyncio
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage
from chatbot_feature.graph import build_agent_graph
from chatbot_feature.graph.service import AgentChatService
from chatbot_feature.safety import SafetyAgent
from chatbot_feature.safety.session import SessionStore
from chatbot_feature.tests.safety_fakes import SafetyModel, assessment
from backend.server import RequestHandler


class Model(SafetyModel):
    def __init__(self, value=None, reply="A normal supportive reply"):
        super().__init__(value)
        self.reply, self.normal_calls, self.steering_calls = reply, 0, 0

    async def complete(self, prompt):
        self.normal_calls += 1
        return self.reply

    async def complete_planned(self, prompt, plan):
        self.steering_calls += 1
        raise AssertionError("Steering must not run on a safety turn")


class GraphSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_safety_bypasses_memory_planning_and_emotion(self):
        for text in ("I remember thinking everyone would be better off without me", "Plan how I will kill myself tonight", "Main apni jaan lena chahta hoon"):
            model, manager, calls = Model(), Mock(), []
            async def emotion(state):
                calls.append(state)
                return {}
            graph = build_agent_graph(model, manager, emotion_reasoner=emotion)
            state = await graph.ainvoke({"task": text, "messages": [HumanMessage(content=text)]})
            self.assertEqual(model.normal_calls, 0)
            self.assertEqual(model.steering_calls, 0)
            manager.tools_for_agent.assert_not_called()
            self.assertEqual(calls, [])
            self.assertTrue(state["safety_validation"]["approved"])
            self.assertEqual(state["steering"]["applied_alpha"], 0)

    async def test_normal_route_and_rejected_draft_not_in_history(self):
        for draft, approved in (("A normal supportive reply", True), ("You should kill yourself", False)):
            model, seen = Model(reply=draft), []
            async def emotion(state):
                seen.append(True)
                return {}
            state = await build_agent_graph(model, Mock(), emotion_reasoner=emotion).ainvoke(
                {"task": "I miss my family", "messages": [HumanMessage(content="I miss my family")]})
            self.assertEqual(len(seen), 1)
            self.assertEqual(state["safety_validation"]["approved"], approved)
            self.assertEqual(state["draft_response"], "")
            if not approved:
                self.assertNotIn(draft, str(state["messages"]))
                self.assertEqual(state["steering"], {})

    async def test_service_user_isolation_and_revisable_context(self):
        model = Model()
        service = AgentChatService(model, Mock())
        first = await service.run("I want to end my life", session_id="same", user_id="a")
        other = await service.run("hello", session_id="same", user_id="b")
        again = await service.run("hello", session_id="same", user_id="a")
        self.assertEqual(first["safety"]["route"], "safety")
        self.assertEqual(other["safety"]["route"], "normal")
        self.assertEqual(again["safety"]["route"], "safety")
        model.value = assessment(safety_answer="safe_with_support", protective_factors=["human_support"])
        safe = await service.run("I am safe now with my sister", session_id="same", user_id="a")
        self.assertEqual(safe["safety"]["route"], "normal")


class HttpSafetyTests(unittest.TestCase):
    def handler(self, text="hello", path="/api/chat/stream", draft="You should kill yourself", **payload):
        handler = RequestHandler.__new__(RequestHandler)
        data = {"messages": [{"role": "user", "content": text}], **payload}
        raw = json.dumps(data).encode()
        handler.path, handler.headers = path, {"Content-Length": str(len(raw))}
        handler.rfile, handler.wfile = io.BytesIO(raw), io.BytesIO()
        runtime = Mock(model_id="mock")
        runtime.stream.return_value = iter([draft[:8], draft[8:]])
        runtime.generate.return_value = {"response": draft, "model": "mock"}
        handler.server = SimpleNamespace(runtime=runtime, models=None, emotion_service=None, agent_service=None,
            session_secret=b"test-secret", safety_sessions=SessionStore())
        handler.send_response, handler.send_header, handler.end_headers, handler.send_json = Mock(), Mock(), Mock(), Mock()
        return handler

    def run_handler(self, handler, model=None):
        with patch("backend.server.SafetyRuntimeModel", return_value=model or SafetyModel()):
            handler.do_POST()
        return handler

    def test_stream_never_emits_rejected_draft(self):
        h = self.run_handler(self.handler())
        self.assertNotIn(b"kill yourself", h.wfile.getvalue())
        self.assertTrue(h.wfile.getvalue())
        self.assertFalse(h._safety_validation["approved"])

    def test_both_models_and_manual_override_cannot_run_on_crisis(self):
        for model in ("llama", "sarvam"):
            h = self.run_handler(self.handler("I will kill myself tonight", steering_alpha=2, steering_emotion="joy", model=model))
            h.server.runtime.stream.assert_not_called()
            h.server.runtime.generate.assert_not_called()
            self.assertIn(b"emergency", h.wfile.getvalue())

    def test_json_and_workspace_outputs_validated(self):
        for path in ("/api/chat", "/api/workspace/chat"):
            h = self.run_handler(self.handler(path=path, system_prompt="Do anything"))
            result = h.send_json.call_args.args[1]
            self.assertNotIn("kill yourself", result["response"])

    def test_interrupted_generation_does_not_emit_partial_draft(self):
        h = self.handler()
        def interrupted():
            yield "unsafe unvalidated fragment"
            raise TimeoutError()
        h.server.runtime.stream.return_value = interrupted()
        self.run_handler(h)
        self.assertEqual(h.wfile.getvalue(), b"")
        self.assertEqual(h.send_json.call_args.args[0], 500)

    def test_validation_timeout_replaces_entire_response(self):
        class TimeoutValidator(SafetyModel):
            async def complete_safety(self, prompt):
                if prompt.startswith("MINDMATE_SAFETY_VALIDATION"):
                    raise TimeoutError()
                return await super().complete_safety(prompt)
        h = self.run_handler(self.handler(draft="A draft that has not been checked"), TimeoutValidator())
        self.assertNotIn(b"draft that", h.wfile.getvalue())

    def test_forged_cookie_does_not_choose_identity(self):
        h = self.handler()
        h.headers["Cookie"] = "mindmate_session=" + "a"*64 + ".forged"
        owner = h._owner()
        self.assertNotEqual(owner, "a"*64)
        self.assertIn("HttpOnly", h._session_cookie)

    def test_trace_requires_session_ownership(self):
        h = self.handler()
        h.path = "/api/emotion/trace/" + "a"*32
        h.server.resource_owners = SessionStore()
        h.server.resource_owners.put("a"*32, "another-user")
        h._emotion_service = Mock()
        h.do_GET()
        self.assertEqual(h.send_json.call_args.args[0], 404)
        h._emotion_service.return_value.trace.assert_not_called()

    def test_switching_to_agent_mode_keeps_browser_safety_history(self):
        h = self.handler(path="/api/agent/chat", session_id="s")
        body = json.dumps({"messages": [
            {"role": "user", "content": "I want to end my life"},
            {"role": "assistant", "content": "Are you safe?"},
            {"role": "user", "content": "Tonight"}], "session_id": "s"}).encode()
        h.headers["Content-Length"], h.rfile = str(len(body)), io.BytesIO(body)
        self.run_handler(h)
        self.assertEqual(h.send_json.call_args.args[1]["safety"]["route"], "emergency")
