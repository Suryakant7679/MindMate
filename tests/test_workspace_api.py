import io
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock
from backend.server import RequestHandler


class WorkspaceEndpointTests(unittest.TestCase):
    def request(self, path, payload):
        handler = RequestHandler.__new__(RequestHandler)
        body = json.dumps(payload).encode()
        handler.path = path
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.server = Mock()
        handler.server.agent_service = None
        handler.server.runtime.generate.return_value = {"response": "Hello", "model": "llama"}
        handler.send_json = Mock()
        handler.do_POST()
        return handler

    def test_workspace_prompt_reaches_model(self):
        handler = self.request('/api/workspace/chat', {
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'system_prompt': 'You are AgroMind.'
        })
        self.assertEqual(handler.send_json.call_args.args[0], 200)
        self.assertEqual(handler.server.runtime.generate.call_args.kwargs['system_prompt'], 'You are AgroMind.')

    def test_workspace_requires_prompt(self):
        handler = self.request('/api/workspace/chat', {'messages': [{'role': 'user', 'content': 'Hi'}]})
        self.assertEqual(handler.send_json.call_args.args[0], 400)
        handler.server.runtime.generate.assert_not_called()

    def test_mindmate_ignores_prompt_override(self):
        handler = self.request('/api/chat', {
            'messages': [{'role': 'user', 'content': 'Hi'}], 'system_prompt': 'Override'
        })
        self.assertNotIn('system_prompt', handler.server.runtime.generate.call_args.kwargs)

    def test_agent_chat_requires_feature_and_session(self):
        handler = self.request('/api/agent/chat', {
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'session_id': 'valid-session',
        })
        self.assertEqual(handler.send_json.call_args.args[0], 503)
        handler = self.request('/api/agent/chat', {
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'session_id': '../invalid',
        })
        self.assertEqual(handler.send_json.call_args.args[0], 400)


class FrontendTests(unittest.TestCase):
    def test_unbuilt_frontend_returns_helpful_error(self):
        handler = RequestHandler.__new__(RequestHandler)
        handler.path = "/"
        handler.frontend_root = Path("/nonexistent-mindmate-build")
        handler.send_json = Mock()
        handler.do_GET()
        self.assertEqual(handler.send_json.call_args.args[0], 404)
        self.assertIn("npm run build", handler.send_json.call_args.args[1]["error"])

    def test_serves_frontend_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dist"
            root.mkdir()
            (root / "index.html").write_text("<html>MindMate</html>")
            (Path(directory) / "secret.txt").write_text("private")
            handler = RequestHandler.__new__(RequestHandler)
            handler.frontend_root = root
            handler.path = "/"
            handler.wfile = io.BytesIO()
            handler.send_response = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            handler.send_json = Mock()
            handler.do_GET()
            self.assertEqual(handler.wfile.getvalue(), b"<html>MindMate</html>")
            handler.path = "/assets/%2e%2e/%2e%2e/secret.txt"
            handler.do_GET()
            self.assertEqual(handler.send_json.call_args.args[0], 404)
