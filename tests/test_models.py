"""Model registry, on-disk detection and switching (no weights are loaded)."""
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.config import MODEL_REGISTRY
from backend.models import ModelManager, ModelUnavailable, is_downloaded
from backend.server import RequestHandler, attach_runtime


class FakeRuntime:
    def __init__(self, key, panels=None):
        self.key, self.model_id = key, MODEL_REGISTRY[key].config.model_id
        self.lock = threading.Lock()
        self.model = object()
        self._emotion_panels = panels or {}

    def generate(self, messages, **_):
        return {"response": f"reply from {self.key}", "model": self.model_id}


def manager(downloaded=("llama", "emollama"), **kwargs):
    made = []

    def factory(key, allow_download=False, **options):
        made.append(key)
        return FakeRuntime(key)
    on_disk = {MODEL_REGISTRY[key].config.model_id for key in downloaded}
    patcher = patch("backend.models.is_downloaded", side_effect=lambda model_id: model_id in on_disk)
    patcher.start()
    return ModelManager(factory=factory, **kwargs), made, patcher


class RegistryTests(unittest.TestCase):
    def test_registry_lists_the_four_models_with_separate_result_folders(self):
        self.assertEqual(set(MODEL_REGISTRY), {"llama", "llama2", "emollama", "sarvam"})
        tags = [spec.config.results_tag for spec in MODEL_REGISTRY.values()]
        self.assertEqual(len(set(tags)), 4)
        self.assertEqual(MODEL_REGISTRY["llama"].config.results_tag, "llama31_8b")       # existing results kept
        self.assertIsNotNone(MODEL_REGISTRY["emollama"].config.chat_template)

    def test_download_detection_requires_complete_weights(self):
        with tempfile.TemporaryDirectory() as home:
            snapshot = Path(home) / "hub" / "models--org--model" / "snapshots" / "abc"
            snapshot.mkdir(parents=True)
            with patch.dict(os.environ, {"HF_HUB_CACHE": str(Path(home) / "hub")}):
                (snapshot / "config.json").write_text("{}")
                self.assertFalse(is_downloaded("org/model"))                 # metadata only
                (snapshot / "model.safetensors").write_text("x")
                self.assertTrue(is_downloaded("org/model"))
                blobs = snapshot.parents[1] / "blobs"
                blobs.mkdir()
                (blobs / "abc.incomplete").write_text("")
                self.assertFalse(is_downloaded("org/model"))                 # download in progress


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.models, self.made, patcher = manager()
        self.addCleanup(patcher.stop)

    def test_switching_loads_one_model_at_a_time(self):
        first = self.models.activate("llama")
        self.assertIs(self.models.activate("llama"), first)                  # no reload
        second = self.models.activate("emollama")
        self.assertEqual(self.made, ["llama", "emollama"])
        self.assertIsNone(first.model)                                       # previous model released
        self.assertIs(self.models.runtime, second)
        self.assertEqual(self.models.active, "emollama")

    def test_models_not_on_disk_are_listed_but_not_loaded(self):
        catalog = {row["key"]: row for row in self.models.catalog()}
        self.assertTrue(catalog["emollama"]["available"])
        self.assertFalse(catalog["sarvam"]["available"])
        self.assertFalse(catalog["llama2"]["downloaded"])
        with self.assertRaises(ModelUnavailable):
            self.models.activate("sarvam")
        with self.assertRaises(ModelUnavailable):
            self.models.activate("gpt-unknown")
        self.assertEqual(self.made, [])

    def test_allow_download_makes_every_model_selectable(self):
        models, made, patcher = manager(downloaded=(), allow_download=True)
        self.addCleanup(patcher.stop)
        self.assertTrue(all(row["available"] for row in models.catalog()))
        models.activate("sarvam")
        self.assertEqual(made, ["sarvam"])


class ServerTests(unittest.TestCase):
    def server(self):
        server = Mock(agent_service=None, emotion_service=None, runtime=None)
        server.models, self.made, patcher = manager(on_change=lambda key, runtime: attach_runtime(server, key, runtime))
        self.addCleanup(patcher.stop)
        return server

    def request(self, server, method, path, payload=None):
        handler = RequestHandler.__new__(RequestHandler)
        body = json.dumps(payload or {}).encode()
        handler.path, handler.server = path, server
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
        handler.send_json = Mock()
        getattr(handler, method)()
        return handler.send_json.call_args.args

    def test_models_endpoint_and_selection_per_message(self):
        server = self.server()
        server.models.activate("llama")
        status, payload = self.request(server, "do_GET", "/api/models")
        self.assertEqual((status, payload["active"]), (200, "llama"))
        self.assertEqual(len(payload["models"]), 4)
        status, payload = self.request(server, "do_POST", "/api/chat",
                                       {"messages": [{"role": "user", "content": "Hi"}], "model": "emollama"})
        self.assertEqual((status, payload["response"]), (200, "reply from emollama"))   # answered by the new model
        self.assertEqual(server.models.active, "emollama")
        self.assertIs(server.runtime, server.models.runtime)

    def test_unknown_or_missing_model_is_a_client_error(self):
        server = self.server()
        server.models.activate("llama")
        for model in ("sarvam", "nope", 3):
            status, payload = self.request(server, "do_POST", "/api/chat",
                                           {"messages": [{"role": "user", "content": "Hi"}], "model": model})
            self.assertEqual(status, 400)
        self.assertEqual(server.models.active, "llama")

    def test_steering_service_only_for_models_with_their_own_panels(self):
        server = self.server()
        attach_runtime(server, "emollama", FakeRuntime("emollama"))
        self.assertIsNone(server.emotion_service)                           # plain baseline chat
        with patch("backend.emotion_service.EmotionSteeringService") as service:
            attach_runtime(server, "emollama", FakeRuntime("emollama", panels={"love": {}}))
        service.assert_called_once()


if __name__ == "__main__":
    unittest.main()
