"""Model pool: lazy single loads, reuse, leases, LRU and memory-aware residency, idle unloading."""
import threading
import time
import unittest
from unittest.mock import patch

from backend.config import MODEL_REGISTRY
from backend.models import ModelBusy, ModelManager


class Runtime:
    def __init__(self, key):
        self.key, self.model_id = key, MODEL_REGISTRY[key].config.model_id
        self.lock = threading.Lock()
        self.model = object()


def pool(**kwargs):
    made, loaded, unloaded = [], [], []

    def factory(key, allow_download=False, **options):
        made.append((key, options))
        return Runtime(key)
    patcher = patch("backend.models.is_downloaded", return_value=True)
    patcher.start()
    manager = ModelManager(factory=factory, on_load=lambda key, rt: loaded.append(key),
                           on_unload=lambda key, rt: unloaded.append(key), free_memory=lambda: None, **kwargs)
    return manager, made, loaded, unloaded, patcher


class PoolTests(unittest.TestCase):
    def make(self, **kwargs):
        manager, made, loaded, unloaded, patcher = pool(**kwargs)
        self.addCleanup(patcher.stop)
        return manager, made, loaded, unloaded

    def test_lazy_load_and_reuse(self):
        manager, made, loaded, _ = self.make()
        self.assertEqual(made, [])                       # nothing is loaded until a request needs it
        with manager.lease("llama") as first:
            pass
        with manager.lease("llama") as again:
            self.assertIs(again, first)
        self.assertEqual([key for key, _ in made], ["llama"])
        self.assertEqual(manager.loads, {"llama": 1})

    def test_concurrent_requests_for_one_model_load_it_once(self):
        manager, made, _, _ = self.make()
        gate = threading.Event()
        original = manager.factory

        def slow(key, **options):
            gate.wait(2)
            return original(key, **options)
        manager.factory = slow
        results = []
        threads = [threading.Thread(target=lambda: results.append(manager.activate("emollama"))) for _ in range(5)]
        for thread in threads:
            thread.start()
        time.sleep(0.1)
        gate.set()
        for thread in threads:
            thread.join(5)
        self.assertEqual(len({id(runtime) for runtime in results}), 1)
        self.assertEqual([key for key, _ in made], ["emollama"])

    def test_default_keeps_one_model_and_unloads_the_previous(self):
        manager, made, _, unloaded = self.make()
        first = manager.activate("llama")
        manager.activate("emollama")
        self.assertEqual(unloaded, ["llama"])
        self.assertIsNone(first.model)
        self.assertEqual(sorted(manager.loaded()), ["emollama"])

    def test_two_resident_models_evict_least_recently_used(self):
        manager, _, _, unloaded = self.make(max_loaded=2)
        manager.activate("llama")
        manager.activate("emollama")
        manager.activate("llama")                       # llama is now the most recently used
        manager.activate("sarvam")
        self.assertEqual(unloaded, ["emollama"])
        self.assertEqual(sorted(manager.loaded()), ["llama", "sarvam"])
        self.assertEqual(manager.loads, {"llama": 1, "emollama": 1, "sarvam": 1})     # no reload of llama

    def test_a_leased_model_is_never_unloaded_mid_reply(self):
        manager, _, _, unloaded = self.make(lease_timeout=5)
        runtime = manager.acquire("llama")
        switched = threading.Event()

        def other_user():
            manager.activate("emollama")
            switched.set()
        thread = threading.Thread(target=other_user)
        thread.start()
        time.sleep(0.3)
        self.assertFalse(switched.is_set())             # waits for the reply in progress
        self.assertIsNotNone(runtime.model)
        manager.release("llama")
        thread.join(5)
        self.assertTrue(switched.is_set())
        self.assertEqual(unloaded, ["llama"])

    def test_busy_pool_times_out_instead_of_unloading(self):
        manager, _, _, _ = self.make(lease_timeout=0.3)
        manager.acquire("llama")
        with self.assertRaises(ModelBusy):
            manager.activate("emollama")

    def test_memory_check_unloads_idle_models_until_the_new_one_fits(self):
        free = {"gib": 10.0}
        manager, _, _, unloaded = self.make(max_loaded=4, gpu_headroom_gib=2)
        manager._free_memory = lambda: free["gib"]
        free["gib"] = 40
        manager.activate("llama")                      # 16 GiB estimate + 2 headroom fits in 40
        free["gib"] = 24
        manager.activate("emollama")                   # 14 + 2 fits in 24
        free["gib"] = 10                               # Sarvam (20 + 2) does not fit next to them

        def unload_side_effect(key, runtime):
            unloaded.append(key)
            free["gib"] += MODEL_REGISTRY[key].gpu_gib
        manager.on_unload = unload_side_effect
        manager.activate("sarvam")
        self.assertEqual(unloaded, ["llama"])           # LRU first; 10 + 16 = 26 >= 22, so emollama stays
        self.assertEqual(sorted(manager.loaded()), ["emollama", "sarvam"])

    def test_idle_unloading(self):
        manager, _, _, unloaded = self.make(idle_unload_seconds=60)
        manager.activate("llama")
        self.assertEqual(manager.unload_idle(time.monotonic() + 30), [])
        self.assertEqual(manager.unload_idle(time.monotonic() + 61), ["llama"])
        self.assertEqual(unloaded, ["llama"])

    def test_start_options_apply_to_every_load(self):
        manager, made, _, _ = self.make(overrides={"llama": {"load_in_4bit": True}}, placement={"device_map": "cpu"})
        manager.activate("llama")
        manager.activate("emollama")
        manager.activate("llama")
        self.assertEqual(made[0][1], {"load_in_4bit": True, "placement": {"device_map": "cpu"}})
        self.assertEqual(made[2][1], {"load_in_4bit": True, "placement": {"device_map": "cpu"}})
        self.assertEqual(made[1][1], {"placement": {"device_map": "cpu"}})

    def test_catalog_reports_capabilities(self):
        manager, _, _, _ = self.make()
        manager.activate("llama")
        rows = {row["key"]: row for row in manager.catalog()}
        self.assertTrue(rows["llama"]["loaded"])
        self.assertIsNotNone(rows["llama"]["resident"])
        self.assertNotIn("safety_evaluated", rows["llama"])
        self.assertEqual(rows["emollama"]["context_tokens"], 2048)
        self.assertTrue(rows["llama"]["generation"]["deterministic"])
        self.assertIn(rows["sarvam"]["steering_status"]["status"],
                      {"policy_gated", "disabled_by_calibration", "uncalibrated", "unavailable"})


class OfflineTests(unittest.TestCase):
    def test_stay_offline_switches_off_hub_requests_even_after_import(self):
        import os
        import transformers.utils.hub as hub
        from backend.models import stay_offline
        saved = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        flag = hub._is_offline_mode
        try:
            hub._is_offline_mode = False
            stay_offline()
            self.assertTrue(hub.is_offline_mode())            # transformers was imported before the call
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        finally:
            hub._is_offline_mode = flag
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
