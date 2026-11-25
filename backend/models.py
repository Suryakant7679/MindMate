"""Chat-model registry at run time: which models are on disk, loading them lazily, reusing and unloading them.

``ModelManager`` is a small pool of loaded runtimes:

* **Lazy, single load.**  A model is loaded the first time a request needs it; concurrent requests for the same
  model wait for that one load (never two copies, never a repeated download).  Nothing is downloaded unless the
  server was started with ``--allow-download``.
* **Leases.**  A request holds a lease on the runtime it generates with (``with manager.lease(key) as runtime``).  A
  leased model is never unloaded, so a reply in progress can never lose its model to another user's switch.
* **Memory-aware residency.**  At most ``max_loaded`` models stay loaded (default 1, the previous one-model-at-a-time
  behaviour; raise it only when the GPU can hold several).  Before a load, idle models are unloaded in
  least-recently-used order until the count limit is met and the free GPU memory (measured with
  ``torch.cuda.mem_get_info``, so other processes on the GPU are accounted for) covers the model's footprint plus
  ``gpu_headroom_gib``.  A model's footprint is its measured size from an earlier load, else the registry estimate.
* **Idle unloading** (optional): models unused for ``idle_unload_seconds`` are unloaded by ``unload_idle()``.

Placement (``device_map`` / ``max_memory``) is passed to the runtime constructor; see ``HuggingFaceRuntime``.
"""
from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import MODEL_REGISTRY, ModelSpec


def cache_dir() -> Path:
    home = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    return Path(home)


def is_downloaded(model_id: str) -> bool:
    """True if a complete checkpoint is available locally (a local folder, or weights in the HF cache).

    A cache entry with only metadata, or with a download still in progress, does not count.
    """
    if Path(model_id).is_dir():
        return True
    root = cache_dir() / ("models--" + model_id.replace("/", "--"))
    if any((root / "blobs").glob("*.incomplete")):
        return False
    for snapshot in (root / "snapshots").glob("*"):
        weights = [*snapshot.glob("*.safetensors"), *snapshot.glob("*.bin")]
        if (snapshot / "config.json").exists() and weights and all(path.exists() for path in weights):
            return True
    return False


def stay_offline() -> None:
    """Use only the local Hugging Face cache: no request to the Hub at all.

    ``local_files_only`` alone is not enough: transformers 4.57 still asks the Hub about every tokenizer it loads
    (``_patch_mistral_regex`` calls ``model_info``) unless offline mode is on, so a flaky network made loading a model
    that is fully on disk fail.  Offline mode is read when transformers is imported, so the flags of already
    imported modules are set as well.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import sys
    hub = sys.modules.get("transformers.utils.hub")
    if hub is not None and hasattr(hub, "_is_offline_mode"):
        hub._is_offline_mode = True
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_OFFLINE = True


def create_runtime(key: str, *, model_id: str | None = None, load_in_4bit: bool | None = None,
                   allow_download: bool = False, placement: dict[str, Any] | None = None) -> Any:
    """Load the registry model ``key`` (optionally another checkpoint of it) with its own settings."""
    spec = MODEL_REGISTRY[key]
    model_id = model_id or spec.config.model_id
    if not allow_download and not is_downloaded(model_id):
        raise ModelUnavailable(
            f"{spec.label} ({model_id}) is not downloaded. Download it first, for example "
            f"`hf download {model_id}`, or start the server with --allow-download.")
    local = not allow_download
    if local:
        stay_offline()                    # the checkpoint is on disk: never depend on the network to load it
    extra = {"placement": placement} if placement else {}
    if spec.runtime == "sarvam":
        from .sarvam import SarvamRuntime
        return SarvamRuntime(model_id, load_in_4bit, local_files_only=local, **extra)
    if spec.runtime == "llama":
        from .llama import LlamaRuntime
        return LlamaRuntime(model_id, load_in_4bit, local_files_only=local, **extra)
    from .runtime import HuggingFaceRuntime
    return HuggingFaceRuntime(model_id, load_in_4bit, config=spec.config, local_files_only=local, **extra)


class ModelUnavailable(ValueError):
    """The requested model is unknown or not on disk."""


class ModelBusy(RuntimeError):
    """No loaded model could be unloaded to make room in time (all of them are generating)."""


@dataclass
class _Loaded:
    key: str
    runtime: Any
    model_id: str
    leases: int = 0
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.monotonic)
    load_seconds: float = 0.0
    footprint_gib: float | None = None
    uses: int = 0


def _gpu_free_gib() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / 1024 ** 3
    except Exception:       # noqa: BLE001 - no CUDA, or a mocked torch in tests
        pass
    return None


def _gpu_allocated_gib() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 3
    except Exception:       # noqa: BLE001
        pass
    return None


class ModelManager:
    """Pool of loaded chat runtimes (see the module docstring).

    Callbacks: ``on_load(key, runtime)`` after a model is loaded, ``on_unload(key, runtime)`` before it is released.
    ``on_change(key, runtime)`` is the earlier single-model hook: it is called after every load, and with
    ``("", None)`` before the most recently used model is unloaded.
    """

    def __init__(self, *, allow_download: bool = False, factory: Callable[..., Any] = create_runtime,
                 on_change: Callable[[str, Any], None] | None = None,
                 on_load: Callable[[str, Any], None] | None = None,
                 on_unload: Callable[[str, Any], None] | None = None,
                 max_loaded: int = 1, gpu_headroom_gib: float = 2.0, placement: dict[str, Any] | None = None,
                 idle_unload_seconds: float = 0.0, lease_timeout: float = 900.0,
                 overrides: dict[str, dict[str, Any]] | None = None,
                 free_memory: Callable[[], float | None] = _gpu_free_gib,
                 allocated_memory: Callable[[], float | None] = _gpu_allocated_gib) -> None:
        if max_loaded < 1:
            raise ValueError("max_loaded must be at least 1")
        self.allow_download = allow_download
        self.factory = factory
        self.on_change, self.on_load, self.on_unload = on_change, on_load, on_unload
        self.max_loaded = max_loaded
        self.gpu_headroom_gib = gpu_headroom_gib
        self.placement = dict(placement or {})
        self.idle_unload_seconds = idle_unload_seconds
        self.lease_timeout = lease_timeout
        # Load options that apply to every load of a model (--model-id / --load-in-4bit of the start model).
        self.overrides = {key: dict(value) for key, value in (overrides or {}).items()}
        self._free_memory, self._allocated_memory = free_memory, allocated_memory
        self.active: str | None = None          # most recently used model (what /api/status reports)
        self.loading: str | None = None
        self.loads: dict[str, int] = {}         # completed loads per model since start (reload accounting)
        self._footprints: dict[str, float] = {}
        self._loaded: dict[str, _Loaded] = {}
        self._cv = threading.Condition(threading.RLock())
        self._load_lock = threading.Lock()      # one load / unload sequence at a time

    # ------------------------------------------------------------------ registry

    def spec(self, key: str) -> ModelSpec:
        if not isinstance(key, str) or key not in MODEL_REGISTRY:
            raise ModelUnavailable("Unknown model. Choose one of: " + ", ".join(MODEL_REGISTRY) + ".")
        return MODEL_REGISTRY[key]

    @property
    def runtime(self) -> Any:
        with self._cv:
            entry = self._loaded.get(self.active) if self.active else None
            return entry.runtime if entry else None

    def loaded(self) -> dict[str, Any]:
        with self._cv:
            return {key: entry.runtime for key, entry in self._loaded.items()}

    def get_loaded(self, key: str) -> Any:
        with self._cv:
            entry = self._loaded.get(key)
            return entry.runtime if entry else None

    def catalog(self) -> list[dict[str, Any]]:
        """What the chat composer shows: every registry model, whether it can be used now, and what it supports."""
        from .capabilities import checkpoint_facts, steering_status

        rows = []
        with self._cv:
            loaded = dict(self._loaded)
        for key, spec in MODEL_REGISTRY.items():
            downloaded = is_downloaded(spec.config.model_id)
            entry = loaded.get(key)
            runtime = entry.runtime if entry else None
            facts = checkpoint_facts(spec.config.model_id) if downloaded else {}
            window = getattr(runtime, "context_tokens", None)
            if window is None and facts.get("max_position_embeddings"):
                from .config import CONTEXT_TOKENS
                window = min(CONTEXT_TOKENS, int(facts["max_position_embeddings"]))
            rows.append({
                "key": key, "label": spec.label, "model_id": spec.config.model_id, "purpose": spec.purpose,
                "downloaded": downloaded, "loaded": entry is not None, "loading": key == self.loading,
                "available": entry is not None or downloaded or self.allow_download,
                # Whether replies can actually be steered now: panels and a calibrated policy that steers
                # (``steering_enabled`` is set by the server when it attaches the Emotion Reasoner).
                "steering": bool(runtime is not None and getattr(runtime, "_emotion_panels", None)
                                 and getattr(runtime, "steering_enabled", True)),
                "steering_status": steering_status(spec),
                "gpu_gib": spec.gpu_gib,
                "footprint_gib": self._footprints.get(key),
                "context_tokens": window,
                "generation": {"max_new_tokens": spec.max_new_tokens, "deterministic": spec.deterministic,
                               "temperature": spec.temperature, "top_p": spec.top_p,
                               "system_prompt": "MindMate persona" if spec.persona else "none (model default)"},
                "checkpoint": facts,
                "quantization": "4-bit NF4" if spec.config.load_in_4bit else "none",
                "resident": ({"leases": entry.leases, "uses": entry.uses, "load_seconds": round(entry.load_seconds, 1),
                              "idle_seconds": round(time.monotonic() - entry.last_used, 1)} if entry else None),
            })
        return rows

    def memory(self) -> dict[str, Any]:
        with self._cv:
            resident = {key: {"footprint_gib": entry.footprint_gib, "leases": entry.leases} for key, entry in self._loaded.items()}
        return {"max_loaded": self.max_loaded, "gpu_headroom_gib": self.gpu_headroom_gib,
                "gpu_free_gib": (round(free, 2) if (free := self._free_memory()) is not None else None),
                "placement": self.placement or {"device_map": "auto"}, "resident": resident, "loads": dict(self.loads),
                "idle_unload_seconds": self.idle_unload_seconds}

    # ------------------------------------------------------------------ leases

    def acquire(self, key: str, *, on_loading: Callable[[str], None] | None = None, **options: Any) -> Any:
        """Lease the runtime of ``key``, loading it first if needed.  Pair every call with ``release(key)``."""
        spec = self.spec(key)
        with self._cv:
            entry = self._loaded.get(key)
            if entry is not None:
                return self._lease(entry)
        options = {**self.overrides.get(key, {}), **options}
        model_id = options.get("model_id") or spec.config.model_id
        if not self.allow_download and not is_downloaded(model_id):
            raise ModelUnavailable(
                f"{spec.label} ({model_id}) is not downloaded on this server yet. Download it first "
                f"(`hf download {model_id}`) or start the server with --allow-download.")
        with self._load_lock:
            with self._cv:                          # loaded by a concurrent request while this one waited
                entry = self._loaded.get(key)
                if entry is not None:
                    return self._lease(entry)
            if on_loading is not None:
                on_loading(key)
            self._make_room(key)
            self.loading = key
            try:
                before = self._allocated_memory()
                started = time.perf_counter()
                if self.placement and "placement" not in options:
                    options = {**options, "placement": self.placement}
                runtime = self.factory(key, allow_download=self.allow_download, **options)
                elapsed = time.perf_counter() - started
                after = self._allocated_memory()
            finally:
                self.loading = None
            footprint = round(after - before, 2) if before is not None and after is not None and after > before else None
            if footprint:
                self._footprints[key] = footprint
            entry = _Loaded(key, runtime, model_id, load_seconds=elapsed, footprint_gib=footprint)
            with self._cv:
                self._loaded[key] = entry
                self.loads[key] = self.loads.get(key, 0) + 1
                leased = self._lease(entry)
                self._cv.notify_all()
            print(f"Model {key} loaded in {elapsed:.1f} s" + (f" ({footprint:.1f} GiB on the GPU)." if footprint else "."))
            if self.on_load:
                self.on_load(key, runtime)
            if self.on_change:
                self.on_change(key, runtime)
            return leased

    def _lease(self, entry: _Loaded) -> Any:
        entry.leases += 1
        entry.uses += 1
        entry.last_used = time.monotonic()
        self.active = entry.key
        return entry.runtime

    def release(self, key: str) -> None:
        with self._cv:
            entry = self._loaded.get(key)
            if entry is not None and entry.leases > 0:
                entry.leases -= 1
                entry.last_used = time.monotonic()
            self._cv.notify_all()

    @contextmanager
    def lease(self, key: str, **options: Any) -> Iterator[Any]:
        runtime = self.acquire(key, **options)
        try:
            yield runtime
        finally:
            self.release(key)

    def activate(self, key: str, **options: Any) -> Any:
        """Load ``key`` if needed and make it the most recently used model; returns its runtime (no lease kept)."""
        runtime = self.acquire(key, **options)
        self.release(key)
        return runtime

    # ------------------------------------------------------------------ residency

    def _needed_gib(self, key: str) -> float:
        return float(self._footprints.get(key) or MODEL_REGISTRY[key].gpu_gib or 0.0)

    def _make_room(self, key: str) -> None:
        """Unload idle models (least recently used first) until ``key`` fits; wait for leased ones if necessary."""
        deadline = time.monotonic() + self.lease_timeout
        while True:
            with self._cv:
                resident = list(self._loaded.values())
                free = self._free_memory()
                need = self._needed_gib(key) + self.gpu_headroom_gib
                if len(resident) < self.max_loaded and (free is None or free >= need):
                    return
                idle = sorted((entry for entry in resident if entry.leases == 0), key=lambda entry: entry.last_used)
                if not idle:
                    if not resident:
                        print(f"Warning: about {free:.1f} GiB of GPU memory is free and {MODEL_REGISTRY[key].label} "
                              f"needs about {need:.1f} GiB; loading anyway (device_map may offload layers to the CPU).")
                        return
                    if len(resident) < self.max_loaded:
                        # The count allows another model; only memory is short and nothing idle can be unloaded.
                        print(f"Warning: GPU memory is short for {MODEL_REGISTRY[key].label} while other models are "
                              "generating; loading anyway.")
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ModelBusy("Every loaded model is busy generating; try again in a moment.")
                    self._cv.wait(timeout=min(remaining, 1.0))
                    continue
                victim = idle[0]
                del self._loaded[victim.key]
            self._unload(victim)

    def _unload(self, entry: _Loaded) -> None:
        runtime = entry.runtime
        lock = getattr(runtime, "lock", None)
        # A generation that outlived its request (a timed-out safety check) still holds the runtime lock.
        if lock is not None:
            lock.acquire()
        try:
            if self.on_unload:
                self.on_unload(entry.key, runtime)
            if self.on_change and self.active == entry.key:
                self.on_change("", None)
            if self.active == entry.key:
                self.active = None
            runtime.model = None
        finally:
            if lock is not None:
                lock.release()
        del runtime
        entry.runtime = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:       # noqa: BLE001
            pass
        print(f"Model {entry.key} unloaded.")

    def unload(self, key: str) -> bool:
        """Unload ``key`` now if it is loaded and idle; returns whether it was unloaded."""
        with self._load_lock:
            with self._cv:
                entry = self._loaded.get(key)
                if entry is None or entry.leases:
                    return False
                del self._loaded[key]
            self._unload(entry)
            return True

    def unload_idle(self, now: float | None = None) -> list[str]:
        """Unload models idle for longer than ``idle_unload_seconds`` (no-op when that is 0)."""
        if self.idle_unload_seconds <= 0:
            return []
        now = time.monotonic() if now is None else now
        with self._cv:
            stale = [key for key, entry in self._loaded.items()
                     if entry.leases == 0 and now - entry.last_used >= self.idle_unload_seconds]
        return [key for key in stale if self.unload(key)]
