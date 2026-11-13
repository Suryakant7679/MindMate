"""Chat-model registry at run time: which models are on disk, loading one, switching between them.

The GPU holds one chat model at a time.  Selecting another model in the chat
composer unloads the current one and loads the selected one; nothing is ever
downloaded unless the server was started with ``--allow-download``.
"""
from __future__ import annotations

import gc
import os
import threading
from pathlib import Path
from typing import Any, Callable

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


def create_runtime(key: str, *, model_id: str | None = None, load_in_4bit: bool | None = None,
                   allow_download: bool = False) -> Any:
    """Load the registry model ``key`` (optionally another checkpoint of it) with its own settings."""
    spec = MODEL_REGISTRY[key]
    model_id = model_id or spec.config.model_id
    if not allow_download and not is_downloaded(model_id):
        raise ModelUnavailable(
            f"{spec.label} ({model_id}) is not downloaded. Download it first, for example "
            f"`hf download {model_id}`, or start the server with --allow-download.")
    local = not allow_download
    if spec.runtime == "sarvam":
        from .sarvam import SarvamRuntime
        return SarvamRuntime(model_id, load_in_4bit, local_files_only=local)
    if spec.runtime == "llama":
        from .llama import LlamaRuntime
        return LlamaRuntime(model_id, load_in_4bit, local_files_only=local)
    from .runtime import HuggingFaceRuntime
    return HuggingFaceRuntime(model_id, load_in_4bit, config=spec.config, local_files_only=local)


class ModelUnavailable(ValueError):
    """The requested model is unknown or not on disk."""


class ModelManager:
    """Owns the loaded chat model and swaps it on request.

    ``on_change(key, runtime)`` is called after every load so the server can
    rebuild what depends on the runtime (emotion service, agent model).
    """

    def __init__(self, *, allow_download: bool = False, factory: Callable[..., Any] = create_runtime,
                 on_change: Callable[[str, Any], None] | None = None) -> None:
        self.allow_download = allow_download
        self.factory = factory
        self.on_change = on_change
        self.active: str | None = None
        self.runtime: Any = None
        self.loading: str | None = None
        self._lock = threading.Lock()

    def spec(self, key: str) -> ModelSpec:
        if key not in MODEL_REGISTRY:
            raise ModelUnavailable("Unknown model. Choose one of: " + ", ".join(MODEL_REGISTRY) + ".")
        return MODEL_REGISTRY[key]

    def catalog(self) -> list[dict[str, Any]]:
        """What the chat composer shows: every registry model and whether it can be used now."""
        rows = []
        for key, spec in MODEL_REGISTRY.items():
            downloaded = is_downloaded(spec.config.model_id)
            loaded = key == self.active
            rows.append({
                "key": key, "label": spec.label, "model_id": spec.config.model_id, "purpose": spec.purpose,
                "downloaded": downloaded, "loaded": loaded, "loading": key == self.loading,
                "available": loaded or downloaded or self.allow_download,
                "steering": bool(loaded and getattr(self.runtime, "_emotion_panels", None)),
                "gpu_gib": spec.gpu_gib,
            })
        return rows

    def activate(self, key: str, **options: Any) -> Any:
        """Return the runtime for ``key``, loading it (and unloading the current model) if needed."""
        self.spec(key)
        with self._lock:
            if key == self.active and self.runtime is not None:
                return self.runtime
            spec = MODEL_REGISTRY[key]
            model_id = options.get("model_id") or spec.config.model_id
            if not self.allow_download and not is_downloaded(model_id):
                raise ModelUnavailable(
                    f"{spec.label} ({model_id}) is not downloaded on this server yet. Download it first "
                    f"(`hf download {model_id}`) or start the server with --allow-download.")
            self.loading = key
            try:
                previous = self.runtime
                if previous is not None:
                    # One model at a time: wait for running generations, then free the GPU first.
                    with previous.lock:
                        self.runtime, self.active = None, None
                        if self.on_change:
                            self.on_change("", None)
                        previous.model = None
                    del previous
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                runtime = self.factory(key, allow_download=self.allow_download, **options)
                self.runtime, self.active = runtime, key
                if self.on_change:
                    self.on_change(key, runtime)
                return runtime
            finally:
                self.loading = None
