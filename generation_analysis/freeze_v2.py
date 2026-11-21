#!/usr/bin/env python3
"""Freeze the deployed (v2) steering configuration and its evaluation results.

For each model that has a deployed policy, the panel file, ``policy.json``, the
Mood Tracker evidence file and the earlier held-out / calibration result tables
are copied to ``results/frozen_v2/<tag>/`` together with a SHA-256 manifest of
the *live* files.  Later experiments read the frozen copies and call
``verify()`` first, so an experiment can neither change the deployed
configuration unnoticed nor run against a configuration that has drifted.

  python3 -m generation_analysis.freeze_v2            # freeze (refuses to overwrite an existing freeze)
  python3 -m generation_analysis.freeze_v2 --verify   # check that the deployed files are unchanged
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from backend.config import MODEL_REGISTRY

ROOT = Path("results")
FROZEN = ROOT / "frozen_v2"
MODELS = ("llama", "emollama", "llama2")        # llama2 is frozen after its own pipeline has installed a policy
RESULT_TABLES = ("causal_steering/heldout", "causal_steering/recalibration", "causal_steering/splits")
SKIP = ("generations.jsonl",)          # large raw generations stay where they are (hashed, not copied)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def live_files(key: str) -> dict[str, Path]:
    """The files that define what the chatbot does for this model."""
    config = MODEL_REGISTRY[key].config
    base = ROOT / f"{config.results_tag}_generation_emotion"
    return {"panels": base / "steering_panels" / config.panel_file,
            "policy": base / "steering_policy" / "policy.json",
            "evidence": base / "causal_steering" / "evidence.json"}


def frozen_dir(key: str) -> Path:
    return FROZEN / MODEL_REGISTRY[key].config.results_tag


def frozen_models() -> list[str]:
    """Models that have a freeze (a model is frozen after its own pipeline installed a policy)."""
    return [key for key in MODELS if (frozen_dir(key) / "manifest.json").exists()]


def freeze(key: str) -> None:
    target = frozen_dir(key)
    if (target / "manifest.json").exists():
        raise SystemExit(f"{target} is already frozen; delete it by hand if you really mean to refreeze.")
    config = MODEL_REGISTRY[key].config
    base = ROOT / f"{config.results_tag}_generation_emotion"
    (target / "deployed").mkdir(parents=True)
    manifest = {"model": key, "model_id": config.model_id, "panel_file": config.panel_file,
                "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "deployed": {}, "results": {}}
    for name, path in live_files(key).items():
        shutil.copy2(path, target / "deployed" / path.name)
        manifest["deployed"][name] = {"path": str(path), "sha256": sha256(path)}
    for folder in RESULT_TABLES:
        for path in sorted((base / folder).glob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(base)
            manifest["results"][str(relative)] = sha256(path)
            if path.name not in SKIP:
                (target / relative.parent).mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target / relative)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Froze {key}: {target}")


def verify(key: str) -> dict:
    """Raise SystemExit unless the deployed files are byte-identical to the frozen ones; returns the manifest."""
    path = frozen_dir(key) / "manifest.json"
    if not path.exists():
        raise SystemExit(f"No freeze for {key}. Run: python3 -m generation_analysis.freeze_v2")
    manifest = json.loads(path.read_text())
    changed = [name for name, entry in manifest["deployed"].items() if sha256(Path(entry["path"])) != entry["sha256"]]
    if changed:
        raise SystemExit(f"Deployed {key} files differ from the v2 freeze: {', '.join(changed)}")
    return manifest


def frozen_policy(key: str) -> dict:
    verify(key)
    return json.loads((frozen_dir(key) / "deployed" / "policy.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        help="Default: verify every frozen model; freeze every model that is not frozen yet.")
    args = parser.parse_args()
    frozen = [key for key in MODELS if (frozen_dir(key) / "manifest.json").exists()]
    pending = [key for key in MODELS if key not in frozen and all(path.exists() for path in live_files(key).values())]
    for key in args.models or (frozen if args.verify else pending):
        if args.verify:
            manifest = verify(key)
            print(f"{key}: deployed panels, policy and evidence match the freeze of {manifest['frozen_at']}")
        else:
            freeze(key)


if __name__ == "__main__":
    main()
