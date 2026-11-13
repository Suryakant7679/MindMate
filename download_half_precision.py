#!/usr/bin/env python3
"""Download a Hugging Face model whose weights are stored as float32, keeping them as bfloat16.

Each weight shard is downloaded, converted and removed before the next one, so the
disk never holds more than the half-precision copy plus one shard.  The result sits
in the normal Hugging Face cache under the same file names, so the model loads as
usual (it is quantised to 4-bit at load time anyway).

  python3 download_half_precision.py sarvamai/sarvam-30b
"""
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file


def main(repo: str) -> None:
    files = [item.rfilename for item in HfApi().model_info(repo).siblings]
    weights = sorted(name for name in files if name.endswith(".safetensors"))
    for name in files:
        if not name.endswith((".safetensors", ".bin", ".gguf")):
            hf_hub_download(repo, name)
    snapshot = Path(hf_hub_download(repo, "config.json")).parent
    staging = snapshot.parent.parent / "half_precision_staging"      # moved into place only when complete,
    staging.mkdir(exist_ok=True)                                     # so a partial model never looks usable
    for index, name in enumerate(weights, 1):
        target = staging / name
        if target.exists() or (snapshot / name).exists():
            continue
        with tempfile.TemporaryDirectory(dir=snapshot.parent.parent, prefix="convert-") as scratch:
            source = hf_hub_download(repo, name, local_dir=scratch)
            tensors = {key: value.to(torch.bfloat16) if value.dtype == torch.float32 else value
                       for key, value in load_file(source).items()}
            partial = Path(scratch) / "converted.safetensors"
            save_file(tensors, str(partial), metadata={"format": "pt"})
            shutil.move(str(partial), target)
        print(f"{index}/{len(weights)} {name} -> {target.stat().st_size / 1e9:.2f} GB", flush=True)
    for name in weights:
        if (staging / name).exists():
            shutil.move(str(staging / name), snapshot / name)
    staging.rmdir()
    print(f"Done: {snapshot}")


if __name__ == "__main__":
    main(sys.argv[1])
