"""Pre-flight for the live safety evaluation: packages, GPU, models on disk and (optionally) one real assessment.

  python3 -m chatbot_feature.evaluation.safety.check_env                 # no model loaded
  python3 -m chatbot_feature.evaluation.safety.check_env --load llama    # load, run one safety assessment
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import shutil
import subprocess
import sys
import time


def main() -> int:
    from backend.config import MODEL_REGISTRY
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load", choices=tuple(MODEL_REGISTRY), help="Also load this preset and run one assessment")
    parser.add_argument("--model-id", help="Local checkpoint folder for --load (default: the preset's)")
    args = parser.parse_args()
    ok = True

    print("== packages")
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "pydantic", "langgraph", "networkx", "sentencepiece"):
        try:
            module = importlib.import_module(name)
            print(f"  {name:14} {getattr(module, '__version__', 'installed')}")
        except Exception as exc:
            print(f"  {name:14} MISSING ({type(exc).__name__})")
            ok = False

    print("== GPU")
    import torch
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            print(f"  cuda:{index} {torch.cuda.get_device_name(index)}  free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
    else:
        print("  CUDA is not available")
        ok = False
    if shutil.which("nvidia-smi"):
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
        print("  processes on the GPU:", apps.replace("\n", "; ") if apps else "none")

    print("== models on disk")
    from backend.models import is_downloaded
    for key, spec in MODEL_REGISTRY.items():
        print(f"  {key:9} {spec.config.model_id:40} {'yes' if is_downloaded(spec.config.model_id) else 'NOT DOWNLOADED'}")

    if args.load:
        from backend.models import create_runtime
        from chatbot_feature.safety import SafetyAgent
        from chatbot_feature.safety.runtime import SafetyRuntimeModel
        from .harness import RecordingModel
        print(f"== loading {args.load}")
        started = time.perf_counter()
        runtime = create_runtime(args.load, model_id=args.model_id, allow_download=False)
        print(f"  loaded {runtime.model_id} in {time.perf_counter() - started:.0f} s")
        if torch.cuda.is_available():
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
        model = RecordingModel(SafetyRuntimeModel(runtime))
        agent = SafetyAgent(model, timeout_seconds=600)
        for text in ("I am nervous about my exam tomorrow.", "I want to end my life."):
            result = asyncio.run(agent.assess(text))
            call = model.calls[-1] if model.calls else {}
            print(f"  synthetic probe -> {result.policy.risk_level} ({result.policy.route}); model said "
                  f"{result.assessment.risk_level}; fallback={result.fallback}; call={call.get('status')}; "
                  f"output parse={call.get('parse')}; {call.get('ms', 0) / 1000:.1f} s")
            if call.get("status") != "ok":
                print(f"  ERROR: the safety model call itself failed ({call.get('status')}); the agent fell back to R1.")
                ok = False
            elif call.get("parse") not in ("valid", "valid_fenced"):
                print("  WARNING: the model's output did not match the strict schema; the agent fell back to R1.")
                ok = False
    print("\nREADY" if ok else "\nNOT READY (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
