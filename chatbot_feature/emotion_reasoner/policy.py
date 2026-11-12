"""Calibrated steering policy: strengths chosen from a controlled sweep.

``policy.json`` is written by ``generation_analysis.calibrate_steering_policy``.
For each context cell it holds the strongest steering strength that raised the
target emotion without lowering judged empathy, relevance or fluency, together
with the sweep that supports the choice.  A cell without such a strength has
alpha 0: the policy is then not to steer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

DEFAULT_PATH = (Path(__file__).resolve().parents[2] / "results" / "llama31_8b_generation_emotion"
                / "steering_policy" / "policy.json")


def load_policy(path: str | Path = DEFAULT_PATH) -> dict[str, Any] | None:
    try:
        policy = json.loads(Path(path).read_text(encoding="utf-8"))
        return policy if isinstance(policy.get("cells"), dict) else None
    except (OSError, ValueError, AttributeError):
        return None


def lookup(policy: Mapping[str, Any], strategy: str, response_emotion: str, hinglish: bool
           ) -> tuple[str, dict[str, Any]] | None:
    """Most specific calibrated cell: (strategy, language), else (response emotion, language)."""
    language = "hinglish" if hinglish else "english"
    for key in (f"strategy:{strategy}|{language}", f"{response_emotion}|{language}"):
        cell = policy["cells"].get(key)
        if cell is not None:
            return key, cell
    return None
