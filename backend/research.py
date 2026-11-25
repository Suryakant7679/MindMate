"""Research records of the chatbot: per-turn telemetry, per-neuron records and voluntary feedback.

One ``ResearchStore`` belongs to one loaded model.  It keeps the turns of each conversation in
memory (for the Mood Tracker) and, when persistence is on, appends them to files under
``results/telemetry/<model tag>/``:

  turns.jsonl            one line per replied turn: what was detected, planned, decided, both replies,
                         their classifier scores, the run-time comparison and the per-layer measurements
  neurons/<turn>.json    every steered neuron of a turn: id, layer, activation before and after, shift,
                         normalised shift, write-vector norm and contribution
  feedback.jsonl         voluntary user ratings (own mood for a turn, helpfulness of a reply)
  judged.jsonl           written offline by ``generation_analysis.turn_evaluation``: both judges' ratings of
                         the two replies of a turn; read back here and shown next to the turn

The files hold the user's messages in plain text and stay on this machine (``results/`` is not under
version control).  Set ``MINDMATE_TELEMETRY=0`` to keep everything in memory only.
"""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA = 1
MAX_CONVERSATIONS = 200
ROOT = Path(__file__).resolve().parent.parent / "results" / "telemetry"


def telemetry_enabled() -> bool:
    return os.environ.get("MINDMATE_TELEMETRY", "1").lower() not in {"0", "false", "no", "off"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResearchStore:
    def __init__(self, tag: str, directory: str | Path | None = None, persist: bool = False) -> None:
        self.tag = tag
        self.directory = Path(directory) if directory else Path(os.environ.get("MINDMATE_TELEMETRY_DIR", ROOT)) / tag
        self.persist = persist
        self._lock = threading.Lock()
        self._conversations: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._judged: dict[str, dict[str, Any]] = {}
        self._judged_stamp: float | None = None

    # ------------------------------------------------------------------ files

    def _append(self, name: str, payload: Mapping[str, Any]) -> None:
        if not self.persist:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / name).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:                       # telemetry must never break a reply
            print(f"Telemetry not written ({exc}).")

    def save_neurons(self, turn_id: str, payload: Mapping[str, Any]) -> None:
        """Per-neuron record of one steered reply (also kept for the last turns in memory)."""
        with self._lock:
            self._neurons = getattr(self, "_neurons", OrderedDict())
            self._neurons[turn_id] = dict(payload)
            while len(self._neurons) > 50:
                self._neurons.popitem(last=False)
        if not self.persist:
            return
        try:
            folder = self.directory / "neurons"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{turn_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            print(f"Neuron record not written ({exc}).")

    def neurons(self, turn_id: str) -> dict[str, Any] | None:
        with self._lock:
            found = getattr(self, "_neurons", {}).get(turn_id)
        if found is not None:
            return found
        path = self.directory / "neurons" / f"{turn_id}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except (OSError, ValueError):
            return None

    def judged(self, turn_id: str) -> dict[str, Any] | None:
        """Offline judge results for a turn, if ``turn_evaluation`` has been run since it was recorded."""
        path = self.directory / "judged.jsonl"
        try:
            stamp = path.stat().st_mtime
        except OSError:
            return None
        with self._lock:
            if stamp != self._judged_stamp:
                loaded = {}
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            item = json.loads(line)
                            loaded[item["turn_id"]] = item
                except (OSError, ValueError, KeyError):
                    loaded = {}
                self._judged, self._judged_stamp = loaded, stamp
            return self._judged.get(turn_id)

    # ---------------------------------------------------------- conversations

    def _conversation(self, conversation_id: str) -> dict[str, Any]:
        found = self._conversations.get(conversation_id)
        if found is None:
            found = self._conversations[conversation_id] = {"turns": {}, "mood": {}, "helpfulness": {}}
            while len(self._conversations) > MAX_CONVERSATIONS:
                self._conversations.popitem(last=False)
        else:
            self._conversations.move_to_end(conversation_id)
        return found

    def record(self, conversation_id: str, turn_index: int, record: Mapping[str, Any], *, persist: bool = True) -> None:
        """Store the record of one turn (a regenerated turn replaces the earlier record of that index)."""
        entry = {"schema": SCHEMA, "time": now(), "conversation_id": conversation_id, "turn_index": turn_index, **record}
        with self._lock:
            conversation = self._conversation(conversation_id)
            conversation["turns"][turn_index] = entry
            # A regenerated turn invalidates everything that was said after it.
            for later in [index for index in conversation["turns"] if index > turn_index]:
                del conversation["turns"][later]
        if persist:
            self._append("turns.jsonl", entry)

    def forget(self, conversation_id: str) -> None:
        """Remove a conversation's in-memory records (a user deleted it).  Opt-in telemetry files are not rewritten:
        they are operator-managed research data collected with participant consent."""
        with self._lock:
            self._conversations.pop(conversation_id, None)

    def has_turn(self, conversation_id: str, turn_index: int, text: str | None = None) -> bool:
        with self._lock:
            entry = self._conversations.get(conversation_id, {}).get("turns", {}).get(turn_index)
        return entry is not None and (text is None or entry.get("user", {}).get("text") == text)

    def turns(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            return [conversation["turns"][index] for index in sorted(conversation["turns"])] if conversation else []

    def feedback(self, conversation_id: str, turn_index: int, *, mood_rating: int | None = None,
                 helpfulness: int | None = None) -> dict[str, Any]:
        """Voluntary ratings: the user's own mood when writing turn ``turn_index`` (1-5) and how supportive
        the reply to it was (1-5).  A new rating for the same turn replaces the earlier one."""
        with self._lock:
            conversation = self._conversation(conversation_id)
            if mood_rating is not None:
                conversation["mood"][turn_index] = int(mood_rating)
            if helpfulness is not None:
                conversation["helpfulness"][turn_index] = int(helpfulness)
            turn = conversation["turns"].get(turn_index, {})
        entry = {"schema": SCHEMA, "time": now(), "conversation_id": conversation_id, "turn_index": turn_index,
                 "turn_id": turn.get("turn_id"), "mood_rating": mood_rating, "helpfulness": helpfulness}
        self._append("feedback.jsonl", entry)
        return entry

    def ratings(self, conversation_id: str) -> tuple[dict[int, int], dict[int, int]]:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                return {}, {}
            return dict(conversation["mood"]), dict(conversation["helpfulness"])

    def counts(self) -> dict[str, Any]:
        with self._lock:
            return {"conversations": len(self._conversations),
                    "turns": sum(len(item["turns"]) for item in self._conversations.values()),
                    "persisted": self.persist, "directory": str(self.directory) if self.persist else None}


def load_json(path: str | Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_cards(path: str | Path) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    """Offline per-neuron evidence written by ``generation_analysis.neuron_validation cards``:
    ``{panel: {"<layer>:<neuron>": {...}}}`` -> ``{panel: {(layer, neuron): {...}}}``."""
    payload = load_json(path) or {}
    cards: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for panel, item in (payload.get("panels") or {}).items():
        cards[panel] = {}
        for key, value in (item.get("neurons") or {}).items():
            layer, neuron = key.split(":")
            cards[panel][(int(layer), int(neuron))] = value
    return cards


# --------------------------------------------------------------------------- saved results of every model

MODEL_RESULTS = (("llama", "llama31_8b", "Llama 3.1-8B"), ("llama2", "llama2_13b_chat", "Llama-2-13B-chat"),
                 ("emollama", "emollama_chat_7b", "EmoLLaMA-chat-7B"),
                 ("sarvam", "sarvam_30b", "Sarvam-30B"))
RESULTS = Path(__file__).resolve().parent.parent / "results"
SHOWN_QUALITY = ("appropriateness", "empathy", "relevance", "fluency", "naturalness", "hinglish_naturalness")


def _dose_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"alpha": row["alpha"], "n": row["n"], "emotion": row["emotion"],
            "judges": {name: stat.get("mean") for name, stat in (row.get("emotion_judges") or {}).items()},
            "independent": (row.get("independent") or {}).get("mean"),
            "quality": {metric: (row.get("quality") or {}).get(metric, {}).get("mean") for metric in SHOWN_QUALITY},
            "harm_share": row.get("harm_share"), "preference": row.get("preference"),
            "length_ratio": row.get("length_ratio")}


def _battery(report: Mapping[str, Any]) -> dict[str, Any]:
    pooled = report["pooled"]
    by_name = {row["name"]: row for row in pooled["conditions"]}
    doses = sorted((row for row in pooled["conditions"] if row["family"] == "dose"), key=lambda row: row["alpha"])
    languages: dict[str, Any] = {}
    for block in report["panels"].values():
        for language, part in block["by_language"].items():
            entry = languages.setdefault(language, {"messages": part["messages"], "panels": {}})
            entry["panels"][block["panel"]] = [_dose_row(row) for row in sorted(
                (row for row in part["conditions"] if row["family"] == "dose"), key=lambda row: row["alpha"])]
    return {
        "generated": report["generated"], "messages": report["messages"], "split": report["split"],
        "languages": pooled["languages"], "reference_alpha": report["reference_alpha"], "judges": report["judges"],
        "verdicts": pooled["verdict"], "panels": {name: block["verdict"] for name, block in report["panels"].items()},
        "doses": [_dose_row(row) for row in doses],
        "controls": {key: by_name[name]["emotion"] for key, name in (("sham", "sham"), ("matched_random", "random_mean"))
                     if name in by_name},
        "reverse": next((row["emotion"] for row in pooled["conditions"] if row["family"] == "reverse"), None),
        "usable_alpha": (pooled["verdict"].get("quality") or {}).get("alpha"),
        "by_language": languages, "agreement": report["judge_agreement"], "sampling": report.get("sampling"),
    }


def _sarvam_calibration(root: Path) -> list[dict[str, Any]]:
    """Sarvam's two-judge calibration sweep of the tone panels, per language (saved table; no inference)."""
    import csv
    path = root / "steering" / "calibrate" / "intervention_summary.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "tone":
                continue
            number = lambda key: float(row[key]) if row.get(key) not in (None, "", "nan") else None   # noqa: E731
            rows.append({"language": row["language"], "alpha": number("alpha"), "n": int(float(row["n"])),
                         "emotion": {"mean": number("tone_gain"), "lo": number("tone_gain_lo"), "hi": number("tone_gain_hi"),
                                     "n": int(float(row["n"]))},
                         "quality": {metric: number(f"{metric}_change") for metric in SHOWN_QUALITY},
                         "harm_share": number("harm_share"), "preference": number("preference"),
                         "hinglish_share_change": number("hinglish_share_change")})
    return rows


def model_evaluation(loaded_tag: str | None = None) -> dict[str, Any]:
    """Saved experimental results of the three models, read from disk (opening the dashboard runs no model)."""
    models = []
    for key, tag, label in MODEL_RESULTS:
        root = RESULTS / f"{tag}_generation_emotion"
        report = load_json(root / "neuron_validation" / "report.json")
        policy = load_json(root / "steering_policy" / "policy.json")
        installed = policy is not None
        if policy is None:
            policy = load_json(root / "steering" / "policy_candidate.json")
        evidence = load_json(root / "causal_steering" / "evidence.json") or load_json(root / "steering" / "evidence.json") or {}
        cells = (policy or {}).get("cells") or {}
        steered = [name for name, cell in cells.items() if cell.get("alpha")]
        manifest = load_json(root / "neuron_validation" / "manifest.json") or {}
        in_progress = report is None and (root / "neuron_validation" / "generations.jsonl").exists()
        if not root.exists() and report is None and policy is None:
            continue                               # a model whose pipeline has not been started is not listed
        models.append({
            "key": key, "tag": tag, "label": label, "model": (report or manifest).get("model"), "loaded": tag == loaded_tag,
            "policy": ({"installed": installed, "steering_enabled": bool(steered), "steered_cells": steered,
                        "cells": len(cells), "source": policy.get("source"), "split": policy.get("split"),
                        "note": policy.get("note")} if policy else None),
            "battery": _battery(report) if report else None,
            "battery_status": "complete" if report else "in progress" if in_progress else "not run",
            "held_out": evidence.get("held_out"),
            "calibration": _sarvam_calibration(root) if key == "sarvam" else [],
            # Steering-mode comparison on the fresh held-out set (generation_analysis.steering_modes), when run.
            "steering_modes": load_json(RESULTS / "steering_modes" / tag / "summary.json"),
        })
    return {"models": models, "detector": load_json(RESULTS / "emotion_trajectory" / "validation.json"),
            "steering_modes_set": load_json(RESULTS / "steering_modes" / "messages.json"),
            "harm_rule": "a reply counts as harmed when any quality rating (mean of the judges) falls by more than 0.5 points"}
