"""E5 message sets: a development supplement and a fresh held-out set, with leakage checks.  Built once, frozen.

Development data = the 527 already-used messages of the exploratory and confirmatory studies (they have E3/E4
replies rated by both judges) + a small *supplement* for what those sets lack (Hindi, neutral and authored Hinglish
messages, mixed emotions, self-criticism, indirect risk).  Held-out data = new messages that no earlier split,
study, held-out set or neuron-discovery prompt contains, exactly or nearly.

Sources
  english     EmpatheticDialogues: held-out from the *test* split (then validation), supplement from *train*; ED
              situation labels give the category (positive, sadness, fear, anger, self-criticism)
  mixed       GoEmotions messages labelled with both a positive and a negative emotion (odd rows held-out, even
              rows supplement)
  hinglish    held-out: data/mlt_hinghlish_dataset_clean.csv (flagged: the whole corpus was used for
              *recognition*-neuron selectivity, not for the steering panels) + authored messages
  hindi, neutral, safety_indirect, safety_explicit, safety_mention
              data/e5_authored_messages.csv (written for E5 before any E5 reply existed; split column decides)
  safety_explicit (dataset)  unused messages with a knowledge-graph crisis cue

Leakage checks against every earlier message text (``steering_modes.used_messages`` + the two earlier studies'
items) and the panel-discovery prompts (results/*_generation_emotion/prompts/*.csv):
  exact        lower-cased, whitespace-normalised text
  near         character 3-5-gram TF-IDF cosine >= NEAR_CHAR
  semantic     all-MiniLM-L6-v2 cosine >= NEAR_SEMANTIC, Latin-script text only (an English model: unrelated
               Devanagari sentences score ~0.90, so Hindi relies on the exact and n-gram checks)
A drawn candidate that fails is skipped; an authored message that fails stops the build (rewrite it).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .e5_common import ROOT, SEED

DATA = ROOT / "data"
HELDOUT = DATA / "heldout.csv"
SUPPLEMENT = DATA / "dev_supplement.csv"
MANIFEST = DATA / "manifest.json"
AUTHORED = Path("data/e5_authored_messages.csv")
NEAR_CHAR = 0.85
NEAR_SEMANTIC = 0.90
ED_CATEGORY = {
    "positive": ("joyful", "proud", "excited", "grateful", "content", "hopeful", "confident", "impressed", "trusting",
                 "faithful", "caring"),
    "sadness": ("sad", "lonely", "devastated", "disappointed", "sentimental", "nostalgic"),
    "fear_anxiety": ("afraid", "terrified", "anxious", "apprehensive"),
    "anger_frustration": ("angry", "annoyed", "furious", "disgusted", "jealous"),
    "self_criticism": ("ashamed", "guilty", "embarrassed"),
}
HELDOUT_SIZES = {"positive": 28, "sadness": 28, "fear_anxiety": 24, "anger_frustration": 16, "self_criticism": 24,
                 "multi_turn": 40, "mixed": 30, "hinglish_dataset": 40, "safety_explicit_dataset": 12}
SUPPLEMENT_SIZES = {"self_criticism": 24, "mixed": 24}
POSITIVE = ("admiration", "amusement", "approval", "caring", "excitement", "gratitude", "joy", "love", "optimism",
            "pride", "relief")
NEGATIVE = ("anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief",
            "nervousness", "remorse", "sadness")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


# ------------------------------------------------------------------------------------------- used texts

def discovery_prompts() -> set[str]:
    texts = set()
    for path in Path("results").glob("*_generation_emotion/prompts/*.csv"):
        frame = pd.read_csv(path)
        column = "text" if "text" in frame else frame.columns[0]
        texts.update(normalise(text) for text in frame[column].dropna().astype(str))
    return texts


def earlier_texts() -> tuple[set[str], set[str]]:
    from . import steering_modes as sm
    texts, conversations = sm.used_messages()
    for path in (Path("results/steering_modes/items.csv"), Path("results/confirmatory/items.csv")):
        frame = pd.read_csv(path)
        texts |= set(frame.text.map(normalise))
        conversations |= set(frame.conv_id.astype(str))
    return {normalise(text) for text in texts}, conversations


class LeakageIndex:
    """Exact, character-n-gram and MiniLM near-duplicate checks against a set of earlier texts."""

    def __init__(self, texts: set[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.texts = sorted(texts)
        self.exact = set(self.texts)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self._encoder = None
        self._embeddings = None

    def _semantic(self, candidates: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if self._encoder is None:
            self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._embeddings = self._encoder.encode(self.texts, batch_size=256, normalize_embeddings=True,
                                                    show_progress_bar=False)
        query = self._encoder.encode(candidates, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        best = np.zeros(len(candidates))
        for start in range(0, len(candidates), 512):
            best[start:start + 512] = (query[start:start + 512] @ self._embeddings.T).max(axis=1)
        return best

    def check(self, candidates: list[str]) -> pd.DataFrame:
        """Per candidate: exact match, best char-n-gram cosine and best semantic cosine against the earlier texts."""
        normalised = [normalise(text) for text in candidates]
        char = np.zeros(len(candidates))
        if candidates:
            query = self.vectorizer.transform(normalised)
            for start in range(0, len(candidates), 1024):
                char[start:start + 1024] = (query[start:start + 1024] @ self.matrix.T).max(axis=1).toarray().ravel()
        # MiniLM is an English model: Devanagari text falls to near-identical embeddings (unrelated Hindi sentences
        # score ~0.90), so the semantic check covers Latin-script text only; Devanagari relies on exact + n-gram.
        latin = [i for i, text in enumerate(normalised) if not re.search(r"[\u0900-\u097f]", text)]
        semantic = np.full(len(candidates), np.nan)
        if latin:
            semantic[latin] = self._semantic([normalised[i] for i in latin])
        frame = pd.DataFrame({"text": candidates, "exact": [text in self.exact for text in normalised],
                              "char_cosine": char, "semantic_cosine": semantic})
        frame["leak"] = frame.exact | (frame.char_cosine >= NEAR_CHAR) | (frame.semantic_cosine.fillna(0) >= NEAR_SEMANTIC)
        return frame


# ----------------------------------------------------------------------------------------------- pools

def ed_pool(split: str, conversations: set[str], seed: int) -> list[dict[str, Any]]:
    from .steering_lab import _ed_items
    items = _ed_items(split, seed, 10 ** 6, 10 ** 6)
    frame = pd.read_parquet(Path("data/external/empathetic_dialogues/default") / split / "0000.parquet")
    context = frame.drop_duplicates("conv_id").set_index("conv_id").context
    out = []
    for item in items:
        if item["conv_id"] in conversations:
            continue
        situation = context.get(item["conv_id"])
        category = next((name for name, labels in ED_CATEGORY.items() if situation in labels), None)
        if category is None:
            continue
        item.update(gold=situation, category="multi_turn" if item["turns"] == "multi" else category, group="english",
                    language="english")
        out.append(item)
    return out


def mixed_pool(parity: int) -> list[dict[str, Any]]:
    from .steering_lab import _clean
    frame = pd.read_csv("data/goemotions_hardlabels.csv")
    positive = frame[list(POSITIVE)].sum(axis=1) > 0
    negative = frame[list(NEGATIVE)].sum(axis=1) > 0
    frame = frame[positive & negative & (frame.index % 2 == parity)]
    out = []
    for index, row in frame.sample(frac=1.0, random_state=SEED).iterrows():
        text = _clean(row.text)
        if not 25 <= len(text) <= 240 or "[name]" in text.lower() or sum(c.isalpha() for c in text) < 15:
            continue
        labels = [name for name in (*POSITIVE, *NEGATIVE) if row[name]]
        out.append({"source": "goemotions_hardlabels.csv", "conv_id": f"goemotions:{index}", "gold": "+".join(labels),
                    "language": "english", "turns": "single", "text": text, "history": [], "category": "mixed",
                    "group": "english"})
    return out


def authored(split: str) -> list[dict[str, Any]]:
    from .steering_lab import _clean
    frame = pd.read_csv(AUTHORED)
    frame = frame[frame.split == split]
    group = {"hindi": "hindi", "hinglish": "hinglish", "english": "english"}
    out = []
    for index, row in frame.iterrows():
        category = row.category
        out.append({"source": AUTHORED.name, "conv_id": f"authored:{index}", "gold": row.label,
                    "language": row.language, "turns": "single", "text": _clean(row.text), "history": [],
                    "category": category,
                    "group": "safety" if category.startswith("safety") else group[row.language]})
    return out


def crisis_pool(used: set[str]) -> list[dict[str, Any]]:
    from . import steering_modes as sm
    from .steering_lab import _clean
    goemotions = pd.read_csv("data/goemotions_hardlabels.csv", usecols=["text"])
    sources = [("goemotions_hardlabels.csv", f"goemotions:{i}", _clean(str(t))) for i, t in goemotions.text.items()]
    mlt = pd.read_csv(sm.MLT_CSV)
    sources += [(sm.MLT_CSV.name, f"mlt:{i}", _clean(str(t))) for i, t in mlt.hinglish_genz_text.items()]
    for split in ("test", "validation"):
        frame = pd.read_parquet(Path("data/external/empathetic_dialogues/default") / split / "0000.parquet")
        sources += [(f"empathetic_dialogues/{split}", f"{conv}:{index}", _clean(str(text)))
                    for (conv, index, text) in frame[["conv_id", "utterance_idx", "utterance"]].itertuples(index=False)]
    items = sm.crisis_candidates(sources, used)
    for item in items:
        item.update(category="safety_explicit", group="safety")
    return items


# ----------------------------------------------------------------------------------------------- build

def _finish(frame: pd.DataFrame, prefix: str, split: str) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    frame.insert(0, "item_id", [f"{prefix}-{index:03d}" for index in range(len(frame))])
    frame.insert(1, "split", split)
    frame["history"] = frame.history.map(lambda value: json.dumps(value, ensure_ascii=False))
    return frame


def build(dry_run: bool = False, out_dir: Path | None = None) -> dict[str, Any]:
    """Draw, plan, check and (unless ``dry_run``) freeze the supplement and held-out sets."""
    from chatbot_feature.emotion_reasoner.risk_screen import screen
    from chatbot_feature.emotion_reasoner.selective import language_of
    from .common import write_json
    from .freeze_v2 import sha256
    from .steering_lab import Planner, _balanced
    out_dir = out_dir or DATA
    heldout_path, supplement_path, manifest_path = out_dir / HELDOUT.name, out_dir / SUPPLEMENT.name, out_dir / MANIFEST.name
    if heldout_path.exists() and not dry_run:
        raise SystemExit(f"{heldout_path} exists; the E5 message sets are frozen and never rebuilt.")
    used, conversations = earlier_texts()
    discovery = discovery_prompts()
    index = LeakageIndex(used | discovery)
    mlt_corpus = {normalise(text) for text in pd.read_csv("data/mlt_hinghlish_dataset_clean.csv").hinglish_genz_text.astype(str)}

    # Authored messages must be new: stop if any repeats an earlier message (exactly or nearly) or another authored one.
    authored_rows = authored("dev") + authored("heldout")
    texts = [row["text"] for row in authored_rows]
    leaks = index.check(texts)
    duplicates = pd.Series([normalise(t) for t in texts]).duplicated(keep=False)
    if leaks.leak.any() or duplicates.any():
        bad = pd.concat([leaks[leaks.leak], leaks[duplicates.to_numpy()]]).drop_duplicates("text")
        raise SystemExit("Authored messages overlap earlier data or each other; rewrite them:\n" + bad.to_string())

    rng = np.random.default_rng(SEED)
    pools = {"heldout_ed": ed_pool("test", conversations, SEED) + ed_pool("validation", conversations, SEED + 1),
             "supplement_ed": ed_pool("train", conversations, SEED + 2),
             "heldout_mixed": mixed_pool(1), "supplement_mixed": mixed_pool(0)}
    from . import steering_modes as sm
    mlt = pd.read_csv(sm.MLT_CSV)
    pools["heldout_hinglish"] = [dict(item, category="hinglish_dataset", group="hinglish")
                                 for item in sm.hinglish_candidates(mlt, used, 600, SEED)]
    pools["heldout_crisis"] = crisis_pool(used)
    # Leakage filter on every drawn pool (cap the pools first: the checks cost time).
    filtered, excluded = {}, []
    for name, items in pools.items():
        items = [items[i] for i in rng.permutation(len(items))[:2500]] if len(items) > 2500 else items
        found = index.check([item["text"] for item in items])
        keep = [item for item, leak in zip(items, found.leak) if not leak]
        excluded += [{"pool": name, **row} for row in found[found.leak].to_dict("records")]
        print(f"leakage filter {name}: {len(items)} drawn, {int(found.leak.sum())} excluded "
              f"(exact {int(found.exact.sum())}, n-gram {int((found.char_cosine >= NEAR_CHAR).sum())}, "
              f"semantic {int((found.semantic_cosine >= NEAR_SEMANTIC).sum())})", flush=True)
        filtered[name] = keep
    planner = Planner()
    for items in (*filtered.values(), authored_rows):
        for item in items:
            item.update(planner.plan_fields(item))
    planner.close()

    def frame(items: list[dict[str, Any]]) -> pd.DataFrame:
        out = pd.DataFrame(items).drop(columns=["kg_prompt"], errors="ignore")
        if out.empty:
            return out
        out["lang3"] = out.text.map(language_of)
        out["risk_category"] = [screen(text, history)["category"] for text, history in zip(out.text, out.history)]
        out["discovery_overlap"] = ["recognition_corpus" if normalise(text) in mlt_corpus else "none" for text in out.text]
        return out

    def ordinary(part: pd.DataFrame) -> pd.DataFrame:
        return part[~part.safety_flag.astype(bool) & part.target.notna() & (part.target != "")]

    heldout_ed, supplement_ed = frame(filtered["heldout_ed"]), frame(filtered["supplement_ed"])
    parts = []
    for category in ("positive", "sadness", "fear_anxiety", "anger_frustration", "self_criticism", "multi_turn"):
        pool = ordinary(heldout_ed[heldout_ed.category == category])
        parts.append(_balanced(pool, HELDOUT_SIZES[category]))
    mixed = ordinary(frame(filtered["heldout_mixed"]))
    parts.append(_balanced(mixed, HELDOUT_SIZES["mixed"]))
    hinglish = ordinary(frame(filtered["heldout_hinglish"]))
    parts.append(_balanced(hinglish, HELDOUT_SIZES["hinglish_dataset"]))
    crisis = frame(filtered["heldout_crisis"])
    if len(crisis):
        crisis = crisis[crisis.safety_flag.astype(bool)]
        parts.append(crisis.sample(n=min(HELDOUT_SIZES["safety_explicit_dataset"], len(crisis)), random_state=SEED))
    authored_frame = frame(authored_rows)
    parts.append(authored_frame[authored_frame.conv_id.isin([r["conv_id"] for r in authored("heldout")])])
    heldout = _finish(pd.concat(parts, ignore_index=True), "e5h", "e5_heldout")
    supplement_parts = [_balanced(ordinary(supplement_ed[supplement_ed.category == "self_criticism"]),
                                  SUPPLEMENT_SIZES["self_criticism"]),
                        _balanced(ordinary(frame(filtered["supplement_mixed"])), SUPPLEMENT_SIZES["mixed"]),
                        authored_frame[authored_frame.conv_id.isin([r["conv_id"] for r in authored("dev")])]]
    supplement = _finish(pd.concat(supplement_parts, ignore_index=True), "e5d", "e5_dev_supplement")
    every = pd.concat([heldout.text, supplement.text]).map(normalise)
    if every.duplicated().any():
        raise SystemExit("The supplement and held-out sets share a message; refusing to write them.")
    cross = LeakageIndex(set(supplement.text.map(normalise))).check(heldout.text.tolist())
    if cross.leak.any():
        raise SystemExit("A held-out message nearly repeats a development message:\n" + cross[cross.leak].to_string())

    def composition(part: pd.DataFrame) -> dict[str, Any]:
        return {"n": int(len(part)),
                "by_group_category": part.groupby(["group", "category"]).size().reset_index(name="n").to_dict("records"),
                "by_language": part.lang3.value_counts().to_dict(), "by_strategy": part.strategy.value_counts().to_dict(),
                "by_target": part.target.fillna("none").value_counts().to_dict(),
                "by_user_emotion": part.user_emotion.value_counts().to_dict(),
                "by_risk_category": part.risk_category.value_counts().to_dict(),
                "crisis_flagged": int(part.safety_flag.astype(bool).sum()),
                "intensity_quartiles": part.intensity.quantile([0.25, 0.5, 0.75]).round(3).tolist(),
                "discovery_overlap": part.discovery_overlap.value_counts().to_dict()}
    manifest = {
        "seed": SEED, "heldout": composition(heldout), "dev_supplement": composition(supplement),
        "leakage": {"earlier_texts": len(used), "earlier_conversations": len(conversations),
                    "discovery_prompts": len(discovery), "thresholds": {"char_ngram_cosine": NEAR_CHAR,
                                                                        "semantic_cosine": NEAR_SEMANTIC},
                    "excluded_candidates": len(excluded),
                    "max_char_cosine_heldout": float(index.check(heldout.text.tolist()).char_cosine.max()),
                    "max_semantic_cosine_heldout_latin": float(index.check(heldout.text.tolist()).semantic_cosine.max()),
                    "heldout_vs_supplement_max_char": float(cross.char_cosine.max())},
        "sources": __doc__.split("Sources")[1].split("Leakage")[0].strip(),
        "caveats": {"authored": "Hindi, part of the Hinglish, neutral and safety messages were written for E5 by an AI "
                                "assistant (not native speakers or clinicians); review before freezing if possible",
                    "hinglish_dataset": "machine-generated GenZ Hinglish from the corpus whose every row was used for "
                                        "recognition-neuron selectivity (not for the steering panels)",
                    "crisis": "dataset crisis-cue messages are few; authored explicit / indirect / mention messages complete the group"},
        "note": "Frozen before any E5 policy was fitted on development data and before any model replied to these messages.",
    }
    if dry_run:
        return {"manifest": manifest, "heldout": heldout, "supplement": supplement, "excluded": excluded}
    out_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(heldout_path, index=False)
    supplement.to_csv(supplement_path, index=False)
    pd.DataFrame(excluded).to_csv(out_dir / "leakage_excluded.csv", index=False)
    manifest["sha256"] = {"heldout": sha256(heldout_path), "dev_supplement": sha256(supplement_path),
                          "authored_source": sha256(AUTHORED)}
    write_json(manifest_path, manifest)
    return {"manifest": manifest, "heldout": heldout, "supplement": supplement, "excluded": excluded}


def load(name: str, limit: int | None = None) -> pd.DataFrame:
    """``heldout`` or ``dev_supplement``, checked against the frozen hash."""
    from .freeze_v2 import sha256
    path = {"heldout": HELDOUT, "dev_supplement": SUPPLEMENT}[name]
    if not path.exists():
        raise SystemExit(f"{path} is missing: run --stages build first.")
    manifest = json.loads(MANIFEST.read_text())
    if sha256(path) != manifest["sha256"][name]:
        raise SystemExit(f"{path} changed after it was frozen.")
    frame = pd.read_csv(path)
    frame["history"] = frame.history.map(json.loads)
    return frame.head(limit) if limit else frame


__all__ = ["build", "load", "LeakageIndex", "HELDOUT", "SUPPLEMENT", "MANIFEST", "AUTHORED"]
