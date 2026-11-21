#!/usr/bin/env python3
"""Single-neuron interventions: does any individual panel neuron change the emotion of replies on its own?

The panel-level battery (``neuron_validation``) cannot say this.  Testing every neuron by generation is not
feasible (thousands of neurons x messages x replies), so the study has two stages:

  screen     Teacher-forced and classifier-free, for many neurons.  Candidates per panel: the K top-ranked panel
             neurons (gradient estimate where one is saved, else the selection score) and K control neurons
             outside every panel, each matched to one candidate (same layer, similar neutral activation, same
             shift value).  For each neuron and each message the model scores two saved replies to the same
             knowledge-graph prompt: the reply prompted for the emotion and the plain reply.  The screen effect is

                 [logp(emotional reply | intervention) - logp(emotional reply)]
               - [logp(plain reply     | intervention) - logp(plain reply)]        (nats per token)

             i.e. how much the intervention moves the model toward the emotional reply rather than toward text
             in general.  Interventions: the neuron alone at 1x, 8x and 32x the panel's reference strength
             (sufficiency, with dose), and the full panel without the neuron against the full panel (necessity).
  generate   Replies under the strongest single-neuron dose for the M best-screened neurons of each panel and M
             of the control neurons, plus the panel without each of those neurons, the full panel and no steering.
  score      Both judges and the independent classifier, as in the battery.
  analyse    Per neuron: screen effect with CI, Holm correction over the tested neurons of the panel, comparison
             with the control neurons' null distribution, judged effect against control neurons.

A neuron is reported as individually supported only if (1) its screen effect has a CI above zero after Holm
correction, (2) it exceeds the 95th percentile of the control neurons at the same dose, and (3) the judges' emotion
rating of generated replies rises more than for control neurons (CI above zero).  Passing only (1)-(2) is
"likelihood only".  Untested neurons stay "not tested".  A 32x shift of one neuron is far outside normal
activations, so (3) is read together with the quality ratings.

Needs the battery of the model (its neutral means and saved replies):
  python3 -m generation_analysis.single_neuron --model llama all
"""
from __future__ import annotations

import os
import sys

if "--model" in sys.argv:                 # the experiment helpers read the analysed model at import time
    _key = sys.argv[sys.argv.index("--model") + 1]
    if _key in {"llama", "llama2", "emollama", "sarvam"}:
        os.environ["MINDMATE_MODEL"] = _key

import argparse  # noqa: E402
import json  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import neuron_validation as nv  # noqa: E402
from .steering_lab import holm, release  # noqa: E402

DOSES = (1.0, 8.0, 32.0)          # multiples of the panel's reference strength for one neuron alone
NULL_PERCENTILE = 95.0


def single_paths(key: str) -> dict[str, Path]:
    root = nv.model_paths(key)["root"] / "single"
    return {"root": root, "candidates": root / "candidates.json", "screen": root / "screen.csv",
            "generations": root / "generations.jsonl", "scores": root / "scores.csv", "report": root / "report.json"}


# --------------------------------------------------------------------------- candidates

def neuron_scores(key: str, name: str, panel: nv.Panel) -> tuple[dict[int, dict[int, float]], str]:
    """Score each panel neuron is ranked by for candidate choice, and what that score is."""
    cards = nv.model_paths(key)["cards"]
    if cards.exists():
        saved = json.loads(cards.read_text())["panels"].get(name, {}).get("neurons", {})
        if saved and all("gradient_effect" in card for card in saved.values()):
            return ({layer: {n: float(saved[f"{layer}:{n}"]["gradient_effect"]) for n in neurons}
                     for layer, (neurons, _) in panel.items()}, "gradient estimate (dL/d shift x shift)")
    return nv.ranking(key, name, panel), ("attribution score" if nv.MODELS[key]["source"] == "sarvam" else "stored shift magnitude")


def choose(panel: nv.Panel, scores: dict[int, dict[int, float]], count: int) -> list[tuple[int, int, float]]:
    """The ``count`` best-scored neurons of the whole panel as (layer, neuron, stored shift)."""
    flat = [(scores[layer][neuron], layer, neuron, delta) for layer, (neurons, deltas) in panel.items()
            for neuron, delta in zip(neurons, deltas)]
    return [(layer, neuron, float(delta)) for _, layer, neuron, delta in sorted(flat, key=lambda item: -item[0])[:count]]


def controls(chosen: list[tuple[int, int, float]], banned: dict[int, set[int]], widths: dict[int, int], neutral: np.ndarray,
             start: dict[int, int], rng: np.random.Generator) -> list[tuple[int, int, float]]:
    """One control neuron per candidate: same layer, outside every panel, nearest neutral activation magnitude
    among a random window, same shift value."""
    out, used = [], {layer: set() for layer in widths}
    for layer, neuron, delta in chosen:
        magnitude = np.abs(neutral[start[layer]:start[layer] + widths[layer]])
        pool = np.array([n for n in range(widths[layer]) if n not in banned.get(layer, set()) and n not in used[layer]])
        order = pool[np.argsort(np.abs(magnitude[pool] - magnitude[neuron]), kind="stable")][:max(8, len(pool) // 50)]
        pick = int(order[int(rng.integers(len(order)))])
        used[layer].add(pick)
        out.append((layer, pick, delta))
    return out


def one(layer: int, neuron: int, delta: float) -> nv.Panel:
    return {layer: ([neuron], [delta])}


def without(panel: nv.Panel, layer: int, neuron: int) -> nv.Panel:
    out = dict(panel)
    neurons, deltas = panel[layer]
    keep = [i for i, n in enumerate(neurons) if n != neuron]
    out[layer] = ([neurons[i] for i in keep], [deltas[i] for i in keep])
    return out


def build_candidates(key: str, runtime, args) -> dict[str, Any]:
    from .sarvam_lab import sites
    paths = single_paths(key)
    if paths["candidates"].exists():
        return json.loads(paths["candidates"].read_text())
    widths = sites(runtime)
    start = nv.site_offsets(widths)
    neutral = np.load(nv.model_paths(key)["neutral"])
    names = nv.panel_names(key, runtime)
    banned: dict[int, set[int]] = {}
    for by_layer in runtime._emotion_panels.values():
        for layer, (neurons, _) in by_layer.items():
            banned.setdefault(layer, set()).update(neurons)
    payload: dict[str, Any] = {"model_key": key, "reference": nv.MODELS[key]["reference"], "doses": list(DOSES), "panels": {}}
    for name in names:
        panel = runtime._emotion_panels[name]
        scores, basis = neuron_scores(key, name, panel)
        rng = np.random.default_rng(zlib.crc32(f"single|{key}|{name}".encode()))
        chosen = choose(panel, scores, args.neurons)
        payload["panels"][name] = {"ranked_by": basis, "panel": [list(item) for item in chosen],
                                   "control": [list(item) for item in controls(chosen, banned, widths, neutral, start, rng)]}
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["candidates"].write_text(json.dumps(payload, indent=1))
    return payload


def saved_replies(key: str) -> pd.DataFrame:
    """Per message and panel: the plain reply and the reply prompted for the emotion (from the battery)."""
    rows = [json.loads(line) for line in nv.model_paths(key)["generations"].open(encoding="utf-8")]
    frame = pd.DataFrame([row for row in rows if row["name"] in ("none", "styled")])
    wide = frame.pivot_table(index=["item_id", "panel"], columns="name", values="reply", aggfunc="first").reset_index()
    return wide.dropna(subset=["none", "styled"])


def pick_messages(key: str, count: int) -> pd.DataFrame:
    """Development messages that have saved battery replies, balanced over languages (round-robin)."""
    os.environ["MINDMATE_VALIDATION_HINGLISH"] = "1"        # Llama family: include the Hinglish messages if the battery has them
    frame = nv.load_messages(key, None)
    frame = frame[frame.item_id.isin(set(saved_replies(key).item_id))]
    groups = [part.reset_index(drop=True) for _, part in frame.groupby("language", sort=True)]
    order = [group.iloc[index] for index in range(max(map(len, groups), default=0)) for group in groups if index < len(group)]
    return pd.DataFrame(order[:count]).reset_index(drop=True)


# --------------------------------------------------------------------------- screen

def log_likelihoods(runtime, prompt_ids, reply_ids, specs: list[dict], widths: dict[int, int], batch: int) -> np.ndarray:
    """Mean log-probability of the reply tokens under each intervention (teacher-forced; shift on decode positions).

    ``specs`` use the battery's format: ``{"shift": [(panel, alpha), ...]}``; an empty dict is no intervention.
    Reply token 0 is predicted from the prompt, before any shift acts, so tokens 1.. are scored.
    """
    import torch
    from backend.config import STEERING_GAIN
    device = prompt_ids.device
    ids = torch.cat([prompt_ids, reply_ids])
    length = len(prompt_ids)
    out = []
    for begin in range(0, len(specs), batch):
        chunk = specs[begin:begin + batch]
        deltas: dict[int, Any] = {}
        for row, spec in enumerate(chunk):
            for panel, alpha in spec.get("shift", []):
                for layer, (neurons, shifts) in panel.items():
                    if layer not in deltas:
                        deltas[layer] = torch.zeros(len(chunk), widths[layer])
                    deltas[layer][row, neurons] += float(alpha) * STEERING_GAIN * torch.tensor(shifts)

        def hook_for(layer: int):
            def hook(_module, inputs):
                hidden = inputs[0]
                if hidden.dim() != 3 or hidden.shape[1] != len(ids):
                    return inputs
                mask = torch.zeros(len(ids), 1, device=hidden.device, dtype=hidden.dtype)
                mask[length:] = 1
                return (hidden + mask * deltas[layer].to(device=hidden.device, dtype=hidden.dtype)[:, None, :], *inputs[1:])
            return hook
        handles = [runtime.steering_module(layer).register_forward_pre_hook(hook_for(layer)) for layer in deltas]
        try:
            with runtime.lock, torch.inference_mode():
                batch_ids = ids[None].repeat(len(chunk), 1)
                logits = runtime.model(input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids)).logits[:, length:-1].float()
                logp = torch.log_softmax(logits, dim=-1)
                target = reply_ids[1:][None, :, None].expand(len(chunk), -1, 1)
                out.append(logp.gather(2, target)[..., 0].mean(dim=1).cpu().numpy())
        finally:
            for handle in handles:
                handle.remove()
    return np.concatenate(out)


def screen_specs(panel: nv.Panel, block: dict[str, Any], reference: float) -> list[tuple[str, str, int, int, float | None, dict]]:
    """(condition, role, layer, neuron, dose, spec) for one panel: baseline, full panel, each neuron alone and left out."""
    specs: list[tuple[str, str, int, int, float | None, dict]] = [("none", "", -1, -1, None, {}),
                                                                 ("full", "", -1, -1, None, {"shift": [(panel, reference)]})]
    for role in ("panel", "control"):
        for layer, neuron, delta in block[role]:
            for dose in DOSES:
                specs.append(("solo", role, int(layer), int(neuron), dose, {"shift": [(one(int(layer), int(neuron), float(delta)), reference * dose)]}))
            if role == "panel":
                specs.append(("loo", role, int(layer), int(neuron), None, {"shift": [(without(panel, int(layer), int(neuron)), reference)]}))
    return specs


def screen(args) -> None:
    import torch
    from .sarvam_lab import sites
    key = args.model
    paths = single_paths(key)
    runtime = nv.load_runtime(key)
    widths = sites(runtime)
    candidates = build_candidates(key, runtime, args)
    replies = saved_replies(key)
    messages = pick_messages(key, args.messages).set_index("item_id")
    wanted = list(messages.index)
    done = set()
    if paths["screen"].exists():
        done = set(map(tuple, pd.read_csv(paths["screen"], dtype={"item_id": str})[["item_id", "panel"]].drop_duplicates().to_numpy()))
    device = next(runtime.model.parameters()).device
    for position, item_id in enumerate(wanted):
        row = messages.loc[item_id]
        encoded = runtime._encode_conversation([*row.history, {"role": "user", "content": row.text}], row.kg_prompt, args.max_new_tokens)
        prompt_ids = encoded["input_ids"][0]
        out = []
        for name, block in candidates["panels"].items():
            found = replies[(replies.item_id == item_id) & (replies.panel == name)]
            if found.empty or (item_id, name) in done:
                continue
            specs = screen_specs(runtime._emotion_panels[name], block, candidates["reference"])
            scores = {}
            for kind in ("styled", "none"):
                tokens = runtime.tokenizer(found[kind].iloc[0], add_special_tokens=False)["input_ids"][:args.max_new_tokens]
                if len(tokens) < 3:
                    break
                scores[kind] = log_likelihoods(runtime, prompt_ids, torch.tensor(tokens, device=device), [s[5] for s in specs], widths, args.batch)
            if len(scores) < 2:
                continue
            for index, (condition, role, layer, neuron, dose, _) in enumerate(specs):
                out.append({"item_id": item_id, "panel": name, "language": row.language, "condition": condition, "role": role,
                            "layer": layer, "neuron": neuron, "dose": dose,
                            "emotional": float(scores["styled"][index] - scores["styled"][0]),
                            "plain": float(scores["none"][index] - scores["none"][0])})
        if out:
            pd.DataFrame(out).to_csv(paths["screen"], mode="a", header=not paths["screen"].exists(), index=False)
        print(f"screened {item_id} ({position + 1}/{len(wanted)})", flush=True)
    del runtime
    release()


# --------------------------------------------------------------------------- screen statistics

def screen_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per message: the screen effect of every intervention (emotional minus plain log-likelihood change)."""
    frame = frame.copy()
    frame["effect"] = frame.emotional - frame.plain
    full = frame[frame.condition == "full"].set_index(["item_id", "panel"]).effect
    loo = frame.condition == "loo"
    # Necessity: how much of the panel's effect is lost when the neuron is left out (positive = the neuron mattered).
    frame.loc[loo, "effect"] = full.reindex(list(zip(frame.loc[loo, "item_id"], frame.loc[loo, "panel"]))).to_numpy() - frame.loc[loo, "effect"].to_numpy()
    return frame


def screen_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Per panel: one entry per tested neuron with its screen statistics and the control neurons' null."""
    table = screen_table(frame)
    top = max(DOSES)
    panels = {}
    for name, part in table.groupby("panel"):
        solo = part[(part.condition == "solo") & (part.dose == top)]
        null = solo[solo.role == "control"].groupby(["layer", "neuron"]).effect.mean()
        threshold = float(np.percentile(null, NULL_PERCENTILE)) if len(null) else float("nan")
        neurons = []
        for (role, layer, neuron), group in solo.groupby(["role", "layer", "neuron"]):
            entry = {"role": role, "layer": int(layer), "neuron": int(neuron), "screen": nv.stat(group.effect),
                     "emotional": nv._num(group.emotional.mean()), "plain": nv._num(group.plain.mean()), "doses": {}}
            for dose in DOSES:
                values = part[(part.condition == "solo") & (part.dose == dose) & (part.layer == layer) & (part.neuron == neuron)].effect
                entry["doses"][f"{dose:g}"] = nv.stat(values)
            left = part[(part.condition == "loo") & (part.layer == layer) & (part.neuron == neuron)].effect
            entry["leave_out"] = nv.stat(left) if len(left) else None
            neurons.append(entry)
        tested = [entry for entry in neurons if entry["role"] == "panel"]
        adjusted = holm([entry["screen"]["p"] if entry["screen"]["p"] is not None else 1.0 for entry in tested]) if tested else []
        for entry, p in zip(tested, adjusted):
            entry["screen"]["p_holm"] = nv._num(p)
        for entry in neurons:
            stat = entry["screen"]
            entry["above_null"] = bool(stat["mean"] is not None and stat["mean"] > threshold)
            entry["screen_status"] = screen_status(stat, entry["above_null"]) if entry["role"] == "panel" else "control"
        full = part[part.condition == "full"].effect
        panels[name] = {"null_threshold": nv._num(threshold), "null_mean": nv._num(null.mean()) if len(null) else None,
                        "full_panel": nv.stat(full), "messages": int(part.item_id.nunique()), "neurons": neurons}
    return panels


def screen_status(stat: dict[str, Any], above_null: bool) -> str:
    """``supported``: CI above zero, Holm p < 0.05 and above the control neurons' 95th percentile."""
    if stat.get("lo") is None:
        return "inconclusive"
    if stat["lo"] > 0 and (stat.get("p_holm") if stat.get("p_holm") is not None else 1.0) < 0.05 and above_null:
        return "supported"
    if stat["hi"] <= 0 or not above_null:
        return "not_supported"
    return "inconclusive"


# --------------------------------------------------------------------------- generate and score

def generate(args) -> None:
    from .sarvam_lab import generate_rows
    key = args.model
    paths = single_paths(key)
    if not paths["screen"].exists():
        raise SystemExit("Run the screen stage first.")
    summary = screen_summary(pd.read_csv(paths["screen"], dtype={"item_id": str}))
    candidates = json.loads(paths["candidates"].read_text())
    reference, top = candidates["reference"], max(DOSES)
    runtime = nv.load_runtime(key)
    messages = pick_messages(key, args.messages)
    done = set()
    if paths["generations"].exists():
        done = {(r["item_id"], r["panel"]) for r in map(json.loads, paths["generations"].open(encoding="utf-8"))}
    with paths["generations"].open("a", encoding="utf-8") as handle:
        for position, row in enumerate(messages.itertuples()):
            for name, block in summary.items():
                if (row.item_id, name) in done:
                    continue
                panel = runtime._emotion_panels[name]
                shifts = {(int(l), int(n)): float(d) for role in ("panel", "control") for l, n, d in candidates["panels"][name][role]}
                best = sorted((e for e in block["neurons"] if e["role"] == "panel"), key=lambda e: -(e["screen"]["mean"] or -1e9))[:args.confirm]
                random = [e for e in block["neurons"] if e["role"] == "control"][:args.confirm]
                plan = [("none", "none", "", None, {}, None, 0), ("full", "dose", "", reference, {"shift": [(panel, reference)]}, "none", nv.size(panel))]
                for entry in best + random:
                    layer, neuron = entry["layer"], entry["neuron"]
                    family = "solo" if entry["role"] == "panel" else "solo_control"
                    plan.append((f"{family}_L{layer}_N{neuron}", family, f"L{layer} N{neuron}", reference * top,
                                 {"shift": [(one(layer, neuron, shifts[(layer, neuron)]), reference * top)]}, "none", 1))
                for entry in best:
                    layer, neuron = entry["layer"], entry["neuron"]
                    plan.append((f"loo_L{layer}_N{neuron}", "leave_out", f"L{layer} N{neuron}", reference,
                                 {"shift": [(without(panel, layer, neuron), reference)]}, "full", nv.size(panel) - 1))
                target, description = nv.describe(key, name)
                replies = generate_rows(runtime, nv.messages_of(row), row.kg_prompt, [item[4] for item in plan], args.max_new_tokens)
                for (condition, family, group, alpha, _, ref, neurons), reply in zip(plan, replies):
                    handle.write(json.dumps({"item_id": row.item_id, "panel": name, "prompt": "A", "name": condition, "family": family,
                                             "group": group, "alpha": alpha, "reference": ref, "seed": None,
                                             "label": f"{family.replace('_', ' ')} {group}".strip(), "neurons": neurons,
                                             "sample_seed": None, "text": row.text, "language": row.language,
                                             "history": list(row.history), "target": target, "description": description,
                                             "reply": reply}, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"generated {row.item_id} ({position + 1}/{len(messages)})", flush=True)
    del runtime
    release()


def score(args) -> None:
    nv.score_files(single_paths(args.model), args.force)


# --------------------------------------------------------------------------- analyse

def judged_block(part: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    """Judged effect of each generated single-neuron intervention, against the control neurons."""
    summary = nv.summarise(part, reference="full")
    series = summary["series"]
    rows = {row["name"]: row for row in summary["conditions"]}
    control = [name for name in series if name.startswith("solo_control_")]
    null = pd.concat([series[name] for name in control], axis=1).mean(axis=1) if control else None
    out = {}
    for name, row in rows.items():
        if not name.startswith("solo_L"):
            continue
        layer, neuron = int(name.split("_")[1][1:]), int(name.split("_")[2][1:])
        against = nv.stat(series[name] - null) if null is not None else None
        left = rows.get(f"loo_L{layer}_N{neuron}")
        out[(layer, neuron)] = {**row["emotion"], "vs_control": against, "harm_share": row["harm_share"], "preference": row["preference"],
                                "fluency": (row["quality"].get("fluency") or {}).get("mean"),
                                "judges": {judge: stat["mean"] for judge, stat in row["emotion_judges"].items()},
                                # The reference of a leave-out reply is the full-panel reply, so a negative change means the neuron mattered.
                                "leave_out": left["emotion"] if left else None}
    return out


def final_status(screen: str, judged: dict[str, Any] | None) -> tuple[str, str]:
    if judged is None:
        return ("inconclusive", "Passed the likelihood screen; not tested by generation.") if screen == "supported" else (
            screen, "Likelihood screen only.")
    against = judged.get("vs_control") or {}
    confirmed = judged.get("lo") is not None and judged["lo"] > 0 and against.get("lo") is not None and against["lo"] > 0
    if screen == "supported" and confirmed:
        return "supported", "Raises the emotion alone in the likelihood screen and in judged replies, beyond control neurons."
    if confirmed:
        return "inconclusive", "Judged replies show an effect but the likelihood screen did not pass."
    if screen == "supported":
        return "inconclusive", "Likelihood only: the screen passed but judged replies do not show an effect beyond control neurons."
    return "not_supported", "No individual effect beyond control neurons."


def analyse(args) -> None:
    key = args.model
    paths = single_paths(key)
    candidates = json.loads(paths["candidates"].read_text())
    panels = screen_summary(pd.read_csv(paths["screen"], dtype={"item_id": str}))
    judged: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    judges_used: dict[str, str] = {}
    if paths["scores"].exists() and (paths["root"] / "ratings_judge2.csv").exists():
        from .v3_common import JUDGES
        table, _ = nv.load_score_files(paths)
        judges_used = {name: model for name, (model, _) in JUDGES.items()}
        for name, part in table.groupby("panel"):
            judged[name] = judged_block(part)
    for name, block in panels.items():
        block["ranked_by"] = candidates["panels"][name]["ranked_by"]
        for entry in block["neurons"]:
            entry["judged"] = judged.get(name, {}).get((entry["layer"], entry["neuron"])) if entry["role"] == "panel" else None
            entry["status"], entry["reading"] = final_status(entry["screen_status"], entry["judged"]) if entry["role"] == "panel" else ("control", "Control neuron outside every panel.")
        tested = [entry for entry in block["neurons"] if entry["role"] == "panel"]
        block["tested"] = len(tested)
        block["supported"] = sum(entry["status"] == "supported" for entry in tested)
        block["screen_supported"] = sum(entry["screen_status"] == "supported" for entry in tested)
        block["generated"] = sum(entry["judged"] is not None for entry in tested)
        block["verdict"] = verdict(block)
    report = {"model_key": key, "model": nv.MODELS[key]["label"], "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
              "reference": candidates["reference"], "doses": list(DOSES), "null_percentile": NULL_PERCENTILE, "judges": judges_used,
              "panels": panels,
              "method": ("Screen: change of the emotional reply's log-likelihood minus change of the plain reply's (nats per token), "
                         f"one neuron shifted alone at {max(DOSES):g}x the reference strength; control neurons are outside every panel, "
                         "matched on layer, neutral activation and shift. Confirmation: judged emotion rating of generated replies, "
                         "against the same intervention on control neurons."),
              "notes": ["Only the listed neurons were tested; every other neuron remains not tested.",
                        f"A {max(DOSES):g}x shift of one neuron is far outside normal activations; read the judged effect with its quality ratings.",
                        "'Leave out' is the panel's effect lost when the neuron is removed from the panel."]}
    paths["report"].write_text(json.dumps(report, indent=1, default=str))
    nv.write_cards(key, nv.model_paths(key))
    print(f"wrote {paths['report']}", flush=True)
    for name, block in panels.items():
        print(f"  {name}: {block['verdict']['text']}", flush=True)


def verdict(block: dict[str, Any]) -> dict[str, str]:
    tested, passed, screened = block["tested"], block["supported"], block["screen_supported"]
    if not block["generated"]:
        status = "inconclusive" if screened else "not_supported"
        text = (f"{screened} of {tested} tested neurons pass the likelihood screen alone; none was tested by generation yet.")
    elif passed:
        status, text = "supported", f"{passed} of {tested} tested neurons raise the emotion on their own in the screen and in judged replies."
    else:
        status = "not_supported"
        text = (f"None of the {tested} tested neurons raises the emotion on its own in judged replies beyond control neurons "
                f"({screened} pass the likelihood screen). The effect is a property of the panel.")
    return {"status": status, "text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=("screen", "generate", "score", "analyse", "all"))
    parser.add_argument("--model", choices=tuple(nv.MODELS), default="llama")
    parser.add_argument("--neurons", type=int, default=20, help="Top-ranked neurons screened per panel (and as many controls).")
    parser.add_argument("--confirm", type=int, default=5, help="Best-screened neurons per panel tested by generation.")
    parser.add_argument("--messages", type=int, default=24, help="Development messages used.")
    parser.add_argument("--batch", type=int, default=16, help="Interventions scored per forward pass in the screen.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    stages = ["screen", "generate", "score", "analyse"] if "all" in args.stages else args.stages
    for stage in stages:
        {"screen": screen, "generate": generate, "score": score, "analyse": analyse}[stage](args)


if __name__ == "__main__":
    main()
