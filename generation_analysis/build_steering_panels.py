#!/usr/bin/env python3
"""Build the chatbot's emotion-steering neuron panels.

The earlier panels were chosen by firing selectivity, which finds neurons that
*respond* to an emotion.  Steering them had no measurable effect on generated
text (see ``steering/steering_summary.csv``).  This script instead selects
neurons that *write* the emotion into the residual stream:

1. Emotion direction.  The saved matched responses (ten style conditions per
   user message) are re-run teacher-forced.  For each layer l,
   ``d[e, l] = mean residual(e) - mean residual(neutral)`` over generated tokens.
2. Attribution.  For neuron u in layer l with output weight ``w_u`` (column of
   ``mlp.down_proj``) and decode-step mean shift
   ``dmu_u = mean act(e) - mean act(neutral)``:
   ``score_u = dmu_u * (w_u . d_hat[e, l])``, i.e. how much the neuron's
   natural change of activity pushes the residual along the emotion direction.
3. Panel.  The top ``--top-percent`` neurons per layer in the middle band
   ``--layers`` form the panel; each stores its own ``dmu_u``.  At run time the
   backend adds ``alpha * gain * dmu_u`` to those neurons on every decode step.

Run from the project root:  python3 -m generation_analysis.build_steering_panels
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .common import (MODEL_ID, NEUTRAL, RESULTS_DIR, input_device, load_model, model_layers, prompt_token_ids,
                     read_json, read_jsonl, special_token_ids, top_k, write_json)

PANEL_DIR = "steering_panels"


def residual_condition_means(model, tokenizer, rows: list[dict], labels: list[str], batch_size: int) -> np.ndarray:
    """[condition, layer + 1, hidden] mean residual stream over generated, non-special tokens."""
    device = input_device(model)
    special = torch.as_tensor(sorted(special_token_ids(tokenizer)))
    sums = torch.zeros(len(labels), model.config.num_hidden_layers + 1, model.config.hidden_size, dtype=torch.float64)
    counts = torch.zeros(len(labels), dtype=torch.float64)
    for start in tqdm(range(0, len(rows), batch_size), desc="Residual means"):
        chunk = rows[start:start + batch_size]
        prompts = [prompt_token_ids(tokenizer, row["user_text"], row["condition"]) for row in chunk]
        sequences = [prompt + row["generated_ids"] for prompt, row in zip(prompts, chunk)]
        length = max(map(len, sequences))
        ids = torch.full((len(chunk), length), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        generated = torch.zeros_like(ids, dtype=torch.bool)
        for i, (prompt, sequence) in enumerate(zip(prompts, sequences)):
            ids[i, :len(sequence)] = torch.as_tensor(sequence)
            attention[i, :len(sequence)] = 1
            generated[i, len(prompt):len(sequence)] = True
        generated &= ~torch.isin(ids, special)
        with torch.inference_mode():
            states = model.model(input_ids=ids.to(device), attention_mask=attention.to(device),
                                 output_hidden_states=True).hidden_states
        for i, row in enumerate(chunk):
            condition = labels.index(row["condition"])
            counts[condition] += generated[i].sum()
            sums[condition] += torch.stack([state[i][generated[i].to(state.device)].double().sum(0).cpu()
                                            for state in states])
    return (sums / counts[:, None, None]).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--layers", type=int, nargs=2, default=[8, 18], metavar=("FIRST", "LAST"),
                        help="Inclusive band of transformer layers to steer (default: 8 18).")
    parser.add_argument("--top-percent", type=float, default=1.0, help="Neurons kept per layer (default: 1%%).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--recompute", action="store_true", help="Ignore the cached residual means.")
    args = parser.parse_args()

    output = Path(args.output_dir)
    panel_dir = output / PANEL_DIR
    panel_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_json(output / "capture_metadata.json")
    if metadata["model_id"] != args.model_id:
        raise SystemExit(f"Activations were captured with {metadata['model_id']}, not {args.model_id}.")
    labels = metadata["conditions"]
    neutral = labels.index(NEUTRAL)
    means = np.load(output / "neuron_sets" / "decode_condition_means.npy")

    model, tokenizer = load_model(args.model_id)
    layers = model_layers(model)
    first, last = args.layers
    if not 0 <= first <= last < len(layers):
        parser.error(f"--layers must lie within 0..{len(layers) - 1}.")
    residual_path = panel_dir / "residual_condition_means.npy"
    if residual_path.exists() and not args.recompute:
        residual = np.load(residual_path)
    else:
        residual = residual_condition_means(model, tokenizer, read_jsonl(output / "activations" / "responses.jsonl"),
                                            labels, args.batch_size)
        np.save(residual_path, residual.astype(np.float32))

    count = top_k(means.shape[2], args.top_percent)
    panels, summary = {}, []
    for emotion in (label for label in labels if label != NEUTRAL):
        e = labels.index(emotion)
        panels[emotion] = {}
        for layer in range(first, last + 1):
            weight = layers[layer].mlp.down_proj.weight.float()                     # [hidden, neurons]
            direction = torch.as_tensor(residual[e, layer + 1] - residual[neutral, layer + 1],
                                        dtype=torch.float32, device=weight.device)
            unit = direction / direction.norm()
            shift = torch.as_tensor(means[e, layer] - means[neutral, layer], dtype=torch.float32,
                                    device=weight.device)
            score = shift * (unit @ weight)
            chosen = score.topk(count).indices
            push = weight[:, chosen] @ shift[chosen]
            panels[emotion][str(layer)] = {"neurons": chosen.cpu().tolist(),
                                           "deltas": [round(float(v), 6) for v in shift[chosen].cpu()]}
            summary.append({"emotion": emotion, "layer": layer, "neurons": count,
                            "direction_norm": float(direction.norm()), "push_norm": float(push.norm()),
                            "push_along_direction": float(push @ unit),
                            "attribution_share": float(score[chosen].sum() / score.clamp_min(0).sum())})
    write_json(panel_dir / "panels.json", {
        "model_id": args.model_id, "layers": [first, last], "top_percent": args.top_percent,
        "neurons_per_layer": count, "selection": "top dmu_u * (w_u . d_hat) per layer",
        "delta": "decode-step mean activation(emotion) - mean activation(neutral), per neuron",
        "intervention": "add alpha * gain * delta to the input of layer.mlp.down_proj on decode steps",
        "panels": panels,
    })
    import pandas as pd
    pd.DataFrame(summary).to_csv(panel_dir / "panel_summary.csv", index=False)
    print(f"Saved {len(panels)} panels x {last - first + 1} layers x {count} neurons to {panel_dir / 'panels.json'}")


if __name__ == "__main__":
    main()
