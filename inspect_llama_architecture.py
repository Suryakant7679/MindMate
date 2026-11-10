#!/usr/bin/env python3
"""Save an inspectable architecture report for a Hugging Face causal LM.

The default is Llama 3.1 8B Instruct.  By default this downloads only the
small model configuration, not the 8B parameter weights.  Use
``--load-weights`` to additionally enumerate every loaded module and count
parameters; that option needs enough RAM/VRAM for the checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT = "results/llama31_8b_architecture/architecture.json"


def serializable(value: Any) -> Any:
    """Convert Transformers config values into JSON-compatible data."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    return str(value)


def layer_report(layer: Any, index: int) -> dict[str, Any]:
    mlp = layer.mlp
    attention = layer.self_attn
    return {
        "index": index,
        "layer_type": type(layer).__name__,
        "attention_type": type(attention).__name__,
        "mlp_type": type(mlp).__name__,
        "input_layernorm": type(layer.input_layernorm).__name__,
        "post_attention_layernorm": type(layer.post_attention_layernorm).__name__,
        "attention_projections": {
            name: list(getattr(attention, name).weight.shape)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        },
        "mlp_projections": {
            name: list(getattr(mlp, name).weight.shape)
            for name in ("gate_proj", "up_proj", "down_proj")
        },
        "mlp_neurons": int(mlp.down_proj.in_features),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--load-weights", action="store_true",
        help="Load weights and save per-layer projection shapes and parameter counts.",
    )
    args = parser.parse_args()

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model_id)
    report: dict[str, Any] = {
        "model_id": args.model_id,
        "config_class": type(config).__name__,
        "architecture": list(getattr(config, "architectures", [])),
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_word_embeddings,
        "activation_function": config.hidden_act,
        "load_weights": args.load_weights,
        "full_transformers_config": serializable(config.to_dict()),
    }

    if args.load_weights:
        import torch
        from transformers import AutoModelForCausalLM

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, device_map="auto", torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model.eval()
        layers = model.model.layers
        report["loaded_model_class"] = type(model).__name__
        report["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        report["trainable_parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        report["embedding_shape"] = list(model.model.embed_tokens.weight.shape)
        report["lm_head_shape"] = list(model.lm_head.weight.shape)
        report["layers"] = [layer_report(layer, index) for index, layer in enumerate(layers)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved Llama architecture report to {output}")
    print(
        f"{report['num_hidden_layers']} layers; hidden={report['hidden_size']}; "
        f"MLP={report['intermediate_size']}; attention heads={report['num_attention_heads']}; "
        f"KV heads={report['num_key_value_heads']}"
    )


if __name__ == "__main__":
    main()
