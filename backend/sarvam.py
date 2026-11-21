"""Load Sarvam 30B in 4-bit and generate replies without a persona prompt."""
from .runtime import HuggingFaceRuntime
from .sarvam_config import SARVAM_CONFIG


class SarvamRuntime(HuggingFaceRuntime):
    reasoning = bool(SARVAM_CONFIG.enable_thinking)

    def __init__(self, model_id=SARVAM_CONFIG.model_id, load_in_4bit=None, *, local_files_only=False):
        super().__init__(model_id, load_in_4bit, config=SARVAM_CONFIG, local_files_only=local_files_only)
        self._accept_output_dtype()
        self.set_reasoning(bool(SARVAM_CONFIG.enable_thinking))

    # Sarvam-30B is a mixture-of-experts model: layer 0 has an ordinary MLP, every later layer has 128
    # routed experts (6 per token) and one shared expert that runs on every token.  Steering panels use
    # the neurons that are always active: the layer-0 MLP and the shared experts.
    steering_location = ("input of the shared expert's down_proj (model.layers[L].mlp.shared_experts.down_proj; "
                         "layer 0: model.layers[0].mlp.down_proj), decode steps only")

    def steering_module(self, layer: int):
        mlp = self.model.model.layers[layer].mlp
        return mlp.shared_experts.down_proj if hasattr(mlp, "shared_experts") else mlp.down_proj

    def _load_emotion_panels(self):
        """Neuron panels plus residual-stream directions (``directions`` in the panel file).

        A direction panel ``<tone>:direction`` adds alpha * ||h|| * unit to the output of one layer and
        restores the norm; ``<tone>:direction+hinglish`` also shifts the Hinglish language neurons.
        """
        import json
        from pathlib import Path
        panels = super()._load_emotion_panels()
        self._emotion_directions = {}
        path = (Path(__file__).resolve().parent.parent / "results" / f"{self.results_tag}_generation_emotion"
                / "steering_panels" / self.panel_file) if self.results_tag else None
        if panels and path is not None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Sarvam's panels are named after response tones; the panel file says which strategy uses which.
            self.strategy_panels = {str(k): str(v) for k, v in (payload.get("strategy_tone") or {}).items() if v in panels}
            for name, item in payload.get("directions", {}).items():
                self._emotion_directions[name] = (int(item["layer"]), [float(value) for value in item["unit"]])
                panels.setdefault(name, {})
        return panels

    def generate_candidates(self, messages, max_new_tokens, system_prompt, steering_emotion, alphas, panel=None):
        """Greedy replies at several strengths, decoded together, with the measurements the Mood Tracker shows.

        Same contract as the base method, for Sarvam's sites: shared-expert / layer-0 neurons (pre-hook on their
        down_proj) and, for direction panels, the residual stream after one layer (forward hook).
        """
        import time
        from .config import STEERING_GAIN
        torch = self.torch
        neurons_by_layer = panel if panel is not None else (self._emotion_panels.get(steering_emotion) or {})
        direction = None if panel is not None else self._emotion_directions.get(steering_emotion)
        if not neurons_by_layer and direction is None:
            raise ValueError("No steering panel is available for the requested emotion.")
        encoded = self._encode_conversation(messages, system_prompt, max_new_tokens)
        rows = len(alphas)
        input_ids = encoded["input_ids"].repeat(rows, 1)
        attention = encoded["attention_mask"].repeat(rows, 1)
        scale = torch.tensor([float(alpha) for alpha in alphas], dtype=torch.float32)
        recorded: dict = {}
        prompt_seen: dict = {}
        handles = []

        def neuron_hook(layer, neurons, shifts):
            cache: dict = {}

            def hook(_module, inputs):
                if not prompt_seen.get(layer):
                    prompt_seen[layer] = True
                    return inputs
                hidden = inputs[0]
                if not cache:
                    cache["index"] = torch.tensor(neurons, device=hidden.device, dtype=torch.long)
                    base = torch.tensor(shifts, device=hidden.device, dtype=torch.float32) * STEERING_GAIN
                    cache["delta"] = (scale.to(hidden.device)[:, None, None] * base).to(hidden.dtype)
                recorded.setdefault(layer, []).append(hidden[:, -1, cache["index"]].float())
                modified = hidden.clone()
                modified[..., cache["index"]] += cache["delta"]
                return (modified, *inputs[1:])
            return hook

        shown: list[int] = []
        norms: list = []

        def direction_hook(unit):
            state: dict = {}

            def hook(_module, _inputs, output):
                if not state.get("prompt"):
                    state["prompt"] = True
                    return output
                hidden = output[0] if isinstance(output, tuple) else output
                values = hidden.float()
                if "unit" not in state:
                    state["unit"] = torch.tensor(unit, device=values.device, dtype=torch.float32)
                    state["scale"] = scale.to(values.device)[:, None, None]
                    state["index"] = torch.tensor(shown, device=values.device, dtype=torch.long)
                norm = values.norm(dim=-1, keepdim=True)
                recorded.setdefault("direction", []).append(values[:, -1, state["index"]])
                norms.append(norm[:, -1, 0])
                updated = values + state["scale"] * norm * state["unit"]
                updated = updated * (norm / updated.norm(dim=-1, keepdim=True).clamp_min(1e-8))
                updated = torch.where(state["scale"] != 0, updated, values).to(hidden.dtype)
                return (updated, *output[1:]) if isinstance(output, tuple) else updated
            return hook

        settings = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": self.tokenizer.pad_token_id,
                    **self.generation_settings()}
        started = time.perf_counter()
        try:
            for layer, (neurons, shifts) in neurons_by_layer.items():
                handles.append(self.steering_module(layer).register_forward_pre_hook(neuron_hook(layer, neurons, shifts)))
            if direction is not None:
                layer, unit = direction
                order = sorted(range(len(unit)), key=lambda index: -abs(unit[index]))[:64]
                shown.extend(sorted(order))                 # the residual dimensions the direction moves most
                handles.append(self.model.model.layers[layer].register_forward_hook(direction_hook(unit)))
            with self.lock, torch.inference_mode():
                output = self.model.generate(input_ids=input_ids, attention_mask=attention, **settings)
        finally:
            for handle in handles:
                handle.remove()
        elapsed = time.perf_counter() - started
        stop = set(settings.get("eos_token_id", [])) | {self.tokenizer.eos_token_id}
        results = []
        writes = ({layer: self.write_contribution(layer, neurons, shifts) for layer, (neurons, shifts) in neurons_by_layer.items()}
                  if any(alphas) and hasattr(self, "lock") else {})
        for row, alpha in enumerate(alphas):
            tokens = output[row, input_ids.shape[1]:]
            listed = tokens.tolist()
            length = next((index + 1 for index, token in enumerate(listed) if token in stop), len(listed))
            steps = max(length - 1, 0)

            def mean(values):
                values = values[:steps]
                return torch.stack([step[row] for step in values]).mean(0).cpu().tolist() if values else None
            layers = []
            for layer, (neurons, shifts) in sorted(neurons_by_layer.items()):
                before = mean(recorded.get(layer, []))
                layers.append({"layer": layer, "neurons": list(neurons), "decode_positions": min(steps, len(recorded.get(layer, []))),
                               "applied_shift": [round(float(alpha) * STEERING_GAIN * shift, 5) for shift in shifts],
                               "mean_activation_before_shift": [round(float(v), 5) for v in (before or [0.0] * len(neurons))],
                               **({"write_norm": writes[layer][0],
                                   "residual_update_norm": round(abs(float(alpha)) * STEERING_GAIN * writes[layer][1], 5)}
                                  if alpha and writes.get(layer) else {})})
            location = self.steering_location if neurons_by_layer else ""
            if direction is not None:
                layer, unit = direction
                before = mean(recorded.get("direction", []))
                norm = float(torch.stack([step[row] for step in norms[:steps]]).mean()) if norms[:steps] else 0.0
                layers.append({"layer": layer, "site": "residual stream", "neurons": list(shown),
                               "decode_positions": min(steps, len(norms)), "residual_update_norm": round(abs(float(alpha)) * norm, 5),
                               "applied_shift": [round(float(alpha) * norm * unit[index], 5) for index in shown],
                               "mean_activation_before_shift": [round(float(v), 5) for v in (before or [0.0] * len(shown))]})
                location = ((location + "; ") if location else "") + (
                    f"residual stream after model.layers[{layer}] (tone direction: alpha x the activation norm, norm "
                    "restored; the 64 most-moved dimensions are listed), decode steps only")
            results.append({"response": self.decode_response(tokens[:length - 1] if length and listed[length - 1] in stop else tokens[:length]),
                            "generated_tokens": length, "steering_alpha": float(alpha), "latency_seconds": round(elapsed, 3),
                            "trace": {"emotion": steering_emotion, "alpha": float(alpha), "gain": STEERING_GAIN,
                                      "location": location, "layers": sorted(layers, key=lambda item: item["layer"])}})
        if not results[0]["response"]:
            raise RuntimeError("The model returned an empty response.")
        return results

    def set_reasoning(self, enabled: bool) -> None:
        """Reasoning on: the model writes a <think> block first (about 1,300 tokens).  Off: the block is
        pre-filled empty, because the checkpoint ignores its own no-thinking flag."""
        self.reasoning = enabled
        self.enable_thinking = enabled

    def prompt_suffix(self) -> str:
        return "" if self.reasoning else "<think>\n</think>\n"

    def _accept_output_dtype(self):
        """Let 4-bit layers take the ``output_dtype`` argument of Sarvam's own down-projection.

        The checkpoint's MoE code calls ``down_proj(x, output_dtype=float32)`` on its Linear subclass.
        Quantisation replaces that layer by a bitsandbytes ``Linear4bit``, which does not know the
        argument; the result is cast instead.
        """
        import functools
        import inspect
        try:
            modules = iter(self.model.modules())
        except TypeError:
            return
        for module in modules:
            if type(module).__name__ != "Linear4bit" or "output_dtype" in inspect.signature(module.forward).parameters:
                continue
            original = module.forward

            @functools.wraps(original)
            def forward(hidden, output_dtype=None, _original=original):
                output = _original(hidden)
                return output if output_dtype is None else output.to(output_dtype)
            module.forward = forward

    def generation_settings(self):
        # The checkpoint config and tokenizer define different end markers.
        # Stop on either; otherwise an end-of-sequence can be decoded away while
        # generation continues into another response.
        stop_ids = {self.tokenizer.eos_token_id}
        for token in ("<eos>", "<|end_of_turn|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id != self.tokenizer.unk_token_id:
                stop_ids.add(token_id)
        return {
            "eos_token_id": sorted(i for i in stop_ids if isinstance(i, int)),
        }

    def decode_response(self, new_tokens):
        # Parse token IDs before skip_special_tokens can remove delimiters.
        # Reasoning never enters the browser or subsequent conversation history.
        ids = new_tokens.tolist()
        opening = self.tokenizer.convert_tokens_to_ids("<think>")
        closing = self.tokenizer.convert_tokens_to_ids("</think>")
        if not self.reasoning:
            # The think block was pre-filled empty, so no reasoning can follow.  A stray boundary token
            # is a degenerate answer (seen under strong steering): drop it and let the quality gates
            # judge the text instead of aborting the turn.
            return self.tokenizer.decode([t for t in ids if t not in (opening, closing)], skip_special_tokens=True).strip()
        depth = 0
        answer = []
        for token_id in ids:
            if token_id == opening:
                depth += 1
            elif token_id == closing:
                if depth == 0:
                    raise RuntimeError("Malformed reasoning boundary in model response.")
                depth -= 1
            elif depth == 0:
                answer.append(token_id)
        if depth:
            raise RuntimeError("Sarvam reached the generation limit before completing its answer.")
        return self.tokenizer.decode(answer, skip_special_tokens=True).strip()

    def generate(self, messages, max_new_tokens=4096, deterministic=True,
                 temperature=0.7, system_prompt=None):
        # Only the workspace endpoint supplies an explicit system prompt.
        return super().generate(messages, max_new_tokens, deterministic,
                                temperature, system_prompt=system_prompt)
