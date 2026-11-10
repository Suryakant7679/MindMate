"""Shared Hugging Face Llama and Sarvam inference, with optional Llama emotion-neuron steering."""
import os
import threading
import time
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from .config import (CONTEXT_TOKENS, DEFAULT_STEERING_TARGET, STEERING_GAIN, SUPPORTIVE_TARGET,
                     SYSTEM_PROMPT, ModelConfig)

class HuggingFaceRuntime:
    enable_thinking: bool | None = None

    def generation_settings(self) -> dict[str, Any]:
        return {}

    def decode_response(self, new_tokens) -> str:
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def __init__(self, model_id: str, load_in_4bit: bool | None = None, *, config: ModelConfig | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Install dependencies: python3 -m pip install torch transformers accelerate"
            ) from exc

        config = config or ModelConfig(model_id=model_id)
        if load_in_4bit is None:
            load_in_4bit = config.load_in_4bit
        self.enable_thinking = config.enable_thinking
        self.torch = torch
        self.model_id = model_id
        self.lock = threading.Lock()
        token = os.environ.get("HF_TOKEN")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"Loading {model_id} on {gpu_name} ({gpu_gib:.1f} GiB)")
        else:
            dtype = torch.float32
            print("Warning: CUDA is unavailable; generation will be slow.")

        model_kwargs: dict[str, Any] = {
            "token": token,
            "trust_remote_code": config.trust_remote_code,
            "device_map": "auto",
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError as exc:
                raise SystemExit(
                    "4-bit loading requires bitsandbytes. Install backend/requirements.txt."
                ) from exc
            print("Quantization: 4-bit NF4 with double quantization.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except OSError as exc:
            message = str(exc).lower()
            if "gated repo" in message or "401 client error" in message:
                raise SystemExit(
                    "Hugging Face access is missing. Accept the model terms, then run "
                    "`hf auth login` before starting this server."
                ) from exc
            raise
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self._emotion_panels = self._load_emotion_panels()
        self._emotion_probe = None
        self._probe_load_attempted = False
        print("Model ready.")

    def _load_emotion_panels(self) -> dict[str, dict[int, tuple[list[int], list[float]]]]:
        """Load the attribution-selected steering panels.

        Built by ``python3 -m generation_analysis.build_steering_panels``.  Each
        emotion maps layers to (neuron indices, per-neuron activation shifts);
        a shift is the neuron's mean decode-step activation under that emotion
        minus its mean under the neutral style.
        """
        if "llama" not in self.model_id.lower():
            return {}
        path = (Path(__file__).resolve().parent.parent / "results" / "llama31_8b_generation_emotion"
                / "steering_panels" / "panels.json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                emotion: {int(layer): ([int(n) for n in item["neurons"]], [float(d) for d in item["deltas"]])
                          for layer, item in by_layer.items()}
                for emotion, by_layer in payload["panels"].items()
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Emotion steering unavailable ({exc}). "
                  "Run: python3 -m generation_analysis.build_steering_panels")
            return {}

    def _load_emotion_probe(self) -> dict[str, Any] | None:
        if self._probe_load_attempted:
            return self._emotion_probe
        self._probe_load_attempted = True
        path = (Path(__file__).resolve().parent.parent / "results" / "llama31_8b_neuron_analysis"
                / "04_probe" / "probe.pt")
        try:
            checkpoint = self.torch.load(path, map_location="cpu", weights_only=True)
            state = checkpoint["state_dict"]
            labels = checkpoint["labels"]
            if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
                raise ValueError("Probe checkpoint has no label list.")
            self._emotion_probe = {"weight": state["weight"], "bias": state["bias"], "labels": labels}
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            print(f"Emotion recognition unavailable: {exc}")
        return self._emotion_probe

    def detect_emotion(self, text: str) -> dict[str, Any]:
        """Classify one user message using the saved frozen 10-class probe."""
        probe = self._load_emotion_probe()
        if probe is None:
            raise ValueError("The saved 10-emotion probe is unavailable for this model.")
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}
        mask = encoded["attention_mask"].bool().clone()
        for token_id in self.tokenizer.all_special_ids:
            mask &= encoded["input_ids"].ne(token_id)
        with self.lock, self.torch.inference_mode():
            output = self.model.model(**encoded, output_hidden_states=False, use_cache=False)
            hidden = output.last_hidden_state
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            logits = pooled.float() @ probe["weight"].to(device).float().T + probe["bias"].to(device).float()
            probabilities = self.torch.softmax(logits, dim=-1)[0]
        index = int(probabilities.argmax().item())
        scores = {
            label: round(float(probabilities[position].item()), 4)
            for position, label in enumerate(probe["labels"])
        }
        return {
            "emotion": probe["labels"][index],
            "confidence": round(float(probabilities[index].item()), 4),
            "scores": scores,
        }

    def steering_target(self, detected: str, requested: str | None = None) -> str:
        """Emotion whose panel is steered for a user message detected as ``detected``.

        ``requested`` may name a panel, or be "mirror" (the detected emotion
        itself).  The default, "auto", maps the detected emotion to a response
        emotion suited to a support conversation (see SUPPORTIVE_TARGET).
        """
        if requested in (None, "auto"):
            target = SUPPORTIVE_TARGET.get(detected, DEFAULT_STEERING_TARGET)
        elif requested == "mirror":
            target = detected if detected in self._emotion_panels else DEFAULT_STEERING_TARGET
        else:
            target = requested
        if target not in self._emotion_panels:
            raise ValueError(f"No steering panel is available for {target!r}.")
        return target

    def panel_size(self, emotion: str) -> int:
        return sum(len(neurons) for neurons, _ in self._emotion_panels.get(emotion, {}).values())

    @contextmanager
    def _emotion_steering(self, alpha: float, emotion: str | None, trace: dict[str, Any] | None = None):
        """Shift one emotion panel's MLP neurons while the reply is generated.

        Prompt processing is left unchanged; the intervention starts on the
        first autoregressive decoding step.  Each panel neuron's activation
        (input of ``mlp.down_proj``) receives alpha * STEERING_GAIN * its
        emotion-minus-neutral mean shift.  Negative alpha pushes away from the
        emotion.

        If ``trace`` is a dict, the hooks also measure the panel neurons'
        activation on every decode step (before the shift is added) and the
        dict is filled on exit.  With alpha 0 and a trace, nothing is modified:
        the same neurons are only observed, giving the unsteered reference.
        """
        if (not alpha and trace is None) or emotion is None:
            yield
            return
        if emotion not in self._emotion_panels:
            raise ValueError("No steering panel is available for the requested emotion.")
        handles = []
        prompt_seen: dict[int, bool] = {}
        measured: dict[int, dict[str, Any]] = {}

        def hook_for_layer(layer: int, neurons: list[int], shifts: list[float]):
            cache: dict[str, Any] = {}

            def hook(_module, inputs):
                if not inputs:
                    return inputs
                if not prompt_seen.get(layer, False):
                    prompt_seen[layer] = True
                    return inputs
                hidden = inputs[0]
                if "indices" not in cache:
                    if max(neurons) >= hidden.shape[-1]:
                        raise RuntimeError("Configured emotion neuron is outside this checkpoint's MLP width.")
                    cache["indices"] = self.torch.tensor(neurons, device=hidden.device, dtype=self.torch.long)
                    cache["deltas"] = self.torch.tensor(
                        [float(alpha) * STEERING_GAIN * shift for shift in shifts],
                        device=hidden.device, dtype=hidden.dtype,
                    )
                if trace is not None:
                    values = hidden[..., cache["indices"]].float().reshape(-1, len(neurons))
                    entry = measured.setdefault(layer, {"sum": 0.0, "positions": 0})
                    entry["sum"] = entry["sum"] + values.sum(0)
                    entry["positions"] += values.shape[0]
                if not alpha:
                    return inputs
                modified = hidden.clone()
                modified[..., cache["indices"]] += cache["deltas"]
                return (modified, *inputs[1:])
            return hook

        try:
            for layer, (neurons, shifts) in self._emotion_panels[emotion].items():
                module = self.model.model.layers[layer].mlp.down_proj
                handles.append(module.register_forward_pre_hook(hook_for_layer(layer, neurons, shifts)))
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            for handle in handles:
                handle.remove()
            raise ValueError("This checkpoint does not expose the expected Llama MLP layers.") from exc
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            if trace is not None:
                layers = []
                for layer, (neurons, shifts) in sorted(self._emotion_panels[emotion].items()):
                    entry = measured.get(layer)
                    positions = entry["positions"] if entry else 0
                    means = (entry["sum"] / positions).cpu().tolist() if positions else [0.0] * len(neurons)
                    layers.append({
                        "layer": layer, "neurons": list(neurons), "decode_positions": positions,
                        "applied_shift": [round(float(alpha) * STEERING_GAIN * shift, 5) for shift in shifts],
                        "mean_activation_before_shift": [round(float(value), 5) for value in means],
                    })
                trace.update({"emotion": emotion, "alpha": float(alpha), "gain": STEERING_GAIN,
                              "location": "input of model.layers[L].mlp.down_proj, decode steps only",
                              "layers": layers})

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        deterministic: bool,
        temperature: float,
        system_prompt: str | None = SYSTEM_PROMPT,
        steering_alpha: float = 0.0,
        steering_emotion: str | None = None,
        steering_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        torch = self.torch
        conversation = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)
        first_turn = 1 if system_prompt else 0
        # Remove oldest complete turns, preserving the system prompt and latest user turn.
        while True:
            formatted = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
                **({"enable_thinking": self.enable_thinking} if self.enable_thinking is not None else {})
            )
            encoded = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
            if encoded["input_ids"].shape[1] <= CONTEXT_TOKENS - max_new_tokens:
                break
            if len(conversation) <= first_turn + 1:
                raise ValueError("Message exceeds the model context budget.")
            del conversation[first_turn:first_turn + 2]
        model_device = next(self.model.parameters()).device
        # Sarvam's tokenizer emits token_type_ids, but its custom causal-LM
        # implementation does not accept that argument during generation.
        encoded = {
            key: value.to(model_device)
            for key, value in encoded.items()
            if key != "token_type_ids"
        }
        settings: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not deterministic,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if not deterministic:
            settings.update(temperature=temperature, top_p=0.9)
        settings.update(self.generation_settings())

        started = time.perf_counter()
        with self.lock, torch.inference_mode(), self._emotion_steering(steering_alpha, steering_emotion, steering_trace):
            output = self.model.generate(**encoded, **settings)
        elapsed = time.perf_counter() - started
        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        response = self.decode_response(new_tokens)
        if not response:
            raise RuntimeError("The model returned an empty response.")
        return {
            "response": response,
            "model": self.model_id,
            "latency_seconds": round(elapsed, 3),
            "generated_tokens": int(new_tokens.shape[0]),
            "deterministic": deterministic,
            "steering_alpha": steering_alpha,
            "steering_emotion": steering_emotion,
        }

    def stream(self, messages, max_new_tokens, deterministic, temperature,
               system_prompt=SYSTEM_PROMPT, steering_alpha=0.0, steering_emotion=None, steering_trace=None):
        """Yield visible response text as the model generates it."""
        from transformers import TextIteratorStreamer

        conversation = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)
        first_turn = 1 if system_prompt else 0
        formatted = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": self.enable_thinking} if self.enable_thinking is not None else {})
        )
        encoded = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        if encoded["input_ids"].shape[1] > CONTEXT_TOKENS - max_new_tokens:
            raise ValueError("Message exceeds the model context budget.")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}
        settings = {"max_new_tokens": max_new_tokens, "do_sample": not deterministic,
                    "pad_token_id": self.tokenizer.pad_token_id}
        if not deterministic:
            settings.update(temperature=temperature, top_p=0.9)
        settings.update(self.generation_settings())
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        settings["streamer"] = streamer
        errors = []

        def run():
            try:
                with self.lock, self.torch.inference_mode(), self._emotion_steering(steering_alpha, steering_emotion, steering_trace):
                    self.model.generate(**encoded, **settings)
            except Exception as exc:
                errors.append(exc)
                streamer.end()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        hidden = True
        pending = ""
        for chunk in streamer:
            pending += chunk
            if hidden:
                if "</think>" in pending:
                    pending = pending.split("</think>", 1)[1]
                    hidden = False
                elif "<think>" not in pending:
                    hidden = False
                else:
                    continue
            if pending:
                yield pending
                pending = ""
        worker.join()
        if errors:
            raise errors[0]
