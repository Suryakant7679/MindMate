"""Shared Hugging Face Llama and Sarvam inference, with optional Llama emotion-neuron steering."""
import os
import threading
import time
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from .config import (CONTEXT_TOKENS, DEFAULT_STEERING_TARGET, STEERING_GAIN, SUPPORTIVE_TARGET, SYSTEM_PROMPT,
                     ModelConfig)

class HuggingFaceRuntime:
    enable_thinking: bool | None = None

    def generation_settings(self) -> dict[str, Any]:
        return {}

    def decode_response(self, new_tokens) -> str:
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def __init__(self, model_id: str, load_in_4bit: bool | None = None, *, config: ModelConfig | None = None,
                 local_files_only: bool = False) -> None:
        """``local_files_only`` refuses to download: the checkpoint must already be in the local cache."""
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
        # Result folders (panels, probe) belong to one checkpoint; a different model id gets none.
        self.results_tag = config.results_tag if model_id == config.model_id else None
        self.panel_file = config.panel_file
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
        if local_files_only:
            model_kwargs["local_files_only"] = True
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
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=token, **({"local_files_only": True} if local_files_only else {}))
            if config.chat_template and not getattr(self.tokenizer, "chat_template", None):
                self.tokenizer.chat_template = config.chat_template
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
        # Prompt-trimming budget: the app default, never more than the checkpoint's own window
        # (EmoLLaMA and Llama-2 have 2,048 / 4,096 positions).
        window = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        self.context_tokens = min(CONTEXT_TOKENS, window) if isinstance(window, int) and window > 0 else CONTEXT_TOKENS
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
        if not self.results_tag:
            return {}
        path = (Path(__file__).resolve().parent.parent / "results" / f"{self.results_tag}_generation_emotion"
                / "steering_panels" / self.panel_file)
        if not path.exists():
            return {}       # the generation analysis has not been run for this model: plain chat
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("model_id") != self.model_id:
                raise ValueError(f"panels were built for {payload.get('model_id')!r}, not {self.model_id!r}")
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
        path = (Path(__file__).resolve().parent.parent / "results" / f"{self.results_tag}_neuron_analysis"
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
            if encoded["input_ids"].shape[1] <= self._context_budget(max_new_tokens):
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

    def _context_budget(self, max_new_tokens: int) -> int:
        """Prompt tokens allowed for a reply of up to ``max_new_tokens`` (at least a quarter of the window)."""
        window = getattr(self, "context_tokens", CONTEXT_TOKENS)
        return max(window - max_new_tokens, window // 4)

    def _encode_conversation(self, messages: list[dict[str, str]], system_prompt: str | None,
                             max_new_tokens: int) -> dict[str, Any]:
        """Chat-template the conversation, dropping the oldest complete turns until it fits the context."""
        conversation = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)
        first_turn = 1 if system_prompt else 0
        while True:
            formatted = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
                **({"enable_thinking": self.enable_thinking} if self.enable_thinking is not None else {})
            )
            encoded = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
            if encoded["input_ids"].shape[1] <= self._context_budget(max_new_tokens):
                break
            if len(conversation) <= first_turn + 1:
                raise ValueError("Message exceeds the model context budget.")
            del conversation[first_turn:first_turn + 2]
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}

    def generate_candidates(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        system_prompt: str | None,
        steering_emotion: str,
        alphas: list[float],
        panel: dict[int, tuple[list[int], list[float]]] | None = None,
    ) -> list[dict[str, Any]]:
        """Greedy replies to one prompt at several steering strengths, decoded together.

        Row i of the batch is steered with ``alphas[i]`` (0 = unsteered), so an
        unsteered reply and its steered candidates cost about one generation
        instead of one each.  The rows share the prompt, so no padding is
        involved; a row that finishes early is ignored from then on.

        Each result holds the reply and a hook trace in the format of
        ``_emotion_steering``: the panel neurons' mean activation before the
        shift over that row's own decode steps, and the shift applied.
        """
        torch = self.torch
        panel = panel if panel is not None else self._emotion_panels.get(steering_emotion)
        if not panel:
            raise ValueError("No steering panel is available for the requested emotion.")
        encoded = self._encode_conversation(messages, system_prompt, max_new_tokens)
        rows = len(alphas)
        input_ids = encoded["input_ids"].repeat(rows, 1)
        attention = encoded["attention_mask"].repeat(rows, 1)
        prompt_seen: dict[int, bool] = {}
        recorded: dict[int, list[Any]] = {}
        handles = []

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
                    cache["indices"] = torch.tensor(neurons, device=hidden.device, dtype=torch.long)
                    base = torch.tensor(shifts, device=hidden.device, dtype=torch.float32) * STEERING_GAIN
                    scale = torch.tensor([float(a) for a in alphas], device=hidden.device, dtype=torch.float32)
                    cache["deltas"] = (scale[:, None, None] * base).to(hidden.dtype)      # [rows, 1, neurons]
                recorded.setdefault(layer, []).append(hidden[:, -1, cache["indices"]].float())
                modified = hidden.clone()
                modified[..., cache["indices"]] += cache["deltas"]
                return (modified, *inputs[1:])
            return hook

        settings: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False,
                                    "pad_token_id": self.tokenizer.pad_token_id}
        settings.update(self.generation_settings())
        started = time.perf_counter()
        try:
            for layer, (neurons, shifts) in panel.items():
                module = self.model.model.layers[layer].mlp.down_proj
                handles.append(module.register_forward_pre_hook(hook_for_layer(layer, neurons, shifts)))
            with self.lock, torch.inference_mode():
                output = self.model.generate(input_ids=input_ids, attention_mask=attention, **settings)
        finally:
            for handle in handles:
                handle.remove()
        elapsed = time.perf_counter() - started
        stop_ids = self.model.generation_config.eos_token_id
        stop_ids = set(stop_ids if isinstance(stop_ids, (list, tuple)) else [stop_ids]) | {self.tokenizer.eos_token_id}
        results = []
        for row, alpha in enumerate(alphas):
            new_tokens = output[row, input_ids.shape[1]:].tolist()
            length = next((index + 1 for index, token in enumerate(new_tokens) if token in stop_ids), len(new_tokens))
            steps = max(length - 1, 0)        # decode step s reads token s-1; the last token is not read
            layers = []
            for layer, (neurons, shifts) in sorted(panel.items()):
                values = recorded.get(layer, [])[:steps]
                means = (torch.stack([step[row] for step in values]).mean(0).cpu().tolist()
                         if values else [0.0] * len(neurons))
                layers.append({
                    "layer": layer, "neurons": list(neurons), "decode_positions": len(values),
                    "applied_shift": [round(float(alpha) * STEERING_GAIN * shift, 5) for shift in shifts],
                    "mean_activation_before_shift": [round(float(value), 5) for value in means],
                })
            results.append({
                "response": self.decode_response(new_tokens[:length]), "generated_tokens": length,
                "steering_alpha": float(alpha), "latency_seconds": round(elapsed, 3),
                "trace": {"emotion": steering_emotion, "alpha": float(alpha), "gain": STEERING_GAIN,
                          "location": "input of model.layers[L].mlp.down_proj, decode steps only", "layers": layers},
            })
        if not results[0]["response"]:
            raise RuntimeError("The model returned an empty response.")
        return results

    def stream(self, messages, max_new_tokens, deterministic, temperature,
               system_prompt=SYSTEM_PROMPT, steering_alpha=0.0, steering_emotion=None, steering_trace=None):
        """Yield visible response text as the model generates it."""
        from transformers import TextIteratorStreamer

        # Same context trimming as generate(): oldest complete turns are dropped when the prompt is too long.
        encoded = self._encode_conversation(messages, system_prompt, max_new_tokens)
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
