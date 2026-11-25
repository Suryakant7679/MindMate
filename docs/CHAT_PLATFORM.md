# MindMate chat platform: multi-model, persistent, multi-user

This document describes the chat infrastructure added on top of the existing research system (Emotion Reasoner,
emotion-neuron steering).  The Safety Agent was removed on 2026-11-22. Research code, panels, policies and results are unchanged; the chat
reads them.

## Architecture

```
browser (React)                       backend.server (ThreadingHTTPServer)
  sidebar: history, account   ─HTTP─▶  chat_api.ChatApiMixin      users, consent, conversations, SSE turns
  chat: stream, stop, model           │   store.Store (SQLite)    users · sessions · conversations · messages
  Mood Tracker (unchanged)            │   store.ConversationStore(None)   unsaved conversations, memory only
                                      │   chat_engine.TurnEngine  one turn:
                                      │     1 Emotion Reasoner plan (strategy, response emotion, strength)
                                      │     2 policy-gated steering → generation, streamed as it is written
                                      │   context.fit             history fitted to the model's window
                                      │   models.ModelManager     lazy pool, leases, LRU + GPU-memory checks
                                      │     emotion_services[key] one EmotionSteeringService per loaded model
                                      └── legacy endpoints (/api/chat, /api/chat/stream, /api/agent/chat, ...)
```

The legacy endpoints are unchanged in behaviour (client sends the history, nothing stored). They now hold a lease
on their model for the whole request. The LangGraph/MCP agent mode still uses `/api/agent/chat`; its turns are not
stored.

## Model registry and loading

`backend/config.py: MODEL_REGISTRY` (facts below were read from the cached checkpoints' `config.json`):

| key | checkpoint | architecture | layers | window (app) | precision | GPU (approx.) |
|---|---|---|---|---|---|---|
| `llama` | meta-llama/Llama-3.1-8B-Instruct | LlamaForCausalLM, dense | 32 | 131,072 (16,384) | bf16 | 16 GiB |
| `llama2` | meta-llama/Llama-2-13b-chat-hf | LlamaForCausalLM, dense | 40 | 4,096 | fp16 | 26 GiB |
| `emollama` | lzw1008/Emollama-chat-7b | LlamaForCausalLM, dense (no chat template; registry template) | 32 | 2,048 | fp16 | 14 GiB |
| `sarvam` | sarvamai/sarvam-30b | SarvamMoEForCausalLM, 128 routed experts + shared expert | 19 | 131,072 (16,384) | 4-bit NF4 | 20 GiB |

Each entry carries its own tokenizer / chat template (from the checkpoint, or the registry template for EmoLLaMA),
`max_new_tokens`, and generation settings. Greedy decoding stays the default for every model because every
calibration and held-out comparison was produced with it.

`ModelManager` (backend/models.py):

* **Lazy, single load.** A model loads when a message first needs it; concurrent requests for the same model wait
  for one load. Nothing is downloaded without `--allow-download`.
* **Leases.** A request generates with a leased runtime; a leased model is never unloaded.
* **Residency.** `--max-loaded-models` (default 1: one model at a time, as before). Before a load, idle models are
  unloaded least-recently-used first until the count fits and `torch.cuda.mem_get_info()` (which sees other
  processes on the GPU) covers the model's measured footprint (or registry estimate) plus `--gpu-headroom-gib`.
* **Placement.** `--device auto|cuda|cpu`, `--gpu-max-memory-gib` (layers beyond the cap are offloaded to the CPU).
* **Idle unloading.** `--idle-unload-minutes`. **Start options** (`--model-id`, `--load-in-4bit`) apply to every
  reload of the start model.
* `GET /api/models` reports each model's capabilities, measured footprint, loads since start and residency.

## Emotion-neuron steering per model

Status read from `results/<tag>_generation_emotion/` (`backend/capabilities.py`); the per-turn decision is still
made by the Emotion Reasoner, the calibrated policy and the run-time gate (`EmotionSteeringService`).

| model | neuron discovery | panels for this checkpoint | calibrated policy | chat behaviour |
|---|---|---|---|---|
| Llama 3.1-8B | yes | `panels_causal.json`: 786–1,573 neurons per panel | v2, steers `love|english` 0.5, `compassionate_comfort|english` 0.25, `grounding_reassurance|english` 0.75 | policy-gated steering |
| Llama-2-13B | yes | `panels.json`: 1,794 neurons, layers 10–22 | v2, steers `grounding_reassurance|english` 0.25 | policy-gated steering |
| EmoLLaMA-7B | yes | `panels.json`: 1,210 neurons, layers 8–18 | v2, steers `reflective_validation|english` 0.5 | policy-gated steering |
| Sarvam-30B | yes | `panels.json`: 1,328 neurons/tone (layer-0 MLP + shared experts) + layer-10 directions | `sarvam-1`, steers nothing | never steered |

**Steering modes** (per conversation; server default `--steering-mode always`):

* `always` (default): every reply is steered. The strength follows the user's emotional state: the Emotion
  Reasoner's state rule (strategy base strength x intensity of the user's emotion x panel sensitivity, +0.15 when
  distress persists across turns, -0.10 when the mood lifts, lower cap for Hinglish) gives a level 0-1, which is
  mapped onto the model's steering range (`STATE_ALPHA_RANGE`: 0.25-0.75 for the Llama family, 2-3 for Sarvam's tone
  panels; the strengths their calibrations tested). A crisis message gets the crisis-support strategy (a warm,
  hopeful reply that encourages the user not to act on the thoughts and to reach a trusted person or helpline) and
  is steered toward warmth at the highest base strength. The steered reply is sent unless it is broken (not a
  conversational answer, much more repetitive, collapsed in length, or left the user's Hinglish); then the
  unsteered one is sent and the reason recorded.
* `policy`: the calibrated policy decides (steered only in its calibrated contexts, after the run-time gate).
* `off`: unsteered baseline (same strategy and prompt, strength 0).

**Tone before and after steering.** The unsteered and steered versions of a reply are written in one batched pass
and both are scored by the GoEmotions classifier. Every assistant message stores `metadata.tone`: dominant emotion,
top emotions, expected valence and target-emotion score of each version, and the change. The chat shows it under
each reply ("Emotional tone: sadness -> love · love +0.35"); the Mood Tracker shows the full comparison. These are
estimates of the reply text, not of the user.

Panels are used only for the checkpoint they were built for (`model_id` check); nothing transfers between models.
Every assistant message stores its steering record: mode (`auto` / `off`), status, reason, planned and applied
strength, panel, response emotion, neuron count, layers, gain and site. The per-conversation **Steering: off**
switch gives the unsteered baseline (same strategy and prompt, strength 0); research-compare mode still produces a
never-sent steered candidate for the Mood Tracker.

Concurrency fix: `generate_candidates` (base and Sarvam) registered the steering hooks *before* taking the runtime
lock, so a concurrent request's generation could run with another request's steering. On a tiny model, the original
code contaminated 11 of 25 concurrent unsteered replies. Hooks are now registered and removed only under the lock
(`tests/test_runtime_generation.py`). Steering state (hook closures) is per call.

## Conversation context

`context.normalize` turns the stored history into strict user/assistant alternation (consecutive user messages, e.g.
after a stopped reply, are joined, not dropped). `context.fit` keeps the newest complete turns that fit the selected
model's prompt budget counted with *its* tokenizer; the runtime's exact templated check stays the final guard.
Switching models therefore keeps the conversation: the next model receives the same stored history, trimmed to its
own window when shorter (stored, and shown as "saw last N of M messages"). Older turns are not summarised.

The Emotion Reasoner receives the same history (it uses the last three user turns).

## Streaming and Stop

With steering (always the case in mode `always`) the unsteered reply and the steered candidate are decoded together,
the gate picks one, and that reply is then sent; no text of a rejected candidate is shown. Without steering (modes
`off`, or `policy` where it does not steer) the reply is streamed token by token as the model writes it. Events: `start`, `status` (`loading_model`,
`generating`), `progress` (decode steps, from a stopping criterion that does not change outputs), `delta`, `done`
(stored message, emotion summary), `title`, `cancelled`, `error`.

**Stop** (`POST /api/conversations/<id>/cancel`, or closing the connection) sets the request's cancel flag; the
stopping criterion ends `model.generate` at the next token in every path (plain, batched candidates, stream worker).
The text already shown is stored as an assistant message with status `cancelled` (no research record is written for
it); if nothing was shown, nothing is stored.

## Users, sessions, consent and retention

* Every browser gets an anonymous user and a server-side session (HttpOnly, SameSite=Strict cookie with a random
  token; only its SHA-256 is stored; sessions survive restarts). Optional username + password (scrypt) gives access
  from another browser. State-changing requests from another origin are refused.
* Server-side authorisation: every query is scoped by the session's user; another user's conversation answers 404.
* Consent first: nothing is stored until the user chooses **Save my conversations** (retention 7/30/90/365 days of
  inactivity, or until deleted) or **Don't save** (in-memory store, dropped on restart or after 6 h of inactivity).
* Delete a conversation, export everything (JSON) or delete all data (cascades; also clears in-memory Emotion
  Reasoner records). An hourly purge applies retention.
* The database (`chatbot_feature/data/mindmate_chat.sqlite`, git-ignored) is created with mode 0600. Opt-in research
  telemetry (`--research-telemetry`) is separate and operator-managed.

## Database schema

```sql
users(id PK, username UNIQUE NOCASE NULL, password_hash NULL, created_at, last_seen_at,
      consent_version NULL, consent_at NULL, store_history 0|1, retention_days)
sessions(token_hash PK, user_id → users ON DELETE CASCADE, created_at, expires_at, last_used_at)
conversations(id PK, user_id → users ON DELETE CASCADE, title, title_source pending|auto|fallback|user,
              model_key, steering_mode auto|off, created_at, updated_at)
messages(id PK, conversation_id → conversations ON DELETE CASCADE, user_id → users ON DELETE CASCADE,
         seq, role user|assistant, content, status complete|cancelled|error, model_key, model_id, turn_id,
         metadata JSON {generation, steering, context, context_window}, created_at, UNIQUE(conversation_id, seq))
```

## API

| method | path | purpose |
|---|---|---|
| GET | `/api/me` | user, consent state, retention options |
| POST | `/api/me/consent` | `{store_history, retention_days}` |
| PATCH | `/api/me` | `{store_history?, retention_days?}` |
| GET | `/api/me/export` | all conversations as JSON |
| DELETE | `/api/me` | delete the user and everything stored |
| POST | `/api/auth/register` · `/login` · `/logout` | optional account |
| GET / POST | `/api/conversations` | list (most recent first) / create `{model?, title?, steering?}` |
| GET / PATCH / DELETE | `/api/conversations/<id>` | messages · `{title?, model?, steering?}` · delete |
| POST | `/api/conversations/<id>/messages` | `{content?, model?, steering?}` → `text/event-stream`; no `content` regenerates the reply to the last unanswered message |
| POST | `/api/conversations/<id>/cancel` | stop the reply being written |
| GET | `/api/models` | registry, capabilities, residency, memory |

## Running

```sh
cd frontend && npm run build && cd ..
python3 -m backend.server --port 8080                              # Sarvam preloaded, one model at a time
python3 -m backend.server --lazy-load --max-loaded-models 2        # load on demand, keep two when memory allows
```

Tests: `python3 -m unittest tests.test_store tests.test_context tests.test_model_pool tests.test_chat_engine
tests.test_chat_api tests.test_runtime_generation` (no GPU). Real models: `python3 tests/e2e_multimodel_chat.py
--models llama,emollama,llama2,sarvam` (needs the GPU to itself for Llama-2 in fp16 and Sarvam).

## Known limitations

* One GPU: with the default of one loaded model, switching models costs a load (tens of seconds); users of
  different models wait for each other's leases.
* Older turns beyond a model's window are left out, not summarised.
* There is no safety assessment of messages or replies. Crisis handling is the Emotion Reasoner's knowledge-graph
  crisis cue (phrase list, English / Hinglish / Hindi), which selects the crisis-support strategy. Phrasings not in
  the list are not recognised as a crisis.
* Mode `always` steers contexts where the calibration found every tested strength lowered judged quality
  (Hinglish, joy, admiration; all of EmoLLaMA and Sarvam). The reply checks catch broken replies, not subtle
  quality loss.
* Emotion Reasoner traces (Mood Tracker) live in server memory. When a model is unloaded its service is kept
  without weights (newest 8), so traces of earlier turns stay readable after model switches; after a restart, stored
  conversations show their steering records but older traces are unavailable.
* Agent (MCP) mode turns are not stored.
