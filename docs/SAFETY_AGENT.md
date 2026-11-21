# Agent 1: Safety and Risk Assessment

This implementation is an engineering research prototype, not clinical validation, diagnosis, or a C-SSRS implementation. Agent 2 (the broader Response Critic) is not implemented.

## Architecture inspected

- `backend/server.py`: threaded HTTP server; `/api/chat`, `/api/chat/stream`, `/api/workspace/chat`, and `/api/agent/chat`. Browser messages carry recent history. The frontend previously received raw runtime tokens on unsteered turns.
- `chatbot_feature/graph/{builder,state,routing,service}.py`: LangGraph supervisor, six keyword-routed specialists (including Memory and Planning), and response synthesis. Previously `AgentChatService` checkpointed messages and emotion state in SQLite under client-provided `user_id:session_id`.
- `chatbot_feature/agents/emotion.py` and `emotion_reasoner/{agent,knowledge_graph,policy}.py`: emotion probe probabilities feed knowledge-graph needs/strategy selection and calibrated steering strengths. The graph has 17 crisis phrases but no contextual risk classifier.
- `backend/emotion_service.py`: direct chat's Emotion Reasoner, candidate generation/comparison, research traces, and streaming callbacks. Reply emotion scoring uses GoEmotions XLM-R or the model's frozen emotion probe. Neither is a validated suicide-risk detector.
- `backend/runtime.py`, `llama.py`, `sarvam.py`: shared generation lock and temporary steering hooks; Sarvam additionally supports direction/shared-expert steering. The research/checkpoint/neuron files remain untouched by this implementation.
- `backend/research.py`: existing optional raw conversation/reply research persistence. Frontend chat is React state; `MessageBubble.tsx` renders escaped React text, not raw HTML.
- Existing datasets concern emotions/steering. Cached models include Llama 3.1, Sarvam 30B, emotion classifiers, and general language encoders; no suitable trained risk classifier was found.

## Implemented flow

```text
HTTP validation + signed anonymous browser identity + recent history
  -> safety screen + contextual assessment + deterministic policy
       -> R0: existing Emotion Reasoner / KG / routing / steering / draft
       -> R1: supportive clarification template
       -> R2/R3: support + direct safety question + human support template
       -> R4: immediate emergency template (no LLM/steering call)
  -> deterministic outgoing checks + semantic check for model-written drafts
  -> approved text, or checked fixed replacement
  -> HTTP JSON / buffered text stream
```

Actual LangGraph edges in `graph/builder.py`:

```text
START -> safety_assessment
  normal -> emotion_reasoner (when configured) -> supervisor -> specialists -> response
  override -> crisis_response
response / crisis_response -> outgoing_safety -> END
```

`next_node` rechecks policy at delegation boundaries. Specialists enforce `tools_allowed` before connecting/calling MCP. `ResponseAgent` refuses normal generation when steering is forbidden. Only outgoing validation appends the assistant message; rejected drafts never enter chat history. Draft text in steering comparisons is also checked/redacted before it is exposed. Crisis replies never invoke Llama or Sarvam generation, so configured steering strengths or model-specific hooks cannot reactivate.

## Classification and policy

`ProvisionalScreen` is a replaceable, **untrained** compositional screen. It separates clauses, limited negation and attribution, historical language, intent, behavior, means and time. It supplies a risk floor, not a calibrated score. Its multilingual normalization preserves the original message and is supplemental. Every noncritical message, including screen negatives, goes to structured semantic assessment; this is not a keyword-only detector. Explicit critical evidence uses the immediate template without waiting for inference.

`ContextualAssessor` uses the already loaded local chat model, with deterministic unsteered generation, recent original-language turns, and a minimal unresolved policy record. Pydantic requires schema-valid JSON and enumerated evidence codes, rejecting extra fields or malformed outputs. Its confidence is an uncalibrated self-estimate. No third-party assessment API or additional model download is introduced. The selected local Llama/Sarvam checkpoint retains its existing license and limitations; no clinical intended-use or validation status is claimed for either. A replacement model can implement async `complete_safety(prompt)`; a replacement screen implements `screen(text) -> Screen`.

The deterministic policy conservatively combines model judgment and the screen floor. Attempt/injury, or present intent plus plan/access and near-term timing, can independently force R4. Disagreement cannot reduce a higher floor. Low confidence, missing models, invalid outputs, or timeouts produce clarification, never unchecked normal replies. This intentionally increases false positives during outages. R1 gets a short clarification without emergency numbers. R2/R3 get safety support; R4 gets urgent action before questions.

Unresolved context survives topic changes. Explicit current safety, human support, and an appropriate model assessment can revise prior concerns. A current attempt overrides a safety assertion. Bounded contextual rules also recognize “decided” / “tonight” following risk evidence. This is limited language coverage, not a substitute for trained models or expert review.

## Responses and referrals

Templates cover English, Hindi and romanized Hindi/Hinglish. `Resources` and the injected directory support country-specific configuration. The UI lets users explicitly choose India; language never selects location. Unknown locations receive generic local-emergency guidance and a country question. India uses Tele-MANAS 14416 as primary mental-health support, alternate 1800-891-4416 in the directory, and 112 for immediate emergency assistance.

Sources checked during implementation: [Government of India Tele-MANAS release](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2022057&lang=2&reg=48), [Ministry of Home Affairs emergency response system](https://www.mha.gov.in/emergency-response-support-system-erss). Directory entries need periodic operator review. The app does not contact anyone or claim that help has been dispatched.

## Outgoing validation and streaming

Outgoing checks target harmful instructions/encouragement, minimization, shaming, unsupported diagnosis/clinical authority, discouraging care, false confidentiality/rescue promises, missing urgent help, and prohibited crisis steering. Fixed exact-match internal templates receive deterministic validation. Model-written text must also pass semantic review, in two steps. First the draft is reviewed without the user's message, for every category; a self-contradictory answer counts as a rejection. Then it is reviewed with the user's message, and that review can reject only for minimization, shaming or missing urgent help, the categories that depend on what the user said. In live tests, the user's wording swayed the in-context review into approving unsafe drafts, and on safe drafts it named encouragement or harm instructions most of the time. A failed check, unparseable output or timeout replaces the entire draft with a checked template, with no regeneration loop. The validation result records rejection even though a safe replacement is sent.

All HTTP chat streams now buffer complete responses. The text/plain frontend protocol remains compatible; first visible text arrives later. The production server never supplies live network callbacks to unvalidated generation. Internal runtime and EmotionSteeringService experiment APIs still support raw callbacks for research; they must not be exposed directly as public chat endpoints. Chunk safety is not claimed. Model-generation failures return an error without exposing partial text.

A bounded independent inference worker makes assessment timeout return promptly without waiting for GPU shutdown. Timeout does **not** cancel an already running GPU inference. While that worker remains busy, later safety calls fail closed rather than queue indefinitely. Ordinary model-generation failures can still reduce availability. The existing runtime lock serializes hooks and safety inference.

## Privacy and access boundaries

- New safety state is memory-only, bounded (256 sessions), with configurable inactivity expiration. Expired items are removed on the next store operation; process restart clears the store. Retained assessments contain enum codes, not quotations.
- `AgentChatService` no longer writes safety-enriched LangGraph states into SQLite. Its `checkpoint_path` argument remains accepted for compatibility but is unused. Its recent chat context is also ephemeral and bounded to 24 messages. Existing SQLite files are not deleted. Explicit external callers using `build_agent_graph(checkpointer=...)` remain responsible for consent and retention; the deployed service does not do so.
- Signed random HttpOnly/SameSite browser cookies isolate server state. Client `user_id` cannot select another namespace. Keys hash the structured user/session pair to avoid concatenation collisions. Agent mode switches still pass through the shared HTTP safety gate with browser history.
- Trace, neuron and conversation read endpoints and feedback require ownership of an issued resource. This is anonymous session isolation, **not account authentication**. Shared browsers share identity; cookie theft, transport protection, authorization for research settings, and multi-process session coordination require deployment work. Bind to localhost by default; remote deployment needs HTTPS, Secure cookies and authenticated access.
- Raw crisis text is not added to ordinary diagnostic logs. Generation errors log exception types only. Safety overrides skip emotion research recording. Diagnostic results do not contain raw assessment explanations.
- Raw research persistence is now opt-in with `--research-telemetry`; obtain participant consent before enabling it. Existing research files and in-memory emotion histories have their own lifecycle. Clearing safety context does not erase those files or existing SQLite records. The UI explains this distinction.
- UI: “Clear safety context and start a new chat” deletes temporary direct and agent safety context and clears the current chat view. Location is optional, in page state. User/model text uses React escaping. No automatic third-party notifications exist. Ordinary R0 tool workflows still use the project's pre-existing MCP integrations and permissions.

## Configuration

| Setting | Default | Meaning |
|---|---:|---|
| `MINDMATE_SAFETY_TIMEOUT` | 30 seconds | Structured assessment/review timeout |
| `MINDMATE_SAFETY_CONTEXT_TURNS` | 12 messages | Recent role-preserving assessment context |
| `MINDMATE_SAFETY_RETENTION_SECONDS` | 1800 | Ephemeral session inactivity lifetime |
| `SafetyAgent(confidence_threshold=...)` | 0.65 | Uncalibrated uncertainty threshold |
| `SafetyAgent(model=..., screen=..., directory=...)` | loaded local model / provisional screen / India directory | Replaceable dependencies |
| `--model llama` / `--model sarvam` | llama | Existing local model selection |
| `--research-telemetry` | off | Explicit raw research record persistence |

## Files created or changed for Agent 1

| Files | Purpose |
|---|---|
| `chatbot_feature/safety/schemas.py` | Typed triage, dimensions, policy, validation and timing |
| `safety/screening.py`, `assessor.py`, `policy.py` | Replaceable screen, contextual semantic assessment, deterministic authority |
| `safety/resources.py`, `responses.py`, `validator.py` | Country directory, fixed response templates, outgoing guard |
| `safety/agent.py`, `runtime.py`, `session.py`, `__init__.py` | Orchestration, local bounded inference, ephemeral sessions, exports |
| `chatbot_feature/agents/safety.py` | Assessment, crisis and outgoing LangGraph nodes |
| `agents/model.py`, `response.py`, `specialist.py` | Safety model adapter, draft-only synthesis, tool gate |
| `graph/builder.py`, `routing.py`, `state.py`, `service.py` | Mandatory safety edges, typed fields, transient isolated state |
| `backend/server.py`, `backend/emotion_service.py` | All endpoint gates, buffered delivery, metadata checks, privacy boundaries |
| `backend/requirements.txt` | Explicit Pydantic dependency already installed transitively |
| `frontend/src/api/chat.ts`, `components/Chat/ChatPage.tsx`, `styles.css` | Country option, privacy notice, context deletion |
| `chatbot_feature/tests/safety_fakes.py`, `test_safety_agent.py`, `test_safety_integration.py` | CPU-only schema and control-flow regressions |
| Existing graph/emotion/workspace tests | Updated mocks and expectations for mandatory safety gates |
| `chatbot_feature/evaluation/safety/{cases.json,evaluate.py,offline_report.json}` | Synthetic cases, measured baseline pipeline, saved results |
| This document, `README.md` | Operation, architecture, evaluation and limitations |

The workspace already contained unrelated modifications on arrival; they were preserved. The diff against HEAD also includes those pre-existing changes.

## Reproducible evaluation

`cases.json` has 60 manually curated synthetic development scenarios (68 total user turns), four language groups, and 13 outgoing cases. Coverage includes ordinary distress, R1–R4, indirect language, figures of speech, negation/quotes/fiction, third-person reports, escalation, de-escalation, and topic changes. It is neither expert-annotated nor a held-out benchmark. Models must not be trained/tuned on this file and then described as evaluated on independent data.

Measured CPU results in `offline_report.json`:

| System | Macro F1 | Active/imminent recall | Ordinary-message false-positive rate |
|---|---:|---:|---:|
| A: existing 17-phrase cue | 0.1421 | 0.3333 | 0.15 |
| B: context-free provisional screen | 0.8872 | 0.8333 | 0.15 |
| C: offline failure fallback, **not hybrid model inference** | 0.6818 | 1.0000 | 1.00 |

A was binary; assigning its positive cases R3 is an explicitly documented severity proxy. Its macro-F1 is therefore not an equivalent five-class model comparison. B's development-set performance is optimistic and says nothing about clinical reliability. Offline C's perfect high-risk recall is accompanied by **100% ordinary-message escalation to clarification**. It demonstrates conservative failure handling, not a useful or validated classifier. Outgoing fallback blocked all 11 unsafe drafts **and both safe drafts**; deterministic checks alone missed one indirect unsafe case. Do not interpret the zero escape rate under total model outage as quality evidence.

The JSON includes class precision/recall/F1, weighted F1, confusion matrix, language-specific metrics, missed multi-turn escalation, timeout/fallback rates, and local timing. No calibrated risk probability is reported. Full chatbot latency, GPU inference latency/cost, model accuracy, and real-world effectiveness are unmeasured (`null` where appropriate). CUDA was unavailable (`torch.cuda.is_available() == False`), so local Llama/Sarvam hybrid inference was not executed. CPU-only tests use scripted model outputs to test control flow; these are not evaluation predictions.

Use `--mode live --model llama` or `sarvam` on the intended model machine to measure the real hybrid. Ablations are `--ablation no_context`, `no_llm`, `no_screen`, and `no_normalization`; output separate files for every run. `no_llm` is labeled failure fallback. Store hardware, checkpoint revision, prompt revision, dataset hash and operator configuration with research runs. Live validation and input assessment share the same general model and may share failure modes.

A suitable external benchmark candidate is the [UMD Reddit Suicidality Dataset v2](https://psresnik.github.io/umd_reddit_suicidality_dataset.html), including expert annotations. Access requires an application and institutional ethics approval; it is not an unrestricted downloadable benchmark. Obtain and follow its data-use agreement, do not redistribute source posts, and review label mapping. Reddit user-level English risk labels differ from conversational R0–R4 engineering triage and do not establish Hindi/Hinglish validity. No external dataset was downloaded or clinical data created.

### Live GPU evaluation (`live_eval.py`)

`chatbot_feature/evaluation/safety/live_eval.py` evaluates the deployed configuration: the chat model of a preset (`--model llama|sarvam|llama2|emollama`) is loaded once and acts as assessor and validator through `SafetyRuntimeModel`, as in the server. `benchmark.json` (148 held-out synthetic cases, 175 user turns, 18 multi-turn conversations with per-turn labels, 36 outgoing drafts) was written after the screen rules and with different wording; `cases.json` stays the development set. Parts:

- `classify`: A (`A_keyword17`), B (`B_rule_screen`; `B_llm_context_free` = the agent on the current message only), C (`C_full`) and ablations (`no_prior`, `no_screen`, `no_normalization`, `no_llm`). Case- and turn-level metrics, 95% case-bootstrap intervals, R3–R4 false-negative rate, R0 false-positive rate, per language/category, steering/Emotion-Reasoner gating implied by each policy, and the parse status and latency of every safety-model call (including the share slower than the 30 s production timeout).
- `outgoing`: deterministic-only, model-only and hybrid validation of the drafts, plus whether `finalize` replaces unsafe drafts by the validated template (safety-override accuracy).
- `pipeline`: real requests to `/api/chat` and `/api/chat/stream` on an in-process `AppServer` with the loaded model and real assistant replies in multi-turn conversations. Instrumentation records the assessment, every `finalize`, `begin_turn` calls, requested steering strengths and chat generations. It checks that off-normal turns produce only the template (no Emotion Reasoner, steering or chat generation), that the streamed body equals the approved reply and its first byte follows approval, and that no rejected draft leaks. Fault injection forces every unsafe normal-route draft of the dataset (or `--inject N` of them) as the chat model's reply (never the safety model's) on a carrier message that the main pass routed to normal chat on the same endpoint, preferring the draft's language, so each draft actually reaches the guard.

Outputs (`results/safety_eval/<run>/<dataset>/`): `predictions_turns.csv`, `predictions_cases.csv`, `confusion_<system>.csv`, `failures.csv`, `outgoing.csv`, `pipeline.csv`, `fault_injection.csv`, `llm_calls.csv`, `summary.json`, and `manifest.json` (model, commit, GPU, dataset hashes). Files carry case ids and codes, never message text. `report.py` builds `results/safety_eval/report/REPORT.md` and comparison tables. `run_safety_eval.sh` runs the stages with logs. The harness itself is tested with scripted fakes in `chatbot_feature/tests/test_safety_live_eval.py`; those tests produce no model-quality numbers.

Model-free result on the held-out set (`--no-model`, measured): the rule screen falls from macro-F1 0.887 on `cases.json` to 0.188 on `benchmark.json`, with an R3–R4 false-negative rate of 0.898 (it was written on the development phrasing). Without a model (`C_no_llm`) 86% of R3–R4 held-out cases get only the R1 clarification. High-risk recall therefore rests on the model's assessment, so the live run's schema-parse failure and timeout rates matter as much as its accuracy.

## Commands

From the repository root, using the project's Python environment:

```bash
# Fresh environment: existing dependencies plus explicit Pydantic.
python3 -m pip install -r backend/requirements.txt

# CPU safety tests; no weight downloads.
python3 -m unittest chatbot_feature.tests.test_safety_agent chatbot_feature.tests.test_safety_integration -v

# Existing feature and focused backend/steering regressions.
python3 -m unittest discover -s chatbot_feature/tests -v
python3 -m unittest tests.test_workspace_api tests.test_emotion_service tests.test_chat tests.test_sarvam_steering -v

# Reproduce the offline result; deliberately no LLM.
python3 -m chatbot_feature.evaluation.safety.evaluate

# Actual hybrid inference on a machine with the cached model and suitable GPU memory.
python3 -m chatbot_feature.evaluation.safety.evaluate --mode live --model llama --output /tmp/safety-llama.json
python3 -m chatbot_feature.evaluation.safety.evaluate --mode live --model sarvam --output /tmp/safety-sarvam.json

# Build and serve UI and backend together. No downloads unless explicitly enabled.
npm --prefix frontend install
npm --prefix frontend run build
python3 -m backend.server --model llama --no-telemetry
# Alternative:
python3 -m backend.server --model sarvam --no-telemetry
# Enable configured MCP tools when needed:
MCP_ENABLED=true python3 -m backend.server --model llama --no-telemetry
```

Open `http://127.0.0.1:8080/chatbot`. Select India explicitly to exercise its directory. Try ordinary exam stress; a passive wish; an explicit imminent disclosure; and the multi-turn “cannot keep going / already decided / tonight” scenario. Check that safety turns have no emotion steering/tool activity, urgent replies appear before any unnecessary question, and switching agent mode cannot bypass safety. Clear context between independent scenarios. Safe full-response buffering increases time to first visible text. For connectivity and template-only failure handling, `python3 -m backend.server --no-model` runs without loading weights; this is not normal chat or a hybrid evaluation.

## Remaining limits

The screen has limited linguistic scope, idiom, temporal and quotation handling; the general LLM can misunderstand multilingual euphemisms, hallucinate classifications, or follow prompt injections despite instructions. Both false positives and false negatives remain possible. Templates and uncertainty thresholds require expert and lived-experience review. No probability calibration, clinical study, adversarial certification, long-term patient risk record, automatic rescue, or full therapeutic-quality critic is supplied. Safety buffering cannot guarantee semantic safety when both detection layers miss a harmful reply. Research interfaces remain localhost-oriented and need deployment authentication, secure transport, consent/retention procedures, operational monitoring, and measured GPU tests before real-world use.

## Verification completed

- Feature test discovery: **56 passed, 0 failed, 0 skipped**, including **32** safety unit/integration cases.
- Focused backend/workspace/chat/Sarvam regression modules: **46 passed, 0 failed, 0 skipped**.
- Total completed targeted checks: **102 passed**. Python compilation and `git diff --check` passed.
- `npm --prefix frontend run build` passed. Vite emitted its existing large-chunk advisory.
- Synthetic offline evaluation executed and saved; live GPU evaluation was not run.
