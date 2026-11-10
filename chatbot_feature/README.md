# Chatbot feature layer

This directory is the isolated home for capabilities around the existing
MindMate model runtime. It now contains the opt-in LangGraph multi-agent and MCP
integration. Direct chat remains the fallback when `MCP_ENABLED` is false. See
[`MCP_SETUP.md`](../MCP_SETUP.md) for configuration and verified server commands.

## Structure

| Area | Responsibility |
| --- | --- |
| `emotion_reasoner/` | Knowledge-graph Emotion Reasoner Agent: understands the user's emotion, chooses a response strategy, response emotion and neuron-steering strength; `agents/emotion.py` is its LangGraph node |
| `mcp_connector/` | Connect to MCP servers; expose approved tools, resources, and prompts |
| `memory/` | Recent turns, summaries, semantic recall, retention, and deletion |
| `rag/` | Ingest documents, chunk/embed them, retrieve evidence, and return citations |
| `tools/` | Tool schemas, argument validation, permissions, execution, and audit logs |
| `safety/` | Mental-health safety, crisis flow, privacy, moderation, and injection defense |
| `conversation/` | Sessions, context budgeting, prompt construction, and response streaming |
| `personalization/` | Explicit user preferences, language, tone, and accessibility settings |
| `observability/` | Privacy-safe logs, tracing, model/tool latency, errors, and quality metrics |
| `evaluation/` | Automated and human tests for quality, safety, RAG, memory, and tools |
| `multimodal/` | Future support for documents, images, and audio |

The shared contracts live in `protocols.py`, common request/response types in
`types.py`, optional feature ordering in `registry.py`, and the model-facing
pipeline boundary in `orchestrator.py`.

## Recommended implementation order

1. **Safety and privacy baseline** — crisis escalation, consent, data redaction,
   retention rules, authorization, and prompt-injection tests.
2. **Conversation state** — stable session IDs, context-window budgeting,
   streaming/cancellation, and conversation summaries.
3. **Memory** — begin with in-memory/session memory; add durable or semantic
   memory only with explicit consent, view/delete controls, and expiry.
4. **RAG** — source ingestion, chunking, embeddings, hybrid retrieval, reranking,
   citations, access control, and an abstain-when-unsupported behavior.
5. **Tools and MCP** — allowlisted servers/tools, schema validation, timeouts,
   user confirmation for sensitive actions, output sanitization, and audit logs.
6. **Observability and evaluation** — latency/error traces and regression suites
   for hallucination, retrieval quality, memory leakage, safety, and tool misuse.
7. **Personalization and multimodal input** — add only after the data and safety
   boundaries above are working.

## Design rules

- Keep model providers behind the `ChatModel` protocol.
- Treat MCP output, retrieved documents, memories, and user uploads as untrusted.
- Never let retrieved text override the system or safety policy.
- Apply least privilege: each connector and tool gets only the access it needs.
- Do not store sensitive mental-health content by default. Make memory opt-in,
  inspectable, editable, deletable, and time-bounded.
- Require citations for knowledge-base claims and preserve source metadata.
- Keep feature hooks small; business logic belongs inside its feature package.
- Give each production feature unit, integration, safety, and failure-mode tests.

## Adding a feature

Implement the `ChatFeature` protocol and register it in the intended order:

```python
from chatbot_feature import FeatureRegistry

registry = FeatureRegistry([
    safety_feature,
    conversation_feature,
    memory_feature,
    rag_feature,
    tool_feature,
])
```

`before_generation` hooks run in registration order. `after_generation` hooks
run in reverse order, following normal middleware behavior.

Run the scaffold tests from the repository root:

```sh
python3 -m unittest discover -s chatbot_feature/tests -v
```
