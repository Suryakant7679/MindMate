import { useEffect, useRef, useState } from "react";
import { CloudOff, Leaf, RotateCcw } from "lucide-react";
import type { ChatMessage, ToolActivity } from "../../types/chat";
import type { EmotionTurn } from "../../types/emotion";
import { sendAgentMessage, getModelStatus, getModels, fetchEmotionTrace, streamMessage } from "../../api/chat";
import type { ChatModel } from "../../api/chat";
import type { Conversation, Phase, SteeringMode, StoredMessage } from "../../api/conversations";
import { ApiError, cancelGeneration, createConversation, getConversation, streamReply, updateConversation } from "../../api/conversations";
import { MessageBubble } from "./MessageBubble";
import { ChatInput } from "./ChatInput";

const clock = (date: Date = new Date()) => date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export function modelLabel(model: string | null): string {
  if (!model) return "local model";
  if (model.toLowerCase().includes("sarvam")) return "Sarvam 30B";
  if (model.toLowerCase().includes("emollama")) return "EmoLLaMA";
  if (model.toLowerCase().includes("llama-2")) return "Llama-2";
  if (model.toLowerCase().includes("llama")) return "Llama";
  return model;
}

function fromStored(message: StoredMessage): ChatMessage {
  const context = message.metadata?.context;
  return {
    id: message.id, role: message.role, content: message.content, timestamp: clock(new Date(message.created_at * 1000)),
    modelKey: message.model_key, modelId: message.model_id, steering: message.metadata?.steering, turnId: message.turn_id,
    stopped: message.status === "cancelled", tone: message.metadata?.tone ?? null,
    contextTrimmed: context?.trimmed ? { sent: context.messages_sent, total: context.messages_total } : undefined,
  };
}

function phaseText(phase: Phase | null, tokens: number, loadingLabel: string | null): string {
  switch (phase) {
    case "loading_model": return `Loading ${loadingLabel ?? "the model"}… the first message to a model can take a minute`;
    case "generating": return tokens ? `Writing a reply… ${tokens} tokens` : "Writing a reply…";
    default: return "MindMate is thinking…";
  }
}

export function ChatPage({
  active, onEmotionTurn, onResetTurns, conversationId, onConversation, storeHistory, persistence,
}: {
  active: boolean;
  onEmotionTurn: (turn: EmotionTurn) => void;
  /** The Mood Tracker shows the turns of the open conversation only. */
  onResetTurns: () => void;
  /** Open conversation (null: a new chat that is created with the first message). */
  conversationId: string | null;
  /** A conversation was created or changed (title, model, time) by this page. */
  onConversation: (conversation: Partial<Conversation> & { id: string }, created?: boolean) => void;
  storeHistory: boolean;
  /** The server stores conversations (false: consent not given yet or storage unavailable). */
  persistence: boolean;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<Phase | null>(null);
  const [writing, setWriting] = useState(false);
  const [tokens, setTokens] = useState(0);
  const [loadingLabel, setLoadingLabel] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [canRetry, setCanRetry] = useState(false);
  const [ready, setReady] = useState<boolean | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [models, setModels] = useState<ChatModel[]>([]);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [steering, setSteering] = useState<SteeringMode>("always");
  // The model the next message is answered with; the server loads it when that message is sent.
  const [selectedModel, setSelectedModel] = useState<string | undefined>(() => localStorage.getItem("mindmate-model") ?? undefined);
  const [mcpEnabled, setMcpEnabled] = useState(false);
  const [agentMode, setAgentMode] = useState(false);
  const [activities, setActivities] = useState<ToolActivity[]>([]);
  const end = useRef<HTMLDivElement>(null);
  const controller = useRef<AbortController | null>(null);
  const current = useRef<string | null>(conversationId);
  const streaming = useRef<string | null>(null);
  const justCreated = useRef<string | null>(null);
  // Without conversation storage (consent pending is handled by the dialog; an older server or a storage failure),
  // chat falls back to the earlier endpoint that receives the whole history from the browser and stores nothing.
  const legacyConversation = useRef(crypto.randomUUID());
  const stoppedWithText = useRef(false);

  const labelFor = (key: string | null | undefined, id?: string | null) =>
    models.find((item) => item.key === key)?.label ?? modelLabel(id ?? key ?? null);

  useEffect(() => {
    const statusController = new AbortController();
    const check = () =>
      getModelStatus(statusController.signal)
        .then((status) => { setReady(status.ready); setModel(status.model); setMcpEnabled(status.mcpEnabled); })
        .then(() => getModels(statusController.signal))
        .then((list) => {
          setModels(list.models);
          setSelectedModel((selected) =>
            selected && list.models.some((item) => item.key === selected && item.available) ? selected : list.active ?? undefined);
        })
        .catch(() => { if (!statusController.signal.aborted) setReady(false); });
    void check();
    const timer = window.setInterval(check, 15000);
    return () => { clearInterval(timer); statusController.abort(); controller.current?.abort(); };
  }, []);

  // Open a conversation: load its stored messages and rebuild the Mood Tracker from turns the server still holds.
  useEffect(() => {
    current.current = conversationId;
    setError("");
    setCanRetry(false);
    setActivities([]);
    if (justCreated.current && justCreated.current === conversationId) return;   // created by the message being sent
    onResetTurns();
    if (!conversationId) {
      setMessages([]);
      setConversation(null);
      setSteering("always");
      return;
    }
    const abort = new AbortController();
    getConversation(conversationId, abort.signal)
      .then(({ conversation: loaded, messages: stored }) => {
        if (current.current !== conversationId) return;
        setConversation(loaded);
        setSteering(loaded.steering_mode);
        if (loaded.model_key) setSelectedModel(loaded.model_key);
        const shown = stored.map(fromStored);
        setMessages(shown);
        const last = stored[stored.length - 1];
        if (last && last.role === "user" && !loaded.generating) {
          setError("This message has no reply yet.");
          setCanRetry(true);
        }
        stored.forEach((message, index) => {
          if (message.role !== "assistant" || !message.turn_id) return;
          const userText = stored[index - 1]?.role === "user" ? stored[index - 1].content : "";
          const turnId = message.turn_id;
          onEmotionTurn({ id: turnId, status: "analysing", userText, turnId });
          void fetchEmotionTrace(turnId)
            .then((trace) => onEmotionTurn(trace
              ? { id: turnId, status: "ready", userText, ...trace, conversationId: trace.conversation_id, turnIndex: trace.turn_index, turnId }
              : { id: turnId, status: "unavailable", userText, turnId }))
            .catch(() => onEmotionTurn({ id: turnId, status: "unavailable", userText, turnId }));
        });
      })
      .catch((cause) => { if (!abort.signal.aborted) setError(cause instanceof Error ? cause.message : "Could not open the conversation."); });
    return () => abort.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  useEffect(() => {
    if (active) end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy, phase, error, active]);

  function appendAssistant(id: string, text: string) {
    setMessages((all) => all.map((message) => (message.id === id ? { ...message, content: message.content + text } : message)));
  }

  async function ensureConversation(): Promise<string> {
    if (conversationId) return conversationId;
    const created = await createConversation({ model: selectedModel, steering });
    justCreated.current = created.id;
    current.current = created.id;
    setConversation(created);
    onConversation(created, true);
    return created.id;
  }

  async function generate(content: string | null, userBubbleId?: string, history?: ChatMessage[]) {
    if (busy) return;
    setBusy(true);
    setWriting(false);
    setError("");
    setCanRetry(false);
    setPhase(null);
    setTokens(0);
    setActivities([]);
    controller.current = new AbortController();
    const userText = content ?? [...messages].reverse().find((item) => item.role === "user")?.content ?? "";
    const draftId = crypto.randomUUID();
    try {
      if (mcpEnabled && agentMode) {
        await generateAgent(userText, draftId, history ?? messages);
        return;
      }
      if (!persistence) {
        await generateLegacy(userText, draftId, history ?? messages);
        return;
      }
      const id = await ensureConversation();
      streaming.current = id;
      const visible = () => current.current === id;
      let started = false;
      const outcome = await streamReply(id, { content: content ?? undefined, model: selectedModel, steering }, {
        onStart: (data) => {
          if (!visible()) return;
          setLoadingLabel(data.label);
          if (data.user_message && userBubbleId) {
            const stored = fromStored(data.user_message);
            setMessages((all) => all.map((message) => (message.id === userBubbleId ? { ...stored, id: userBubbleId } : message)));
          }
        },
        onStatus: (next, data) => {
          if (!visible()) return;
          setPhase(next);
          if (next === "loading_model" && typeof data.label === "string") setLoadingLabel(data.label);
        },
        onProgress: (count) => { if (visible()) setTokens(count); },
        onDelta: (text) => {
          if (!visible()) return;
          if (!started) {
            started = true;
            setPhase(null);
            setWriting(true);
            setMessages((all) => [...all, { id: draftId, role: "assistant", content: "", timestamp: clock() }]);
          }
          appendAssistant(draftId, text);
        },
        onDone: (data) => {
          onConversation({ id, model_key: data.message.model_key ?? undefined, updated_at: data.message.created_at });
          if (!visible()) return;
          const final = fromStored(data.message);
          // The stored reply (model, steering record); the draft id stays the list key.
          setMessages((all) => all.map((message) => (message.id === draftId ? { ...final, id: draftId } : message)));
          if (data.emotion?.turn_id) {
            const turnId = data.emotion.turn_id;
            const preview = { emotion: data.emotion.user_emotion, confidence: data.emotion.confidence,
                              steeringEmotion: data.message.metadata.steering?.response_emotion ?? undefined,
                              alpha: data.message.metadata.steering?.applied_alpha };
            onEmotionTurn({ id: turnId, status: "analysing", userText, preview, turnId });
            void fetchEmotionTrace(turnId)
              .then((trace) => onEmotionTurn(trace
                ? { id: turnId, status: "ready", userText, preview, ...trace, conversationId: trace.conversation_id, turnIndex: trace.turn_index, turnId }
                : { id: turnId, status: "unavailable", userText, preview, turnId }))
              .catch(() => onEmotionTurn({ id: turnId, status: "unavailable", userText, preview, turnId }));
          }
        },
        onCancelled: (stored) => {
          if (!visible()) return;
          if (stored) {
            const kept = fromStored(stored);
            setMessages((all) => all.map((message) => (message.id === draftId ? { ...kept, id: draftId } : message)));
          }
          stoppedWithText.current = Boolean(stored);
        },
        onTitle: (title) => onConversation({ id, title }),
      }, controller.current.signal);
      if (outcome === "cancelled" && visible()) {
        setError(stoppedWithText.current ? "Stopped. The text shown so far stays in the conversation."
                                         : "Stopped before any reply text was written.");
        setCanRetry(true);
      }
      setReady(true);
      void getModels().then((list) => {
        setModels(list.models);
        if (list.active) setModel(list.models.find((item) => item.key === list.active)?.model_id ?? null);
      }).catch(() => undefined);
    } catch (cause) {
      const aborted = cause instanceof Error && cause.name === "AbortError";
      if (current.current === streaming.current || !streaming.current) {
        setMessages((all) => all.map((message) => (message.id === draftId ? { ...message, stopped: aborted } : message))
          .filter((message) => message.id !== draftId || message.content.trim()));
        setError(aborted ? "Stopped. The text shown so far stays in the conversation."
          : cause instanceof ApiError && cause.data?.consent_required ? "Choose whether your conversations are saved first."
          : cause instanceof Error ? cause.message : "Could not reach the chat model.");
        setCanRetry(true);
      }
    } finally {
      streaming.current = null;
      justCreated.current = null;
      setBusy(false);
      setWriting(false);
      setPhase(null);
      setTokens(0);
    }
  }

  async function generateLegacy(userText: string, draftId: string, all: ChatMessage[]) {
    let response = "";
    setMessages((list) => [...list, { id: draftId, role: "assistant", content: "", timestamp: clock(), unsaved: true }]);
    setWriting(true);
    const result = await streamMessage(all, (chunk) => { response += chunk; appendAssistant(draftId, chunk); },
      controller.current?.signal, selectedModel, legacyConversation.current);
    if (!response.trim()) throw new Error("The model returned an empty response. Please try again.");
    setMessages((list) => list.map((message) => (message.id === draftId
      ? { ...message, modelKey: selectedModel, steeringAlpha: result.steeringAlpha,
          steeringEmotion: result.steeringAlpha ? result.steeringEmotion : undefined } : message)));
    if (result.turnId) {
      const turnId = result.turnId;
      onEmotionTurn({ id: turnId, status: "analysing", userText, turnId });
      void fetchEmotionTrace(turnId)
        .then((trace) => onEmotionTurn(trace
          ? { id: turnId, status: "ready", userText, ...trace, conversationId: trace.conversation_id, turnIndex: trace.turn_index, turnId }
          : { id: turnId, status: "unavailable", userText, turnId }));
    }
  }

  async function generateAgent(userText: string, draftId: string, all: ChatMessage[]) {
    // LangGraph / MCP agent mode keeps its own endpoint; these turns are not stored.
    const history = all.filter((message) => message.content.trim());
    const result = await sendAgentMessage(history, getBrowserId("mindmate-agent-session"), getBrowserId("mindmate-agent-user"),
      controller.current?.signal, selectedModel, conversationId ?? undefined);
    setActivities(result.activities);
    if (!result.response.trim()) throw new Error("The model returned an empty response. Please try again.");
    const plan = result.emotionPlan;
    const applied = result.steering?.decision?.applied_alpha ?? 0;
    setMessages((all) => [...all, { id: draftId, role: "assistant", content: result.response, timestamp: clock(), unsaved: true,
      modelKey: selectedModel, steeringAlpha: applied, steeringEmotion: applied ? plan?.response_emotion ?? undefined : undefined }]);
    if (plan) onEmotionTurn({
      id: result.turnId ?? crypto.randomUUID(), status: "ready", userText, plan, steering: result.steering,
      decision: result.steering?.decision, effect: result.steering?.effect, evidence: result.steering?.evidence,
      comparison: result.comparison ?? result.steering?.comparison, trajectory: result.trajectory, summary: result.summary,
      conversationId: result.conversationId, turnIndex: result.turnIndex, turnId: result.turnId,
    });
  }

  function send(content: string) {
    // Shown at once; the "start" event confirms it was stored.
    const bubble: ChatMessage = { id: crypto.randomUUID(), role: "user", content, timestamp: clock(),
                                  unsaved: (mcpEnabled && agentMode) || !persistence };
    setMessages((all) => [...all, bubble]);
    void generate(content, bubble.id, [...messages, bubble]);
  }

  function stop() {
    const id = streaming.current;
    const active = controller.current;
    if (!id) {                                   // agent mode or the earlier endpoint: just close the connection
      active?.abort();
      return;
    }
    // The server stops generating, keeps the text already shown and ends the stream; closing the connection is only
    // the fallback if it does not answer.
    void cancelGeneration(id).catch(() => active?.abort());
    window.setTimeout(() => { if (streaming.current === id) active?.abort(); }, 4000);
  }

  async function setSteeringMode(mode: SteeringMode) {
    setSteering(mode);
    if (conversationId) {
      try {
        const updated = await updateConversation(conversationId, { steering: mode });
        setConversation(updated);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Could not change the steering setting.");
      }
    }
  }

  const chosen = models.find((item) => item.key === selectedModel);
  const switching = chosen && !chosen.loaded ? chosen : null;
  function selectModel(key: string) {
    setSelectedModel(key);
    localStorage.setItem("mindmate-model", key);
  }
  const viewingStream = busy && (!streaming.current || streaming.current === current.current);
  const steeringStatus = chosen?.steering_status?.status;
  // What the selected mode means for the selected model (panels and calibration are read from disk).
  const steeringLabel = !chosen ? ""
    : steering === "off" ? " · steering off (baseline)"
    : steeringStatus === "unavailable" ? " · no emotion steering (no neuron panels for this model)"
    : steering === "always" ? " · every reply steered"
    : steeringStatus === "policy_gated" ? " · calibrated steering" : " · calibrated policy steers nothing for this model";
  return (
    <section className="chat-page" aria-label="Chat conversation" hidden={!active}>
      <div className="conversation-scroll">
        <div className="conversation">
          {(conversation && !conversation.saved) || (!conversation && !storeHistory && persistence) ? (
            <div className="unsaved-banner" role="note"><CloudOff size={14} /> This conversation is not saved. It stays in server memory for this session only.</div>
          ) : !persistence && ready ? (
            <div className="unsaved-banner" role="note"><CloudOff size={14} /> Conversation history is unavailable on this server; this chat is not saved.</div>
          ) : null}
          {messages.length > 0 && (
            <div className="conversation-label">
              <span />
              {conversation?.title && conversation.title !== "New chat" ? conversation.title.toUpperCase() : "YOUR CONVERSATION"}
              <span />
            </div>
          )}
          <div className="messages" role="log" aria-label="Messages" aria-live="polite">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} modelLabel={labelFor} />
            ))}
            {viewingStream && phase !== null && (
              <div className="thinking" role="status">
                <Leaf size={17} /> {phaseText(phase, tokens, loadingLabel)}
              </div>
            )}
            {viewingStream && phase === null && !writing && (
              <div className="thinking" role="status">
                <Leaf size={17} /> {switching ? `Loading ${switching.label}… this can take a minute` : "MindMate is thinking…"}
              </div>
            )}
          </div>
          {error && (
            <div className="chat-error" role="alert">
              <p>{error}</p>
              {canRetry && (
                <button onClick={() => void generate(null)} disabled={busy}>
                  <RotateCcw size={15} /> {error.startsWith("Stopped") ? "Write the reply again" : "Try again"}
                </button>
              )}
            </div>
          )}
          {activities.length > 0 && (
            <div className="tool-activity" aria-label="Agent tool activity">
              {activities.map((item, index) => (
                <div key={`${item.agent}-${item.server}-${item.tool}-${index}`} className={`tool-event ${item.status}`}>
                  <span>{item.agent.replace(/^./, (letter) => letter.toUpperCase())} Agent</span>
                  <span>› {item.server}</span>
                  <span>› {item.tool}</span>
                  <span>{item.status === "running" ? "Running…" : item.status}</span>
                </div>
              ))}
            </div>
          )}
          <div ref={end} />
        </div>
      </div>
      <div className="connection" role="status">
        <span className={ready ? "status-dot online" : "status-dot"} />
        {ready === null
          ? "Connecting to local model…"
          : ready
            ? switching
              ? `${modelLabel(model)} is loaded · ${switching.label} will load with your next message`
              : `Connected to ${chosen?.label ?? modelLabel(model)}${steeringLabel}`
            : "Local model is unavailable. Start the model server to chat."}
        <label className="steering-toggle" title={"Every reply: each reply is steered, with a strength set by your emotional "
          + "state; it is sent unless it comes out broken. Calibrated policy: steer only where the calibration found it "
          + `quality-safe${steeringStatus === "policy_gated" ? "" : " (this model's policy steers nothing)"}. `
          + "Off: unsteered baseline (strategy and prompt unchanged)."}>
          Steering{" "}
          <select value={steering} onChange={(event) => void setSteeringMode(event.target.value as SteeringMode)} disabled={busy}>
            <option value="always">every reply</option>
            <option value="policy">calibrated policy</option>
            <option value="off">off (baseline)</option>
          </select>
        </label>
        {mcpEnabled && (
          <button type="button" className={agentMode ? "agent-mode active" : "agent-mode"} aria-pressed={agentMode}
                  onClick={() => { setAgentMode((value) => !value); setActivities([]); }} disabled={busy}>
            Agents &amp; tools {agentMode ? "on" : "off"}
          </button>
        )}
      </div>
      <div className="safety-privacy">
        <p>MindMate is a research assistant, not an emergency service. No one is contacted automatically.</p>
      </div>
      <ChatInput onSend={send} busy={busy} onStop={stop}
                 models={models} selectedModel={selectedModel} onSelectModel={selectModel} />
    </section>
  );
}

function getBrowserId(key: string): string {
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const value = crypto.randomUUID();
  localStorage.setItem(key, value);
  return value;
}
