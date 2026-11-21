import { useEffect, useRef, useState } from "react";
import { Leaf, RotateCcw } from "lucide-react";
import type { ChatMessage, ToolActivity } from "../../types/chat";
import type { EmotionTurn } from "../../types/emotion";
import { streamMessage, sendAgentMessage, getModelStatus, getModels, fetchEmotionTrace, clearSafetyContext } from "../../api/chat";
import type { ChatModel } from "../../api/chat";
import { MessageBubble } from "./MessageBubble";
import { ChatInput } from "./ChatInput";
const timestamp = () =>
  new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
function modelLabel(model: string | null): string {
  if (!model) return "local model";
  if (model.toLowerCase().includes("sarvam")) return "Sarvam 30B";
  if (model.toLowerCase().includes("emollama")) return "EmoLLaMA";
  if (model.toLowerCase().includes("llama-2")) return "Llama-2";
  if (model.toLowerCase().includes("llama")) return "Llama";
  return model;
}
export function ChatPage({ active, onEmotionTurn }: { active: boolean; onEmotionTurn: (turn: EmotionTurn) => void }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [supportCountry, setSupportCountry] = useState("");
  const [error, setError] = useState("");
  const [ready, setReady] = useState<boolean | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [models, setModels] = useState<ChatModel[]>([]);
  // The model the next message is answered with; the server loads it when that message is sent.
  const [selectedModel, setSelectedModel] = useState<string | undefined>(() => localStorage.getItem("mindmate-model") ?? undefined);
  const [mcpEnabled, setMcpEnabled] = useState(false);
  const [agentMode, setAgentMode] = useState(false);
  const [activities, setActivities] = useState<ToolActivity[]>([]);
  const end = useRef<HTMLDivElement>(null);
  // One id per conversation on this page, so the server can keep its turns together for the Mood Tracker.
  const conversationId = useRef(crypto.randomUUID());
  const pending = useRef(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => {
    const statusController = new AbortController();
    const check = () =>
      getModelStatus(statusController.signal)
        .then((status) => { setReady(status.ready); setModel(status.model); setMcpEnabled(status.mcpEnabled); })
        .then(() => getModels(statusController.signal))
        .then((list) => {
          setModels(list.models);
          // Keep the user's choice while it is usable; otherwise follow the model the server has loaded.
          setSelectedModel((current) =>
            current && list.models.some((item) => item.key === current && item.available) ? current : list.active ?? undefined);
        })
        .catch(() => {
          if (!statusController.signal.aborted) setReady(false);
        });
    void check();
    const timer = window.setInterval(check, 15000);
    return () => {
      clearInterval(timer);
      statusController.abort();
      controller.current?.abort();
    };
  }, []);
  useEffect(() => {
    if (active)
      end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy, active]);
  async function generate(history: ChatMessage[]) {
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    setError("");
    controller.current = new AbortController();
    const timer = window.setTimeout(() => controller.current?.abort(), 600000);
    const userText = history[history.length - 1]?.content ?? "";
    try {
      let response = "";
      const draft = (extra: Partial<ChatMessage> = {}) =>
        setMessages([...history, { id: crypto.randomUUID(), role: "assistant", content: response, timestamp: timestamp(), ...extra }]);
      draft();
      setActivities([]);
      if (mcpEnabled && agentMode) {
        const sessionId = getBrowserId("mindmate-agent-session");
        const userId = getBrowserId("mindmate-agent-user");
        const result = await sendAgentMessage(history, sessionId, userId, controller.current?.signal, selectedModel, conversationId.current, supportCountry || undefined);
        response = result.response;
        setActivities(result.activities);
        if (!response.trim()) throw new Error("The model returned an empty response. Please try again.");
        const plan = result.emotionPlan;
        // Agent mode uses the same steering selection and safeguards as direct chat and reports the same analysis.
        const applied = result.steering?.decision?.applied_alpha ?? 0;
        draft({ steeringAlpha: applied, steeringEmotion: applied ? plan?.response_emotion ?? undefined : undefined, strategy: plan?.strategy_title });
        if (plan) onEmotionTurn({
          id: result.turnId ?? crypto.randomUUID(), status: "ready", userText, plan, steering: result.steering,
          decision: result.steering?.decision, effect: result.steering?.effect, evidence: result.steering?.evidence,
          comparison: result.comparison ?? result.steering?.comparison, trajectory: result.trajectory, summary: result.summary,
          conversationId: result.conversationId, turnIndex: result.turnIndex, turnId: result.turnId,
        });
      } else {
        const stream = await streamMessage(history, (chunk) => { response += chunk; draft(); }, controller.current.signal, selectedModel, conversationId.current, supportCountry || undefined);
        if (!response.trim()) throw new Error("The model returned an empty response. Please try again.");
        draft({ steeringAlpha: stream.steeringAlpha, steeringEmotion: stream.steeringAlpha ? stream.steeringEmotion : undefined });
        const turnId = stream.turnId;
        if (turnId) {
          const preview = { emotion: stream.emotion, confidence: stream.confidence, steeringEmotion: stream.steeringEmotion, alpha: stream.steeringAlpha };
          onEmotionTurn({ id: turnId, status: "analysing", userText, preview });
          void fetchEmotionTrace(turnId)
            .then((trace) => onEmotionTurn(trace
              ? { id: turnId, status: "ready", userText, preview, ...trace, conversationId: trace.conversation_id,
                  turnIndex: trace.turn_index, turnId }
              : { id: turnId, status: "unavailable", userText, preview }))
            .catch(() => onEmotionTurn({ id: turnId, status: "unavailable", userText, preview }));
        }
      }
      setReady(true);
      void getModels().then((list) => { setModels(list.models); if (list.active) setModel(list.models.find((item) => item.key === list.active)?.model_id ?? null); }).catch(() => undefined);
    } catch (cause) {
      setError(
        cause instanceof Error && cause.name === "AbortError"
          ? "The response took too long. Please try again."
          : cause instanceof Error
            ? cause.message
            : "Could not reach the chat model.",
      );
    } finally {
      clearTimeout(timer);
      pending.current = false;
      setBusy(false);
    }
  }
  function send(content: string) {
    const history = [
      ...messages,
      {
        id: crypto.randomUUID(),
        role: "user" as const,
        content,
        timestamp: timestamp(),
      },
    ];
    setMessages(history);
    void generate(history);
  }
  const chosen = models.find((item) => item.key === selectedModel);
  // Set while a message is waiting for a model that is not on the GPU yet.
  const switching = chosen && !chosen.loaded ? chosen : null;
  function selectModel(key: string) {
    setSelectedModel(key);
    localStorage.setItem("mindmate-model", key);
  }
  return (
    <section
      className="chat-page"
      aria-label="Chat conversation"
      hidden={!active}
    >
      <div className="conversation-scroll">
        <div className="conversation">
          {messages.length > 0 && (
            <div className="conversation-label">
              <span />
              TODAY · YOUR CONVERSATION
              <span />
            </div>
          )}
          <div
            className="messages"
            role="log"
            aria-label="Messages"
            aria-live="polite"
          >
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {busy && (
              <div className="thinking" role="status">
                <Leaf size={17} />{" "}
                {switching ? `Loading ${switching.label}… this can take a minute` : "MindMate is thinking…"}
              </div>
            )}
          </div>
          {error && (
            <div className="chat-error" role="alert">
              <p>{error}</p>
              <button onClick={() => void generate(messages)} disabled={busy}>
                <RotateCcw size={15} /> Try again
              </button>
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
              : `Connected to ${chosen?.label ?? modelLabel(model)}${chosen ? (chosen.steering ? " · emotion steering on" : " · no emotion steering") : ""}`
            : "Local model is unavailable. Start the model server to chat."}
        {mcpEnabled && (
          <button
            type="button"
            className={agentMode ? "agent-mode active" : "agent-mode"}
            aria-pressed={agentMode}
            onClick={() => { setAgentMode((value) => !value); setActivities([]); }}
            disabled={busy}
          >
            Agents &amp; tools {agentMode ? "on" : "off"}
          </button>
        )}
      </div>
      <div className="safety-privacy">
        <p>MindMate is a research assistant, not an emergency service. Safety context is kept temporarily in server memory
          and expires after inactivity (30 minutes by default). Replies are checked before delivery. No one is contacted automatically.</p>
        <label>Support location (optional): {" "}
          <select value={supportCountry} onChange={(event) => setSupportCountry(event.target.value)} disabled={busy}>
            <option value="">Not specified / elsewhere</option><option value="IN">India</option>
          </select>
        </label>{" "}
        <button type="button" disabled={busy} onClick={() => {
          void clearSafetyContext(conversationId.current, getBrowserId("mindmate-agent-session")).then(() => {
            setMessages([]); setError(""); setActivities([]);
            conversationId.current = crypto.randomUUID();
            localStorage.setItem("mindmate-agent-session", crypto.randomUUID());
          }).catch((cause) => setError(String(cause)));
        }}>Clear safety context and start a new chat</button>
        <p>This clears temporary safety context and this chat view. Existing research records are managed separately by the operator.</p>
      </div>
      <ChatInput onSend={send} busy={busy || !!error} models={models} selectedModel={selectedModel} onSelectModel={selectModel} />
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
