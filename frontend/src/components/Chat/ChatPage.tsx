import { useEffect, useRef, useState } from "react";
import { Leaf, RotateCcw } from "lucide-react";
import type { ChatMessage, ToolActivity } from "../../types/chat";
import type { EmotionTurn } from "../../types/emotion";
import { streamMessage, sendAgentMessage, getModelStatus, fetchEmotionTrace } from "../../api/chat";
import { MessageBubble } from "./MessageBubble";
import { ChatInput } from "./ChatInput";
const timestamp = () =>
  new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
function modelLabel(model: string | null): string {
  if (!model) return "local model";
  if (model.toLowerCase().includes("sarvam")) return "Sarvam 30B";
  if (model.toLowerCase().includes("llama")) return "Llama";
  return model;
}
export function ChatPage({ active, onEmotionTurn }: { active: boolean; onEmotionTurn: (turn: EmotionTurn) => void }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [ready, setReady] = useState<boolean | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [mcpEnabled, setMcpEnabled] = useState(false);
  const [agentMode, setAgentMode] = useState(false);
  const [activities, setActivities] = useState<ToolActivity[]>([]);
  const end = useRef<HTMLDivElement>(null);
  const pending = useRef(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => {
    const statusController = new AbortController();
    const check = () =>
      getModelStatus(statusController.signal)
        .then((status) => { setReady(status.ready); setModel(status.model); setMcpEnabled(status.mcpEnabled); })
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
        const result = await sendAgentMessage(history, sessionId, userId, controller.current?.signal);
        response = result.response;
        setActivities(result.activities);
        if (!response.trim()) throw new Error("The model returned an empty response. Please try again.");
        const plan = result.emotionPlan;
        draft({ steeringAlpha: plan?.steering_alpha, steeringEmotion: plan?.response_emotion ?? undefined, strategy: plan?.strategy_title });
        // Agent mode reports the plan and the measured intervention; it does not generate an unsteered reply.
        if (plan) onEmotionTurn({ id: crypto.randomUUID(), status: "ready", userText, plan, steering: result.steering });
      } else {
        const stream = await streamMessage(history, (chunk) => { response += chunk; draft(); }, controller.current.signal);
        if (!response.trim()) throw new Error("The model returned an empty response. Please try again.");
        draft({ steeringAlpha: stream.steeringAlpha, steeringEmotion: stream.steeringEmotion });
        const turnId = stream.turnId;
        if (turnId) {
          const preview = { emotion: stream.emotion, confidence: stream.confidence, steeringEmotion: stream.steeringEmotion, alpha: stream.steeringAlpha };
          onEmotionTurn({ id: turnId, status: "analysing", userText, preview });
          // The analysis re-generates the reply without steering, so it arrives after the reply itself.
          void fetchEmotionTrace(turnId)
            .then((trace) => onEmotionTurn(trace
              ? { id: turnId, status: "ready", userText, preview, ...trace }
              : { id: turnId, status: "unavailable", userText, preview }))
            .catch(() => onEmotionTurn({ id: turnId, status: "unavailable", userText, preview }));
        }
      }
      setReady(true);
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
                <Leaf size={17} /> MindMate is thinking…
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
            ? `Connected to ${modelLabel(model)}`
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
      <ChatInput onSend={send} busy={busy || !!error} />
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
