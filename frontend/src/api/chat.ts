import type { ChatMessage, ChatResponse, ToolActivity } from "../types/chat";
import type { EmotionPlan, SteeringEffect, SteeringSummary } from "../types/emotion";
export async function sendMessage(
  messages: ChatMessage[],
  signal?: AbortSignal,
): Promise<ChatResponse> {
  // Preserve complete turns within the API's 41-message limit.
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages
        .slice(-41)
        .map(({ role, content }) => ({ role, content })),
    }),
    signal,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(data?.error || "Could not reach the chat model. Please try again.");
  if (typeof data?.response !== "string" || !data.response.trim())
    throw new Error("The model returned an empty response. Please try again.");
  return data;
}
export interface StreamResult {
  turnId?: string;
  emotion?: string;
  confidence?: number;
  steeringEmotion?: string;
  steeringAlpha?: number;
  neuronCount?: number;
}
/** Streams the reply. Strategy, response emotion and steering strength are chosen by the Emotion Reasoner Agent. */
export async function streamMessage(
  messages: ChatMessage[],
  onChunk: (chunk: string) => void,
  signal?: AbortSignal,
): Promise<StreamResult> {
  const response = await fetch("/api/chat/stream", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.slice(-41).map(({ role, content }) => ({ role, content })),
    }), signal,
  });
  if (!response.ok || !response.body) throw new Error("Could not reach the chat model. Please try again.");
  const header = (name: string) => response.headers.get(name) || undefined;
  const number = (name: string) => {
    const value = response.headers.get(name);
    return value !== null && Number.isFinite(Number(value)) ? Number(value) : undefined;
  };
  const result: StreamResult = {
    turnId: header("X-Turn-Id"),
    emotion: header("X-Detected-Emotion"),
    confidence: number("X-Emotion-Confidence"),
    steeringEmotion: header("X-Steering-Emotion"),
    steeringAlpha: number("X-Steering-Alpha"),
    neuronCount: number("X-Steering-Neuron-Count"),
  };
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    onChunk(decoder.decode(value, { stream: true }));
  }
  return result;
}
/** Plan, measured intervention and before/after estimate for a finished turn. May take as long as a reply. */
export async function fetchEmotionTrace(turnId: string): Promise<{
  plan: EmotionPlan; steering: SteeringSummary; effect: SteeringEffect;
} | null> {
  const response = await fetch(`/api/emotion/trace/${encodeURIComponent(turnId)}`);
  if (!response.ok) return null;
  return response.json();
}
export async function sendAgentMessage(
  messages: ChatMessage[], sessionId: string, userId: string, signal?: AbortSignal,
): Promise<{ response: string; activities: ToolActivity[]; errors: string[]; emotionPlan?: EmotionPlan; steering?: SteeringSummary }> {
  const response = await fetch("/api/agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.slice(-41).map(({ role, content }) => ({ role, content })),
      session_id: sessionId,
      user_id: userId,
    }),
    signal,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(data?.error || "Agent chat failed.");
  return {
    response: typeof data?.response === "string" ? data.response : "",
    activities: Array.isArray(data?.activities) ? data.activities : [],
    errors: Array.isArray(data?.errors) ? data.errors : [],
    emotionPlan: data?.emotion_plan ?? undefined,
    steering: data?.steering ?? undefined,
  };
}

export async function getModelStatus(signal?: AbortSignal): Promise<{ ready: boolean; model: string | null; mcpEnabled: boolean }> {
  const response = await fetch("/api/status", { signal });
  if (!response.ok) return { ready: false, model: null, mcpEnabled: false };
  const data = await response.json();
  return { ready: data.ready === true, model: typeof data.model === "string" ? data.model : null, mcpEnabled: data.mcp_enabled === true };
}
