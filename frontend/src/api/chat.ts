import type { ChatMessage, ChatResponse, ToolActivity } from "../types/chat";
import type {
  ConversationSummary, EmotionPlan, ModelEvaluation, NeuronRecord, PanelView, ReplyComparison, ResearchOverview,
  SteeringDecision, SteeringEffect, SteeringEvidence, SteeringSummary, Trajectory,
} from "../types/emotion";
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
  model?: string,
  conversationId?: string,
  country?: string,
): Promise<StreamResult> {
  const response = await fetch("/api/chat/stream", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.slice(-41).map(({ role, content }) => ({ role, content })),
      ...(model ? { model } : {}),
      ...(conversationId ? { conversation_id: conversationId } : {}),
      ...(country ? { country } : {}),
    }), signal,
  });
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error || "Could not reach the chat model. Please try again.");
  }
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
export interface EmotionTrace {
  plan: EmotionPlan; decision: SteeringDecision; steering: SteeringSummary; effect: SteeringEffect;
  evidence?: SteeringEvidence | null;
  comparison?: ReplyComparison | null;
  trajectory?: Trajectory;
  summary?: ConversationSummary;
  conversation_id?: string;
  turn_index?: number;
  turn_id?: string;
}
/** Plan, steering decision, measured intervention, comparison and conversation trajectory for a finished turn. */
export async function fetchEmotionTrace(turnId: string): Promise<EmotionTrace | null> {
  const response = await fetch(`/api/emotion/trace/${encodeURIComponent(turnId)}`);
  if (!response.ok) return null;
  return response.json();
}
/** Model-level research material: settings, policy, offline evidence and the causal validation report. */
export async function fetchResearch(signal?: AbortSignal): Promise<ResearchOverview | null> {
  const response = await fetch("/api/emotion/research", { signal });
  if (!response.ok) return null;
  return response.json();
}
/** Research mode: also generate a (never sent) steered candidate on turns the policy does not steer. */
export async function setResearchCompare(enabled: boolean): Promise<ResearchOverview["settings"] | null> {
  const response = await fetch("/api/emotion/settings", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ research_compare: enabled }),
  });
  if (!response.ok) return null;
  return response.json();
}
/** Voluntary rating: own mood when writing a turn (1-5) and/or how supportive the reply was (1-5). */
export async function sendFeedback(
  conversationId: string, turnIndex: number, rating: { mood_rating?: number; helpfulness?: number },
): Promise<{ trajectory: Trajectory; summary: ConversationSummary } | null> {
  const response = await fetch("/api/emotion/feedback", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, turn_index: turnIndex, ...rating }),
  });
  if (!response.ok) return null;
  return response.json();
}
export async function sendAgentMessage(
  messages: ChatMessage[], sessionId: string, userId: string, signal?: AbortSignal, model?: string, conversationId?: string, country?: string,
): Promise<{
  response: string; activities: ToolActivity[]; errors: string[]; emotionPlan?: EmotionPlan;
  /** Agent replies go through the same steering path as direct chat, so the same analysis is returned. */
  steering?: SteeringSummary & {
    decision?: SteeringDecision; effect?: SteeringEffect; evidence?: SteeringEvidence | null; comparison?: ReplyComparison | null;
  };
  comparison?: ReplyComparison | null; trajectory?: Trajectory; summary?: ConversationSummary;
  turnId?: string; conversationId?: string; turnIndex?: number;
}> {
  const response = await fetch("/api/agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.slice(-41).map(({ role, content }) => ({ role, content })),
      session_id: sessionId,
      user_id: userId,
      ...(model ? { model } : {}),
      ...(conversationId ? { conversation_id: conversationId } : {}),
      ...(country ? { country } : {}),
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
    comparison: data?.comparison ?? undefined,
    trajectory: data?.trajectory ?? undefined,
    summary: data?.summary ?? undefined,
    turnId: typeof data?.turn_id === "string" ? data.turn_id : undefined,
    conversationId: typeof data?.conversation_id === "string" ? data.conversation_id : undefined,
    turnIndex: typeof data?.turn_index === "number" ? data.turn_index : undefined,
  };
}

export async function getModelStatus(signal?: AbortSignal): Promise<{ ready: boolean; model: string | null; mcpEnabled: boolean }> {
  const response = await fetch("/api/status", { signal });
  if (!response.ok) return { ready: false, model: null, mcpEnabled: false };
  const data = await response.json();
  return { ready: data.ready === true, model: typeof data.model === "string" ? data.model : null, mcpEnabled: data.mcp_enabled === true };
}

/** A chat model the server can serve. Only one is loaded on the GPU at a time. */
export interface ChatModel {
  key: string;
  label: string;
  model_id: string;
  purpose: string;
  /** Weights are on the server's disk. */
  downloaded: boolean;
  loaded: boolean;
  loading: boolean;
  /** Can be selected now (loaded, or on disk, or the server may download it). */
  available: boolean;
  /** Emotion-neuron steering is active for this model. */
  steering: boolean;
}
export async function getModels(signal?: AbortSignal): Promise<{ models: ChatModel[]; active: string | null }> {
  const response = await fetch("/api/models", { signal });
  if (!response.ok) return { models: [], active: null };
  const data = await response.json();
  return { models: Array.isArray(data?.models) ? data.models : [], active: typeof data?.active === "string" ? data.active : null };
}

async function readJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error || `Request failed (${response.status}).`);
  }
  return response.json();
}
/** Saved experimental results of Llama, EmoLLaMA and Sarvam (read from disk; no model runs). */
export const fetchModelEvaluation = (signal?: AbortSignal) => readJson<ModelEvaluation>("/api/emotion/models", signal);
/** Every neuron of one steering panel of the loaded model, with its offline evidence. */
export const fetchPanel = (name: string, signal?: AbortSignal) =>
  readJson<PanelView>(`/api/emotion/panel/${encodeURIComponent(name)}`, signal);
/** Every steered neuron of one turn, as measured by the model hooks. */
export const fetchNeuronRecord = (turnId: string, signal?: AbortSignal) =>
  readJson<NeuronRecord>(`/api/emotion/neurons/${encodeURIComponent(turnId)}`, signal);

/** Delete ephemeral safety context. Ordinary research records have a separate retention policy. */
export async function clearSafetyContext(conversationId: string, sessionId: string): Promise<void> {
  const response = await fetch("/api/safety/clear", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, session_id: sessionId }),
  });
  if (!response.ok) throw new Error("Could not clear safety context.");
}
