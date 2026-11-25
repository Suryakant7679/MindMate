/** Persistent conversations, consent, accounts and streamed replies (see backend/chat_api.py). */

/** always: every reply steered, strength from the user's emotional state; policy: the calibrated policy decides;
 *  off: unsteered baseline. */
export type SteeringMode = "always" | "policy" | "off";

export interface User {
  id: string;
  username: string | null;
  anonymous: boolean;
  store_history: boolean;
  retention_days: number;
  consent_version: string | null;
  consent_at: number | null;
  created_at: number;
}
export interface Me {
  user: User;
  consent: { required: boolean; version: string; outdated: boolean };
  retention_options: number[];
}
export interface Conversation {
  id: string;
  title: string;
  title_source: "pending" | "auto" | "fallback" | "user";
  model_key: string | null;
  steering_mode: SteeringMode;
  created_at: number;
  updated_at: number;
  message_count?: number;
  /** Stored in the database (true) or only in server memory for this session (false). */
  saved: boolean;
  generating?: boolean;
}
/** What steering did for one reply (stored with the message). */
export interface SteeringRecord {
  mode: SteeringMode;
  supported: boolean;
  policy_enabled: boolean;
  applied: boolean;
  status?: string;
  reason?: string;
  disabled_by?: string;
  planned_alpha: number;
  applied_alpha: number;
  panel?: string | null;
  response_emotion?: string | null;
  strategy?: string | null;
  neurons?: number;
  layers?: number[];
  direction_layer?: number;
  gain?: number;
  shadow?: boolean;
  shadow_alpha?: number;
  /** "always" mode: the user's emotional state as a level 0-1, and what the calibrated policy would have used. */
  state_level?: number;
  policy_alpha?: number;
  factors?: { factor: string; effect: string }[];
}
/** Emotional tone of one version of a reply, estimated by the GoEmotions classifier (describes the text). */
export interface ToneState {
  emotion: string;
  expected_valence: number;
  target_score: number | null;
  top: { emotion: string; share: number }[];
  distribution: Record<string, number>;
}
/** Tone before steering (unsteered version) and after steering (steered version) of the same reply. */
export interface ToneRecord {
  method: string;
  target: string | null;
  sent: string;
  shadow: boolean;
  before: ToneState | null;
  after: ToneState | null;
  change: { target_score?: number; expected_valence?: number } | null;
}
export interface StoredMessage {
  id: string;
  conversation_id: string;
  seq: number;
  role: "user" | "assistant";
  content: string;
  status: "complete" | "cancelled" | "error";
  model_key: string | null;
  model_id: string | null;
  turn_id: string | null;
  created_at: number;
  metadata: {
    steering?: SteeringRecord;
    generation?: { latency_seconds: number; generated_tokens: number; max_new_tokens: number; deterministic: boolean };
    context?: { messages_total: number; messages_sent: number; messages_left_out: number; trimmed: boolean };
    context_window?: number | null;
    tone?: ToneRecord | null;
  };
}

export class ApiError extends Error {
  constructor(message: string, public status: number, public data?: Record<string, unknown>) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(data?.error || `Request failed (${response.status}).`, response.status, data ?? undefined);
  return data as T;
}

export const getMe = (signal?: AbortSignal) => request<Me>("GET", "/api/me", undefined, signal);
export const giveConsent = (storeHistory: boolean, retentionDays: number) =>
  request<Me>("POST", "/api/me/consent", { store_history: storeHistory, retention_days: retentionDays });
export const updateSettings = (changes: { store_history?: boolean; retention_days?: number }) =>
  request<Me>("PATCH", "/api/me", changes);
export const deleteAllData = () => request<{ deleted: boolean }>("DELETE", "/api/me");
export const register = (username: string, password: string) =>
  request<Me>("POST", "/api/auth/register", { username, password });
export const login = (username: string, password: string) => request<Me>("POST", "/api/auth/login", { username, password });
export const logout = () => request<{ signed_out: boolean }>("POST", "/api/auth/logout", {});

export const listConversations = (signal?: AbortSignal) =>
  request<{ conversations: Conversation[] }>("GET", "/api/conversations", undefined, signal).then((data) => data.conversations);
export const createConversation = (options: { model?: string; steering?: SteeringMode }) =>
  request<{ conversation: Conversation }>("POST", "/api/conversations", options).then((data) => data.conversation);
export const getConversation = (id: string, signal?: AbortSignal) =>
  request<{ conversation: Conversation; messages: StoredMessage[] }>("GET", `/api/conversations/${id}`, undefined, signal);
export const updateConversation = (id: string, changes: { title?: string; model?: string; steering?: SteeringMode }) =>
  request<{ conversation: Conversation }>("PATCH", `/api/conversations/${id}`, changes).then((data) => data.conversation);
export const deleteConversation = (id: string) => request<{ deleted: boolean }>("DELETE", `/api/conversations/${id}`);
export const cancelGeneration = (id: string) => request<{ cancelled: boolean }>("POST", `/api/conversations/${id}/cancel`, {});
export const exportUrl = "/api/me/export";

/** Phases the server reports before the reply text streams in. */
export type Phase = "loading_model" | "generating";
export interface StreamHandlers {
  onStart?: (data: { model: string; label: string; user_message: StoredMessage | null }) => void;
  onStatus?: (phase: Phase, data: Record<string, unknown>) => void;
  onProgress?: (tokens: number) => void;
  onDelta?: (text: string) => void;
  onDone?: (data: {
    message: StoredMessage;
    emotion: { turn_id: string; user_emotion?: string; confidence?: number; response_emotion?: string; strategy?: string } | null;
  }) => void;
  onTitle?: (title: string) => void;
  /** Stopped by the user; ``message`` holds the text already shown (null when nothing was shown). */
  onCancelled?: (message: StoredMessage | null) => void;
}
export type StreamOutcome = "done" | "cancelled";

/** Send a message (or regenerate the reply to the last unanswered one when ``content`` is omitted). */
export async function streamReply(
  conversationId: string,
  body: { content?: string; model?: string; steering?: SteeringMode },
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<StreamOutcome> {
  const response = await fetch(`/api/conversations/${conversationId}/messages`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  });
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => null);
    throw new ApiError(data?.error || "Could not reach the chat model. Please try again.", response.status, data ?? undefined);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let outcome: StreamOutcome | null = null;
  let failure: string | null = null;
  const dispatch = (block: string) => {
    let name = "";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) name = line.slice(7);
      else if (line.startsWith("data: ")) data.push(line.slice(6));
    }
    if (!name) return;                                // keep-alive comment
    const payload = data.length ? JSON.parse(data.join("\n")) : {};
    if (name === "start") handlers.onStart?.(payload);
    else if (name === "status") handlers.onStatus?.(payload.phase, payload);
    else if (name === "progress") handlers.onProgress?.(payload.tokens);
    else if (name === "delta") handlers.onDelta?.(payload.text);
    else if (name === "done") { outcome = "done"; handlers.onDone?.(payload); }
    else if (name === "title") handlers.onTitle?.(payload.title);
    else if (name === "cancelled") { outcome = "cancelled"; handlers.onCancelled?.(payload.message ?? null); }
    else if (name === "error") failure = payload.error || "Generation failed.";
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index: number;
    while ((index = buffer.indexOf("\n\n")) >= 0) {
      dispatch(buffer.slice(0, index));
      buffer = buffer.slice(index + 2);
    }
  }
  if (buffer.trim()) dispatch(buffer);
  if (failure) throw new ApiError(failure, 500);
  if (!outcome) throw new ApiError("The connection closed before the reply was complete.", 0);
  return outcome;
}
