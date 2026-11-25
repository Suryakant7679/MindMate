import type { EmotionState, EmotionControllerState } from "./emotion";
import type { SteeringRecord, ToneRecord } from "../api/conversations";
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  steeringAlpha?: number;
  steeringEmotion?: string;
  strategy?: string;
  emotionState?: EmotionState;
  controllerState?: EmotionControllerState;
  /** Registry key and checkpoint of the model that wrote this reply. */
  modelKey?: string | null;
  modelId?: string | null;
  /** What steering did for this reply. */
  steering?: SteeringRecord;
  /** Emotional tone before and after steering. */
  tone?: ToneRecord | null;
  /** Older turns were left out of this model's prompt (its context window is shorter). */
  contextTrimmed?: { sent: number; total: number };
  /** Emotion Reasoner turn (Mood Tracker trace), while the server still holds it. */
  turnId?: string | null;
  /** Agent-mode replies go through the legacy endpoint and are not stored. */
  unsaved?: boolean;
  /** The user stopped this reply; it holds the text shown until then. */
  stopped?: boolean;
}
export interface ChatResponse {
  response: string;
  model: string;
  emotion?: EmotionState;
  controller?: EmotionControllerState;
}
export interface ToolActivity {
  agent: string;
  server: string;
  tool: string;
  status: "running" | "completed" | "error";
  error?: string;
}
