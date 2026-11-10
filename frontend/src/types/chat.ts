import type { EmotionState, EmotionControllerState } from "./emotion";
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
