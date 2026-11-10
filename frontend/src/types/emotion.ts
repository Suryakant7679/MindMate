export interface EmotionState {
  primaryEmotion: string;
  intensity: number;
  valence: number;
  arousal: number;
  confidence?: number;
}
export interface ResponseParameters {
  calmness: number;
  empathy: number;
  supportiveness: number;
  assertiveness: number;
}
export interface EmotionControllerState {
  detectedEmotion: EmotionState;
  responseDirection: string;
  responseStrength: number;
  internalParameters: ResponseParameters;
}

/** Plan produced by the Emotion Reasoner Agent for one user turn. */
export interface EmotionPlan {
  user_emotion: string;
  confidence: number;
  intensity: number;
  valence: number;
  arousal: number;
  probe_scores: Record<string, number>;
  fused_scores: Record<string, number>;
  causes: { cause: string; description: string; cues: string[]; weight: number }[];
  needs: { need: string; description: string; score: number }[];
  strategy: string;
  strategy_title: string;
  technique: string;
  guidance: string;
  response_emotion: string | null;
  steering_alpha: number;
  strength_factors: { factor: string; effect: string; reason: string }[];
  reasoning_paths: {
    score: number;
    hops: { from: string; from_kind: string; relation: string; to: string; to_kind: string }[];
  }[];
  safety_flag: boolean;
}
/** Measured by forward hooks on the steered neurons. */
export interface SteeringSummary {
  applied: boolean;
  emotion: string | null;
  alpha: number;
  gain?: number;
  location?: string;
  neurons_total: number;
  layers: {
    layer: number;
    neuron_count: number;
    neuron_ids: number[];
    decode_positions: number;
    neurons_raised: number;
    neurons_lowered: number;
    mean_abs_applied_shift: number;
    mean_abs_measured_change: number | null;
  }[];
  top_neurons: {
    layer: number;
    neuron: number;
    applied_shift: number;
    steered_activation: number;
    unsteered_activation: number | null;
  }[];
}
/** Estimated by a text classifier on the replies. */
export interface SteeringEffect {
  method: string;
  target: string | null;
  before: Record<string, number> | null;
  after: Record<string, number>;
  target_before?: number;
  target_after?: number;
  steered_reply: string;
  unsteered_reply: string | null;
}
export interface EmotionTurn {
  id: string;
  status: "analysing" | "ready" | "unavailable";
  userText: string;
  plan?: EmotionPlan;
  steering?: SteeringSummary;
  effect?: SteeringEffect;
  /** Known from response headers before the analysis arrives. */
  preview?: { emotion?: string; confidence?: number; steeringEmotion?: string; alpha?: number };
}
