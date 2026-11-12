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
  detector?: string;
  /** Sweep evidence behind the selected strength (absent when no calibrated policy is loaded). */
  calibration?: {
    context: string;
    selected_alpha: number;
    reason: string;
    sweep: {
      alpha: number; n: number; target_gain: number; share_negative_gain?: number;
      empathy_change: number; relevance_change: number; fluency_change: number;
      harm_share: number; semantic_preservation?: number; safe: boolean;
      /** Recalibrated policies: gain on evaluators the run-time gate never sees, with a 95% CI. */
      emotion_rating_gain?: number; emotion_rating_gain_lo?: number; emotion_rating_gain_hi?: number;
      independent_classifier_gain?: number; preference?: number; effective?: boolean; admissible?: boolean;
    }[];
    rule?: { tolerance?: number; ci_margin?: number; max_harm_share?: number; selection_metric?: string } | null;
    ceiling?: { threshold: number } | null;
  } | null;
}
/** What actually happened to the planned steering for this reply. */
export interface SteeringDecision {
  status: "applied" | "skipped" | "rejected" | "not_planned";
  reason: string;
  planned_alpha: number;
  applied_alpha: number;
}
/** Measured by forward hooks on the steered neurons. */
export interface SteeringSummary {
  applied: boolean;
  emotion: string | null;
  alpha: number;
  gain?: number;
  location?: string;
  neurons_total: number;
  /** False when the steered candidate was measured but the unsteered reply was sent. */
  sent?: boolean;
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
  unsteered_reply: string;
  unsteered_scores: Record<string, number>;
  /** Null when no steered candidate was generated. */
  steered_reply: string | null;
  steered_scores: Record<string, number> | null;
  sent: "steered" | "unsteered";
}
/** One intervention result from the offline causal experiments (change vs the unsteered reply, 95% CI). */
export interface CausalEffect {
  label: string;
  alpha?: number;
  neurons?: number;
  layer?: number;
  /** Change of the judge's 1-5 rating of the intended emotion. */
  emotion_rating: number;
  lo: number;
  hi: number;
  /** Change of the judge's 1-5 fluency rating. */
  fluency?: number | null;
}
/** Established offline, not measured on this turn: why this panel, and what steering achieved on held-out messages. */
export interface SteeringEvidence {
  panel: {
    emotion: string;
    name: string;
    selection: string;
    neurons: number;
    layers: number[];
    verdict: string;
    /** Teacher-forced check: change in log-likelihood of emotional replies (nats/token) at this strength. */
    likelihood?: { alpha: number; panel: number; panel_lo: number; panel_hi: number; random: number; neutral_cost: number } | null;
    dose_response: CausalEffect[];
    controls: CausalEffect[];
    ablation: CausalEffect[];
    layer_effects: { layer: number; loss_without: number; lo: number; hi: number; alone: number | null; alone_lo: number | null; alone_hi: number | null }[];
    subsets: CausalEffect[];
  };
  held_out?: {
    messages: number;
    summary: string;
    systems: { system: string; label: string; emotion_rating: number; independent_classifier: number; empathy: number; relevance: number; fluency: number; steered_share: number; latency_median?: number | null }[];
    contrasts: { contrast: string; label: string; metric: string; mean: number; lo: number; hi: number; p?: number | null }[];
  } | null;
  sources?: Record<string, string> | null;
}
export interface EmotionTurn {
  id: string;
  status: "analysing" | "ready" | "unavailable";
  userText: string;
  plan?: EmotionPlan;
  decision?: SteeringDecision;
  steering?: SteeringSummary;
  effect?: SteeringEffect;
  evidence?: SteeringEvidence | null;
  /** Known from response headers before the analysis arrives. */
  preview?: { emotion?: string; confidence?: number; steeringEmotion?: string; alpha?: number };
}
