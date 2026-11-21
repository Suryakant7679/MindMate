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
  /** Neuron panel steered with, when it differs from the response emotion (model-specific tone or language panels). */
  steering_panel?: string | null;
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
  /** Why no steering was planned: calibrated policy, safety turn, no panel, a strength rule or a tool-grounded agent answer. */
  disabled_by?: "policy" | "safety" | "no_panel" | "strength_rule" | "agent";
  /** Research mode: strength and panel of the comparison-only candidate (never sent). */
  shadow_alpha?: number;
  shadow_panel?: string;
  manual?: boolean;
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
  /** True when the candidate was generated only for comparison (research mode). */
  shadow?: boolean;
  direction?: string;
  decode_steps?: number;
  /** What "contribution" ranks neurons by. */
  contribution_basis?: string;
  heatmap?: {
    columns: number;
    max_abs: number;
    metric: string;
    rows: { layer: number; label: string; cells: { neuron: number; shift: number; change: number }[] }[];
  };
  layers: {
    layer: number;
    /** Set when the site is not the layer's neurons (e.g. "residual stream"). */
    site?: string;
    neuron_count: number;
    neuron_ids: number[];
    decode_positions: number;
    neurons_raised: number;
    neurons_lowered: number;
    mean_abs_applied_shift: number;
    mean_abs_measured_change: number | null;
    mean_applied_shift?: number;
    mean_measured_change?: number | null;
    /** How far earlier layers and tokens had already moved these neurons before their own shift. */
    mean_abs_upstream_change?: number | null;
    rms_unsteered_activation?: number | null;
    /** Mean |shift| divided by the RMS of the neurons' unsteered activation. */
    normalized_shift?: number | null;
    /** Norm of what this layer's shift adds to the residual stream. */
    residual_update_norm?: number | null;
    contribution?: number;
    contribution_share?: number;
  }[];
  top_neurons: {
    layer: number;
    site?: string;
    neuron: number;
    applied_shift: number;
    steered_activation: number;
    unsteered_activation: number | null;
    direction?: "raised" | "lowered" | "unchanged";
    measured_change?: number | null;
    upstream_change?: number | null;
    normalized_shift?: number | null;
    write_norm?: number | null;
    contribution?: number;
    contribution_share?: number;
    rank?: number;
    /** Offline evidence for this neuron from the model's own experiments. */
    evidence?: NeuronCard;
  }[];
}
export interface NeuronCard {
  attribution_rank?: number;
  attribution?: number;
  /** First-order causal estimate: gradient of the emotional reply's log-likelihood x the neuron's shift. */
  gradient_effect?: number | null;
  gradient_rank?: number | null;
  /** Paired t statistic of the activation difference (emotion style - neutral style). */
  selectivity_t?: number | null;
  delta?: number;
  /** Result of this neuron's own intervention, if it was one of the tested neurons. */
  single?: { status: "supported" | "not_supported" | "inconclusive"; screen: number | null; judged: number | null };
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
  shadow?: boolean;
}
export type VerdictLabel = "improved" | "harmed" | "no_meaningful_change" | "not_compared";
export interface Verdict {
  label: VerdictLabel;
  basis: "automatic" | "judged" | "none";
  reasons: string[];
  note?: string;
}
export interface ReplyDescription {
  target_score: number | null;
  top_emotion: string | null;
  words: number;
  repetition: number;
  language: string;
  degenerate: boolean;
  questions: number;
}
/** Offline ratings of both replies by the two judge models (written by generation_analysis.turn_evaluation). */
export interface JudgedComparison {
  turn_id: string;
  judges: Record<string, string>;
  ratings: { unsteered: Record<string, Record<string, number | null>>; steered: Record<string, Record<string, number | null>> };
  difference: Record<string, Record<string, number | null>>;
  preference: Record<string, number | null>;
  preservation?: Record<string, number | null>;
  independent?: { model: string; target: string; unsteered: number; steered: number; difference: number } | null;
}
/** Steered against unsteered reply to the same prompt. */
export interface ReplyComparison {
  available: boolean;
  delivered: "steered" | "unsteered";
  shadow: boolean;
  unsteered: ReplyDescription;
  steered: ReplyDescription | null;
  automatic: {
    identical: boolean;
    target_gain: number | null;
    repetition_change: number;
    length_ratio: number;
    lexical_overlap: number;
    user_language: string;
    hinglish_kept: boolean | null;
    hinglish_gained: boolean | null;
    score_changes: Record<string, number>;
  } | null;
  judged: JudgedComparison | null;
  verdict: Verdict;
  automatic_verdict?: Verdict;
}
export interface TrajectoryTurn {
  turn: number;
  text: string;
  language: string;
  emotion: string;
  confidence: number;
  intensity: number;
  label_valence: number;
  distribution: Record<string, number>;
  detector_distribution: Record<string, number>;
  detector_emotion: string;
  detector_agrees: boolean;
  expected_valence: number;
  expected_arousal: number;
  entropy: number;
  margin: number;
  uncertain: boolean;
  causes: string[];
  ema_valence: number;
  delta_valence: number | null;
  js_divergence: number | null;
  label_changed: boolean | null;
  self_rating: number | null;
}
export interface TrajectoryResponse {
  turn: number;
  turn_id: string | null;
  mode: string | null;
  text: string | null;
  emotion?: string;
  expected_valence?: number;
  target?: string | null;
  target_score?: number | null;
  delivered: "steered" | "unsteered" | null;
  status: SteeringDecision["status"] | null;
  reason: string | null;
  disabled_by: string | null;
  planned_alpha: number | null;
  applied_alpha: number | null;
  shadow_alpha: number | null;
  strategy: string | null;
  strategy_title: string | null;
  panel: string | null;
  compared: boolean;
  shadow: boolean;
  verdict: VerdictLabel | null;
  verdict_basis: string | null;
  target_gain: number | null;
  unsteered_target_score: number | null;
  steered_target_score: number | null;
  helpfulness: number | null;
  user_language: string;
}
/** Observational record of the user's detected emotion across the conversation. */
export interface Trajectory {
  turns: TrajectoryTurn[];
  labels: string[];
  mean_distribution: Record<string, number>;
  distribution_std: Record<string, number>;
  frequency: Record<string, number>;
  dominant_emotion: string | null;
  dominant_first_half: string | null;
  dominant_second_half: string | null;
  mood: {
    mean_expected_valence: number | null;
    latest_ema: number | null;
    trend_per_turn: number | null;
    trend_interpretable: boolean;
    std: number | null;
    mean_abs_change: number | null;
    label_switch_rate: number | null;
    uncertain_share: number | null;
    detector_disagreement_share: number | null;
    mean_confidence: number | null;
    mean_entropy: number | null;
  };
  languages: Record<string, number>;
  self_report: {
    n: number;
    pairs: { turn: number; rating: number; reported_valence: number; estimated_valence: number; error: number; same_sign: boolean | null }[];
    mean_abs_error: number | null;
    mean_error: number | null;
    sign_agreement: number | null;
    spearman: number | null;
    note: string;
  };
  method: Record<string, string>;
  caveats: string[];
  responses: TrajectoryResponse[];
  after_response: {
    rows: {
      turn: number; user_emotion: string; next_user_emotion: string; valence_before: number; valence_after: number;
      delta_valence: number; reply_emotion: string | null; reply_valence: number | null; delivered: string | null;
    }[];
    n: number;
    mean_delta_valence: number | null;
    improved_share: number | null;
    worsened_share: number | null;
    by_delivery: Record<string, { n: number; mean_delta_valence: number | null }>;
    note: string;
  };
}
export interface ConversationSummary {
  turns: number;
  tracked_turns: number;
  decisions: Record<SteeringDecision["status"], number>;
  steered_delivered: number;
  steered_delivered_share: number | null;
  comparison: {
    compared: number;
    counts: Record<"improved" | "harmed" | "no_meaningful_change", number>;
    shares: Record<"improved" | "harmed" | "no_meaningful_change", number | null>;
    judged: number;
    delivered_steered: number;
    shadow: number;
  };
  mean_target_gain_compared: number | null;
  mean_target_gain_delivered: number | null;
  by_language: Record<string, { turns: number; steered_delivered: number; compared: number; improved: number; harmed: number; no_meaningful_change: number }>;
  user: {
    dominant_emotion: string | null; mean_expected_valence: number | null; trend_per_turn: number | null;
    trend_interpretable: boolean; mean_distribution: Record<string, number>; uncertain_share: number | null; self_reports: number;
  };
  response: { mean_expected_valence: number | null; mean_target_score: number | null };
  helpfulness: { n: number; mean: number | null };
  research_compare: boolean;
  separation: string[];
}
/** Mean paired change with a 95% bootstrap CI. */
export interface Stat {
  n: number;
  mean: number | null;
  lo: number | null;
  hi: number | null;
  p?: number | null;
  p_holm?: number | null;
  /** Standardised paired effect size (mean / SD of the paired differences). */
  dz?: number | null;
}
export interface ValidationCondition {
  name: string;
  family: string;
  label: string;
  /** Pairs related conditions: the layer band, or the subset size. */
  group?: string | null;
  alpha: number | null;
  neurons: number | null;
  seed?: number | null;
  n: number;
  changed_share: number | null;
  emotion: Stat;
  emotion_judges: Record<string, Stat>;
  independent: Stat;
  quality: Record<string, Stat>;
  preference: number | null;
  harm_share: number | null;
  degenerate_share: number | null;
  length_ratio: number | null;
}
export interface ValidationContrast {
  name: string;
  label: string;
  question: string;
  first: string;
  second: string;
  stat: Stat;
  /** Whether the result supports the causal reading the question asks about (null: not decidable). */
  supports: boolean | null;
}
export interface ValidationVerdict {
  status: "supported" | "not_supported" | "inconclusive" | "not_tested";
  text: string;
}
export interface ValidationPanel {
  panel: string;
  target: string | null;
  neurons: number | null;
  layers: number[];
  messages: number;
  languages: Record<string, number>;
  selection: string | null;
  discovery?: Record<string, unknown> | null;
  conditions: ValidationCondition[];
  contrasts: ValidationContrast[];
  by_language: Record<string, { messages: number; conditions: ValidationCondition[]; contrasts: ValidationContrast[] }>;
  verdict: Record<string, ValidationVerdict> & { summary?: ValidationVerdict };
}
/** Unified causal validation battery of one model (generation_analysis.neuron_validation). */
export interface ValidationReport {
  model: string;
  model_key: string;
  generated: string;
  split: string;
  messages: number;
  judges: Record<string, string>;
  reference_alpha: number;
  alphas: number[];
  site: string;
  seeds: number[];
  architecture?: Record<string, unknown> | null;
  panels: Record<string, ValidationPanel>;
  pooled: ValidationPanel;
  judge_agreement: { metric: string; replies: number; spearman: number | null; mean_abs_difference?: number | null; same_winner_share?: number | null }[];
  sampling?: {
    seeds: number[]; messages: number; temperature: number;
    rows: { seed: number | string; panel: Stat; matched_random: Stat; difference: Stat }[];
    note: string;
  } | null;
  failures?: { name: string; label: string; issue: string }[];
  notes: string[];
}
export interface ResearchOverview {
  model: string | null;
  model_tag: string;
  steering_location: string | null;
  architecture?: Architecture;
  panels: Record<string, number>;
  settings: {
    research_compare: boolean; shadow_alpha: number; allow_overrides: boolean;
    telemetry: { conversations: number; turns: number; persisted: boolean; directory: string | null };
  };
  policy: {
    source?: string; version?: number | string; split?: string; steering_enabled: boolean; note?: string | null;
    gate: Record<string, unknown>;
    cells: { context: string; alpha: number; panel: string | null; reason: string | null }[];
  } | null;
  evidence: { panels?: Record<string, SteeringEvidence["panel"]>; held_out?: SteeringEvidence["held_out"]; sources?: Record<string, string> } | null;
  validation: ValidationReport | null;
  single_neuron?: SingleNeuronReport | null;
  trajectory_validation: TrajectoryValidation | null;
  cross_model: CrossModelSummary | null;
}
/** Detector check on labelled messages (generation_analysis.trajectory_validation). */
export interface TrajectoryValidation {
  generated: string;
  detector: string;
  messages: number;
  by_language: {
    language: string; n: number; accuracy: number; accuracy_lo: number; accuracy_hi: number; fused_accuracy?: number | null;
    polarity_auc: number | null; polarity_accuracy: number | null; valence_correlation: number | null;
    ece: number; mean_confidence: number;
  }[];
  reliability: { language: string; bins: { low: number; high: number; n: number; confidence: number; accuracy: number }[] }[];
  notes: string[];
}
export interface CrossModelSummary {
  generated: string;
  rows: {
    model: string; model_key: string; language: string; source: string; n: number | null;
    steered_share: number | null; emotion_gain: number | null; emotion_gain_lo: number | null; emotion_gain_hi: number | null;
    quality_change: number | null; harmed_share: number | null; policy: string;
  }[];
  notes: string[];
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
  comparison?: ReplyComparison | null;
  trajectory?: Trajectory;
  summary?: ConversationSummary;
  conversationId?: string;
  turnIndex?: number;
  /** Server id of the turn (per-neuron record, offline judging); the list key ``id`` can differ in agent mode. */
  turnId?: string;
  /** Known from response headers before the analysis arrives. */
  preview?: { emotion?: string; confidence?: number; steeringEmotion?: string; alpha?: number };
}

/** One strength of a panel in the validation battery (change against the unsteered reply). */
export interface DoseRow {
  alpha: number;
  n: number;
  emotion: Stat;
  judges: Record<string, number | null>;
  independent: number | null;
  quality: Record<string, number | null>;
  harm_share: number | null;
  preference: number | null;
  length_ratio?: number | null;
}
export interface ModelBattery {
  generated: string;
  messages: number;
  split: string;
  languages: Record<string, number>;
  reference_alpha: number;
  judges: Record<string, string>;
  verdicts: Record<string, ValidationVerdict & { alpha?: number | null }>;
  panels: Record<string, Record<string, ValidationVerdict>>;
  doses: DoseRow[];
  controls: Record<string, Stat>;
  reverse: Stat | null;
  usable_alpha: number | null;
  by_language: Record<string, { messages: number; panels: Record<string, DoseRow[]> }>;
  agreement: ValidationReport["judge_agreement"];
  sampling?: ValidationReport["sampling"];
}
export interface ModelResult {
  key: string;
  tag: string;
  label: string;
  model: string | null;
  loaded: boolean;
  policy: {
    installed: boolean; steering_enabled: boolean; steered_cells: string[]; cells: number;
    source?: string | null; split?: string | null; note?: string | null;
  } | null;
  battery: ModelBattery | null;
  battery_status: string;
  held_out: SteeringEvidence["held_out"];
  calibration: {
    language: string; alpha: number; n: number; emotion: Stat; quality: Record<string, number | null>;
    harm_share: number | null; preference: number | null; hinglish_share_change: number | null;
  }[];
}
/** Saved results of every model, read from disk by the server. */
export interface ModelEvaluation {
  models: ModelResult[];
  detector: TrajectoryValidation | null;
  harm_rule: string;
}
export interface PanelNeuron extends NeuronCard {
  layer: number;
  neuron: number;
  delta: number;
}
export interface PanelView {
  panel: string;
  model: string | null;
  neurons_total: number;
  layers: number[];
  per_layer: Record<string, number>;
  location: string | null;
  selection: string | null;
  has_cards: boolean;
  neurons: PanelNeuron[];
}
/** Every steered neuron of one turn (the telemetry record). */
export interface NeuronRecord {
  turn_id: string;
  model: string | null;
  panel: string | null;
  alpha: number;
  gain?: number | null;
  location?: string | null;
  delivered: boolean;
  shadow: boolean;
  decode_steps?: number;
  contribution_basis?: string;
  neurons: SteeringSummary["top_neurons"];
}
export interface Architecture {
  model: string | null;
  class: string | null;
  layers?: number;
  hidden_size?: number | null;
  site_widths?: number[];
  steerable_neurons?: number;
  mixture_of_experts?: boolean;
  quantized?: boolean;
}

export interface SingleNeuronEntry {
  role: "panel" | "control";
  layer: number;
  neuron: number;
  /** Likelihood screen at the strongest single-neuron dose (nats per token). */
  screen: Stat;
  emotional: number | null;
  plain: number | null;
  doses: Record<string, Stat>;
  /** Panel effect lost when this neuron is left out of the panel (screen). */
  leave_out: Stat | null;
  above_null: boolean;
  screen_status: "supported" | "not_supported" | "inconclusive" | "control";
  /** Judged emotion change of generated replies (only for the neurons tested by generation). */
  judged: (Stat & {
    vs_control: Stat | null; harm_share: number | null; preference: number | null; fluency: number | null;
    judges: Record<string, number | null>; leave_out: Stat | null;
  }) | null;
  status: "supported" | "not_supported" | "inconclusive" | "control";
  reading: string;
}
/** Single-neuron interventions of one model (generation_analysis.single_neuron). */
export interface SingleNeuronReport {
  model_key: string;
  model: string;
  generated: string;
  reference: number;
  doses: number[];
  null_percentile: number;
  judges: Record<string, string>;
  method: string;
  notes: string[];
  panels: Record<string, {
    ranked_by: string; messages: number; null_threshold: number | null; null_mean: number | null; full_panel: Stat;
    tested: number; supported: number; screen_supported: number; generated: number;
    verdict: ValidationVerdict; neurons: SingleNeuronEntry[];
  }>;
}
