import { useState } from "react";
import { Activity, ArrowRight, Brain, Cpu, FlaskConical, Gauge, Info, Sparkles, TrendingUp } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CausalEffect, EmotionTurn } from "../../types/emotion";
import { ParameterStrength } from "./ParameterStrength";

const label = (text: string) => text.replace(/_/g, " ").replace(/^./, (letter) => letter.toUpperCase());
const fixed = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined ? "–" : value.toFixed(digits);
const tooltipStyle = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 12,
  color: "var(--text)",
  fontSize: 12,
};
const tick = { fill: "var(--muted)", fontSize: 11 };
const EMOTIONS = ["love", "joy", "admiration", "surprise", "neutral", "sadness", "fear", "anger", "disapproval", "disgust"];

const DECISION_LABEL = {
  applied: "Steering applied",
  skipped: "Steering skipped",
  rejected: "Steered reply rejected",
  not_planned: "No steering planned",
};
const signed = (value: number, digits = 2) => `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;

function Source({ kind }: { kind: "measured" | "estimated" | "reasoned" | "calibrated" | "validated" }) {
  const text = {
    measured: "MEASURED · MODEL HOOKS",
    estimated: "ESTIMATED · TEXT CLASSIFIER",
    reasoned: "REASONED · KNOWLEDGE GRAPH",
    calibrated: "CALIBRATED · STRENGTH SWEEP",
    validated: "VALIDATED OFFLINE · CONTROLLED EXPERIMENTS",
  }[kind];
  return <span className={`badge source ${kind}`}>{text}</span>;
}
const interval = (low: number, high: number, digits = 2) => `${signed(low, digits)} to ${signed(high, digits)}`;
/** Recharts error bars take [distance below, distance above]. */
const errorBar = (value: number, low: number, high: number) => [value - low, high - value];

function EffectTable({ rows, caption }: { rows: CausalEffect[]; caption: string }) {
  return (
    <table className="data-table">
      <caption className="muted small">{caption}</caption>
      <thead>
        <tr>
          <th>Intervention</th><th className="num">Neurons</th><th className="num">α</th>
          <th className="num">Emotion rating change</th><th className="num">95% CI</th><th className="num">Fluency change</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.label}>
            <th scope="row">{row.label}</th>
            <td className="num">{row.neurons?.toLocaleString() ?? "–"}</td>
            <td className="num">{row.alpha === undefined ? "–" : fixed(row.alpha)}</td>
            <td className="num">{signed(row.emotion_rating)}</td>
            <td className="num">{interval(row.lo, row.hi)}</td>
            <td className="num">{row.fluency === undefined || row.fluency === null ? "–" : signed(row.fluency)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function MoodTrackerPage({ turns }: { turns: EmotionTurn[] }) {
  const [picked, setPicked] = useState<string | null>(null);
  if (!turns.length) return (
    <section className="mood-page" aria-label="Mood Tracker">
      <div className="demo-notice">
        <Info size={19} />
        <div><strong>No conversation yet</strong><p>Send a chat message. The Emotion Reasoner's plan and the steering it applied will appear here.</p></div>
        <span className="badge">LIVE</span>
      </div>
    </section>
  );
  const turn = turns.find((item) => item.id === picked) ?? turns[turns.length - 1];
  const turnNumber = turns.indexOf(turn) + 1;
  const { plan, steering, effect, decision, evidence } = turn;
  const panel = evidence?.panel;
  const heldOut = evidence?.held_out;
  const doseRows = panel
    ? panel.dose_response.map((row) => {
        const control = panel.controls.find((item) => item.alpha === row.alpha);
        return {
          alpha: `α ${fixed(row.alpha)}`,
          panel: row.emotion_rating, panelError: errorBar(row.emotion_rating, row.lo, row.hi),
          random: control?.emotion_rating ?? null,
          randomError: control ? errorBar(control.emotion_rating, control.lo, control.hi) : undefined,
        };
      })
    : [];
  const layerEvidence = panel
    ? panel.layer_effects.map((row) => ({ ...row, error: errorBar(row.loss_without, row.lo, row.hi) }))
    : [];
  const recalibrated = plan?.calibration?.sweep.some((row) => row.emotion_rating_gain !== undefined) ?? false;
  const target = effect?.target ?? null;

  const timeline = turns.map((item, index) => ({
    turn: index + 1,
    mood: item.plan ? Number((item.plan.valence * item.plan.intensity).toFixed(2)) : null,
    emotion: item.plan?.user_emotion ?? item.preview?.emotion ?? "",
    alpha: item.decision?.applied_alpha ?? item.plan?.steering_alpha ?? item.preview?.alpha ?? 0,
    target: item.plan?.response_emotion ?? item.preview?.steeringEmotion ?? "none",
    without: item.effect?.target ? item.effect.unsteered_scores[item.effect.target] : null,
    withSteering: item.effect?.target && item.effect.steered_scores ? item.effect.steered_scores[item.effect.target] : null,
  }));
  const layerRows = steering?.layers ?? [];
  const scoreRows = effect
    ? EMOTIONS.map((emotion) => ({
        emotion: label(emotion) + (emotion === effect.target ? " ★" : ""),
        without: effect.unsteered_scores[emotion],
        withSteering: effect.steered_scores?.[emotion] ?? null,
      }))
    : [];

  return (
    <section className="mood-page" aria-label="Mood Tracker">
      <div className="demo-notice">
        <Info size={19} />
        <div>
          <strong>How the Emotion Reasoner handled each turn</strong>
          <p>
            Neuron activations are measured inside the model. Emotion scores of replies are classifier estimates.
            Causal and held-out results come from controlled offline experiments. This is a research view, not a clinical assessment.
          </p>
        </div>
        <span className="badge">LIVE</span>
      </div>

      <div className="turn-picker" role="tablist" aria-label="Conversation turn">
        {turns.map((item, index) => (
          <button
            key={item.id} role="tab" aria-selected={item.id === turn.id}
            className={item.id === turn.id ? "active" : ""} onClick={() => setPicked(item.id)}
          >
            Turn {index + 1}
            <span>{item.plan?.user_emotion ?? item.preview?.emotion ?? "…"}</span>
          </button>
        ))}
      </div>
      <p className="turn-quote">“{turn.userText}”</p>

      {!plan && (
        <article className="mood-card">
          <p className="muted" role="status">
            {turn.status === "analysing"
              ? "Loading this turn's analysis…"
              : "The analysis for this turn is unavailable."}
            {turn.preview?.steeringEmotion && ` Steering applied: ${turn.preview.steeringEmotion}, α ${fixed(turn.preview.alpha)}.`}
          </p>
        </article>
      )}

      {plan && (
        <>
          <div className="mood-grid">
            <article className="mood-card">
              <div className="card-heading">
                <Activity size={20} />
                <h2>Turn {turnNumber}: what was understood</h2>
              </div>
              <div className="flow">
                <div>
                  <span className="eyebrow">USER EMOTION</span>
                  <strong>{label(plan.user_emotion)}</strong>
                  <small>confidence {fixed(plan.confidence)}</small>
                </div>
                <ArrowRight size={18} aria-hidden="true" />
                <div>
                  <span className="eyebrow">RESPONSE EMOTION</span>
                  <strong>{plan.response_emotion ? label(plan.response_emotion) : "None"}</strong>
                  <small>
                    {decision
                      ? decision.applied_alpha ? `steered, α ${fixed(decision.applied_alpha)}` : "not steered"
                      : plan.response_emotion ? `steering α ${fixed(plan.steering_alpha)}` : "steering off"}
                  </small>
                </div>
              </div>
              <ParameterStrength label="Intensity" value={plan.intensity} />
              <div className="emotion-stats">
                <div>Valence<strong>{fixed(plan.valence)}</strong></div>
                <div>Arousal<strong>{fixed(plan.arousal)}</strong></div>
                <div>Detector alone<strong>{label(Object.entries(plan.probe_scores).sort((a, b) => b[1] - a[1])[0][0])}</strong></div>
              </div>
              {plan.causes.length > 0 && (
                <p className="muted small">
                  Likely context: {plan.causes.map((cause) => `${label(cause.cause)} (“${cause.cues.join("”, “")}”)`).join("; ")}
                </p>
              )}
            </article>

            <article className="mood-card">
              <div className="card-heading">
                <TrendingUp size={20} />
                <h2>Across the conversation</h2>
              </div>
              <div className="mini-charts">
                <figure>
                  <figcaption>User mood (valence × intensity, −1 to 1)</figcaption>
                  <div className="chart mini" role="img" aria-label="User mood score per turn">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={timeline} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}>
                        <CartesianGrid stroke="var(--border)" vertical={false} />
                        <XAxis dataKey="turn" tickLine={false} axisLine={false} tick={tick} tickFormatter={(value) => `T${value}`} />
                        <YAxis domain={[-1, 1]} ticks={[-1, 0, 1]} tickLine={false} axisLine={false} tick={tick} />
                        <Tooltip contentStyle={tooltipStyle} labelFormatter={(value) => `Turn ${value}`}
                          formatter={(value, _name, item) => [`${Number(value).toFixed(2)} (${item.payload.emotion})`, "Mood"]} />
                        <Line dataKey="mood" stroke="var(--viz-with)" strokeWidth={2} dot={{ r: 4, fill: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} connectNulls />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </figure>
                <figure>
                  <figcaption>Steering strength applied (α)</figcaption>
                  <div className="chart mini" role="img" aria-label="Steering strength per turn">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={timeline} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}>
                        <CartesianGrid stroke="var(--border)" vertical={false} />
                        <XAxis dataKey="turn" tickLine={false} axisLine={false} tick={tick} tickFormatter={(value) => `T${value}`} />
                        <YAxis domain={[0, 1.2]} ticks={[0, 0.6, 1.2]} tickLine={false} axisLine={false} tick={tick} />
                        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }} labelFormatter={(value) => `Turn ${value}`}
                          formatter={(value, _name, item) => [`α ${Number(value).toFixed(2)} toward ${item.payload.target}`, "Steering"]} />
                        <Bar dataKey="alpha" fill="var(--viz-with)" radius={[4, 4, 0, 0]} maxBarSize={26} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </figure>
              </div>
              <figure>
                <figcaption>
                  Reply's score for the intended emotion, without and with steering (where a steered reply was generated) <Source kind="estimated" />
                </figcaption>
                <div className="chart mini" role="img" aria-label="Intended-emotion score of the reply without and with steering, per turn">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={timeline} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}>
                      <CartesianGrid stroke="var(--border)" vertical={false} />
                      <XAxis dataKey="turn" tickLine={false} axisLine={false} tick={tick} tickFormatter={(value) => `T${value}`} />
                      <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickLine={false} axisLine={false} tick={tick} />
                      <Tooltip contentStyle={tooltipStyle} labelFormatter={(value) => `Turn ${value}`}
                        formatter={(value, name) => [Number(value).toFixed(2), name]} />
                      <Legend iconType="plainline" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
                      <Line name="Without steering" dataKey="without" stroke="var(--viz-without)" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 4, fill: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} />
                      <Line name="With steering" dataKey="withSteering" stroke="var(--viz-with)" strokeWidth={2} dot={{ r: 4, fill: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </figure>
            </article>
          </div>

          <article className="mood-card">
            <div className="card-heading">
              <Brain size={20} />
              <h2>Emotion Reasoner: strategy</h2>
              <Source kind="reasoned" />
            </div>
            {plan.safety_flag && (
              <p className="safety-note" role="alert">
                Safety response: the message suggests the user may be at risk, so neuron steering was switched off.
              </p>
            )}
            <div className="reasoner-grid">
              <div>
                <span className="eyebrow">RESPONSE STRATEGY</span>
                <div className="strategy-name">{plan.strategy_title}</div>
                <p className="muted">Borrowed from: {plan.technique}.</p>
                <p className="guidance">{plan.guidance}</p>
                {plan.needs.length > 0 && (
                  <p className="muted small">What the user likely needs: {plan.needs.map((need) => need.description).join("; ")}.</p>
                )}
              </div>
              <div>
                <span className="eyebrow">KNOWLEDGE-GRAPH PATHS SUPPORTING IT</span>
                <ul className="kg-paths">
                  {plan.reasoning_paths.map((path, index) => (
                    <li key={index}>
                      <span className={`node ${path.hops[0].from_kind}`}>{label(path.hops[0].from)}</span>
                      {path.hops.map((hop, hopIndex) => (
                        <span key={hopIndex} className="hop">
                          <em>{hop.relation.replace(/_/g, " ")} →</em>
                          <span className={`node ${hop.to_kind}`}>{label(hop.to)}</span>
                        </span>
                      ))}
                      <small>support {fixed(path.score)}</small>
                    </li>
                  ))}
                </ul>
                <span className="eyebrow">HOW THE STEERING STRENGTH WAS PLANNED</span>
                <table className="data-table">
                  <tbody>
                    {plan.strength_factors.map((factor, index) => (
                      <tr key={index}>
                        <th scope="row">{label(factor.factor)}</th>
                        <td className="num">{factor.effect}</td>
                        <td>{factor.reason}</td>
                      </tr>
                    ))}
                    <tr className="total">
                      <th scope="row">Planned</th>
                      <td className="num">α {fixed(plan.steering_alpha)}</td>
                      <td>{plan.response_emotion && plan.steering_alpha ? `toward ${plan.response_emotion}` : "no steering"}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </article>

          {(decision || plan.calibration) && (
            <article className="mood-card">
              <div className="card-heading">
                <Gauge size={20} />
                <h2>Steering decision and effectiveness</h2>
                {plan.calibration && <Source kind="calibrated" />}
              </div>
              {decision && (
                <div className={`decision ${decision.status}`}>
                  <strong>{DECISION_LABEL[decision.status]}</strong>
                  <span>planned α {fixed(decision.planned_alpha)} · applied α {fixed(decision.applied_alpha)}</span>
                  <p>{label(decision.reason)}.</p>
                </div>
              )}
              {plan.calibration && (
                <>
                  <p className="muted small">
                    Strength sweep for this context ({plan.calibration.context.replace(/^strategy:/, "").replace("|", ", ").replace(/_/g, " ")}).
                    Changes are relative to the unsteered reply with the same strategy prompt. Empathy, relevance and fluency
                    are 1–5 ratings by an independent judge model; a strength is safe when none drops by more than{" "}
                    {fixed(plan.calibration.rule?.tolerance ?? 0.05)} on average and at most{" "}
                    {Math.round((plan.calibration.rule?.max_harm_share ?? 0.2) * 100)}% of replies lose more than 0.5.
                    {recalibrated && " The strength is selected on the judge's emotion rating (1–5), which the run-time gate never sees; the gate classifier's score is shown for reference only."}
                  </p>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Strength</th><th className="num">Test replies</th>
                        {recalibrated && <th className="num">Emotion rating (95% CI)</th>}
                        <th className="num">{recalibrated ? "Gate classifier" : "Target emotion"}</th>
                        <th className="num">Empathy</th><th className="num">Relevance</th><th className="num">Fluency</th>
                        <th className="num">Replies harmed</th><th>Verdict</th>
                      </tr>
                    </thead>
                    <tbody>
                      {plan.calibration.sweep.map((row) => (
                        <tr key={row.alpha} className={row.alpha === plan.calibration?.selected_alpha ? "selected" : ""}>
                          <th scope="row">α {fixed(row.alpha)}</th>
                          <td className="num">{row.n}</td>
                          {recalibrated && (
                            <td className="num">
                              {row.emotion_rating_gain === undefined ? "–" : `${signed(row.emotion_rating_gain)} (${interval(row.emotion_rating_gain_lo ?? 0, row.emotion_rating_gain_hi ?? 0)})`}
                            </td>
                          )}
                          <td className="num">{signed(row.target_gain)}</td>
                          <td className="num">{signed(row.empathy_change)}</td>
                          <td className="num">{signed(row.relevance_change)}</td>
                          <td className="num">{signed(row.fluency_change)}</td>
                          <td className="num">{Math.round(row.harm_share * 100)}%</td>
                          <td>
                            {row.alpha === plan.calibration?.selected_alpha
                              ? "✓ selected"
                              : !row.safe ? "✕ lowers quality"
                              : row.effective === false ? "safe, no reliable gain"
                              : row.admissible === false ? "safe, next strength is not"
                              : "safe"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {plan.calibration.selected_alpha === 0 && (
                    <p className="muted small">{label(plan.calibration.reason)}.</p>
                  )}
                </>
              )}
              {heldOut && (
                <>
                  <h3>Measured effectiveness on held-out messages <Source kind="validated" /></h3>
                  <p className="muted small">
                    {heldOut.summary} {heldOut.messages} messages that were never used to choose neurons or strengths.
                    Emotion rating, empathy, relevance and fluency are 1–5 judge ratings; the classifier is one the run-time gate does not use.
                  </p>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>System</th><th className="num">Emotion rating</th><th className="num">Independent classifier</th>
                        <th className="num">Empathy</th><th className="num">Relevance</th><th className="num">Fluency</th>
                        <th className="num">Replies steered</th><th className="num">Median latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {heldOut.systems.map((row) => (
                        <tr key={row.system} className={row.system === "v2" ? "selected" : ""}>
                          <th scope="row">{row.label}</th>
                          <td className="num">{fixed(row.emotion_rating)}</td>
                          <td className="num">{fixed(row.independent_classifier, 3)}</td>
                          <td className="num">{fixed(row.empathy)}</td>
                          <td className="num">{fixed(row.relevance)}</td>
                          <td className="num">{fixed(row.fluency)}</td>
                          <td className="num">{Math.round(row.steered_share * 100)}%</td>
                          <td className="num">{row.latency_median === undefined || row.latency_median === null ? "–" : `${fixed(row.latency_median, 1)} s`}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <details>
                    <summary>Paired differences with 95% confidence intervals</summary>
                    <table className="data-table">
                      <thead>
                        <tr><th>Comparison</th><th>Metric</th><th className="num">Mean difference</th><th className="num">95% CI</th><th className="num">p</th></tr>
                      </thead>
                      <tbody>
                        {heldOut.contrasts.map((row) => (
                          <tr key={`${row.contrast}-${row.metric}`}>
                            <th scope="row">{row.label}</th>
                            <td>{row.metric}</td>
                            <td className="num">{signed(row.mean, 3)}</td>
                            <td className="num">{interval(row.lo, row.hi, 3)}</td>
                            <td className="num">{row.p === undefined || row.p === null ? "–" : row.p < 0.001 ? "<0.001" : fixed(row.p, 3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </details>
                </>
              )}
            </article>
          )}

          <article className="mood-card">
            <div className="card-heading">
              <Cpu size={20} />
              <h2>Neuron steering applied</h2>
              <Source kind="measured" />
            </div>
            {!steering?.applied ? (
              <p className="muted">No neurons were steered for this turn.</p>
            ) : (
              <>
                {steering.sent === false && (
                  <p className="safety-note">
                    These measurements are from a steered candidate reply that was not sent; the user received the unsteered reply.
                  </p>
                )}
                <div className="stat-row">
                  <div><span>Strength</span><strong>α {fixed(steering.alpha)}</strong></div>
                  <div><span>Panel</span><strong>{label(steering.emotion ?? "")}</strong></div>
                  <div><span>Layers</span><strong>{layerRows[0].layer}–{layerRows[layerRows.length - 1].layer}</strong></div>
                  <div><span>Neurons</span><strong>{steering.neurons_total.toLocaleString()}</strong></div>
                  <div><span>Decode steps</span><strong>{layerRows[0].decode_positions}</strong></div>
                </div>
                <p className="muted small">
                  Location: {steering.location}. Each neuron's activation is shifted by α × {steering.gain} × its
                  emotion-minus-neutral mean difference. Activations are averaged over this reply's decode steps;
                  “change” compares them with the same neurons in an unsteered reply to the same prompt.
                </p>
                <figure>
                  <figcaption>Mean absolute activation change per layer</figcaption>
                  <div className="chart" role="img" aria-label="Applied shift and measured activation change per steered layer">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={layerRows} margin={{ top: 8, right: 12, left: -14, bottom: 0 }} barGap={2}>
                        <CartesianGrid stroke="var(--border)" vertical={false} />
                        <XAxis dataKey="layer" tickLine={false} axisLine={false} tick={tick} tickFormatter={(value) => `L${value}`} />
                        <YAxis tickLine={false} axisLine={false} tick={tick} />
                        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }} labelFormatter={(value) => `Layer ${value}`}
                          formatter={(value, name) => [Number(value).toFixed(4), name]} />
                        <Legend iconType="square" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
                        <Bar name="Shift added by the hook" dataKey="mean_abs_applied_shift" fill="var(--viz-without)" radius={[4, 4, 0, 0]} maxBarSize={18} isAnimationActive={false} />
                        <Bar name="Change vs unsteered reply" dataKey="mean_abs_measured_change" fill="var(--viz-with)" radius={[4, 4, 0, 0]} maxBarSize={18} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </figure>
                <details>
                  <summary>Per-layer table and neuron IDs</summary>
                  <table className="data-table">
                    <thead>
                      <tr><th>Layer</th><th className="num">Neurons</th><th className="num">Raised / lowered</th><th className="num">Mean |shift|</th><th className="num">Mean |change|</th><th>Neuron IDs</th></tr>
                    </thead>
                    <tbody>
                      {layerRows.map((row) => (
                        <tr key={row.layer}>
                          <th scope="row">{row.layer}</th>
                          <td className="num">{row.neuron_count}</td>
                          <td className="num">{row.neurons_raised} / {row.neurons_lowered}</td>
                          <td className="num">{fixed(row.mean_abs_applied_shift, 4)}</td>
                          <td className="num">{fixed(row.mean_abs_measured_change, 4)}</td>
                          <td><details><summary>show {row.neuron_count}</summary><code>{row.neuron_ids.join(", ")}</code></details></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
                <h3>Most strongly steered neurons</h3>
                <table className="data-table">
                  <thead>
                    <tr><th>Layer</th><th>Neuron ID</th><th className="num">Shift added</th><th className="num">Activation, unsteered reply</th><th className="num">Activation, steered reply</th></tr>
                  </thead>
                  <tbody>
                    {steering.top_neurons.map((neuron) => (
                      <tr key={`${neuron.layer}-${neuron.neuron}`}>
                        <th scope="row">{neuron.layer}</th>
                        <td>{neuron.neuron}</td>
                        <td className="num">{neuron.applied_shift > 0 ? "+" : ""}{fixed(neuron.applied_shift, 3)}</td>
                        <td className="num">{fixed(neuron.unsteered_activation, 3)}</td>
                        <td className="num">{fixed(neuron.steered_activation, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </article>

          {panel && (
            <article className="mood-card">
              <div className="card-heading">
                <FlaskConical size={20} />
                <h2>Causal evidence for the {panel.emotion} neurons</h2>
                <Source kind="validated" />
              </div>
              <div className="stat-row">
                <div><span>Panel</span><strong>{panel.name}</strong></div>
                <div><span>Neurons</span><strong>{panel.neurons.toLocaleString()}</strong></div>
                <div><span>Layers</span><strong>{panel.layers[0]}–{panel.layers[panel.layers.length - 1]}</strong></div>
                <div><span>This turn</span><strong>{decision?.applied_alpha ? `α ${fixed(decision.applied_alpha)} applied` : "not steered"}</strong></div>
              </div>
              <p className="guidance">{panel.verdict}</p>
              <p className="muted small">
                Selection: {panel.selection}. These results come from interventions on separate development messages, not from this
                turn. “Emotion rating change” is the change in an independent judge model's 1–5 rating of how strongly the reply
                expresses {panel.emotion}, relative to the unsteered reply to the same message; bars and brackets are 95% bootstrap
                confidence intervals.
              </p>
              {panel.likelihood && (
                <p className="muted small">
                  Classifier-free check at α {fixed(panel.likelihood.alpha)}: shifting these neurons raises the log-likelihood of
                  emotional replies by {signed(panel.likelihood.panel)} nats per token ({interval(panel.likelihood.panel_lo, panel.likelihood.panel_hi)});
                  the same shift on random neurons gives {signed(panel.likelihood.random)}. Cost to neutral replies: {signed(panel.likelihood.neutral_cost)}.
                </p>
              )}
              <div className="mini-charts">
                <figure>
                  <figcaption>Activation intervention: effect by steering strength, panel vs random neurons</figcaption>
                  <div className="chart" role="img" aria-label="Change in emotion rating by steering strength for the panel and for random neurons">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={doseRows} margin={{ top: 8, right: 12, left: -14, bottom: 0 }} barGap={2}>
                        <CartesianGrid stroke="var(--border)" vertical={false} />
                        <XAxis dataKey="alpha" tickLine={false} axisLine={false} tick={tick} />
                        <YAxis tickLine={false} axisLine={false} tick={tick} />
                        <ReferenceLine y={0} stroke="var(--muted)" />
                        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }}
                          formatter={(value, name) => [signed(Number(value)), name]} />
                        <Legend iconType="square" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
                        <Bar name="Panel neurons" dataKey="panel" fill="var(--viz-with)" radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false}>
                          <ErrorBar dataKey="panelError" width={4} stroke="var(--text)" />
                        </Bar>
                        <Bar name="Random neurons (control)" dataKey="random" fill="var(--viz-without)" radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false}>
                          <ErrorBar dataKey="randomError" width={4} stroke="var(--text)" />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </figure>
                <figure>
                  <figcaption>Layer ablation: effect lost when one layer is left out of the panel</figcaption>
                  <div className="chart" role="img" aria-label="Emotion rating lost when each layer is removed from the panel">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={layerEvidence} margin={{ top: 8, right: 12, left: -14, bottom: 0 }}>
                        <CartesianGrid stroke="var(--border)" vertical={false} />
                        <XAxis dataKey="layer" tickLine={false} axisLine={false} tick={tick} tickFormatter={(value) => `L${value}`} />
                        <YAxis tickLine={false} axisLine={false} tick={tick} />
                        <ReferenceLine y={0} stroke="var(--muted)" />
                        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }} labelFormatter={(value) => `Without layer ${value}`}
                          formatter={(value) => [signed(Number(value)), "Emotion rating lost"]} />
                        <Bar name="Emotion rating lost" dataKey="loss_without" fill="var(--viz-with)" radius={[4, 4, 0, 0]} maxBarSize={18} isAnimationActive={false}>
                          <ErrorBar dataKey="error" width={4} stroke="var(--text)" />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </figure>
              </div>
              {panel.ablation.length > 0 && (
                <EffectTable rows={panel.ablation}
                  caption="Ablation in replies prompted to express the emotion (change vs the same reply without intervention)" />
              )}
              <details>
                <summary>Neuron subsets, single layers and controls</summary>
                <EffectTable rows={[...panel.dose_response, ...panel.controls, ...panel.subsets]}
                  caption="Activation interventions on the MindMate prompt (change vs the unsteered reply)" />
                <table className="data-table">
                  <caption className="muted small">Each layer: effect lost without it, and effect of steering it alone</caption>
                  <thead>
                    <tr><th>Layer</th><th className="num">Lost without it</th><th className="num">95% CI</th><th className="num">Alone (α 4)</th><th className="num">95% CI</th></tr>
                  </thead>
                  <tbody>
                    {panel.layer_effects.map((row) => (
                      <tr key={row.layer}>
                        <th scope="row">{row.layer}</th>
                        <td className="num">{signed(row.loss_without)}</td>
                        <td className="num">{interval(row.lo, row.hi)}</td>
                        <td className="num">{row.alone === null ? "–" : signed(row.alone)}</td>
                        <td className="num">{row.alone_lo === null || row.alone_hi === null ? "–" : interval(row.alone_lo, row.alone_hi)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
              {evidence?.sources && (
                <p className="muted small">
                  Sources: {Object.entries(evidence.sources).map(([name, path]) => `${name}: ${path}`).join("; ")}.
                </p>
              )}
            </article>
          )}

          {effect && (
            <article className="mood-card">
              <div className="card-heading">
                <Sparkles size={20} />
                <h2>How steering changed the reply's emotion</h2>
                <Source kind="estimated" />
              </div>
              {effect.steered_scores && target ? (
                <p className="headline-change">
                  {label(target)} score: <strong>{fixed(effect.unsteered_scores[target])}</strong> without steering
                  <ArrowRight size={16} aria-hidden="true" /> <strong>{fixed(effect.steered_scores[target])}</strong> with steering
                  <span className="sent-tag">{effect.sent === "steered" ? "steered reply sent" : "unsteered reply sent"}</span>
                </p>
              ) : (
                <p className="muted">
                  No steered reply was generated for this turn{target ? `; the reply's ${target} score is ${fixed(effect.unsteered_scores[target])}` : ""}.
                </p>
              )}
              <div className="chart tall" role="img" aria-label="Emotion scores of the reply without and with steering">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={scoreRows} layout="vertical" margin={{ top: 4, right: 16, left: 24, bottom: 0 }} barGap={2}>
                    <CartesianGrid stroke="var(--border)" horizontal={false} />
                    <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickLine={false} axisLine={false} tick={tick} />
                    <YAxis type="category" dataKey="emotion" tickLine={false} axisLine={false} tick={tick} width={86} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }}
                      formatter={(value, name) => [Number(value).toFixed(2), name]} />
                    <Legend iconType="square" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
                    <Bar name={effect.steered_scores ? "Without steering" : "Reply (not steered)"} dataKey="without" fill="var(--viz-without)" radius={[0, 4, 4, 0]} maxBarSize={9} isAnimationActive={false} />
                    {effect.steered_scores && <Bar name="With steering" dataKey="withSteering" fill="var(--viz-with)" radius={[0, 4, 4, 0]} maxBarSize={9} isAnimationActive={false} />}
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="muted small">★ intended response emotion. Scores come from {effect.method}; they describe the reply text, not the user.</p>
              {effect.steered_reply && (
                <details>
                  <summary>Compare the two replies</summary>
                  <div className="reply-compare">
                    <div><span className="eyebrow">WITHOUT STEERING{effect.sent === "unsteered" ? " (SENT)" : ""}</span><p>{effect.unsteered_reply}</p></div>
                    <div><span className="eyebrow">WITH STEERING{effect.sent === "steered" ? " (SENT)" : ""}</span><p>{effect.steered_reply}</p></div>
                  </div>
                </details>
              )}
            </article>
          )}
        </>
      )}
      <p className="mood-footnote">
        A space for reflection, not a clinical assessment. MindMate is an AI, not a therapist.
      </p>
    </section>
  );
}
