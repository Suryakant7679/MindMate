import { useState } from "react";
import { Activity, ArrowRight, Brain, Cpu, Info, Sparkles, TrendingUp } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EmotionTurn } from "../../types/emotion";
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

function Source({ kind }: { kind: "measured" | "estimated" | "reasoned" }) {
  const text = {
    measured: "MEASURED · MODEL HOOKS",
    estimated: "ESTIMATED · TEXT CLASSIFIER",
    reasoned: "REASONED · KNOWLEDGE GRAPH",
  }[kind];
  return <span className={`badge source ${kind}`}>{text}</span>;
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
  const { plan, steering, effect } = turn;

  const timeline = turns.map((item, index) => ({
    turn: index + 1,
    mood: item.plan ? Number((item.plan.valence * item.plan.intensity).toFixed(2)) : null,
    emotion: item.plan?.user_emotion ?? item.preview?.emotion ?? "",
    alpha: item.plan?.steering_alpha ?? item.preview?.alpha ?? 0,
    target: item.plan?.response_emotion ?? item.preview?.steeringEmotion ?? "none",
    without: item.effect?.target_before ?? null,
    withSteering: item.effect?.target_after ?? null,
  }));
  const layerRows = steering?.layers ?? [];
  const scoreRows = effect
    ? EMOTIONS.map((emotion) => ({
        emotion: label(emotion) + (emotion === effect.target ? " ★" : ""),
        without: effect.before?.[emotion] ?? null,
        withSteering: effect.after[emotion],
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
            This is a research view, not a clinical assessment.
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
              ? "Analysing this turn: generating the same reply without steering and reading the hook measurements…"
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
                  <small>{plan.response_emotion ? `steering α ${fixed(plan.steering_alpha)}` : "steering off"}</small>
                </div>
              </div>
              <ParameterStrength label="Intensity" value={plan.intensity} />
              <div className="emotion-stats">
                <div>Valence<strong>{fixed(plan.valence)}</strong></div>
                <div>Arousal<strong>{fixed(plan.arousal)}</strong></div>
                <div>Probe alone<strong>{label(Object.entries(plan.probe_scores).sort((a, b) => b[1] - a[1])[0][0])}</strong></div>
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
                  Reply's score for the intended emotion, without and with steering <Source kind="estimated" />
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
                <span className="eyebrow">HOW THE STEERING STRENGTH WAS SET</span>
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
                      <th scope="row">Applied</th>
                      <td className="num">α {fixed(plan.steering_alpha)}</td>
                      <td>{plan.response_emotion ? `toward ${plan.response_emotion}` : "no steering"}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </article>

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

          {effect && (
            <article className="mood-card">
              <div className="card-heading">
                <Sparkles size={20} />
                <h2>How steering changed the reply's emotion</h2>
                <Source kind="estimated" />
              </div>
              {effect.before && effect.target ? (
                <p className="headline-change">
                  {label(effect.target)} score: <strong>{fixed(effect.target_before)}</strong> without steering
                  <ArrowRight size={16} aria-hidden="true" /> <strong>{fixed(effect.target_after)}</strong> with steering
                </p>
              ) : (
                <p className="muted">This reply was not steered, so there is no unsteered reply to compare with.</p>
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
                    {effect.before && <Bar name="Without steering" dataKey="without" fill="var(--viz-without)" radius={[0, 4, 4, 0]} maxBarSize={9} isAnimationActive={false} />}
                    <Bar name={effect.before ? "With steering" : "Reply"} dataKey="withSteering" fill="var(--viz-with)" radius={[0, 4, 4, 0]} maxBarSize={9} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="muted small">★ intended response emotion. Scores come from {effect.method}; they describe the reply text, not the user.</p>
              <details>
                <summary>Compare the two replies</summary>
                <div className="reply-compare">
                  {effect.unsteered_reply && <div><span className="eyebrow">WITHOUT STEERING</span><p>{effect.unsteered_reply}</p></div>}
                  <div><span className="eyebrow">WITH STEERING (SENT)</span><p>{effect.steered_reply}</p></div>
                </div>
              </details>
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
