import { Activity, ArrowRight, Brain, Gauge } from "lucide-react";
import type { EmotionTurn } from "../../types/emotion";
import { ParameterStrength } from "./ParameterStrength";
import { Source, fixed, interval, label, signed } from "./shared";

/** What the Emotion Reasoner understood and planned for one turn, with the calibration behind the strength. */
export function TurnDetails({ turn, turnNumber }: { turn: EmotionTurn; turnNumber: number }) {
  const { plan, decision } = turn;
  if (!plan) return null;
  const recalibrated = plan.calibration?.sweep.some((row) => row.emotion_rating_gain !== undefined) ?? false;
  return (
    <>
      <article className="mood-card">
        <div className="card-heading"><Activity size={20} /><h2>Turn {turnNumber}: what was understood</h2><Source kind="estimated" /></div>
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
        <div className="card-heading"><Brain size={20} /><h2>Emotion Reasoner: strategy</h2><Source kind="reasoned" /></div>
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
                  <td>
                    {plan.response_emotion && plan.steering_alpha
                      ? `toward ${plan.response_emotion}${plan.steering_panel && plan.steering_panel !== plan.response_emotion ? ` · neuron panel: ${plan.steering_panel.replace("+", " + ")}` : ""}`
                      : "no steering"}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </article>

      {plan.calibration && (
        <article className="mood-card">
          <div className="card-heading"><Gauge size={20} /><h2>Calibration behind the strength</h2><Source kind="calibrated" /></div>
          <p className="muted small">
            Strength sweep for this context ({plan.calibration.context.replace(/^strategy:/, "").replace("|", ", ").replace(/_/g, " ")}).
            Changes are relative to the unsteered reply with the same strategy prompt. Empathy, relevance and fluency
            are 1–5 ratings by judge models; a strength is safe when none drops by more than{" "}
            {fixed(plan.calibration.rule?.tolerance ?? 0.05)} on average and at most{" "}
            {Math.round((plan.calibration.rule?.max_harm_share ?? 0.2) * 100)}% of replies lose more than 0.5.
            {recalibrated && " The strength is selected on the judges' emotion rating (1–5), which the run-time gate never sees; the gate classifier's score is shown for reference only."}
          </p>
          {plan.calibration.sweep.length > 0 ? (
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
          ) : <p className="muted small">No sweep is recorded for this context.</p>}
          {plan.calibration.selected_alpha === 0 && <p className="muted small">{label(plan.calibration.reason)}.</p>}
        </article>
      )}
    </>
  );
}
