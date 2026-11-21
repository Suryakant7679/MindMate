import { Gauge, History, MessageSquare, Scale } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EmotionTurn, ResearchOverview, Trajectory } from "../../types/emotion";
import { TurnDetails } from "./TurnDetails";
import { EMOTIONS, Empty, Metric, Source, Tile, VerdictBadge, fixed, label, signed, tick, tooltipStyle } from "./shared";

const OUTCOME = {
  applied: "Steered reply accepted and sent",
  rejected: "Steered reply rejected by the safeguards",
  skipped: "Steering skipped",
  not_planned: "Steering disabled for this turn",
} as const;
const DISABLED_BY: Record<string, string> = {
  policy: "by the calibrated policy",
  safety: "because this is a safety turn",
  no_panel: "because no neuron panel exists for the response emotion",
  strength_rule: "by the strength rule",
  agent: "because the agent answered from tool results",
};
const JUDGED_ROWS: [string, string][] = [
  ["emotion", "Intended emotion (1–5)"], ["empathy", "Empathy (1–5)"], ["relevance", "Relevance (1–5)"],
  ["fluency", "Fluency (1–5)"], ["appropriateness", "Appropriateness (1–5)"], ["naturalness", "Naturalness (1–5)"],
  ["hinglish_naturalness", "Hinglish naturalness (1–5)"], ["safety", "Safety (1–5)"],
];

/** B. Steered against unsteered reply for one turn, and how steering did across the conversation. */
export function ImpactSection({ turn, turnNumber, trajectory, research, onToggleCompare, toggling }: {
  turn: EmotionTurn;
  turnNumber: number;
  trajectory?: Trajectory;
  research: ResearchOverview | null;
  onToggleCompare: (enabled: boolean) => void;
  toggling: boolean;
}) {
  const { plan, decision, effect, comparison } = turn;
  const target = effect?.target ?? null;
  const judged = comparison?.judged ?? null;
  const judgeNames = judged ? Object.keys(judged.difference).filter((name) => name !== "both") : [];
  const compare = research?.settings.research_compare ?? false;
  const scoreRows = effect
    ? EMOTIONS.map((emotion) => ({
        emotion: label(emotion) + (emotion === effect.target ? " ★" : ""),
        without: effect.unsteered_scores[emotion],
        withSteering: effect.steered_scores?.[emotion] ?? null,
      }))
    : [];
  const history = (trajectory?.responses ?? []).map((row) => ({
    turn: row.turn, without: row.unsteered_target_score, withSteering: row.compared ? row.steered_target_score : null,
  }));
  return (
    <>
      <article className="mood-card">
        <div className="card-heading"><MessageSquare size={20} /><h2>Turn {turnNumber}: the user's message</h2><Source kind="estimated" /></div>
        <blockquote className="user-message">{turn.userText}</blockquote>
        {plan && (
          <div className="stat-row">
            <Tile name="Detected emotion">{label(plan.user_emotion)}</Tile>
            <Tile name="Confidence" note="detector score, not a probability">{fixed(plan.confidence)}</Tile>
            <Tile name="Intensity">{fixed(plan.intensity)}</Tile>
            <Tile name="Valence">{signed(plan.valence)}</Tile>
            <Tile name="KG strategy">{plan.strategy_title}</Tile>
            <Tile name="Target response emotion">{plan.response_emotion ? label(plan.response_emotion) : "None"}</Tile>
          </div>
        )}
      </article>
      <article className="mood-card">
        <div className="card-heading">
          <Gauge size={20} />
          <h2>Steering decision for this turn</h2>
          <Source kind="calibrated" />
        </div>
        {decision && plan ? (
          <>
            <div className={`decision ${decision.status}`}>
              <strong>
                {OUTCOME[decision.status]}
                {decision.status === "not_planned" && decision.disabled_by ? ` ${DISABLED_BY[decision.disabled_by] ?? ""}` : ""}
              </strong>
              <span>planned α {fixed(decision.planned_alpha)} · applied α {fixed(decision.applied_alpha)}</span>
              <p>{label(decision.reason)}.</p>
            </div>
            <div className="stat-row">
              <Tile name="Strategy">{plan.strategy_title}</Tile>
              <Tile name="Target emotion">{plan.response_emotion ? label(plan.response_emotion) : "None"}</Tile>
              <Tile name="Neuron panel">{label(plan.steering_panel ?? decision.shadow_panel ?? plan.response_emotion ?? "none")}</Tile>
              <Tile name="Planned strength">α {fixed(decision.planned_alpha)}</Tile>
              <Tile name="Applied strength">α {fixed(decision.applied_alpha)}</Tile>
              <Tile name="Reply sent">{effect ? label(effect.sent) : "–"}</Tile>
            </div>
            {effect?.shadow && (
              <p className="safety-note">
                Research comparison: the policy does not steer this context, so a steered candidate was generated at the
                diagnostic strength α {fixed(decision.shadow_alpha)} only to compare it. It was not sent and could not have been.
              </p>
            )}
          </>
        ) : (
          <p className="muted">The decision for this turn is not available.</p>
        )}
        <label className="switch">
          <input type="checkbox" checked={compare} disabled={toggling || !research}
            onChange={(event) => onToggleCompare(event.target.checked)} />
          <span>
            Research comparison on every eligible turn
            <small>
              Also generates a steered candidate on turns the policy does not steer (diagnostic strength α{" "}
              {fixed(research?.settings.shadow_alpha)}), in the same decoding batch as the reply. The candidate is never sent
              and is never generated on a safety turn. Replies on those turns arrive all at once instead of streaming.
            </small>
          </span>
        </label>
      </article>

      <article className="mood-card">
        <div className="card-heading">
          <Scale size={20} />
          <h2>Steered and unsteered reply, side by side</h2>
          <Source kind={judged ? "judged" : "estimated"} />
        </div>
        {!comparison?.available || !effect?.steered_reply ? (
          <>
            {effect && (
              <div className="reply-compare">
                <div className="sent"><span className="eyebrow">UNSTEERED REPLY · DELIVERED</span><p>{effect.unsteered_reply}</p></div>
                <div className="missing">
                  <span className="eyebrow">STEERED REPLY</span>
                  <Empty title="No steered reply was generated">
                    {compare ? "This turn was not eligible (for example a safety turn or a tool answer)."
                      : "Switch on the research comparison above to get a comparison-only candidate on turns like this one."}
                  </Empty>
                </div>
              </div>
            )}
          </>
        ) : (
          <>
            <p className="headline-change">
              <VerdictBadge verdict={comparison.verdict.label} basis={comparison.verdict.basis} />
              <span className="muted">{comparison.verdict.reasons.join("; ")}.</span>
            </p>
            {comparison.automatic_verdict && (
              <p className="muted small">
                Automatic estimate before judging: <VerdictBadge verdict={comparison.automatic_verdict.label} /> ({comparison.automatic_verdict.reasons.join("; ")}).
              </p>
            )}
            <div className="reply-compare">
              <div className={effect.sent === "unsteered" ? "sent" : ""}>
                <span className="eyebrow">UNSTEERED REPLY{effect.sent === "unsteered" ? " · DELIVERED" : " · NOT SENT"}</span>
                <p>{effect.unsteered_reply}</p>
              </div>
              <div className={effect.sent === "steered" ? "sent" : ""}>
                <span className="eyebrow">STEERED REPLY{effect.sent === "steered" ? " · DELIVERED" : effect.shadow ? " · COMPARISON ONLY, NEVER SENT" : " · NOT SENT"}</span>
                <p>{effect.steered_reply}</p>
              </div>
            </div>
            <table className="data-table">
              <caption className="muted small">
                Automatic measures from the two texts (run time). They estimate wording and emotion expression, not empathy or relevance.
              </caption>
              <thead>
                <tr><th>Measure</th><th className="num">Unsteered</th><th className="num">Steered</th><th className="num">Change</th><th>Source</th></tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">{target ? `${label(target)} score` : "Target score"}</th>
                  <td className="num">{fixed(comparison.unsteered.target_score)}</td>
                  <td className="num">{fixed(comparison.steered?.target_score)}</td>
                  <td className="num">{signed(comparison.automatic?.target_gain)}</td>
                  <td>reply-emotion classifier</td>
                </tr>
                <tr>
                  <th scope="row">Strongest emotion</th>
                  <td className="num">{comparison.unsteered.top_emotion ? label(comparison.unsteered.top_emotion) : "–"}</td>
                  <td className="num">{comparison.steered?.top_emotion ? label(comparison.steered.top_emotion) : "–"}</td>
                  <td className="num">–</td><td>reply-emotion classifier</td>
                </tr>
                <tr>
                  <th scope="row">Repetition (repeated word pairs)</th>
                  <td className="num">{fixed(comparison.unsteered.repetition)}</td>
                  <td className="num">{fixed(comparison.steered?.repetition)}</td>
                  <td className="num">{signed(comparison.automatic?.repetition_change)}</td><td>text</td>
                </tr>
                <tr>
                  <th scope="row">Length (words)</th>
                  <td className="num">{comparison.unsteered.words}</td>
                  <td className="num">{comparison.steered?.words ?? "–"}</td>
                  <td className="num">×{fixed(comparison.automatic?.length_ratio)}</td><td>text</td>
                </tr>
                <tr>
                  <th scope="row">Reply language</th>
                  <td className="num">{label(comparison.unsteered.language)}</td>
                  <td className="num">{comparison.steered ? label(comparison.steered.language) : "–"}</td>
                  <td className="num">
                    {comparison.automatic?.hinglish_kept === null || comparison.automatic?.hinglish_kept === undefined
                      ? "–" : comparison.automatic.hinglish_kept ? "Hinglish kept" : "Hinglish lost"}
                  </td>
                  <td>script and word list (user wrote {comparison.automatic ? label(comparison.automatic.user_language) : "–"})</td>
                </tr>
                <tr>
                  <th scope="row">Wording kept (ROUGE-L)</th>
                  <td className="num" colSpan={2}>{fixed(comparison.automatic?.lexical_overlap)}</td>
                  <td className="num">–</td><td>shared word sequence; meaning is judged offline</td>
                </tr>
              </tbody>
            </table>
            {judged ? (
              <table className="data-table">
                <caption className="muted small">
                  Offline ratings of both replies by {judgeNames.map((name) => judged.judges[name] ?? name).join(" and ")} (1–5; change is steered − unsteered).
                </caption>
                <thead>
                  <tr>
                    <th>Rating</th><th className="num">Unsteered</th><th className="num">Steered</th><th className="num">Change (mean)</th>
                    {judgeNames.map((name) => <th key={name} className="num">Change, {name}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {JUDGED_ROWS.filter(([metric]) => judged.difference.both?.[metric] !== null && judged.difference.both?.[metric] !== undefined).map(([metric, title]) => (
                    <tr key={metric}>
                      <th scope="row">{title}</th>
                      <td className="num">{fixed(judged.ratings.unsteered.both?.[metric])}</td>
                      <td className="num">{fixed(judged.ratings.steered.both?.[metric])}</td>
                      <td className="num">{signed(judged.difference.both?.[metric])}</td>
                      {judgeNames.map((name) => <td key={name} className="num">{signed(judged.difference[name]?.[metric])}</td>)}
                    </tr>
                  ))}
                  <tr>
                    <th scope="row">Meaning kept (1–5)</th><td className="num">–</td>
                    <td className="num">{fixed(judged.preservation?.both)}</td><td className="num">–</td>
                    {judgeNames.map((name) => <td key={name} className="num">{fixed(judged.preservation?.[name])}</td>)}
                  </tr>
                  <tr>
                    <th scope="row">Steered reply preferred (0–1)</th><td className="num">–</td>
                    <td className="num">{fixed(judged.preference.both)}</td><td className="num">–</td>
                    {judgeNames.map((name) => <td key={name} className="num">{fixed(judged.preference[name])}</td>)}
                  </tr>
                  {judged.independent && (
                    <tr>
                      <th scope="row">{label(judged.independent.target)} score, independent classifier</th>
                      <td className="num">{fixed(judged.independent.unsteered)}</td>
                      <td className="num">{fixed(judged.independent.steered)}</td>
                      <td className="num">{signed(judged.independent.difference)}</td>
                      <td colSpan={judgeNames.length}>{judged.independent.model}</td>
                    </tr>
                  )}
                </tbody>
              </table>
            ) : (
              <p className="muted small">
                Empathy, relevance, fluency, Hinglish naturalness and meaning are rated offline by Qwen2.5-7B-Instruct and Gemma
                3-27B-IT. Not judged yet for this turn: run <code>python3 -m generation_analysis.turn_evaluation --model &lt;llama | llama2 | emollama | sarvam&gt;</code>,
                then press “Refresh data”.
              </p>
            )}
            <figure>
              <figcaption>Emotion scores of the two replies (★ intended emotion) <Source kind="estimated" /></figcaption>
              <div className="chart tall" role="img" aria-label="Emotion scores of the reply without and with steering">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={scoreRows} layout="vertical" margin={{ top: 4, right: 16, left: 24, bottom: 0 }} barGap={2}>
                    <CartesianGrid stroke="var(--border)" horizontal={false} />
                    <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickLine={false} axisLine={false} tick={tick} />
                    <YAxis type="category" dataKey="emotion" tickLine={false} axisLine={false} tick={tick} width={86} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }} formatter={(value, name) => [Number(value).toFixed(2), name]} />
                    <Legend iconType="square" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
                    <Bar name="Without steering" dataKey="without" fill="var(--viz-without)" radius={[0, 4, 4, 0]} maxBarSize={9} isAnimationActive={false} />
                    <Bar name="With steering" dataKey="withSteering" fill="var(--viz-with)" radius={[0, 4, 4, 0]} maxBarSize={9} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </figure>
            <p className="muted small">Scores come from {effect.method}; they describe the reply text, not the user.</p>
          </>
        )}
      </article>

      <details className="section-details">
        <summary>Why this strategy and strength: the Emotion Reasoner's plan and its calibration</summary>
        <TurnDetails turn={turn} turnNumber={turnNumber} />
      </details>

      {trajectory && trajectory.responses.length > 0 && (
        <article className="mood-card">
          <div className="card-heading">
            <History size={20} />
            <h2>Steering across this conversation</h2>
            <Source kind="estimated" />
          </div>
          <figure>
            <figcaption>Reply's score for the intended emotion, without and with steering, on turns where both replies exist</figcaption>
            <div className="chart mini" role="img" aria-label="Intended-emotion score of the reply without and with steering, per turn">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}>
                  <CartesianGrid stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="turn" tickLine={false} axisLine={false} tick={tick} tickFormatter={(value) => `T${value}`} />
                  <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickLine={false} axisLine={false} tick={tick} />
                  <Tooltip contentStyle={tooltipStyle} labelFormatter={(value) => `Turn ${value}`} formatter={(value, name) => [Number(value).toFixed(2), name]} />
                  <Legend iconType="plainline" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
                  <Line name="Without steering" dataKey="without" stroke="var(--viz-without)" strokeWidth={2}
                    dot={{ r: 4, fill: "var(--viz-without)", stroke: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} connectNulls />
                  <Line name="With steering" dataKey="withSteering" stroke="var(--viz-with)" strokeWidth={2}
                    dot={{ r: 4, fill: "var(--viz-with)", stroke: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </figure>
          <table className="data-table">
            <thead>
              <tr>
                <th>Turn</th><th>Strategy</th><th>Outcome</th><th className="num">Planned α</th><th className="num">Applied α</th>
                <th className="num">Unsteered</th><th className="num">Steered</th><th className="num">Gain</th><th>Verdict</th><th>Why</th>
              </tr>
            </thead>
            <tbody>
              {trajectory.responses.map((row, index) => (
                <tr key={row.turn}>
                  <th scope="row">T{row.turn}</th>
                  <td>{row.strategy_title ?? "–"}</td>
                  <td>{row.status ? (row.status === "not_planned" ? "Disabled" : label(row.status)) + (row.shadow ? " (compared)" : "") : "Not tracked"}</td>
                  <td className="num">{fixed(row.planned_alpha)}</td>
                  <td className="num"><Metric path={`trajectory.responses.${index}.applied_alpha`} value={row.applied_alpha} /></td>
                  <td className="num">{fixed(row.unsteered_target_score)}</td>
                  <td className="num">{row.compared ? fixed(row.steered_target_score) : "–"}</td>
                  <td className="num"><Metric path={`trajectory.responses.${index}.target_gain`} value={row.compared ? row.target_gain : null} format={signed} /></td>
                  <td>{row.compared ? <VerdictBadge verdict={row.verdict} basis={row.verdict_basis} /> : "–"}</td>
                  <td className="wrap">{row.reason ? label(row.reason) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </article>
      )}
    </>
  );
}
