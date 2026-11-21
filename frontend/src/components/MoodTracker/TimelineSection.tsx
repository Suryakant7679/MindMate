import { Activity, HeartPulse, MessageSquareHeart, Repeat } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, LabelList, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Trajectory } from "../../types/emotion";
import { Metric, Source, Tile, fixed, label, percent, sequential, signed, tick, tooltipStyle } from "./shared";

const ratingValence = (rating: number) => (rating - 3) / 2;
const RATING_WORDS = ["very low", "low", "okay", "good", "very good"];
const HELP_WORDS = ["not at all", "a little", "somewhat", "supportive", "very supportive"];

function Rating({ name, words, value, onRate, disabled }: {
  name: string; words: string[]; value: number | null; onRate: (rating: number) => void; disabled: boolean;
}) {
  return (
    <span className="rating" role="radiogroup" aria-label={name}>
      {[1, 2, 3, 4, 5].map((rating) => (
        <button
          key={rating} type="button" role="radio" aria-checked={value === rating} disabled={disabled}
          className={value === rating ? "active" : ""} title={`${rating}: ${words[rating - 1]}`}
          onClick={() => onRate(rating)}
        >
          {rating}
        </button>
      ))}
    </span>
  );
}

/** A. Conversation emotion timeline: what the detector estimated for the user, turn by turn. */
export function TimelineSection({ trajectory, canRate, onFeedback }: {
  trajectory: Trajectory;
  canRate: boolean;
  onFeedback: (turnIndex: number, rating: { mood_rating?: number; helpfulness?: number }) => void;
}) {
  const { turns, responses, mood, self_report: selfReport, after_response: after } = trajectory;
  const rows = turns.map((turn, index) => ({
    turn: turn.turn,
    user: turn.expected_valence,
    reply: responses[index]?.expected_valence ?? null,
    self: turn.self_rating === null ? null : ratingValence(turn.self_rating),
    emotion: turn.emotion,
  }));
  const rated = rows.some((row) => row.self !== null);
  const emotions = Object.keys(trajectory.mean_distribution);
  const average = emotions.map((emotion) => ({ emotion: label(emotion), share: trajectory.mean_distribution[emotion] }));
  return (
    <>
      <article className="mood-card">
        <div className="card-heading">
          <HeartPulse size={20} />
          <h2>Estimated mood across the conversation</h2>
          <Source kind="observed" />
        </div>
        <div className="stat-row">
          <Tile name="Turns"><Metric path="trajectory.turns.length" value={turns.length} digits={0} /></Tile>
          <Tile name="Dominant emotion">{trajectory.dominant_emotion ? label(trajectory.dominant_emotion) : "–"}</Tile>
          <Tile name="Mean estimated valence" note="−1 to +1">
            <Metric path="trajectory.mood.mean_expected_valence" value={mood.mean_expected_valence} format={signed} />
          </Tile>
          <Tile name="Recent level" note="moving average">
            <Metric path="trajectory.mood.latest_ema" value={mood.latest_ema} format={signed} />
          </Tile>
          <Tile name="Change per turn" note={mood.trend_interpretable ? "least-squares slope" : "too few turns to read as a trend"}>
            {mood.trend_interpretable
              ? <Metric path="trajectory.mood.trend_per_turn" value={mood.trend_per_turn} format={signed} />
              : "–"}
          </Tile>
          <Tile name="Uncertain turns">
            <Metric path="trajectory.mood.uncertain_share" value={mood.uncertain_share} format={percent} />
          </Tile>
        </div>
        <figure>
          <figcaption>
            Valence per turn (−1 very negative, +1 very positive). The user line is the detector's estimate from the user's
            message; the reply line is a classifier's reading of the chatbot's reply text{rated ? "; dots are the user's own ratings" : ""}.
          </figcaption>
          <div className="chart" role="img" aria-label="Estimated valence of the user's messages and of the chatbot's replies per turn">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows} margin={{ top: 8, right: 16, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis dataKey="turn" tickLine={false} axisLine={false} tick={tick} tickFormatter={(value) => `T${value}`} />
                <YAxis domain={[-1, 1]} ticks={[-1, -0.5, 0, 0.5, 1]} tickLine={false} axisLine={false} tick={tick} />
                <ReferenceLine y={0} stroke="var(--axis)" />
                <Tooltip contentStyle={tooltipStyle} labelFormatter={(value) => `Turn ${value}`}
                  formatter={(value, name) => [signed(Number(value)), name]} />
                <Legend iconType="plainline" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
                <Line name="User (detector estimate)" dataKey="user" stroke="var(--series-1)" strokeWidth={2}
                  dot={{ r: 4, fill: "var(--series-1)", stroke: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} />
                <Line name="Chatbot reply (classifier on the reply)" dataKey="reply" stroke="var(--series-2)" strokeWidth={2}
                  dot={{ r: 4, fill: "var(--series-2)", stroke: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} connectNulls />
                {rated && (
                  <Line name="User's own rating" dataKey="self" stroke="none" legendType="circle"
                    dot={{ r: 5, fill: "var(--series-3)", stroke: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </figure>
        <ul className="caveats">
          {trajectory.caveats.map((caveat) => <li key={caveat}>{caveat}</li>)}
        </ul>
        <details>
          <summary>How these numbers are computed</summary>
          <table className="data-table">
            <tbody>
              {Object.entries(trajectory.method).map(([name, text]) => (
                <tr key={name}><th scope="row">{label(name)}</th><td>{text}</td></tr>
              ))}
            </tbody>
          </table>
        </details>
      </article>

      <div className="mood-grid even">
        <article className="mood-card">
          <div className="card-heading">
            <Activity size={20} />
            <h2>Emotion confidence per turn</h2>
            <Source kind="estimated" />
          </div>
          <p className="muted small">
            Fused probability of each emotion for every user message (darker is higher). The dominant emotion of a turn is
            printed; all values are in the table below.
          </p>
          <div className="heat-scroll">
            <table className="heat" aria-label="Probability of each emotion per turn">
              <thead>
                <tr><th />{turns.map((turn) => <th key={turn.turn} scope="col">T{turn.turn}</th>)}</tr>
              </thead>
              <tbody>
                {emotions.map((emotion) => (
                  <tr key={emotion}>
                    <th scope="row">{label(emotion)}</th>
                    {turns.map((turn) => {
                      const value = turn.distribution[emotion] ?? 0;
                      return (
                        <td key={turn.turn} tabIndex={0} style={{ background: sequential(value) }}
                          className={value > 0.55 ? "strong" : ""}
                          title={`Turn ${turn.turn} · ${label(emotion)}: ${fixed(value)}`}
                          aria-label={`Turn ${turn.turn}, ${emotion}, probability ${fixed(value)}`}>
                          {turn.emotion === emotion ? fixed(value) : ""}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="heat-legend" aria-hidden="true">
            <span>0</span><i style={{ background: `linear-gradient(to right, ${sequential(0)}, ${sequential(1)})` }} /><span>1</span>
          </div>
        </article>

        <article className="mood-card">
          <div className="card-heading">
            <Repeat size={20} />
            <h2>Average emotion distribution</h2>
            <Source kind="estimated" />
          </div>
          <p className="muted small">Mean probability of each emotion over the {turns.length} user turn{turns.length === 1 ? "" : "s"}.</p>
          <div className="chart tall" role="img" aria-label="Mean probability of each emotion over the conversation">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={average} layout="vertical" margin={{ top: 4, right: 40, left: 24, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" horizontal={false} />
                <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickLine={false} axisLine={false} tick={tick} />
                <YAxis type="category" dataKey="emotion" tickLine={false} axisLine={false} tick={tick} width={86} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }}
                  formatter={(value) => [Number(value).toFixed(2), "Mean probability"]} />
                <Bar dataKey="share" fill="var(--series-1)" radius={[0, 4, 4, 0]} maxBarSize={14} isAnimationActive={false}>
                  <LabelList dataKey="share" position="right" formatter={(value) => Number(value).toFixed(2)}
                    style={{ fill: "var(--muted)", fontSize: 11 }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="muted small">
            Dominant emotion, first half: {trajectory.dominant_first_half ? label(trajectory.dominant_first_half) : "–"} · second
            half: {trajectory.dominant_second_half ? label(trajectory.dominant_second_half) : "–"}.
            Turns per dominant emotion: {Object.entries(trajectory.frequency).map(([emotion, count]) => `${label(emotion)} ${count}`).join(", ") || "–"}.
          </p>
        </article>
      </div>

      <article className="mood-card">
        <div className="card-heading">
          <MessageSquareHeart size={20} />
          <h2>Turn by turn, with your own rating</h2>
          <Source kind="reported" />
        </div>
        <p className="muted small">
          Ratings are optional. “My mood” is how you felt when you wrote that message (1 very low, 5 very good); “Reply” is how
          supportive MindMate's answer felt (1 not at all, 5 very). They are used only to check the detector's estimates.
        </p>
        <table className="data-table">
          <thead>
            <tr>
              <th>Turn</th><th>Detected</th><th>Transition</th><th className="num">Confidence</th><th className="num">Intensity</th><th className="num">Est. valence</th>
              <th className="num">Change</th><th className="num">Shift (JSD)</th><th className="num">Uncertainty</th>
              <th>Detector alone</th><th>Language</th><th>My mood</th><th>Reply</th>
            </tr>
          </thead>
          <tbody>
            {turns.map((turn, index) => (
              <tr key={turn.turn}>
                <th scope="row" title={turn.text}>T{turn.turn}</th>
                <td>{label(turn.emotion)}{turn.uncertain && <span className="flag" title="Top probability is low or the two best labels are close"> uncertain</span>}</td>
                <td>{index === 0 ? "–" : turn.label_changed ? `${label(turns[index - 1].emotion)} → ${label(turn.emotion)}` : "same"}</td>
                <td className="num"><Metric path={`trajectory.turns.${index}.confidence`} value={turn.confidence} /></td>
                <td className="num">{fixed(turn.intensity)}</td>
                <td className="num"><Metric path={`trajectory.turns.${index}.expected_valence`} value={turn.expected_valence} format={signed} /></td>
                <td className="num"><Metric path={`trajectory.turns.${index}.delta_valence`} value={turn.delta_valence} format={signed} /></td>
                <td className="num">{fixed(turn.js_divergence)}</td>
                <td className="num"><Metric path={`trajectory.turns.${index}.entropy`} value={turn.entropy} /></td>
                <td>{label(turn.detector_emotion)}{!turn.detector_agrees && <span className="flag" title="The classifier alone picked another label than the fused estimate"> differs</span>}</td>
                <td>{label(turn.language)}</td>
                <td><Rating name={`My mood at turn ${turn.turn}`} words={RATING_WORDS} value={turn.self_rating} disabled={!canRate}
                  onRate={(rating) => onFeedback(index, { mood_rating: rating })} /></td>
                <td><Rating name={`Supportiveness of reply ${turn.turn}`} words={HELP_WORDS} value={responses[index]?.helpfulness ?? null}
                  disabled={!canRate} onRate={(rating) => onFeedback(index, { helpfulness: rating })} /></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">
          Uncertainty is the normalised entropy of the distribution (0 one label certain, 1 all equally likely). Shift is the
          Jensen–Shannon divergence from the previous turn's distribution (0 same, 1 completely different).
          Mean confidence {fixed(mood.mean_confidence)} · label changed in {percent(mood.label_switch_rate)} of turn pairs ·
          classifier alone disagreed with the fused estimate in {percent(mood.detector_disagreement_share)} of turns.
        </p>
        {selfReport.n > 0 && (
          <>
            <h3>Your ratings against the detector <Source kind="reported" /></h3>
            <div className="stat-row">
              <Tile name="Rated turns"><Metric path="trajectory.self_report.n" value={selfReport.n} digits={0} /></Tile>
              <Tile name="Mean absolute error" note="valence units"><Metric path="trajectory.self_report.mean_abs_error" value={selfReport.mean_abs_error} /></Tile>
              <Tile name="Same sign">{percent(selfReport.sign_agreement)}</Tile>
              <Tile name="Rank correlation" note={selfReport.spearman === null ? "needs four rated turns" : "Spearman"}>{fixed(selfReport.spearman)}</Tile>
            </div>
            <p className="muted small">{selfReport.note}</p>
          </>
        )}
      </article>

      {after.n > 0 && (
        <article className="mood-card">
          <div className="card-heading">
            <Activity size={20} />
            <h2>How the detected emotion moved after each reply</h2>
            <Source kind="observed" />
          </div>
          <p className="safety-note">{after.note}</p>
          <table className="data-table">
            <thead>
              <tr>
                <th>After reply</th><th>Reply sent</th><th>Reply emotion</th><th>User before</th><th>User after</th>
                <th className="num">Valence before</th><th className="num">Valence after</th><th className="num">Change</th><th>Next message reads as</th>
              </tr>
            </thead>
            <tbody>
              {after.rows.map((row, index) => (
                <tr key={row.turn}>
                  <th scope="row">T{row.turn} → T{row.turn + 1}</th>
                  <td>{row.delivered ? label(row.delivered) : "–"}</td>
                  <td>{row.reply_emotion ? label(row.reply_emotion) : "–"}</td>
                  <td>{label(row.user_emotion)}</td>
                  <td>{label(row.next_user_emotion)}</td>
                  <td className="num">{signed(row.valence_before)}</td>
                  <td className="num">{signed(row.valence_after)}</td>
                  <td className="num"><Metric path={`trajectory.after_response.rows.${index}.delta_valence`} value={row.delta_valence} format={signed} /></td>
                  <td>{row.delta_valence > 0.05 ? "More positive" : row.delta_valence < -0.05 ? "More negative" : "No clear change"}</td>
                </tr>
              ))}
              <tr className="total">
                <th scope="row">Mean</th><td colSpan={6}>
                  after steered replies: {signed(after.by_delivery.steered?.mean_delta_valence)} (n={after.by_delivery.steered?.n ?? 0}) ·
                  after unsteered replies: {signed(after.by_delivery.unsteered?.mean_delta_valence)} (n={after.by_delivery.unsteered?.n ?? 0})
                </td>
                <td className="num"><Metric path="trajectory.after_response.mean_delta_valence" value={after.mean_delta_valence} format={signed} /></td>
                <td>{percent(after.improved_share)} more positive · {percent(after.worsened_share)} more negative</td>
              </tr>
            </tbody>
          </table>
        </article>
      )}
    </>
  );
}
