import { ClipboardCheck, Cpu, LayoutDashboard, Users } from "lucide-react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ConversationSummary, ModelEvaluation, ResearchOverview, Trajectory } from "../../types/emotion";
import { Empty, Metric, Source, StatusBadge, Tile, fixed, label, percent, signed, tick, tooltipStyle } from "./shared";

/** One whole split into parts: widths proportional to counts, counts printed where they fit, names in the legend. */
export function SegmentedBar({ parts, name }: { parts: { key: string; label: string; value: number; color: string }[]; name: string }) {
  const total = parts.reduce((sum, part) => sum + part.value, 0);
  if (!total) return null;
  return (
    <div className="segmented" role="img"
      aria-label={`${name}: ${parts.map((part) => `${part.label} ${part.value} (${Math.round((part.value / total) * 100)}%)`).join(", ")}`}>
      <div className="segmented-bar">
        {parts.filter((part) => part.value > 0).map((part) => (
          <span key={part.key} tabIndex={0} style={{ flexGrow: part.value, background: part.color }}
            title={`${part.label}: ${part.value} of ${total} (${Math.round((part.value / total) * 100)}%)`}>
            {part.value / total >= 0.12 ? part.value : ""}
          </span>
        ))}
      </div>
      <ul className="segmented-legend">
        {parts.map((part) => <li key={part.key}><i style={{ background: part.color }} />{part.label}</li>)}
      </ul>
    </div>
  );
}

export type DashboardTab = "overview" | "timeline" | "comparison" | "neurons" | "causal" | "models";

/** Overview: the conversation at a glance, the loaded model's steering status, and which evidence exists. */
export function OverviewSection({ summary, trajectory, research, models, onNavigate }: {
  summary?: ConversationSummary; trajectory?: Trajectory; research: ResearchOverview | null;
  models: ModelEvaluation | null; onNavigate: (tab: DashboardTab) => void;
}) {
  const rows = (trajectory?.turns ?? []).map((turn, index) => ({
    turn: turn.turn, user: turn.expected_valence, reply: trajectory?.responses[index]?.expected_valence ?? null,
  }));
  const loaded = models?.models.find((model) => model.loaded);
  const validation = research?.validation;
  const architecture = research?.architecture;
  const selfReport = trajectory?.self_report;
  return (
    <>
      {summary && trajectory ? (
        <>
          <article className="mood-card">
            <div className="card-heading"><LayoutDashboard size={20} /><h2>This conversation at a glance</h2><Source kind="estimated" /></div>
            <div className="stat-row">
              <Tile name="Turns"><Metric path="summary.turns" value={summary.turns} digits={0} /></Tile>
              <Tile name="Steered replies sent" note={`${percent(summary.steered_delivered_share)} of tracked turns`}>
                <Metric path="summary.steered_delivered" value={summary.steered_delivered} digits={0} />
              </Tile>
              <Tile name="Turns with both replies" note={summary.comparison.shadow ? `${summary.comparison.shadow} comparison-only` : "steered and unsteered"}>
                <Metric path="summary.comparison.compared" value={summary.comparison.compared} digits={0} />
              </Tile>
              <Tile name="Improved" note={`${percent(summary.comparison.shares.improved)} of compared`}>
                <Metric path="summary.comparison.counts.improved" value={summary.comparison.counts.improved} digits={0} />
              </Tile>
              <Tile name="Harmed" note={`${percent(summary.comparison.shares.harmed)} of compared`}>
                <Metric path="summary.comparison.counts.harmed" value={summary.comparison.counts.harmed} digits={0} />
              </Tile>
              <Tile name="No meaningful change" note={`${percent(summary.comparison.shares.no_meaningful_change)} of compared`}>
                <Metric path="summary.comparison.counts.no_meaningful_change" value={summary.comparison.counts.no_meaningful_change} digits={0} />
              </Tile>
              <Tile name="Mean gain when compared" note="intended-emotion score, 0–1">
                <Metric path="summary.mean_target_gain_compared" value={summary.mean_target_gain_compared} format={signed} />
              </Tile>
            </div>
            <div className="mood-grid even flush">
              <figure>
                <figcaption>Steering decisions over {summary.tracked_turns} tracked turn{summary.tracked_turns === 1 ? "" : "s"}</figcaption>
                <SegmentedBar name="Steering decisions" parts={[
                  { key: "applied", label: "Applied", value: summary.decisions.applied, color: "var(--series-1)" },
                  { key: "rejected", label: "Rejected by safeguards", value: summary.decisions.rejected, color: "var(--series-2)" },
                  { key: "skipped", label: "Skipped", value: summary.decisions.skipped, color: "var(--series-3)" },
                  { key: "not_planned", label: "Not planned", value: summary.decisions.not_planned, color: "var(--deemph)" },
                ]} />
              </figure>
              <figure>
                <figcaption>
                  Verdicts on the {summary.comparison.compared} compared turn{summary.comparison.compared === 1 ? "" : "s"}{" "}
                  ({summary.comparison.judged} LLM-judged, {summary.comparison.compared - summary.comparison.judged} automatic estimates)
                </figcaption>
                {summary.comparison.compared ? (
                  <SegmentedBar name="Verdicts" parts={[
                    { key: "improved", label: "Improved", value: summary.comparison.counts.improved, color: "var(--status-good)" },
                    { key: "none", label: "No meaningful change", value: summary.comparison.counts.no_meaningful_change, color: "var(--deemph)" },
                    { key: "harmed", label: "Harmed", value: summary.comparison.counts.harmed, color: "var(--status-critical)" },
                  ]} />
                ) : <p className="muted small">No turn has both replies yet. Switch on the research comparison in Response Comparison.</p>}
              </figure>
            </div>
            <p className="muted small">
              <button type="button" className="link-button" onClick={() => onNavigate("comparison")}>Open the per-turn response comparison</button>
            </p>
          </article>

          <div className="mood-grid even">
            <article className="mood-card">
              <div className="card-heading"><Users size={20} /><h2>User emotion and chatbot tone</h2><Source kind="observed" /></div>
              <div className="stat-row">
                <Tile name="Dominant user emotion">{summary.user.dominant_emotion ? label(summary.user.dominant_emotion) : "–"}</Tile>
                <Tile name="User, mean est. valence"><Metric path="summary.user.mean_expected_valence" value={summary.user.mean_expected_valence} format={signed} /></Tile>
                <Tile name="Chatbot replies, mean valence"><Metric path="summary.response.mean_expected_valence" value={summary.response.mean_expected_valence} format={signed} /></Tile>
              </div>
              <div className="chart mini" role="img" aria-label="Estimated valence of user messages and chatbot replies per turn">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rows} margin={{ top: 8, right: 12, left: -22, bottom: 0 }}>
                    <CartesianGrid stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="turn" tickLine={false} axisLine={false} tick={tick} tickFormatter={(value) => `T${value}`} />
                    <YAxis domain={[-1, 1]} ticks={[-1, 0, 1]} tickLine={false} axisLine={false} tick={tick} />
                    <ReferenceLine y={0} stroke="var(--axis)" />
                    <Tooltip contentStyle={tooltipStyle} labelFormatter={(value) => `Turn ${value}`} formatter={(value, name) => [signed(Number(value)), name]} />
                    <Legend iconType="plainline" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
                    <Line name="User (detector)" dataKey="user" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4, fill: "var(--series-1)", stroke: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} />
                    <Line name="Chatbot reply (classifier)" dataKey="reply" stroke="var(--series-2)" strokeWidth={2} dot={{ r: 4, fill: "var(--series-2)", stroke: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="muted small">
                Both lines are classifier estimates. A change in the user line after a reply is an observation, never evidence that
                the reply or steering caused it.{" "}
                <button type="button" className="link-button" onClick={() => onNavigate("timeline")}>Open the emotion timeline</button>
              </p>
            </article>
            <article className="mood-card">
              <div className="card-heading"><Users size={20} /><h2>What the user reported</h2><Source kind="reported" /></div>
              {selfReport && selfReport.n > 0 ? (
                <div className="stat-row">
                  <Tile name="Rated turns"><Metric path="trajectory.self_report.n" value={selfReport.n} digits={0} /></Tile>
                  <Tile name="Mean gap to the detector" note="valence units">{fixed(selfReport.mean_abs_error)}</Tile>
                  <Tile name="Reply supportiveness" note={`${summary.helpfulness.n} rated, 1–5`}>{fixed(summary.helpfulness.mean)}</Tile>
                </div>
              ) : (
                <Empty title="No ratings given yet">
                  Self-ratings are optional. They are the only user-reported data here and are kept apart from the model's estimates.
                  Add them per turn in the Emotion Timeline.
                </Empty>
              )}
              <ul className="caveats">{summary.separation.map((line) => <li key={line}>{line}</li>)}</ul>
            </article>
          </div>
        </>
      ) : (
        <Empty title="No analysed conversation yet">Send a chat message; the conversation overview appears when the first reply has been analysed.</Empty>
      )}

      <div className="mood-grid even">
        <article className="mood-card">
          <div className="card-heading"><Cpu size={20} /><h2>Loaded model and steering status</h2><Source kind="validated" /></div>
          {research ? (
            <>
              <div className="stat-row">
                <Tile name="Model">{loaded?.label ?? research.model ?? "–"}</Tile>
                <Tile name="Layers">{architecture?.layers ?? "–"}</Tile>
                <Tile name="Steerable neurons">{architecture?.steerable_neurons?.toLocaleString() ?? "–"}</Tile>
                <Tile name="Policy">{research.policy ? (research.policy.steering_enabled ? "Steers some contexts" : "Steering disabled") : "None"}</Tile>
              </div>
              <p className="muted small">
                {architecture?.class}{architecture?.mixture_of_experts ? " · mixture of experts (always-active shared expert steered)" : ""}
                {architecture?.quantized ? " · 4-bit weights" : ""}. Site: {research.steering_location ?? "–"}.
              </p>
              {validation ? (
                <table className="data-table">
                  <caption className="muted small">Causal status of each panel (panel-level evidence; no individual neuron is validated)</caption>
                  <thead><tr><th>Panel</th><th>Panel-level effect</th><th>Vs random neurons</th><th>Usable without harm</th></tr></thead>
                  <tbody>
                    {Object.values(validation.panels).map((panel) => (
                      <tr key={panel.panel}>
                        <th scope="row">{label(panel.panel)}</th>
                        <td><StatusBadge status={panel.verdict.panel_level.status} /></td>
                        <td><StatusBadge status={panel.verdict.specificity.status} /></td>
                        <td>{panel.verdict.quality ? <StatusBadge status={panel.verdict.quality.status} /> : "–"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <Empty title="Validation battery not run for this model" command={`python3 -m generation_analysis.neuron_validation --model ${loaded?.key ?? "<model>"} all`}>
                  Without it, no panel of this model has a causal status here.
                </Empty>
              )}
              <p className="muted small">
                <button type="button" className="link-button" onClick={() => onNavigate("causal")}>Open the causal ablations</button>
              </p>
            </>
          ) : <Empty title="No model with an Emotion Reasoner is loaded" />}
        </article>
        <article className="mood-card">
          <div className="card-heading"><ClipboardCheck size={20} /><h2>Which evidence exists</h2></div>
          <table className="data-table">
            <thead><tr><th>Evidence</th><th>Status</th></tr></thead>
            <tbody>
              {(models?.models ?? []).map((model) => (
                <tr key={model.key}>
                  <th scope="row">Causal battery, {model.label}</th>
                  <td>{model.battery ? `Complete · ${model.battery.messages} messages · ${model.battery.generated.slice(0, 10)}` : <span className="na">{label(model.battery_status)}</span>}</td>
                </tr>
              ))}
              <tr>
                <th scope="row">Detector check on labelled messages</th>
                <td>{models?.detector ? `Complete · ${models.detector.messages.toLocaleString()} messages` : <span className="na">Not run</span>}</td>
              </tr>
              <tr>
                <th scope="row">LLM-judged turns of this conversation</th>
                <td>{summary ? (summary.comparison.judged ? `${summary.comparison.judged} of ${summary.comparison.compared}` : <span className="na">None yet (run turn_evaluation after chatting)</span>) : <span className="na">No conversation</span>}</td>
              </tr>
              <tr>
                <th scope="row">User self-ratings</th>
                <td>{selfReport?.n ? `${selfReport.n} turn${selfReport.n === 1 ? "" : "s"}` : <span className="na">None given</span>}</td>
              </tr>
              <tr>
                <th scope="row">Per-turn records</th>
                <td>{research ? (research.settings.telemetry.persisted ? `Saved to ${research.settings.telemetry.directory}` : "In memory only") : <span className="na">–</span>}</td>
              </tr>
            </tbody>
          </table>
          <p className="muted small">
            <button type="button" className="link-button" onClick={() => onNavigate("models")}>Open the model evaluation</button>
          </p>
        </article>
      </div>
    </>
  );
}
