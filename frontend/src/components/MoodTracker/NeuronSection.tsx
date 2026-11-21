import { useState } from "react";
import { Cpu, Grid3x3, ListOrdered } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EmotionTurn, ResearchOverview } from "../../types/emotion";
import { PanelBrowser, TurnNeuronTable } from "./NeuronExplorer";
import { Empty, Source, StatusBadge, Tile, diverging, fixed, label, percent, signed, tick, tooltipStyle } from "./shared";

/** C. Which neurons were shifted in this turn, by how much, and what that did to their activations. */
export function NeuronSection({ turn, research }: { turn: EmotionTurn; research: ResearchOverview | null }) {
  const { steering, decision, plan } = turn;
  const panelName = steering?.emotion ?? plan?.steering_panel ?? decision?.shadow_panel ?? plan?.response_emotion ?? null;
  const offline = panelName ? research?.evidence?.panels?.[panelName] : undefined;
  const validated = panelName ? research?.validation?.panels?.[panelName] : undefined;
  const [layerFilter, setLayerFilter] = useState<number | null>(null);
  const architecture = research?.architecture;
  const modelCard = research && (
    <article className="mood-card">
      <div className="card-heading"><Cpu size={20} /><h2>Model and steering site</h2><Source kind="measured" /></div>
      <div className="stat-row">
        <Tile name="Model">{research.model ?? "–"}</Tile>
        <Tile name="Architecture">{architecture?.class ?? "–"}</Tile>
        <Tile name="Layers">{architecture?.layers ?? "–"}</Tile>
        <Tile name="Hidden size">{architecture?.hidden_size?.toLocaleString() ?? "–"}</Tile>
        <Tile name="Steerable neurons">{architecture?.steerable_neurons?.toLocaleString() ?? "–"}</Tile>
        <Tile name="Experts">{architecture?.mixture_of_experts ? "Mixture of experts" : "Dense MLP"}</Tile>
      </div>
      <p className="muted small">
        Steering site: {research.steering_location ?? "–"}.
        {architecture?.mixture_of_experts && " Only the always-active neurons are steered (layer-0 MLP and each layer's shared expert); the routed experts run for a few tokens each and are not used."}
        {" "}Neuron IDs are positions at this model's own site; they do not correspond to neurons of any other model.
      </p>
    </article>
  );
  if (!steering?.applied) {
    return (
      <>
        {modelCard}
        <Empty title="No neurons were shifted on this turn">
          {decision ? `${label(decision.reason)}. ` : ""}Nothing was measured, so no activations are shown. Switch on the research
          comparison (Response Comparison) to measure a comparison-only candidate on turns the policy does not steer. The stored
          panels can be browsed below.
        </Empty>
        {research && <PanelBrowser research={research} initial={panelName} />}
      </>
    );
  }
  const layers = steering.layers.map((row) => ({ ...row, label: row.site ? `L${row.layer} (${row.site})` : `L${row.layer}` }));
  const heat = steering.heatmap;
  const hasContribution = steering.top_neurons.some((neuron) => neuron.write_norm !== null && neuron.write_norm !== undefined);
  const hasEvidence = steering.top_neurons.some((neuron) => neuron.evidence);
  return (
    <>
      {modelCard}
      <article className="mood-card">
        <div className="card-heading"><Cpu size={20} /><h2>Neuron intervention in this turn</h2><Source kind="measured" /></div>
        {steering.sent === false && (
          <p className="safety-note">
            These measurements are from a steered candidate that was not sent
            {steering.shadow ? " (generated only for comparison)" : ""}; the user received the unsteered reply.
          </p>
        )}
        <div className="stat-row">
          <Tile name="Panel">{label(steering.emotion ?? "")}</Tile>
          <Tile name="Direction">{steering.alpha > 0 ? "Toward" : "Away"}</Tile>
          <Tile name="Strength">α {fixed(steering.alpha)}</Tile>
          <Tile name="Layers">{layers[0].layer}–{layers[layers.length - 1].layer}</Tile>
          <Tile name="Neurons">{steering.neurons_total.toLocaleString()}</Tile>
          <Tile name="Decode steps affected">{steering.decode_steps ?? layers[0].decode_positions}</Tile>
          <Tile name="Causal validation">
            {validated ? <StatusBadge status={validated.verdict.panel_level?.status ?? "not_tested"} /> : offline ? "Earlier experiments" : "Not run"}
          </Tile>
        </div>
        <p className="muted small">
          Location: {steering.location}. Each neuron's activation is shifted by α × {steering.gain} × its stored
          emotion-minus-neutral difference, on decode steps only; model weights are never changed.
          {(validated?.selection ?? offline?.selection) && ` Selection: ${validated?.selection ?? offline?.selection}.`}
        </p>
        {validated?.verdict.summary && <p className="guidance">{validated.verdict.summary.text}</p>}
        {!validated && offline && <p className="guidance">{offline.verdict}</p>}
        <figure>
          <figcaption>Mean absolute activation change per layer: the shift the hook added, and the change against the unsteered reply</figcaption>
          <div className="chart" role="img" aria-label="Applied shift and measured activation change per steered layer">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={layers} margin={{ top: 8, right: 12, left: -14, bottom: 0 }} barGap={2}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis dataKey="label" tickLine={false} axisLine={false} tick={tick} />
                <YAxis tickLine={false} axisLine={false} tick={tick} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }} formatter={(value, name) => [Number(value).toFixed(4), name]} />
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
              <tr>
                <th>Layer</th><th className="num">Neurons</th><th className="num">Raised / lowered</th><th className="num">Mean |shift|</th>
                <th className="num">Mean |change|</th><th className="num">Already moved upstream</th><th className="num">Shift ÷ activation RMS</th>
                <th className="num">Residual update</th><th className="num">Share of push</th><th>Neuron IDs</th>
              </tr>
            </thead>
            <tbody>
              {layers.map((row) => (
                <tr key={row.label}>
                  <th scope="row">{row.site ? `${row.layer} (${row.site})` : row.layer}</th>
                  <td className="num">{row.neuron_count}</td>
                  <td className="num">{row.neurons_raised} / {row.neurons_lowered}</td>
                  <td className="num">{fixed(row.mean_abs_applied_shift, 4)}</td>
                  <td className="num">{fixed(row.mean_abs_measured_change, 4)}</td>
                  <td className="num">{fixed(row.mean_abs_upstream_change, 4)}</td>
                  <td className="num">{fixed(row.normalized_shift, 3)}</td>
                  <td className="num">{fixed(row.residual_update_norm, 4)}</td>
                  <td className="num">{percent(row.contribution_share)}</td>
                  <td><details><summary>show {row.neuron_count}</summary><code>{row.neuron_ids.join(", ")}</code></details></td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">
            “Already moved upstream” is how far steering at earlier layers and tokens had moved these neurons before their own
            shift was added. “Residual update” is the norm of what the layer's shift writes into the residual stream.
          </p>
        </details>
      </article>

      {heat && heat.rows.length > 0 && (
        <article className="mood-card">
          <div className="card-heading"><Grid3x3 size={20} /><h2>Layer-wise activation heat map</h2><Source kind="measured" /></div>
          <p className="muted small">
            {label(heat.metric)}. Each cell is one neuron, ordered by the size of its shift; blue is raised, red is lowered,
            grey is unchanged. Hover or focus a cell for the neuron ID and values.
          </p>
          <div className="heat-scroll">
            <table className="heat neurons" aria-label="Measured activation change of the most shifted neurons per layer">
              <tbody>
                {heat.rows.map((row) => (
                  <tr key={row.label} className={layerFilter === row.layer ? "picked" : ""}>
                    <th scope="row">
                      <button type="button" className="link-button" title="List this layer's neurons in the table below"
                        onClick={() => setLayerFilter(layerFilter === row.layer ? null : row.layer)}>{row.label}</button>
                    </th>
                    {row.cells.map((cell) => (
                      <td key={cell.neuron} tabIndex={0} style={{ background: diverging(cell.change, heat.max_abs) }}
                        title={`${row.label} · neuron ${cell.neuron}: change ${signed(cell.change, 3)}, shift ${signed(cell.shift, 3)}`}
                        aria-label={`${row.label}, neuron ${cell.neuron}, change ${signed(cell.change, 3)}`} />
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="heat-legend" aria-hidden="true">
            <span>−{fixed(heat.max_abs, 2)}</span>
            <i style={{ background: "linear-gradient(to right, var(--div-neg), var(--div-mid), var(--div-pos))" }} />
            <span>+{fixed(heat.max_abs, 2)}</span>
          </div>
        </article>
      )}

      <article className="mood-card">
        <div className="card-heading"><ListOrdered size={20} /><h2>Neurons pushing hardest</h2><Source kind="measured" /></div>
        <p className="muted small">
          Ranked by {steering.contribution_basis ?? "|applied shift|"}. This ranks how hard each neuron's shift pushes on the
          residual stream; it is not a per-neuron effect on the reply. What neuron subsets do to replies is tested in the
          Causal evidence tab.
        </p>
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th><th>Layer</th><th>Neuron ID</th><th>Direction</th><th className="num">Shift added</th>
              <th className="num">Shift ÷ RMS</th><th className="num">Activation, unsteered</th><th className="num">Activation, steered</th>
              <th className="num">Measured change</th>
              {hasContribution && <th className="num">Write norm</th>}
              <th className="num">Share of push</th>
              {hasEvidence && <><th className="num">Attribution rank</th><th className="num">Gradient estimate</th><th className="num">Selectivity t</th></>}
            </tr>
          </thead>
          <tbody>
            {steering.top_neurons.map((neuron) => (
              <tr key={`${neuron.layer}-${neuron.site ?? ""}-${neuron.neuron}`}>
                <th scope="row">{neuron.rank ?? "–"}</th>
                <td>{neuron.site ? `${neuron.layer} (${neuron.site})` : neuron.layer}</td>
                <td>{neuron.neuron}</td>
                <td>{neuron.direction ? label(neuron.direction) : neuron.applied_shift > 0 ? "Raised" : "Lowered"}</td>
                <td className="num">{signed(neuron.applied_shift, 3)}</td>
                <td className="num">{signed(neuron.normalized_shift, 2)}</td>
                <td className="num">{fixed(neuron.unsteered_activation, 3)}</td>
                <td className="num">{fixed(neuron.steered_activation, 3)}</td>
                <td className="num">{signed(neuron.measured_change, 3)}</td>
                {hasContribution && <td className="num">{fixed(neuron.write_norm, 3)}</td>}
                <td className="num">{percent(neuron.contribution_share)}</td>
                {hasEvidence && (
                  <>
                    <td className="num">{neuron.evidence?.attribution_rank ?? "–"}</td>
                    <td className="num">{signed(neuron.evidence?.gradient_effect, 4)}</td>
                    <td className="num">{fixed(neuron.evidence?.selectivity_t, 1)}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
        {turn.turnId && (
          <p className="muted small">
            Every steered neuron of this turn (ID, layer, activation before and after, shift, contribution):{" "}
            <a href={`/api/emotion/neurons/${turn.turnId}`} target="_blank" rel="noreferrer">open the per-neuron record</a>.
          </p>
        )}
      </article>
      {turn.turnId && (
        <TurnNeuronTable turnId={turn.turnId} total={steering.neurons_total} layerFilter={layerFilter} onLayer={setLayerFilter} />
      )}
      {research && <PanelBrowser research={research} initial={panelName} />}
    </>
  );
}
