import { useEffect, useMemo, useState } from "react";
import { Crosshair, Microscope, Table2 } from "lucide-react";
import { fetchNeuronRecord, fetchPanel } from "../../api/chat";
import type { NeuronRecord, PanelView, ResearchOverview, SingleNeuronReport, ValidationPanel } from "../../types/emotion";
import { Empty, Failure, Loading, Source, StatusBadge, Tile, fixed, interval, label, pValue, percent, signed } from "./shared";

const PAGE = 50;

function Pager({ page, pages, total, onPage }: { page: number; pages: number; total: number; onPage: (page: number) => void }) {
  return (
    <div className="pager">
      <button type="button" className="ghost-button" disabled={page === 0} onClick={() => onPage(page - 1)}>Previous</button>
      <span>{total ? `${page * PAGE + 1}–${Math.min((page + 1) * PAGE, total)} of ${total.toLocaleString()}` : "0 neurons"}</span>
      <button type="button" className="ghost-button" disabled={page >= pages - 1} onClick={() => onPage(page + 1)}>Next</button>
    </div>
  );
}

/** What kind of evidence exists for single neurons, subsets and the panel. */
export function EvidenceLadder({ validated, single }: { validated?: ValidationPanel; single?: SingleNeuronReport["panels"][string] }) {
  return (
    <table className="data-table ladder">
      <caption className="muted small">Three kinds of evidence, from weakest to strongest. Only the last one supports a causal claim.</caption>
      <thead><tr><th>Evidence</th><th>What it shows</th><th>Unit it exists for</th><th>Status here</th></tr></thead>
      <tbody>
        <tr>
          <th scope="row">Activation selectivity</th>
          <td className="wrap">The neuron is more active when the reply has the emotion. A correlation; says nothing about effect.</td>
          <td>Each neuron</td><td><Source kind="measured" /></td>
        </tr>
        <tr>
          <th scope="row">Attribution / gradient estimate</th>
          <td className="wrap">A first-order prediction of how much shifting the neuron would push the reply toward the emotion. An estimate, not a test.</td>
          <td>Each neuron</td><td><Source kind="estimated" /></td>
        </tr>
        <tr>
          <th scope="row">Intervention with controls</th>
          <td className="wrap">The shift is applied and compared with matched random neurons, sham hooks and the reversed shift.</td>
          <td>Whole panel, layer bands, ranked subsets</td>
          <td>{validated ? <StatusBadge status={validated.verdict.panel_level.status} /> : <Source kind="unavailable" />}</td>
        </tr>
        <tr>
          <th scope="row">Single-neuron intervention</th>
          <td className="wrap">
            One neuron shifted alone, against matched control neurons; needed to say that one neuron causes the effect.
            {single ? ` ${single.verdict.text}` : ""}
          </td>
          <td>{single ? `${single.tested} top-ranked neurons of this panel; all others not tested` : "Each neuron"}</td>
          <td><StatusBadge status={single ? single.verdict.status : "not_tested"} /></td>
        </tr>
      </tbody>
    </table>
  );
}

/** Single-neuron interventions of one panel: likelihood screen for the top-ranked neurons, judged replies for the best. */
export function SingleNeuronCard({ report, panelName, modelKey }: { report?: SingleNeuronReport | null; panelName: string; modelKey: string }) {
  const [showControls, setShowControls] = useState(false);
  const block = report?.panels[panelName];
  if (!report || !block) {
    return (
      <article className="mood-card">
        <div className="card-heading"><Crosshair size={20} /><h2>Single-neuron interventions</h2><Source kind="unavailable" /></div>
        <Empty title={report ? `No single-neuron test for the ${label(panelName)} panel` : "Single-neuron interventions have not been run for this model"}
          command={`python3 -m generation_analysis.single_neuron --model ${modelKey} all`}>
          Until then no individual neuron has a causal status: only the panel, its layer bands and ranked subsets were tested.
        </Empty>
      </article>
    );
  }
  const top = String(Math.max(...report.doses));
  const judges = Object.keys(report.judges);
  const rows = block.neurons.filter((entry) => showControls || entry.role === "panel")
    .sort((a, b) => (a.role === b.role ? (b.screen.mean ?? -9) - (a.screen.mean ?? -9) : a.role === "panel" ? -1 : 1));
  return (
    <article className="mood-card">
      <div className="card-heading"><Crosshair size={20} /><h2>Single-neuron interventions: {label(panelName)} panel</h2><Source kind="validated" /></div>
      <div className="stat-row">
        <Tile name="Neurons tested alone" note={`top-ranked by ${block.ranked_by}`}>{block.tested}</Tile>
        <Tile name="Pass the likelihood screen">{block.screen_supported}</Tile>
        <Tile name="Tested in generated replies">{block.generated}</Tile>
        <Tile name="Individually supported">{block.supported}</Tile>
        <Tile name="Full panel, same screen" note="nats per token">{signed(block.full_panel.mean, 3)}</Tile>
        <Tile name="Control-neuron threshold" note={`${report.null_percentile}th percentile`}>{signed(block.null_threshold, 4)}</Tile>
        <Tile name="Finding"><StatusBadge status={block.verdict.status} /></Tile>
      </div>
      <p className="guidance">{block.verdict.text}</p>
      <p className="muted small">{report.method} {block.messages} messages; strongest single-neuron dose {top}× the reference strength α {fixed(report.reference)}.</p>
      <label className="switch compact">
        <input type="checkbox" checked={showControls} onChange={(event) => setShowControls(event.target.checked)} />
        <span>Show the matched control neurons (outside every panel)</span>
      </label>
      <table className="data-table">
        <thead>
          <tr>
            <th>Neuron</th><th>Role</th><th className="num">Screen effect at {top}×</th><th className="num">95% CI</th><th className="num">p (Holm)</th>
            <th className="num">At 1×</th><th className="num">At 8×</th><th>Above control neurons</th><th className="num">Panel effect lost without it</th>
            <th className="num">Judged emotion change</th><th className="num">95% CI</th><th className="num">Minus control neurons</th>
            {judges.map((judge) => <th key={judge} className="num">{judge}</th>)}
            <th className="num">Fluency</th><th className="num">Harmed</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((entry) => (
            <tr key={`${entry.role}-${entry.layer}-${entry.neuron}`} title={entry.reading}>
              <th scope="row">L{entry.layer} · {entry.neuron}</th>
              <td>{entry.role === "panel" ? "Panel" : "Control"}</td>
              <td className="num">{signed(entry.screen.mean, 4)}</td><td className="num">{interval(entry.screen.lo, entry.screen.hi, 4)}</td>
              <td className="num">{entry.role === "panel" ? pValue(entry.screen.p_holm) : "–"}</td>
              <td className="num">{signed(entry.doses["1"]?.mean, 4)}</td><td className="num">{signed(entry.doses["8"]?.mean, 4)}</td>
              <td>{entry.role === "panel" ? (entry.above_null ? "Yes" : "No") : "–"}</td>
              <td className="num">{entry.leave_out ? `${signed(entry.leave_out.mean, 4)} (${interval(entry.leave_out.lo, entry.leave_out.hi, 4)})` : "–"}</td>
              {entry.judged ? (
                <>
                  <td className="num">{signed(entry.judged.mean)}</td><td className="num">{interval(entry.judged.lo, entry.judged.hi)}</td>
                  <td className="num">{entry.judged.vs_control ? `${signed(entry.judged.vs_control.mean)} (${interval(entry.judged.vs_control.lo, entry.judged.vs_control.hi)})` : "–"}</td>
                  {judges.map((judge) => <td key={judge} className="num">{signed(entry.judged?.judges[judge])}</td>)}
                  <td className="num">{signed(entry.judged.fluency)}</td><td className="num">{percent(entry.judged.harm_share)}</td>
                </>
              ) : <td colSpan={5 + judges.length} className="na">{entry.role === "panel" ? "Not tested by generation" : "Control"}</td>}
              <td>{entry.role === "panel" ? <StatusBadge status={entry.status as "supported" | "not_supported" | "inconclusive"} /> : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Screen effect: change of the emotional reply's log-likelihood minus change of the plain reply's, in nats per token
        (classifier-free, <strong>measured</strong>). Judged columns are <strong>LLM-judged</strong> ratings of generated replies.
        A neuron counts as individually supported only when both agree and it beats the control neurons. Hover a row for its reading.
      </p>
      <ul className="caveats">{report.notes.map((note) => <li key={note}>{note}</li>)}</ul>
    </article>
  );
}

/** Every steered neuron of one turn, loaded from the saved record when asked for. */
export function TurnNeuronTable({ turnId, total, layerFilter, onLayer }: {
  turnId: string; total: number; layerFilter: number | null; onLayer: (layer: number | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [record, setRecord] = useState<NeuronRecord | null>(null);
  const [error, setError] = useState("");
  const [direction, setDirection] = useState("all");
  const [sort, setSort] = useState("contribution");
  const [page, setPage] = useState(0);
  useEffect(() => { setRecord(null); setOpen(false); setError(""); }, [turnId]);
  useEffect(() => {
    if (!open || record) return;
    const controller = new AbortController();
    fetchNeuronRecord(turnId, controller.signal).then(setRecord).catch((cause) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Unknown error");
    });
    return () => controller.abort();
  }, [open, record, turnId]);
  useEffect(() => { setPage(0); }, [layerFilter, direction, sort]);
  const rows = useMemo(() => {
    const filtered = (record?.neurons ?? []).filter((neuron) =>
      (layerFilter === null || neuron.layer === layerFilter) && (direction === "all" || neuron.direction === direction));
    const key = (neuron: NeuronRecord["neurons"][number]) =>
      sort === "shift" ? Math.abs(neuron.applied_shift) : sort === "change" ? Math.abs(neuron.measured_change ?? 0)
        : sort === "layer" ? -neuron.layer : neuron.contribution ?? 0;
    return [...filtered].sort((a, b) => key(b) - key(a));
  }, [record, layerFilter, direction, sort]);
  const layers = useMemo(() => [...new Set((record?.neurons ?? []).map((neuron) => neuron.layer))].sort((a, b) => a - b), [record]);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const hasEvidence = rows.some((neuron) => neuron.evidence);
  return (
    <article className="mood-card">
      <div className="card-heading"><Table2 size={20} /><h2>All {total.toLocaleString()} steered neurons of this turn</h2><Source kind="measured" /></div>
      {!open && (
        <button type="button" className="ghost-button" onClick={() => setOpen(true)}>
          Load the full neuron table ({total.toLocaleString()} rows, from the saved record)
        </button>
      )}
      {open && error && <Failure what="The neuron record" error={error} onRetry={() => { setError(""); setRecord(null); }} />}
      {open && !record && !error && <Loading what="the neuron record" />}
      {record && (
        <>
          <div className="filter-row" role="group" aria-label="Neuron filters">
            <label>Layer
              <select value={layerFilter ?? "all"} onChange={(event) => onLayer(event.target.value === "all" ? null : Number(event.target.value))}>
                <option value="all">All layers</option>
                {layers.map((layer) => <option key={layer} value={layer}>Layer {layer}</option>)}
              </select>
            </label>
            <label>Direction
              <select value={direction} onChange={(event) => setDirection(event.target.value)}>
                <option value="all">Raised and lowered</option><option value="raised">Raised</option><option value="lowered">Lowered</option>
              </select>
            </label>
            <label>Sort by
              <select value={sort} onChange={(event) => setSort(event.target.value)}>
                <option value="contribution">Push on the residual stream</option><option value="shift">Size of the shift</option>
                <option value="change">Size of the measured change</option><option value="layer">Layer</option>
              </select>
            </label>
            <span className="muted small">{rows.length.toLocaleString()} neurons match</span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Rank</th><th>Layer</th><th>Neuron ID</th><th>Direction</th><th className="num">Original activation</th>
                <th className="num">Modified activation</th><th className="num">Signed shift</th><th className="num">Shift ÷ RMS</th>
                <th className="num">Measured change</th><th className="num">Moved upstream</th><th className="num">Share of push</th>
                {hasEvidence && <><th className="num">Attribution rank</th><th className="num">Gradient estimate</th><th className="num">Selectivity t</th></>}
              </tr>
            </thead>
            <tbody>
              {rows.slice(page * PAGE, (page + 1) * PAGE).map((neuron) => (
                <tr key={`${neuron.layer}-${neuron.site ?? ""}-${neuron.neuron}`}>
                  <th scope="row">{neuron.rank}</th><td>{neuron.site ? `${neuron.layer} (${neuron.site})` : neuron.layer}</td><td>{neuron.neuron}</td>
                  <td>{neuron.direction ? label(neuron.direction) : "–"}</td>
                  <td className="num">{fixed(neuron.unsteered_activation, 3)}</td><td className="num">{fixed(neuron.steered_activation, 3)}</td>
                  <td className="num">{signed(neuron.applied_shift, 3)}</td><td className="num">{signed(neuron.normalized_shift, 2)}</td>
                  <td className="num">{signed(neuron.measured_change, 3)}</td><td className="num">{signed(neuron.upstream_change, 3)}</td>
                  <td className="num">{percent(neuron.contribution_share)}</td>
                  {hasEvidence && (
                    <>
                      <td className="num">{neuron.evidence?.attribution_rank ?? "–"}</td>
                      <td className="num">{neuron.evidence?.gradient_effect === undefined || neuron.evidence?.gradient_effect === null ? <span className="na">n/a</span> : signed(neuron.evidence.gradient_effect, 5)}</td>
                      <td className="num">{neuron.evidence?.selectivity_t === undefined || neuron.evidence?.selectivity_t === null ? <span className="na">n/a</span> : fixed(neuron.evidence.selectivity_t, 1)}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          <Pager page={page} pages={pages} total={rows.length} onPage={setPage} />
          <p className="muted small">
            Original activation: the neuron in the unsteered reply. Modified: the value the model used in the steered reply.
            “Moved upstream” is the part of the change that steering at earlier layers and tokens had already caused.
            {record.contribution_basis && ` Rank and share of push: ${record.contribution_basis}.`}
          </p>
        </>
      )}
    </article>
  );
}

/** The stored panels of the loaded model: every neuron with its shift and offline evidence (no measurement on a turn). */
export function PanelBrowser({ research, initial }: { research: ResearchOverview; initial?: string | null }) {
  const names = Object.keys(research.panels).filter((name) => research.panels[name] > 0);
  const [name, setName] = useState(initial && names.includes(initial) ? initial : names[0]);
  const [panel, setPanel] = useState<PanelView | null>(null);
  const [error, setError] = useState("");
  const [layer, setLayer] = useState<number | null>(null);
  const [page, setPage] = useState(0);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!name) return;
    const controller = new AbortController();
    setPanel(null); setError(""); setLayer(null); setPage(0);
    fetchPanel(name, controller.signal).then(setPanel).catch((cause) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Unknown error");
    });
    return () => controller.abort();
  }, [name, attempt]);
  const rows = useMemo(() => {
    const filtered = (panel?.neurons ?? []).filter((neuron) => layer === null || neuron.layer === layer);
    return [...filtered].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  }, [panel, layer]);
  const validated = name ? research.validation?.panels[name] : undefined;
  const single = name ? research.single_neuron?.panels[name] : undefined;
  const modelKey = research.validation?.model_key ?? research.single_neuron?.model_key ?? "<llama | llama2 | emollama | sarvam>";
  if (!names.length) return <Empty title="This model has no steering panels" />;
  const peak = Math.max(1, ...Object.values(panel?.per_layer ?? {}));
  return (
    <>
    <article className="mood-card">
      <div className="card-heading"><Microscope size={20} /><h2>Panel explorer: stored panels of this model</h2><Source kind="measured" /></div>
      <div className="filter-row" role="group" aria-label="Panel and layer">
        <label>Panel (emotion)
          <select value={name} onChange={(event) => setName(event.target.value)}>
            {names.map((item) => <option key={item} value={item}>{label(item)} · {research.panels[item].toLocaleString()} neurons</option>)}
          </select>
        </label>
        <label>Layer
          <select value={layer ?? "all"} onChange={(event) => { setLayer(event.target.value === "all" ? null : Number(event.target.value)); setPage(0); }}>
            <option value="all">All layers</option>
            {(panel?.layers ?? []).map((item) => <option key={item} value={item}>Layer {item} · {panel?.per_layer[String(item)]} neurons</option>)}
          </select>
        </label>
      </div>
      {error && <Failure what="The panel" error={error} onRetry={() => setAttempt((value) => value + 1)} />}
      {!panel && !error && <Loading what="the panel" />}
      {panel && (
        <>
          <div className="stat-row">
            <Tile name="Panel">{label(panel.panel)}</Tile>
            <Tile name="Neurons">{panel.neurons_total.toLocaleString()}</Tile>
            <Tile name="Layers">{panel.layers[0]}–{panel.layers[panel.layers.length - 1]}</Tile>
            <Tile name="Panel-level causal status">{validated ? <StatusBadge status={validated.verdict.panel_level.status} /> : "Not run"}</Tile>
            <Tile name="Single neurons" note={single ? `${single.tested} tested` : undefined}><StatusBadge status={single ? single.verdict.status : "not_tested"} /></Tile>
          </div>
          <p className="muted small">
            Site: {panel.location ?? "–"}.{panel.selection ? ` Selection: ${panel.selection}.` : ""} The stored shift of a neuron is its mean
            activation in emotional replies minus neutral replies; steering adds strength × gain × that shift.
          </p>
          <figure>
            <figcaption>Neurons per layer (select a bar's layer in the filter to list its neurons)</figcaption>
            <div className="layer-strip" role="group" aria-label="Neurons per layer">
              {panel.layers.map((item) => (
                <button key={item} type="button" className={layer === item ? "active" : ""} aria-pressed={layer === item}
                  title={`Layer ${item}: ${panel.per_layer[String(item)]} neurons`} onClick={() => { setLayer(layer === item ? null : item); setPage(0); }}>
                  <i style={{ height: `${Math.max(6, (panel.per_layer[String(item)] / peak) * 44)}px` }} />
                  <span>{item}</span>
                </button>
              ))}
            </div>
          </figure>
          <table className="data-table">
            <thead>
              <tr>
                <th>Layer</th><th>Neuron ID</th><th>Direction when steering</th><th className="num">Stored shift</th>
                <th className="num">Rank in layer</th><th className="num">Attribution score</th><th className="num">Gradient estimate</th>
                <th className="num">Gradient rank in layer</th><th className="num">Selectivity t</th><th>Causal status</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(page * PAGE, (page + 1) * PAGE).map((neuron) => (
                <tr key={`${neuron.layer}-${neuron.neuron}`}>
                  <th scope="row">{neuron.layer}</th><td>{neuron.neuron}</td><td>{neuron.delta > 0 ? "Raised" : "Lowered"}</td>
                  <td className="num">{signed(neuron.delta, 4)}</td>
                  <td className="num">{neuron.attribution_rank ?? <span className="na">n/a</span>}</td>
                  <td className="num">{neuron.attribution === undefined ? <span className="na">n/a</span> : signed(neuron.attribution, 4)}</td>
                  <td className="num">{neuron.gradient_effect === undefined || neuron.gradient_effect === null ? <span className="na">n/a</span> : signed(neuron.gradient_effect, 5)}</td>
                  <td className="num">{neuron.gradient_rank ?? <span className="na">n/a</span>}</td>
                  <td className="num">{neuron.selectivity_t === undefined || neuron.selectivity_t === null ? <span className="na">n/a</span> : fixed(neuron.selectivity_t, 1)}</td>
                  <td>{neuron.single ? <StatusBadge status={neuron.single.status} /> : <span className="na">Not tested alone</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Pager page={page} pages={Math.max(1, Math.ceil(rows.length / PAGE))} total={rows.length} onPage={setPage} />
          <p className="muted small">
            Rank in layer orders the panel's neurons by the score they were selected with. The gradient estimate is a
            first-order prediction (gradient of the emotional reply's log-likelihood × the stored shift), and “gradient rank”
            places the neuron among all neurons of its layer. Selectivity t is the paired t value of the activation difference.
            “n/a” means that statistic was not saved for this model or panel. None of these is a causal test of the neuron.
          </p>
          {!panel.has_cards && (
            <p className="muted small">
              Attribution ranks appear after the validation battery has written this model's evidence cards
              (<code>python3 -m generation_analysis.neuron_validation --model &lt;model&gt; analyse</code>).
            </p>
          )}
          <EvidenceLadder validated={validated} single={single} />
        </>
      )}
    </article>
    {name && <SingleNeuronCard report={research.single_neuron} panelName={name} modelKey={modelKey} />}
    </>
  );
}
