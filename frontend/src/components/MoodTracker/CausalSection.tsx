import { useState } from "react";
import { FlaskConical, Layers, ShieldQuestion, Sigma } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { EmotionTurn, ResearchOverview, ValidationCondition, ValidationPanel } from "../../types/emotion";
import {
  EffectTable, Source, StatusBadge, Tile, errorBar, fixed, interval, label, pValue, percent, signed, statError, tick, tooltipStyle,
} from "./shared";

const VERDICT_ROWS: [string, string][] = [
  ["panel_level", "Panel raises the emotion (sufficiency)"],
  ["specificity", "More than a generic perturbation (vs matched random neurons, sham, another emotion's panel)"],
  ["reversibility", "Reversing the shift reverses the effect"],
  ["dose_response", "Effect grows with strength"],
  ["necessity", "Emotion falls when the neurons are held neutral (necessity)"],
  ["layer_level", "Which layers carry the effect"],
  ["subset_level", "Do the top-ranked neurons do better than random panel neurons"],
  ["neuron_level", "Individual neurons"],
  ["quality", "Can it be used without harming replies"],
];
const INSPECT_GROUPS: [string, string[]][] = [
  ["Strength (full panel)", ["dose"]], ["Controls", ["sham", "matched_random_mean", "other_panel", "reverse"]],
  ["Layer ablation", ["band_only", "band_out"]], ["Neuron-subset ablation", ["subset_top", "subset_random", "subset_bottom"]],
  ["Ablation under an emotion prompt", ["clamp", "clamp_zero", "clamp_random", "styled_reverse", "styled_reverse_random"]],
];

/** Everything recorded for one intervention: emotion per judge, independent classifier, every quality rating. */
function Inspector({ conditions, judges, names }: { conditions: ValidationCondition[]; judges: string[]; names: Record<string, string> }) {
  const [picked, setPicked] = useState<string | null>(null);
  const row = conditions.find((item) => item.name === picked) ?? conditions.find((item) => item.family === "dose") ?? conditions[0];
  if (!row) return null;
  const stats: [string, typeof row.emotion, string][] = [
    ["Emotion (mean of judges, 1–5)", row.emotion, "judged"],
    ...judges.map((judge): [string, typeof row.emotion, string] => [`Emotion, ${names[judge] ?? judge}`, row.emotion_judges[judge], "judged"]),
    ["Intended-emotion score, independent classifier (0–1)", row.independent, "estimated"],
    ...QUALITY.filter((metric) => row.quality[metric]).map((metric): [string, typeof row.emotion, string] => [`${label(metric)} (1–5)`, row.quality[metric], "judged"]),
  ];
  return (
    <article className="mood-card">
      <div className="card-heading"><FlaskConical size={20} /><h2>Inspect one intervention</h2><Source kind="validated" /></div>
      <div className="filter-row" role="group" aria-label="Intervention">
        <label>Intervention (layer band, neuron subset or control)
          <select value={row.name} onChange={(event) => setPicked(event.target.value)}>
            {INSPECT_GROUPS.map(([group, families]) => {
              const members = conditions.filter((item) => families.includes(item.family));
              return members.length ? (
                <optgroup key={group} label={group}>
                  {members.map((item) => <option key={item.name} value={item.name}>{item.label}</option>)}
                </optgroup>
              ) : null;
            })}
          </select>
        </label>
      </div>
      <div className="stat-row">
        <Tile name="Messages">{row.n}</Tile>
        <Tile name="Neurons shifted">{row.neurons?.toLocaleString() ?? "–"}</Tile>
        <Tile name="Strength">{row.alpha === null ? "clamp" : `α ${fixed(row.alpha)}`}</Tile>
        <Tile name="Replies changed">{percent(row.changed_share)}</Tile>
        <Tile name="Replies harmed">{percent(row.harm_share)}</Tile>
        <Tile name="Steered reply preferred">{fixed(row.preference)}</Tile>
        <Tile name="Length ratio">×{fixed(row.length_ratio)}</Tile>
      </div>
      <table className="data-table">
        <thead><tr><th>Measure</th><th className="num">Change</th><th className="num">95% CI</th><th className="num">p</th><th className="num">Effect size d</th><th>Evidence</th></tr></thead>
        <tbody>
          {stats.map(([name, stat, kind]) => (
            <tr key={name}>
              <th scope="row">{name}</th><td className="num">{signed(stat?.mean, 3)}</td><td className="num">{interval(stat?.lo, stat?.hi, 3)}</td>
              <td className="num">{pValue(stat?.p_holm ?? stat?.p)}</td><td className="num">{signed(stat?.dz)}</td>
              <td>{kind === "judged" ? "LLM-judged" : "Classifier-estimated"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Change is against the reply without this intervention to the same message, decoded in the same batch. A reply counts as
        harmed when any quality rating falls by more than 0.5 points.
      </p>
    </article>
  );
}
const QUALITY = ["appropriateness", "empathy", "relevance", "fluency", "naturalness", "hinglish_naturalness"];

function bars(conditions: ValidationCondition[]) {
  return conditions.filter((row) => row.emotion.mean !== null).map((row) => ({
    name: row.label, value: row.emotion.mean as number, error: statError(row.emotion), row,
  }));
}

function ConditionTable({ rows, caption, judges }: { rows: ValidationCondition[]; caption: string; judges: string[] }) {
  const quality = QUALITY.filter((metric) => rows.some((row) => row.quality[metric]?.mean !== null && row.quality[metric]?.mean !== undefined));
  return (
    <table className="data-table">
      <caption className="muted small">{caption}</caption>
      <thead>
        <tr>
          <th>Intervention</th><th className="num">n</th><th className="num">Neurons</th><th className="num">α</th>
          <th className="num">Emotion change</th><th className="num">95% CI</th><th className="num">p (Holm)</th><th className="num">Effect size d</th>
          {judges.map((judge) => <th key={judge} className="num">{judge}</th>)}
          <th className="num">Indep. classifier</th>
          {quality.map((metric) => <th key={metric} className="num">{label(metric)}</th>)}
          <th className="num">Replies harmed</th><th className="num">Preferred</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.name}>
            <th scope="row">{row.label}</th>
            <td className="num">{row.n}</td>
            <td className="num">{row.neurons?.toLocaleString() ?? "–"}</td>
            <td className="num">{row.alpha === null ? "–" : fixed(row.alpha)}</td>
            <td className="num">{signed(row.emotion.mean)}</td>
            <td className="num">{interval(row.emotion.lo, row.emotion.hi)}</td>
            <td className="num">{pValue(row.emotion.p_holm ?? row.emotion.p)}</td>
            <td className="num">{signed(row.emotion.dz)}</td>
            {judges.map((judge) => <td key={judge} className="num">{signed(row.emotion_judges[judge]?.mean)}</td>)}
            <td className="num">{signed(row.independent.mean, 3)}</td>
            {quality.map((metric) => <td key={metric} className="num">{signed(row.quality[metric]?.mean)}</td>)}
            <td className="num">{percent(row.harm_share)}</td>
            <td className="num">{fixed(row.preference)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function EffectBars({ data, emphasise, height }: { data: ReturnType<typeof bars>; emphasise: (row: ValidationCondition) => boolean; height?: string }) {
  return (
    <div className={`chart ${height ?? ""}`} role="img" aria-label="Change in emotion rating per intervention with 95% confidence intervals">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: -14, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="name" tickLine={false} axisLine={false} tick={tick} interval={0} />
          <YAxis tickLine={false} axisLine={false} tick={tick} />
          <ReferenceLine y={0} stroke="var(--axis)" />
          <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }}
            formatter={(value, _name, item) => [`${signed(Number(value))} (${interval(item.payload.row.emotion.lo, item.payload.row.emotion.hi)})`, "Emotion rating change"]} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false}>
            {data.map((entry) => <Cell key={entry.name} fill={emphasise(entry.row) ? "var(--series-1)" : "var(--deemph)"} />)}
            <ErrorBar dataKey="error" width={4} stroke="var(--text)" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function grouped(conditions: ValidationCondition[], first: string, second: string) {
  const groups = [...new Set(conditions.filter((row) => row.family === first || row.family === second).map((row) => row.group ?? ""))];
  return groups.map((group) => {
    const a = conditions.find((row) => row.family === first && (row.group ?? "") === group);
    const b = conditions.find((row) => row.family === second && (row.group ?? "") === group);
    return { group, first: a?.emotion.mean ?? null, firstError: statError(a?.emotion), second: b?.emotion.mean ?? null, secondError: statError(b?.emotion) };
  });
}

function GroupedBars({ data, firstName, secondName, aria }: { data: ReturnType<typeof grouped>; firstName: string; secondName: string; aria: string }) {
  return (
    <div className="chart" role="img" aria-label={aria}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: -14, bottom: 0 }} barGap={2}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="group" tickLine={false} axisLine={false} tick={tick} />
          <YAxis tickLine={false} axisLine={false} tick={tick} />
          <ReferenceLine y={0} stroke="var(--axis)" />
          <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }} formatter={(value, name) => [signed(Number(value)), name]} />
          <Legend iconType="square" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
          <Bar name={firstName} dataKey="first" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false}>
            <ErrorBar dataKey="firstError" width={4} stroke="var(--text)" />
          </Bar>
          <Bar name={secondName} dataKey="second" fill="var(--series-2)" radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false}>
            <ErrorBar dataKey="secondError" width={4} stroke="var(--text)" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** D. Causal ablation dashboard: is the panel causally relevant, or would random neurons do the same? */
export function CausalSection({ research, turn }: { research: ResearchOverview | null; turn: EmotionTurn }) {
  const report = research?.validation ?? null;
  const turnPanel = turn.steering?.emotion ?? turn.plan?.steering_panel ?? turn.decision?.shadow_panel ?? turn.plan?.response_emotion ?? "";
  const [picked, setPicked] = useState<string | null>(null);
  const [language, setLanguage] = useState("all");
  if (!report) return <EarlierEvidence research={research} turn={turn} />;
  const names = Object.keys(report.panels);
  const selected = picked ?? (names.includes(turnPanel) ? turnPanel : "pooled");
  const panel: ValidationPanel = selected === "pooled" ? report.pooled : report.panels[selected] ?? report.pooled;
  const slice = language === "all" ? panel : panel.by_language[language];
  const conditions = slice?.conditions ?? [];
  const contrasts = slice?.contrasts ?? [];
  const judges = Object.keys(report.judges);
  const reference = report.reference_alpha;
  const family = (...families: string[]) => conditions.filter((row) => families.includes(row.family));
  const referencePanel = conditions.find((row) => row.family === "dose" && row.alpha === reference);
  const controls = [referencePanel, ...family("sham", "matched_random_mean", "other_panel", "reverse")].filter(Boolean) as ValidationCondition[];
  const dose = family("dose").sort((a, b) => (a.alpha ?? 0) - (b.alpha ?? 0));
  const bands = grouped(conditions, "band_only", "band_out");
  const subsets = grouped(conditions, "subset_top", "subset_random");
  const ablation = family("clamp", "clamp_zero", "clamp_random", "styled_reverse", "styled_reverse_random");
  const inspectable = conditions.filter((item) => INSPECT_GROUPS.some(([, families]) => families.includes(item.family)));
  return (
    <>
      <div className="filter-row" role="group" aria-label="Panel and language">
        <label>Panel
          <select value={selected} onChange={(event) => setPicked(event.target.value)}>
            <option value="pooled">All panels pooled</option>
            {names.map((name) => <option key={name} value={name}>{label(name)}</option>)}
          </select>
        </label>
        <label>Language
          <select value={language} onChange={(event) => setLanguage(event.target.value)}>
            <option value="all">All messages</option>
            {Object.keys(panel.by_language).map((name) => <option key={name} value={name}>{label(name)}</option>)}
          </select>
        </label>
        <span className="muted small">
          {report.model} · {slice?.messages ?? panel.messages} development messages · judges: {judges.map((judge) => report.judges[judge]).join(", ")} ·
          generated {report.generated.slice(0, 10)}
        </span>
      </div>

      <article className="mood-card">
        <div className="card-heading"><ShieldQuestion size={20} /><h2>Are these neurons causally relevant?</h2><Source kind="validated" /></div>
        <div className="stat-row">
          <Tile name="Panel">{selected === "pooled" ? "All pooled" : label(selected)}</Tile>
          <Tile name="Neurons">{panel.neurons?.toLocaleString() ?? "–"}</Tile>
          <Tile name="Layers">{panel.layers.length ? `${panel.layers[0]}–${panel.layers[panel.layers.length - 1]}` : "–"}</Tile>
          <Tile name="Messages">{slice?.messages ?? panel.messages}</Tile>
          <Tile name="Reference strength">α {fixed(reference)}</Tile>
          <Tile name="Sampled-decoding seeds">{report.seeds.length}</Tile>
        </div>
        {panel.verdict.summary && <p className="guidance">{panel.verdict.summary.text}</p>}
        {language !== "all" && <p className="muted small">The findings below are for all messages of this panel; the charts and tables show {label(language)} messages only.</p>}
        <table className="data-table">
          <thead><tr><th>Question</th><th>Finding</th><th>Evidence</th></tr></thead>
          <tbody>
            {VERDICT_ROWS.filter(([key]) => panel.verdict[key]).map(([key, question]) => (
              <tr key={key}>
                <th scope="row" className="wrap">{question}</th>
                <td><StatusBadge status={panel.verdict[key].status} /></td>
                <td className="wrap">{panel.verdict[key].text}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">
          Site: {report.site}. {panel.selection ? `Selection: ${panel.selection}. ` : ""}
          “Emotion change” is the change of the judges' mean 1–5 rating of the intended emotion against the unsteered reply to
          the same message, decoded in the same batch; intervals are 95% bootstrap CIs over messages, p-values are Wilcoxon
          signed-rank with Holm correction within the panel.
        </p>
      </article>

      <div className="mood-grid even">
        <article className="mood-card">
          <div className="card-heading"><FlaskConical size={20} /><h2>Panel against its controls</h2></div>
          <p className="muted small">
            Same strength (α {fixed(reference)}). Matched random neurons keep the layer distribution, neuron count, shift values
            and neutral activation level of the panel (mean of the random draws). The sham installs the hooks with a zero shift.
          </p>
          <EffectBars data={bars(controls)} emphasise={(row) => row.family === "dose"} />
        </article>
        <article className="mood-card">
          <div className="card-heading"><Sigma size={20} /><h2>Dose–response</h2></div>
          <p className="muted small">The panel at increasing strength. Quality ratings at each strength are in the table below.</p>
          <EffectBars data={bars(dose).map((entry) => ({ ...entry, name: `α ${fixed(entry.row.alpha)}` }))} emphasise={() => true} />
        </article>
      </div>
      <article className="mood-card">
        <ConditionTable rows={[...dose, ...family("sham", "reverse", "matched_random_mean", "matched_random", "other_panel")]} judges={judges}
          caption="Panel, dose–response and controls on the deployed prompt (change vs the unsteered reply)" />
        <h3>Paired contrasts</h3>
        <table className="data-table">
          <thead>
            <tr><th>Contrast</th><th>Question</th><th className="num">n</th><th className="num">Difference</th><th className="num">95% CI</th>
              <th className="num">p (Holm)</th><th className="num">Effect size d</th><th>Reading</th></tr>
          </thead>
          <tbody>
            {contrasts.map((row) => (
              <tr key={row.name}>
                <th scope="row" className="wrap">{row.label}</th>
                <td className="wrap">{row.question}</td>
                <td className="num">{row.stat.n}</td>
                <td className="num">{signed(row.stat.mean)}</td>
                <td className="num">{interval(row.stat.lo, row.stat.hi)}</td>
                <td className="num">{pValue(row.stat.p_holm ?? row.stat.p)}</td>
                <td className="num">{signed(row.stat.dz)}</td>
                <td>{row.supports === null ? "–" : <StatusBadge status={row.supports ? "supported" : "not_supported"} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </article>

      <Inspector key={`${selected}-${language}`} conditions={inspectable} judges={judges} names={report.judges} />

      <div className="mood-grid even">
        <article className="mood-card">
          <div className="card-heading"><Layers size={20} /><h2>Layer ablation</h2></div>
          <p className="muted small">Steering one band of the panel's layers alone, and the full panel with that band left out (α {fixed(reference)}).</p>
          {bands.length ? <GroupedBars data={bands} firstName="Band alone" secondName="Panel without the band" aria="Emotion change when one layer band is steered alone or left out" />
            : <p className="muted">Not run for this selection.</p>}
        </article>
        <article className="mood-card">
          <div className="card-heading"><Layers size={20} /><h2>Neuron-subset ablation</h2></div>
          <p className="muted small">The top-ranked share of the panel's neurons against a random share of the same size (α {fixed(reference)}).</p>
          {subsets.length ? <GroupedBars data={subsets} firstName="Top-ranked neurons" secondName="Random panel neurons" aria="Emotion change for top-ranked and random subsets of the panel" />
            : <p className="muted">Not run for this selection.</p>}
        </article>
      </div>
      <article className="mood-card">
        <ConditionTable rows={family("band_only", "band_out", "subset_top", "subset_random", "subset_bottom", "subset_out")} judges={judges}
          caption="Layer bands and neuron subsets (change vs the unsteered reply)" />
        {ablation.length > 0 && (
          <>
            <h3>Necessity: ablation while the prompt asks for the emotion</h3>
            <p className="muted small">
              The prompt itself asks for the emotion; the intervention then holds neurons at their neutral value, sets them to
              zero, or reverses the shift. Change is against the same prompted reply without intervention. If the panel were
              necessary, the emotion would fall more than with random neurons.
            </p>
            <EffectBars data={bars(ablation)} emphasise={(row) => !row.family.includes("random")} height="mini-tall" />
            <ConditionTable rows={ablation} judges={judges} caption="Ablations under an emotion-requesting prompt" />
          </>
        )}
        {report.sampling && (
          <>
            <h3>Replication with sampled decoding</h3>
            <p className="muted small">{report.sampling.note}</p>
            <table className="data-table">
              <thead><tr><th>Seed</th><th className="num">Panel</th><th className="num">95% CI</th><th className="num">Matched random</th><th className="num">Panel − random</th><th className="num">95% CI</th></tr></thead>
              <tbody>
                {report.sampling.rows.map((row) => (
                  <tr key={row.seed}>
                    <th scope="row">{row.seed}</th>
                    <td className="num">{signed(row.panel.mean)}</td><td className="num">{interval(row.panel.lo, row.panel.hi)}</td>
                    <td className="num">{signed(row.matched_random.mean)}</td>
                    <td className="num">{signed(row.difference.mean)}</td><td className="num">{interval(row.difference.lo, row.difference.hi)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        <h3>Judge agreement on these replies</h3>
        <table className="data-table">
          <thead><tr><th>Rating</th><th className="num">Replies</th><th className="num">Spearman</th><th className="num">Mean |difference|</th><th className="num">Same winner</th></tr></thead>
          <tbody>
            {report.judge_agreement.map((row) => (
              <tr key={row.metric}>
                <th scope="row">{label(row.metric)}</th><td className="num">{row.replies}</td><td className="num">{fixed(row.spearman)}</td>
                <td className="num">{fixed(row.mean_abs_difference)}</td><td className="num">{percent(row.same_winner_share)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {report.failures && report.failures.length > 0 && (
          <>
            <h3>Failure cases</h3>
            <ul className="caveats">{report.failures.map((item) => <li key={item.name}><strong>{item.label}:</strong> {item.issue}</li>)}</ul>
          </>
        )}
        <ul className="caveats">{report.notes.map((note) => <li key={note}>{note}</li>)}</ul>
      </article>
    </>
  );
}

/** Shown until the unified validation battery has been run for the loaded model: the earlier per-panel experiments. */
function EarlierEvidence({ research, turn }: { research: ResearchOverview | null; turn: EmotionTurn }) {
  const turnPanel = turn.steering?.emotion ?? turn.plan?.steering_panel ?? turn.decision?.shadow_panel ?? "";
  const panels = research?.evidence?.panels ?? {};
  const [picked, setPicked] = useState<string | null>(null);
  const names = Object.keys(panels);
  const selected = picked ?? (names.includes(turnPanel) ? turnPanel : turn.evidence?.panel.emotion ?? names[0]);
  const panel = panels[selected] ?? turn.evidence?.panel;
  const notice = (
    <article className="mood-card">
      <div className="card-heading"><ShieldQuestion size={20} /><h2>Causal ablation dashboard</h2><Source kind="validated" /></div>
      <p className="safety-note">
        The unified validation battery (matched random controls over several seeds, sham, reverse steering, another emotion's
        panel, layer and neuron-subset ablations, two judges) has not been run for {research?.model ?? "this model"} yet. Run{" "}
        <code>python3 -m generation_analysis.neuron_validation --model &lt;llama | llama2 | emollama | sarvam&gt; all</code>.
        {panel ? " Below are this model's earlier causal experiments." : " No earlier causal experiments are recorded for it either."}
      </p>
    </article>
  );
  if (!panel) return notice;
  const doseRows = panel.dose_response.map((row) => {
    const control = panel.controls.find((item) => item.alpha === row.alpha && /random/i.test(item.label));
    return {
      alpha: `α ${fixed(row.alpha)}`, panel: row.emotion_rating, panelError: errorBar(row.emotion_rating, row.lo, row.hi),
      random: control?.emotion_rating ?? null, randomError: control ? errorBar(control.emotion_rating, control.lo, control.hi) : undefined,
    };
  });
  const layerEvidence = panel.layer_effects.map((row) => ({ ...row, error: errorBar(row.loss_without, row.lo, row.hi) }));
  return (
    <>
      {notice}
      {names.length > 1 && (
        <div className="filter-row" role="group" aria-label="Panel">
          <label>Panel
            <select value={selected} onChange={(event) => setPicked(event.target.value)}>
              {names.map((name) => <option key={name} value={name}>{label(name)}</option>)}
            </select>
          </label>
        </div>
      )}
      <article className="mood-card">
        <div className="card-heading"><FlaskConical size={20} /><h2>Earlier causal experiments: {panel.emotion} neurons</h2><Source kind="validated" /></div>
        <div className="stat-row">
          <Tile name="Panel">{panel.name}</Tile>
          <Tile name="Neurons">{panel.neurons.toLocaleString()}</Tile>
          <Tile name="Layers">{panel.layers[0]}–{panel.layers[panel.layers.length - 1]}</Tile>
        </div>
        <p className="guidance">{panel.verdict}</p>
        <p className="muted small">
          Selection: {panel.selection}. Results come from interventions on separate development messages. “Emotion rating
          change” is the change in a judge model's 1–5 rating against the unsteered reply to the same message; brackets are 95% bootstrap CIs.
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
            <figcaption>Effect by steering strength, panel vs random neurons</figcaption>
            <div className="chart" role="img" aria-label="Change in emotion rating by steering strength for the panel and for random neurons">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={doseRows} margin={{ top: 8, right: 12, left: -14, bottom: 0 }} barGap={2}>
                  <CartesianGrid stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="alpha" tickLine={false} axisLine={false} tick={tick} />
                  <YAxis tickLine={false} axisLine={false} tick={tick} />
                  <ReferenceLine y={0} stroke="var(--axis)" />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }} formatter={(value, name) => [signed(Number(value)), name]} />
                  <Legend iconType="square" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
                  <Bar name="Panel neurons" dataKey="panel" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false}>
                    <ErrorBar dataKey="panelError" width={4} stroke="var(--text)" />
                  </Bar>
                  <Bar name="Random neurons (control)" dataKey="random" fill="var(--deemph)" radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false}>
                    <ErrorBar dataKey="randomError" width={4} stroke="var(--text)" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </figure>
          {layerEvidence.length > 0 && (
            <figure>
              <figcaption>Layer ablation: effect lost when one layer is left out of the panel</figcaption>
              <div className="chart" role="img" aria-label="Emotion rating lost when each layer is removed from the panel">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={layerEvidence} margin={{ top: 8, right: 12, left: -14, bottom: 0 }}>
                    <CartesianGrid stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="layer" tickLine={false} axisLine={false} tick={tick} />
                    <YAxis tickLine={false} axisLine={false} tick={tick} />
                    <ReferenceLine y={0} stroke="var(--axis)" />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }} labelFormatter={(value) => `Without layer ${value}`}
                      formatter={(value) => [signed(Number(value)), "Emotion rating lost"]} />
                    <Bar dataKey="loss_without" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={18} isAnimationActive={false}>
                      <ErrorBar dataKey="error" width={4} stroke="var(--text)" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </figure>
          )}
        </div>
        <EffectTable rows={[...panel.dose_response, ...panel.controls]} caption="Dose–response and controls (change vs the unsteered reply)" />
        {panel.ablation.length > 0 && <EffectTable rows={panel.ablation} caption="Ablation in replies prompted to express the emotion" />}
        {panel.subsets.length > 0 && <EffectTable rows={panel.subsets} caption="Neuron subsets and layer bands" />}
        {research?.evidence?.sources && (
          <p className="muted small">Sources: {Object.entries(research.evidence.sources).map(([name, path]) => `${name}: ${path}`).join("; ")}.</p>
        )}
      </article>
    </>
  );
}
