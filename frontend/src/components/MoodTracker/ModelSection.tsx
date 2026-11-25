import { useState } from "react";
import { Database, Globe, Languages, Scale, Trophy } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DoseRow, ModeStat, ModelEvaluation, ModelResult } from "../../types/emotion";
import { Empty, Source, StatusBadge, fixed, interval, label, pValue, percent, signed, statError, tick, tooltipStyle } from "./shared";

const QUALITY: [string, string][] = [
  ["appropriateness", "Appropriateness"], ["empathy", "Empathy"], ["relevance", "Relevance"], ["fluency", "Fluency"],
  ["naturalness", "Naturalness"], ["hinglish_naturalness", "Hinglish naturalness"],
];
const command = (model: ModelResult) => `python3 -m generation_analysis.neuron_validation --model ${model.key} all`;
const modesCommand = (model: ModelResult) => `python3 -m generation_analysis.steering_modes --model ${model.key} --stages generate score report`;
const MODE_SUBSETS: [string, string][] = [["all", "All messages"], ["english", "English"], ["hinglish", "Hinglish"], ["crisis", "Crisis"]];
const MODE_METRICS: [string, string][] = [["judge_emotion", "Emotion"], ["appropriateness", "Appropriateness"], ["empathy", "Empathy"],
  ["safety", "Safety"], ["naturalness", "Naturalness"]];
const change = (stat: ModeStat | null | undefined) =>
  stat ? <span title={`95% CI ${interval(stat.ci_low, stat.ci_high)}${stat.p_holm != null ? ` · p (Holm) ${pValue(stat.p_holm)}` : ` · p ${pValue(stat.p)}`}`}
    className={stat.ci_low > 0 ? "good" : stat.ci_high < 0 ? "bad" : undefined}>{signed(stat.mean)}</span> : <span className="na">n/a</span>;

/** Every reply steered vs the calibrated policy vs no steering, on messages no earlier experiment used. */
function SteeringModesCard({ models, built }: { models: ModelResult[]; built: ModelEvaluation["steering_modes_set"] }) {
  return (
    <article className="mood-card">
      <div className="card-heading"><Scale size={20} /><h2>Steering every reply: fresh held-out comparison</h2><Source kind="judged" /></div>
      <p className="muted small">
        Mode “every reply” steers each reply with a strength set by the user's emotional state and sends it unless it is
        broken. Each row is its change against the same messages answered without steering (mean of the two judges, 1–5
        scales; hover a value for its 95% CI). Messages: {built
          ? built.sizes.map((row) => `${row.n} ${row.group}${row.group === "english" ? ` ${row.turns}-turn` : ""}`).join(", ") + ", used by no earlier experiment."
          : <>not built yet: run <code>python3 -m generation_analysis.steering_modes --stages build</code>.</>}
      </p>
      <table className="data-table">
        <thead>
          <tr>
            <th>Model</th><th>Messages</th><th className="num">n</th>
            {MODE_METRICS.map(([key, name]) => <th key={key} className="num">{name}</th>)}
            <th className="num">Steered replies harmed</th><th className="num">Preferred over unsteered</th><th className="num">Policy steers</th>
          </tr>
        </thead>
        <tbody>
          {models.flatMap((model) => {
            const summary = model.steering_modes;
            if (!summary) return [<tr key={model.key}><th scope="row">{model.label}</th>
              <td colSpan={MODE_METRICS.length + 5} className="wrap"><span className="na">Not run</span> · <code>{modesCommand(model)}</code></td></tr>];
            return MODE_SUBSETS.filter(([key]) => summary.subsets[key]).map(([key, name], index) => {
              const entry = summary.subsets[key];
              const contrast = entry.contrasts["always - off"] ?? {};
              const preference = contrast["preference_minus_0.5"];
              return (
                <tr key={`${model.key}-${key}`}>
                  {index === 0 ? <th scope="row" rowSpan={MODE_SUBSETS.filter(([k]) => summary.subsets[k]).length}>{model.label}</th> : null}
                  <td>{name}</td><td className="num">{entry.n}</td>
                  {MODE_METRICS.map(([metric]) => <td key={metric} className="num">{change(contrast[metric])}</td>)}
                  <td className="num">{percent(entry.harmed_share?.always_vs_off ?? null)}</td>
                  <td className="num">{preference ? fixed(0.5 + preference.mean) : "–"}</td>
                  <td className="num">{percent(entry.steered_share.policy ?? null)}</td>
                </tr>
              );
            });
          })}
        </tbody>
      </table>
      {built?.caveats && <p className="muted small">{Object.entries(built.caveats).map(([group, text]) => <span key={group}><strong>{label(group)}:</strong> {text}. </span>)}</p>}
      <p className="muted small">Harmed: a steered reply whose quality or safety rating falls by more than 0.5 against the unsteered reply. Preferred: the judges' probability of choosing the steered reply (0.5 = no preference). “Policy steers”: share of the same messages the calibrated policy would steer.</p>
    </article>
  );
}

function DoseChart({ model }: { model: ModelResult }) {
  const data = (model.battery?.doses ?? []).map((row) => ({
    alpha: `α ${fixed(row.alpha, 1)}`, gain: row.emotion.mean, error: statError(row.emotion), harmed: row.harm_share,
  }));
  return (
    <figure className="small-multiple">
      <figcaption>
        <strong>{model.label}</strong>
        {model.battery ? ` · ${model.battery.messages} messages` : ""}
      </figcaption>
      {model.battery ? (
        <>
          <div className="chart mini-tall" role="img" aria-label={`${model.label}: emotion rating change by steering strength with 95% confidence intervals`}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis dataKey="alpha" tickLine={false} axisLine={false} tick={tick} />
                <YAxis domain={[0, 2]} ticks={[0, 0.5, 1, 1.5, 2]} tickLine={false} axisLine={false} tick={tick} />
                <ReferenceLine y={0} stroke="var(--axis)" />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }}
                  formatter={(value, _name, item) => [`${signed(Number(value))} · ${percent(item.payload.harmed)} of replies harmed`, "Emotion change"]} />
                <Bar dataKey="gain" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false}>
                  <ErrorBar dataKey="error" width={4} stroke="var(--text)" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <table className="data-table compact">
            <tbody>
              <tr><th scope="row">Replies harmed</th>{data.map((row) => <td key={row.alpha} className="num">{percent(row.harmed)}</td>)}</tr>
            </tbody>
          </table>
        </>
      ) : <Empty title={label(model.battery_status)} command={command(model)} />}
    </figure>
  );
}

function JudgeChart({ model }: { model: ModelResult }) {
  const battery = model.battery;
  if (!battery) return null;
  const names = Object.keys(battery.judges);
  const data = battery.doses.map((row) => ({ alpha: `α ${fixed(row.alpha, 1)}`, first: row.judges[names[0]], second: row.judges[names[1]] }));
  return (
    <figure className="small-multiple">
      <figcaption><strong>{model.label}</strong></figcaption>
      <div className="chart mini-tall" role="img" aria-label={`${model.label}: emotion change by strength, rated by each judge`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }} barGap={2}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey="alpha" tickLine={false} axisLine={false} tick={tick} />
            <YAxis tickLine={false} axisLine={false} tick={tick} />
            <ReferenceLine y={0} stroke="var(--axis)" />
            <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--track)" }} formatter={(value, name) => [signed(Number(value)), name]} />
            <Legend iconType="square" wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
            <Bar name={battery.judges[names[0]]} dataKey="first" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={18} isAnimationActive={false} />
            <Bar name={battery.judges[names[1]]} dataKey="second" fill="var(--series-2)" radius={[4, 4, 0, 0]} maxBarSize={18} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

/** Model Evaluation: every model from its saved experiments (nothing is computed on opening). */
export function ModelSection({ evaluation }: { evaluation: ModelEvaluation }) {
  const [modelKey, setModelKey] = useState("all");
  const [language, setLanguage] = useState("all");
  const models = evaluation.models.filter((model) => modelKey === "all" || model.key === modelKey);
  const detector = evaluation.detector;
  // English vs Hinglish rows: the battery's language slices, and Sarvam's two-judge calibration sweep.
  const languageRows: { model: string; language: string; source: string; panel: string; row: DoseRow | null; n: number; hinglish?: number | null }[] = [];
  for (const model of models) {
    for (const name of ["english", "hinglish"]) {
      if (language !== "all" && language !== name) continue;
      const slice = model.battery?.by_language[name];
      const calibration = model.calibration.filter((row) => row.language === name);
      if (slice) {
        for (const [panel, doses] of Object.entries(slice.panels)) {
          const row = doses.find((dose) => dose.alpha === (model.battery?.usable_alpha ?? doses[0]?.alpha)) ?? doses[0];
          if (row) languageRows.push({ model: model.label, language: name, source: "Validation battery (dev)", panel, row, n: row.n });
        }
      }
      for (const row of calibration.filter((item) => item.alpha === 2)) {
        languageRows.push({
          model: model.label, language: name, source: "Calibration sweep, tone panels", panel: "tone panels",
          row: { alpha: row.alpha, n: row.n, emotion: row.emotion, judges: {}, independent: null, quality: row.quality, harm_share: row.harm_share, preference: row.preference },
          n: row.n, hinglish: row.hinglish_share_change,
        });
      }
      if (!slice && !calibration.length) languageRows.push({ model: model.label, language: name, source: "", panel: "", row: null, n: 0 });
    }
  }
  return (
    <>
      <div className="filter-row" role="group" aria-label="Model and language">
        <label>Model
          <select value={modelKey} onChange={(event) => setModelKey(event.target.value)}>
            <option value="all">All models</option>
            {evaluation.models.map((model) => <option key={model.key} value={model.key}>{model.label}</option>)}
          </select>
        </label>
        <label>Language
          <select value={language} onChange={(event) => setLanguage(event.target.value)}>
            <option value="all">English and Hinglish</option>
            <option value="english">English</option>
            <option value="hinglish">Hinglish</option>
          </select>
        </label>
        <span className="muted small">Saved experiment results, read from disk. Opening this page runs no model.</span>
      </div>

      <article className="mood-card">
        <div className="card-heading"><Trophy size={20} /><h2>Does steering work, and at what cost?</h2><Source kind="validated" /></div>
        <p className="muted small">
          One row per model from its own validation battery on development messages: its own panels, its own strengths, two
          judges. “Usable strength” is the strongest strength that raises the emotion while at most 20% of replies lose
          quality ({evaluation.harm_rule}).
        </p>
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th><th>Deployed policy</th><th>Panel-level causal effect</th><th>Beats random neurons</th><th>Reverses</th>
              <th className="num">Usable strength</th><th className="num">Emotion gain there (95% CI)</th>
              <th className="num">Replies harmed</th><th className="num">Steered preferred</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => {
              const battery = model.battery;
              const usable = battery?.doses.find((row) => row.alpha === battery.usable_alpha);
              return (
                <tr key={model.key}>
                  <th scope="row">{model.label}{model.loaded && <span className="flag">loaded</span>}</th>
                  <td className="wrap">
                    {model.policy
                      ? `${model.policy.steering_enabled ? `Steers ${model.policy.steered_cells.length} of ${model.policy.cells} contexts` : "Steers nothing"}${model.policy.installed ? "" : " (candidate, not installed)"}`
                      : <span className="na">No policy</span>}
                  </td>
                  {battery ? (
                    <>
                      <td><StatusBadge status={battery.verdicts.panel_level.status} /></td>
                      <td><StatusBadge status={battery.verdicts.specificity.status} /></td>
                      <td><StatusBadge status={battery.verdicts.reversibility.status} /></td>
                      <td className="num">{usable ? `α ${fixed(usable.alpha, 1)}` : "none"}</td>
                      <td className="num">{usable ? `${signed(usable.emotion.mean)} (${interval(usable.emotion.lo, usable.emotion.hi)})` : "–"}</td>
                      <td className="num">{usable ? percent(usable.harm_share) : "–"}</td>
                      <td className="num">{usable ? fixed(usable.preference) : "–"}</td>
                    </>
                  ) : <td colSpan={7} className="wrap"><span className="na">Battery {model.battery_status}</span> · run <code>{command(model)}</code></td>}
                </tr>
              );
            })}
          </tbody>
        </table>
        {models.filter((model) => model.battery).map((model) => (
          <p key={model.key} className="muted small"><strong>{model.label}:</strong> {model.battery?.verdicts.quality?.text}</p>
        ))}
        <figure>
          <figcaption>Emotion rating change (judges' mean, 1–5 scale) by steering strength, with 95% CIs. Strengths are each model's own scale and are not comparable across models.</figcaption>
          <div className="multiples">{models.map((model) => <DoseChart key={model.key} model={model} />)}</div>
        </figure>
      </article>

      <SteeringModesCard models={models} built={evaluation.steering_modes_set} />

      <article className="mood-card">
        <div className="card-heading"><Scale size={20} /><h2>Emotion expression and response quality by strength</h2><Source kind="judged" /></div>
        <p className="muted small">Change against the unsteered reply to the same message (mean of the two judges, 1–5 scales; the independent classifier is a 0–1 score).</p>
        {models.map((model) => model.battery ? (
          <table className="data-table" key={model.key}>
            <caption className="muted small">{model.label} · all panels pooled · {model.battery.messages} messages ({Object.entries(model.battery.languages).map(([name, count]) => `${count} ${name}`).join(", ")})</caption>
            <thead>
              <tr>
                <th>Strength</th><th className="num">Emotion</th><th className="num">95% CI</th><th className="num">p (Holm)</th><th className="num">Effect size d</th>
                <th className="num">Indep. classifier</th>
                {QUALITY.map(([key, name]) => <th key={key} className="num">{name}</th>)}
                <th className="num">Harmed</th><th className="num">Preferred</th>
              </tr>
            </thead>
            <tbody>
              {model.battery.doses.map((row) => (
                <tr key={row.alpha} className={row.alpha === model.battery?.usable_alpha ? "selected" : ""}>
                  <th scope="row">α {fixed(row.alpha, 1)}{row.alpha === model.battery?.usable_alpha ? " · usable" : ""}</th>
                  <td className="num">{signed(row.emotion.mean)}</td><td className="num">{interval(row.emotion.lo, row.emotion.hi)}</td>
                  <td className="num">{pValue(row.emotion.p_holm)}</td><td className="num">{signed(row.emotion.dz)}</td>
                  <td className="num">{signed(row.independent, 3)}</td>
                  {QUALITY.map(([key]) => <td key={key} className="num">{row.quality[key] === null || row.quality[key] === undefined ? <span className="na">n/a</span> : signed(row.quality[key])}</td>)}
                  <td className="num">{percent(row.harm_share)}</td><td className="num">{fixed(row.preference)}</td>
                </tr>
              ))}
              {model.battery.controls.matched_random && (
                <tr><th scope="row">Matched random neurons</th><td className="num">{signed(model.battery.controls.matched_random.mean)}</td>
                  <td className="num">{interval(model.battery.controls.matched_random.lo, model.battery.controls.matched_random.hi)}</td><td colSpan={QUALITY.length + 5} /></tr>
              )}
              {model.battery.reverse && (
                <tr><th scope="row">Reversed shift</th><td className="num">{signed(model.battery.reverse.mean)}</td>
                  <td className="num">{interval(model.battery.reverse.lo, model.battery.reverse.hi)}</td><td colSpan={QUALITY.length + 5} /></tr>
              )}
            </tbody>
          </table>
        ) : <Empty key={model.key} title={`${model.label}: ${model.battery_status}`} command={command(model)} />)}
      </article>

      <article className="mood-card">
        <div className="card-heading"><Languages size={20} /><h2>English and Hinglish</h2><Source kind="judged" /></div>
        <p className="muted small">
          Per language, at each model's usable (or weakest) strength. Rows marked not available have no saved experiment for
          that language; nothing is estimated in their place.
        </p>
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th><th>Language</th><th>Panel</th><th>Source</th><th className="num">Messages</th><th className="num">Strength</th>
              <th className="num">Emotion gain</th><th className="num">95% CI</th><th className="num">Fluency</th><th className="num">Naturalness</th>
              <th className="num">Hinglish naturalness</th><th className="num">Harmed</th><th className="num">Preferred</th>
            </tr>
          </thead>
          <tbody>
            {languageRows.map((entry, index) => entry.row ? (
              <tr key={index}>
                <th scope="row">{entry.model}</th><td>{label(entry.language)}</td><td>{label(entry.panel)}</td><td className="wrap">{entry.source}</td>
                <td className="num">{entry.n}</td><td className="num">α {fixed(entry.row.alpha, 1)}</td>
                <td className="num">{signed(entry.row.emotion.mean)}</td><td className="num">{interval(entry.row.emotion.lo, entry.row.emotion.hi)}</td>
                <td className="num">{signed(entry.row.quality.fluency)}</td><td className="num">{signed(entry.row.quality.naturalness)}</td>
                <td className="num">{entry.row.quality.hinglish_naturalness === null || entry.row.quality.hinglish_naturalness === undefined ? <span className="na">n/a</span> : signed(entry.row.quality.hinglish_naturalness)}</td>
                <td className="num">{percent(entry.row.harm_share)}</td><td className="num">{fixed(entry.row.preference)}</td>
              </tr>
            ) : (
              <tr key={index}>
                <th scope="row">{entry.model}</th><td>{label(entry.language)}</td>
                <td colSpan={11}><span className="na">Not available</span> · this model's battery has no {entry.language} development messages
                  {entry.language === "hinglish" ? <> · add them with <code>MINDMATE_VALIDATION_HINGLISH=1</code> and re-run generate, score --force and analyse</> : null}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </article>

      <article className="mood-card">
        <div className="card-heading"><Scale size={20} /><h2>Qwen2.5-7B-Instruct and Gemma 3-27B-IT as judges</h2><Source kind="judged" /></div>
        <p className="muted small">The same replies rated by each judge. Where the bars part, the size of the effect depends on the judge; the direction should not.</p>
        <div className="multiples">{models.filter((model) => model.battery).map((model) => <JudgeChart key={model.key} model={model} />)}</div>
        {models.filter((model) => model.battery).map((model) => (
          <table className="data-table" key={model.key}>
            <caption className="muted small">{model.label}: agreement between the two judges on the same replies</caption>
            <thead><tr><th>Rating</th><th className="num">Replies</th><th className="num">Rank correlation</th><th className="num">Mean |difference|</th><th className="num">Same winner</th></tr></thead>
            <tbody>
              {model.battery?.agreement.map((row) => (
                <tr key={row.metric}>
                  <th scope="row">{label(row.metric)}</th><td className="num">{row.replies.toLocaleString()}</td><td className="num">{fixed(row.spearman)}</td>
                  <td className="num">{fixed(row.mean_abs_difference)}</td><td className="num">{percent(row.same_winner_share)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ))}
        {!models.some((model) => model.battery) && <Empty title="No two-judge battery results for this selection" />}
      </article>

      <article className="mood-card">
        <div className="card-heading"><Globe size={20} /><h2>Held-out comparison of the deployed systems</h2><Source kind="judged" /></div>
        {models.map((model) => model.held_out ? (
          <div key={model.key}>
            <h3>{model.label}</h3>
            <p className="muted small">{model.held_out.summary} {model.held_out.messages} messages never used to choose neurons or strengths.</p>
            <table className="data-table">
              <thead>
                <tr><th>System</th><th className="num">Emotion rating</th><th className="num">Independent classifier</th><th className="num">Empathy</th>
                  <th className="num">Relevance</th><th className="num">Fluency</th><th className="num">Replies steered</th></tr>
              </thead>
              <tbody>
                {model.held_out.systems.map((row) => (
                  <tr key={row.system}>
                    <th scope="row">{row.label}</th><td className="num">{fixed(row.emotion_rating)}</td><td className="num">{fixed(row.independent_classifier, 3)}</td>
                    <td className="num">{fixed(row.empathy)}</td><td className="num">{fixed(row.relevance)}</td><td className="num">{fixed(row.fluency)}</td>
                    <td className="num">{percent(row.steered_share)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <details>
              <summary>Paired differences with 95% confidence intervals</summary>
              <table className="data-table">
                <thead><tr><th>Comparison</th><th>Metric</th><th className="num">Mean difference</th><th className="num">95% CI</th><th className="num">p</th></tr></thead>
                <tbody>
                  {model.held_out.contrasts.map((row) => (
                    <tr key={`${row.contrast}-${row.metric}`}>
                      <th scope="row">{row.label}</th><td>{row.metric}</td><td className="num">{signed(row.mean, 3)}</td>
                      <td className="num">{interval(row.lo, row.hi, 3)}</td><td className="num">{pValue(row.p)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </div>
        ) : <Empty key={model.key} title={`${model.label}: no held-out comparison saved`} />)}
      </article>

      <article className="mood-card">
        <div className="card-heading"><Database size={20} /><h2>How far the emotion detector can be trusted</h2><Source kind="estimated" /></div>
        {detector ? (
          <>
            <p className="muted small">
              {detector.detector} against {detector.messages.toLocaleString()} labelled messages. Calibration error is the average gap
              between the detector's confidence and how often it is right; polarity AUC is how well the estimated valence separates
              positive from negative messages.
            </p>
            <table className="data-table">
              <thead>
                <tr><th>Language</th><th className="num">Messages</th><th className="num">Label accuracy</th><th className="num">95% CI</th>
                  <th className="num">Mean confidence</th><th className="num">Calibration error</th><th className="num">Polarity AUC</th><th className="num">Valence correlation</th></tr>
              </thead>
              <tbody>
                {detector.by_language.filter((row) => language === "all" || row.language === language).map((row) => (
                  <tr key={row.language}>
                    <th scope="row">{label(row.language)}</th><td className="num">{row.n}</td><td className="num">{percent(row.accuracy)}</td>
                    <td className="num">{percent(row.accuracy_lo)} to {percent(row.accuracy_hi)}</td><td className="num">{fixed(row.mean_confidence)}</td>
                    <td className="num">{fixed(row.ece, 3)}</td><td className="num">{fixed(row.polarity_auc)}</td><td className="num">{fixed(row.valence_correlation)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <ul className="caveats">{detector.notes.map((note) => <li key={note}>{note}</li>)}</ul>
          </>
        ) : <Empty title="The detector has not been checked against labelled messages" command="python3 -m generation_analysis.trajectory_validation" />}
      </article>
    </>
  );
}
