import type { ReactNode } from "react";
import { CheckCircle2, CircleSlash, MinusCircle, XCircle } from "lucide-react";
import type { CausalEffect, Stat, VerdictLabel } from "../../types/emotion";

export const label = (text: string) => text.replace(/_/g, " ").replace(/^./, (letter) => letter.toUpperCase());
export const fixed = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value) ? "–" : value.toFixed(digits);
export const signed = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value) ? "–" : `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
export const percent = (value: number | null | undefined) =>
  value === null || value === undefined || Number.isNaN(value) ? "–" : `${Math.round(value * 100)}%`;
export const interval = (low: number | null | undefined, high: number | null | undefined, digits = 2) =>
  low === null || low === undefined || high === null || high === undefined ? "–" : `${signed(low, digits)} to ${signed(high, digits)}`;
export const pValue = (value: number | null | undefined) =>
  value === null || value === undefined || Number.isNaN(value) ? "–" : value < 0.001 ? "<0.001" : value.toFixed(3);
/** Recharts error bars take [distance below, distance above]. */
export const errorBar = (value: number, low: number, high: number) => [value - low, high - value];
export const statError = (stat: Stat | undefined) =>
  stat && stat.mean !== null && stat.lo !== null && stat.hi !== null ? errorBar(stat.mean, stat.lo, stat.hi) : undefined;

export const tooltipStyle = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 12,
  color: "var(--text)",
  fontSize: 12,
};
export const tick = { fill: "var(--muted)", fontSize: 11 };
export const EMOTIONS = ["love", "joy", "admiration", "surprise", "neutral", "sadness", "fear", "anger", "disapproval", "disgust"];

export type SourceKind =
  | "measured" | "estimated" | "reasoned" | "calibrated" | "validated" | "judged" | "observed" | "reported" | "unavailable";
/** Where a number comes from. Five evidence classes; the suffix says how it was obtained. */
export function Source({ kind }: { kind: SourceKind }) {
  const [text, tone] = {
    measured: ["MEASURED · MODEL HOOKS", "measured"],
    estimated: ["CLASSIFIER-ESTIMATED", "estimated"],
    observed: ["CLASSIFIER-ESTIMATED · OBSERVATIONAL", "estimated"],
    reasoned: ["KNOWLEDGE-GRAPH RULE", "reasoned"],
    calibrated: ["LLM-JUDGED · CALIBRATION SWEEP", "judged"],
    validated: ["LLM-JUDGED · CONTROLLED EXPERIMENT", "judged"],
    judged: ["LLM-JUDGED", "judged"],
    reported: ["USER-REPORTED", "reported"],
    unavailable: ["NOT AVAILABLE", "unavailable"],
  }[kind];
  return <span className={`badge source ${tone}`}>{text}</span>;
}

/** An explicit empty state: what is missing, why, and how to produce it. */
export function Empty({ title, children, command }: { title: string; children?: ReactNode; command?: string }) {
  return (
    <div className="empty-state" role="note">
      <CircleSlash size={18} aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        {children && <p>{children}</p>}
        {command && <code>{command}</code>}
      </div>
      <Source kind="unavailable" />
    </div>
  );
}
export function Loading({ what }: { what: string }) {
  return <p className="muted loading" role="status">Loading {what}…</p>;
}
export function Failure({ what, error, onRetry }: { what: string; error: string; onRetry?: () => void }) {
  return (
    <div className="empty-state error" role="alert">
      <XCircle size={18} aria-hidden="true" />
      <div><strong>{what} could not be loaded</strong><p>{error}</p></div>
      {onRetry && <button type="button" className="ghost-button" onClick={onRetry}>Try again</button>}
    </div>
  );
}

/** A number that the browser check compares with the raw record: ``path`` addresses it in the conversation JSON. */
export function Metric({ path, value, digits = 2, format }: {
  path: string; value: number | null | undefined; digits?: number; format?: (value: number | null | undefined) => string;
}) {
  return (
    <span data-metric={path} data-value={value === null || value === undefined ? "" : String(value)}>
      {format ? format(value) : fixed(value, digits)}
    </span>
  );
}

export function Tile({ name, children, note }: { name: string; children: ReactNode; note?: string }) {
  return (
    <div>
      <span>{name}</span>
      <strong>{children}</strong>
      {note && <small className="tile-note">{note}</small>}
    </div>
  );
}

const VERDICT = {
  improved: { text: "Improved", icon: CheckCircle2, tone: "good" },
  harmed: { text: "Harmed", icon: XCircle, tone: "critical" },
  no_meaningful_change: { text: "No meaningful change", icon: MinusCircle, tone: "neutral" },
  not_compared: { text: "Not compared", icon: CircleSlash, tone: "neutral" },
} as const;
/** Status is carried by icon and words; the colour only repeats it. */
export function VerdictBadge({ verdict, basis }: { verdict: VerdictLabel | null | undefined; basis?: string | null }) {
  if (!verdict) return <span className="verdict neutral">–</span>;
  const { text, icon: Icon, tone } = VERDICT[verdict];
  return (
    <span className={`verdict ${tone}`}>
      <Icon size={14} aria-hidden="true" /> {text}
      {basis && basis !== "none" && <small>{basis === "judged" ? "judged" : "automatic"}</small>}
    </span>
  );
}

const STATUS = {
  supported: { text: "Supported", icon: CheckCircle2, tone: "good" },
  not_supported: { text: "Not supported", icon: XCircle, tone: "critical" },
  inconclusive: { text: "Inconclusive", icon: MinusCircle, tone: "neutral" },
  not_tested: { text: "Not tested", icon: CircleSlash, tone: "neutral" },
} as const;
export function StatusBadge({ status }: { status: keyof typeof STATUS }) {
  const { text, icon: Icon, tone } = STATUS[status] ?? STATUS.not_tested;
  return <span className={`verdict ${tone}`}><Icon size={14} aria-hidden="true" /> {text}</span>;
}

/** Sequential cell colour: one hue, more is darker, zero recedes into the surface. */
export const sequential = (share: number) =>
  `color-mix(in srgb, var(--series-1) ${Math.round(Math.max(0, Math.min(1, share)) * 100)}%, var(--surface))`;
/** Diverging cell colour: two opposite hues around a neutral grey midpoint. */
export const diverging = (value: number, peak: number) => {
  const share = peak ? Math.max(-1, Math.min(1, value / peak)) : 0;
  return `color-mix(in srgb, var(${share >= 0 ? "--div-pos" : "--div-neg"}) ${Math.round(Math.abs(share) * 100)}%, var(--div-mid))`;
};

export function EffectTable({ rows, caption }: { rows: CausalEffect[]; caption: string }) {
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
