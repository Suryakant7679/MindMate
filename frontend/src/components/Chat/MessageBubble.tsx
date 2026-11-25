import { Leaf, UserRound } from "lucide-react";
import type { ReactNode } from "react";
import type { ChatMessage } from "../../types/chat";
import type { ToneRecord, ToneState } from "../../api/conversations";

const signed = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;

function ToneColumn({ title, state, target }: { title: string; state: ToneState; target: string | null }) {
  return (
    <div className="tone-column">
      <h4>{title}</h4>
      <ul>
        {state.top.map((item) => (
          <li key={item.emotion} className={item.emotion === target ? "tone-target" : undefined}>
            <span className="tone-label">{item.emotion}</span>
            <span className="tone-bar" aria-hidden="true"><span style={{ width: `${Math.round(item.share * 100)}%` }} /></span>
            <span className="tone-value">{Math.round(item.share * 100)}%</span>
          </li>
        ))}
      </ul>
      <p>valence {signed(state.expected_valence)}{target && state.target_score !== null ? ` · ${target} score ${state.target_score.toFixed(2)}` : ""}</p>
    </div>
  );
}

/** The reply's emotional tone before steering (unsteered version) and after steering (steered version). */
function ToneDetails({ tone }: { tone: ToneRecord }) {
  const { before, after, change, target } = tone;
  if (!before) return null;
  const summary = after
    ? `${before.emotion} → ${after.emotion}` + (change?.target_score !== undefined && target ? ` · ${target} ${signed(change.target_score)}` : "")
    : `${before.emotion} (no steered version)`;
  return (
    <details className="tone">
      <summary>Emotional tone: {summary}</summary>
      <div className="tone-grid">
        <ToneColumn title="Before steering" state={before} target={target} />
        {after && <ToneColumn title="After steering" state={after} target={target} />}
      </div>
      <p className="tone-note">
        Classifier estimate of each reply text, not of the user. {tone.sent === "steered"
          ? "The steered version was sent."
          : after ? "The unsteered version was sent (the steered one did not pass the reply checks)." : "This reply was not steered."}
      </p>
    </details>
  );
}

function inlineMarkdown(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
    part.startsWith("**") && part.endsWith("**")
      ? <strong key={index}>{part.slice(2, -2)}</strong>
      : part,
  );
}

function formattedContent(content: string): ReactNode {
  const lines = content.split("\n");
  const nodes: React.ReactNode[] = [];
  let bullets: string[] = [];
  const flush = () => {
    if (!bullets.length) return;
    nodes.push(
      <ul key={`bullets-${nodes.length}`}>
        {bullets.map((bullet, index) => <li key={index}>{inlineMarkdown(bullet)}</li>)}
      </ul>,
    );
    bullets = [];
  };
  lines.forEach((line, index) => {
    const bullet = line.match(/^\s*(?:[*-]|•)\s+(.+)$/);
    if (bullet) {
      bullets.push(bullet[1]);
      return;
    }
    flush();
    if (line.trim()) nodes.push(<p key={`line-${index}`}>{inlineMarkdown(line)}</p>);
  });
  flush();
  return nodes;
}

function steeringNote(message: ChatMessage): { text: string; title: string } | null {
  const steering = message.steering;
  if (steering?.applied) {
    return {
      text: `steered toward ${steering.response_emotion ?? steering.panel} · α ${steering.applied_alpha.toFixed(2)}`,
      title: `${steering.neurons ?? "?"} neurons in ${steering.layers?.length ?? "?"} layers` +
        (steering.layers?.length ? ` (layers ${steering.layers[0]}–${steering.layers[steering.layers.length - 1]})` : "") +
        ". Chosen by the Emotion Reasoner from this model's calibrated policy; details are in the Mood Tracker.",
    };
  }
  if (steering && steering.mode === "off") return { text: "unsteered baseline", title: "Steering is switched off for this conversation." };
  if (!steering && message.steeringEmotion && message.steeringAlpha !== undefined)
    return { text: `steered toward ${message.steeringEmotion} · α ${message.steeringAlpha.toFixed(2)}`,
             title: "Chosen by the Emotion Reasoner. Details are in the Mood Tracker." };
  return null;
}

export function MessageBubble({ message, modelLabel }: { message: ChatMessage; modelLabel?: (key: string | null | undefined, id?: string | null) => string }) {
  const note = message.role === "assistant" ? steeringNote(message) : null;
  const author = message.role === "assistant"
    ? (message.modelKey || message.modelId ? `MindMate · ${modelLabel ? modelLabel(message.modelKey, message.modelId) : message.modelKey}` : "MindMate")
    : "You";
  return (
    <article className={`message ${message.role}`}>
      <span className="avatar" aria-hidden="true">
        {message.role === "assistant" ? (
          <Leaf size={19} />
        ) : (
          <UserRound size={18} />
        )}
      </span>
      <div className="message-body">
        <div className="message-meta">
          <span>{author}</span>
          {note && <span className="meta-chip" title={note.title}>{note.text}</span>}
          {message.contextTrimmed && (
            <span className="meta-chip muted" title="This model's context window is shorter, so older turns were left out of its prompt. They are still saved.">
              saw last {message.contextTrimmed.sent} of {message.contextTrimmed.total} messages
            </span>
          )}
          {message.unsaved && <span className="meta-chip muted" title="This message is not stored (agent mode, or conversation history is unavailable).">not saved</span>}
          {message.stopped && <span className="meta-chip muted" title="You stopped this reply; it shows the text written until then.">stopped</span>}
          <time>{message.timestamp}</time>
        </div>
        <div className="bubble">{formattedContent(message.content)}</div>
        {message.role === "assistant" && message.tone && <ToneDetails tone={message.tone} />}
      </div>
    </article>
  );
}
