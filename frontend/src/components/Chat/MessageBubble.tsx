import { Leaf, UserRound } from "lucide-react";
import type { ReactNode } from "react";
import type { ChatMessage } from "../../types/chat";

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

export function MessageBubble({ message }: { message: ChatMessage }) {
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
          <span>{message.role === "assistant" ? "MindMate" : "You"}</span>
          {message.role === "assistant" && message.steeringEmotion && message.steeringAlpha !== undefined && (
            <span title="Chosen by the Emotion Reasoner. Details are in the Mood Tracker.">
              steered toward {message.steeringEmotion} · α {message.steeringAlpha.toFixed(2)}
            </span>
          )}
          <time>{message.timestamp}</time>
        </div>
        <div className="bubble">{formattedContent(message.content)}</div>
      </div>
    </article>
  );
}
