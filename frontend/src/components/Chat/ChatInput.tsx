import { Paperclip, ArrowUp, Square } from "lucide-react";
import { useState } from "react";
import type { ChatModel } from "../../api/chat";
export function ChatInput({
  onSend,
  busy,
  blocked = false,
  onStop,
  models = [],
  selectedModel,
  onSelectModel,
}: {
  onSend: (text: string) => void;
  /** A reply is being prepared: the send button becomes a stop button. */
  busy: boolean;
  /** Sending is not possible right now (e.g. an error must be resolved first). */
  blocked?: boolean;
  onStop?: () => void;
  /** Models offered in the selector; hidden when the server lists none. */
  models?: ChatModel[];
  selectedModel?: string;
  onSelectModel?: (key: string) => void;
}) {
  const [text, setText] = useState("");
  function submit() {
    if (!text.trim() || busy || blocked) return;
    onSend(text.trim());
    setText("");
  }
  return (
    <div className="composer-wrap">
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <span
          className="attachment"
          title="Attachments are not supported yet"
          aria-label="Attachments are not supported yet"
        >
          <Paperclip size={21} />
        </span>
        <textarea
          aria-label="Message"
          placeholder="Type your message here..."
          rows={1}
          maxLength={8000}
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              submit();
            }
          }}
        />
        {models.length > 0 && (
          <select
            className="model-select"
            aria-label="Model used to answer"
            title={models.find((item) => item.key === selectedModel)?.purpose}
            value={selectedModel ?? ""}
            disabled={busy}
            onChange={(event) => onSelectModel?.(event.target.value)}
          >
            {models.map((item) => (
              <option key={item.key} value={item.key} disabled={!item.available}>
                {item.label}{item.loaded ? " · loaded" : item.available ? "" : " · not downloaded"}
              </option>
            ))}
          </select>
        )}
        {busy && onStop ? (
          <button type="button" className="send-button stop" aria-label="Stop generating" title="Stop generating" onClick={onStop}>
            <Square size={16} fill="currentColor" />
          </button>
        ) : (
          <button
            className="send-button"
            aria-label="Send message"
            disabled={!text.trim() || busy || blocked}
          >
            <ArrowUp size={22} />
          </button>
        )}
      </form>
      <p className="composer-note">
        A little space to be yourself.{" "}
        <span>MindMate is an AI, not a therapist.</span>
      </p>
    </div>
  );
}
