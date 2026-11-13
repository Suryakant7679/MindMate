import { Paperclip, ArrowUp } from "lucide-react";
import { useState } from "react";
import type { ChatModel } from "../../api/chat";
export function ChatInput({
  onSend,
  busy,
  models = [],
  selectedModel,
  onSelectModel,
}: {
  onSend: (text: string) => void;
  busy: boolean;
  /** Models offered in the selector; hidden when the server lists none. */
  models?: ChatModel[];
  selectedModel?: string;
  onSelectModel?: (key: string) => void;
}) {
  const [text, setText] = useState("");
  function submit() {
    if (!text.trim() || busy) return;
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
        <button
          className="send-button"
          aria-label="Send message"
          disabled={!text.trim() || busy}
        >
          <ArrowUp size={22} />
        </button>
      </form>
      <p className="composer-note">
        A little space to be yourself.{" "}
        <span>MindMate is an AI, not a therapist.</span>
      </p>
    </div>
  );
}
