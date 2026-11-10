import { Paperclip, ArrowUp } from "lucide-react";
import { useState } from "react";
export function ChatInput({
  onSend,
  busy,
}: {
  onSend: (text: string) => void;
  busy: boolean;
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
