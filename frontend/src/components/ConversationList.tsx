import { useState } from "react";
import { Check, CloudOff, Pencil, Plus, Trash2, X } from "lucide-react";
import type { Conversation } from "../api/conversations";

function group(updated: number): string {
  const days = (Date.now() / 1000 - updated) / 86400;
  if (days < 1) return "Today";
  if (days < 7) return "Previous 7 days";
  if (days < 30) return "Previous 30 days";
  return "Older";
}

export function ConversationList({
  conversations, activeId, busyId, onSelect, onNew, onRename, onDelete, modelLabel,
}: {
  conversations: Conversation[];
  activeId: string | null;
  /** Conversation whose reply is being written (it cannot be deleted meanwhile). */
  busyId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  modelLabel: (key: string | null) => string;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [confirming, setConfirming] = useState<string | null>(null);
  const [error, setError] = useState("");
  const groups: [string, Conversation[]][] = [];
  for (const item of conversations) {
    const name = group(item.updated_at);
    const last = groups[groups.length - 1];
    if (last && last[0] === name) last[1].push(item);
    else groups.push([name, [item]]);
  }
  async function saveTitle(id: string) {
    const title = draft.trim();
    if (!title) return setEditing(null);
    try {
      await onRename(id, title);
      setEditing(null);
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not rename the conversation.");
    }
  }
  async function remove(id: string) {
    try {
      await onDelete(id);
      setConfirming(null);
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not delete the conversation.");
    }
  }
  return (
    <section className="history" aria-label="Conversation history">
      <button type="button" className="new-chat" onClick={onNew}>
        <Plus size={17} /> New chat
      </button>
      {error && <p className="history-error" role="alert">{error}</p>}
      {conversations.length === 0 && <p className="history-empty">Your conversations will appear here.</p>}
      <div className="history-scroll">
        {groups.map(([name, items]) => (
          <div key={name} className="history-group">
            <h2>{name}</h2>
            <ul>
              {items.map((item) => (
                <li key={item.id} className={item.id === activeId ? "history-item active" : "history-item"}>
                  {editing === item.id ? (
                    <form className="history-rename" onSubmit={(event) => { event.preventDefault(); void saveTitle(item.id); }}>
                      <input
                        aria-label="Conversation title" value={draft} maxLength={120} autoFocus
                        onChange={(event) => setDraft(event.target.value)}
                        onKeyDown={(event) => { if (event.key === "Escape") setEditing(null); }}
                      />
                      <button type="submit" aria-label="Save title"><Check size={15} /></button>
                      <button type="button" aria-label="Cancel renaming" onClick={() => setEditing(null)}><X size={15} /></button>
                    </form>
                  ) : confirming === item.id ? (
                    <div className="history-confirm" role="group" aria-label={`Delete ${item.title}?`}>
                      <span>Delete permanently?</span>
                      <button type="button" className="danger" onClick={() => void remove(item.id)}>Delete</button>
                      <button type="button" onClick={() => setConfirming(null)}>Keep</button>
                    </div>
                  ) : (
                    <>
                      <button type="button" className="history-open" onClick={() => onSelect(item.id)}
                              aria-current={item.id === activeId ? "true" : undefined}
                              title={`${item.title} · ${modelLabel(item.model_key)}`}>
                        <span className="history-title">{item.title}</span>
                        <span className="history-meta">
                          {!item.saved && <CloudOff size={11} aria-label="Not saved" />}
                          {modelLabel(item.model_key)}
                        </span>
                      </button>
                      <span className="history-actions">
                        <button type="button" aria-label={`Rename ${item.title}`}
                                onClick={() => { setEditing(item.id); setDraft(item.title); setConfirming(null); }}>
                          <Pencil size={14} />
                        </button>
                        <button type="button" aria-label={`Delete ${item.title}`} disabled={busyId === item.id}
                                onClick={() => { setConfirming(item.id); setEditing(null); }}>
                          <Trash2 size={14} />
                        </button>
                      </span>
                    </>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}
