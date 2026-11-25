import { useEffect, useRef, useState } from "react";
import { ShieldCheck, UserRound, X } from "lucide-react";
import type { Me } from "../api/conversations";
import { deleteAllData, exportUrl, login, logout, register, updateSettings } from "../api/conversations";

export function retentionLabel(days: number): string {
  if (days === 0) return "until I delete them";
  if (days === 365) return "1 year after my last message";
  return `${days} days after my last message`;
}

/** Shown before anything is stored: the user decides whether conversations are saved and for how long. */
export function ConsentDialog({ me, onDecide }: { me: Me; onDecide: (store: boolean, days: number) => Promise<void> }) {
  const [days, setDays] = useState(me.user.retention_days || 30);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function decide(store: boolean) {
    setBusy(true);
    try {
      await onDecide(store, days);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save your choice.");
      setBusy(false);
    }
  }
  return (
    <div className="dialog-backdrop">
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="consent-title">
        <ShieldCheck size={28} className="dialog-icon" />
        <h2 id="consent-title">Before we talk: your conversations</h2>
        <p>Conversations with MindMate can include personal and mental-health information. You decide whether they are kept.</p>
        <ul>
          <li><strong>What is saved:</strong> your messages, MindMate's replies, which model answered and the emotion-steering
            settings used.</li>
          <li><strong>Where:</strong> only in this server's database, on the operator's machine. Nothing is shared with third parties.</li>
          <li><strong>Your control:</strong> rename or delete any conversation, export everything, or delete all your data at any time.</li>
          <li>MindMate is a research assistant, not an emergency service, and no one is contacted automatically.</li>
        </ul>
        <label className="dialog-field">Keep saved conversations{" "}
          <select value={days} onChange={(event) => setDays(Number(event.target.value))} disabled={busy}>
            {me.retention_options.map((option) => <option key={option} value={option}>{retentionLabel(option)}</option>)}
          </select>
        </label>
        {error && <p className="dialog-error" role="alert">{error}</p>}
        <div className="dialog-actions">
          <button type="button" className="primary" disabled={busy} onClick={() => void decide(true)}>Save my conversations</button>
          <button type="button" disabled={busy} onClick={() => void decide(false)}>Don't save (this session only)</button>
        </div>
        <p className="dialog-note">Unsaved conversations stay in server memory only and disappear after a restart or a few hours of inactivity.</p>
      </div>
    </div>
  );
}

/** Account, history settings, export and deletion. */
export function AccountDialog({ me, onClose, onChanged }: { me: Me; onClose: () => void; onChanged: () => Promise<void> }) {
  const [mode, setMode] = useState<"none" | "register" | "login">("none");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    close.current?.focus();                     // keyboard users start inside the dialog
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  async function run(action: () => Promise<unknown>, done?: string) {
    setBusy(true);
    setMessage("");
    try {
      await action();
      await onChanged();
      if (done) setMessage(done);
      setMode("none");
      setPassword("");
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }
  const user = me.user;
  return (
    <div className="dialog-backdrop" onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="account-title">
        <button type="button" ref={close} className="dialog-close" aria-label="Close" onClick={onClose}><X size={18} /></button>
        <UserRound size={26} className="dialog-icon" />
        <h2 id="account-title">Account and privacy</h2>
        <p>{user.anonymous
          ? "You are using MindMate anonymously in this browser. Create an account to reach your conversations from another browser."
          : <>Signed in as <strong>{user.username}</strong>.</>}</p>
        {mode === "none" ? (
          <div className="dialog-actions">
            {user.anonymous && <button type="button" onClick={() => setMode("register")} disabled={busy}>Create account</button>}
            {user.anonymous && <button type="button" onClick={() => setMode("login")} disabled={busy}>Sign in</button>}
            {!user.anonymous && <button type="button" onClick={() => void run(logout, "Signed out.")} disabled={busy}>Sign out</button>}
          </div>
        ) : (
          <form className="dialog-form" onSubmit={(event) => {
            event.preventDefault();
            void run(() => (mode === "register" ? register(username.trim(), password) : login(username.trim(), password)),
                     mode === "register" ? "Account created. Your conversations are kept." : "Signed in.");
          }}>
            <label>Username <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username"
                                   minLength={3} maxLength={32} required /></label>
            <label>Password <input type="password" value={password} onChange={(event) => setPassword(event.target.value)}
                                   autoComplete={mode === "register" ? "new-password" : "current-password"} minLength={8} required /></label>
            {mode === "login" && user.anonymous && <p className="dialog-note">Conversations of this anonymous session stay with it after you sign in.</p>}
            <div className="dialog-actions">
              <button type="submit" className="primary" disabled={busy}>{mode === "register" ? "Create account" : "Sign in"}</button>
              <button type="button" onClick={() => setMode("none")} disabled={busy}>Cancel</button>
            </div>
          </form>
        )}
        <h3>Conversation history</h3>
        <label className="dialog-field">
          <input type="checkbox" checked={user.store_history} disabled={busy}
                 onChange={(event) => void run(() => updateSettings({ store_history: event.target.checked }))} />
          {" "}Save new conversations
        </label>
        <label className="dialog-field">Keep saved conversations{" "}
          <select value={user.retention_days} disabled={busy}
                  onChange={(event) => void run(() => updateSettings({ retention_days: Number(event.target.value) }), "Retention updated.")}>
            {me.retention_options.map((option) => <option key={option} value={option}>{retentionLabel(option)}</option>)}
          </select>
        </label>
        <div className="dialog-actions">
          <a className="button-link" href={exportUrl} download>Export my data</a>
          {!confirmDelete
            ? <button type="button" className="danger" onClick={() => setConfirmDelete(true)} disabled={busy}>Delete all my data</button>
            : <>
                <button type="button" className="danger" disabled={busy}
                        onClick={() => void run(deleteAllData, "All your data was deleted.")}>Yes, delete everything</button>
                <button type="button" onClick={() => setConfirmDelete(false)} disabled={busy}>Cancel</button>
              </>}
        </div>
        {message && <p className="dialog-status" role="status">{message}</p>}
      </div>
    </div>
  );
}
