import type { ReactNode } from "react";
import { Leaf, ChartNoAxesCombined, MessageCircle, UserRound } from "lucide-react";
export type ActiveSection = "chat" | "mood";
export function Sidebar({
  active,
  onChange,
  children,
  account,
  onAccount,
}: {
  active: ActiveSection;
  onChange: (section: ActiveSection) => void;
  /** Conversation history, shown below the navigation. */
  children?: ReactNode;
  /** Label of the account button (hidden when the server stores no conversations). */
  account?: string;
  onAccount?: () => void;
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-line">
          <span className="brand-mark">
            <Leaf size={27} />
          </span>
          <span>MindMate</span>
        </div>
        <p>A calmer you, every day.</p>
      </div>
      <nav aria-label="Main navigation">
        {(
          [
            { id: "mood", label: "Mood Tracker", icon: ChartNoAxesCombined },
            { id: "chat", label: "Chat", icon: MessageCircle },
          ] as const
        ).map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onChange(id)}
            className={`nav-item ${active === id ? "selected" : ""}`}
            aria-current={active === id ? "page" : undefined}
          >
            <Icon size={21} />
            {label}
            {active === id && <span className="nav-dot" />}
          </button>
        ))}
      </nav>
      {children}
      {account && (
        <button type="button" className="account-button" onClick={onAccount}>
          <UserRound size={17} /> <span>{account}</span>
        </button>
      )}
    </aside>
  );
}
