import { Leaf, ChartNoAxesCombined, MessageCircle } from "lucide-react";
export type ActiveSection = "chat" | "mood";
export function Sidebar({
  active,
  onChange,
}: {
  active: ActiveSection;
  onChange: (section: ActiveSection) => void;
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
    </aside>
  );
}
