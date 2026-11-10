import { lazy, Suspense, useState } from "react";
import { Sidebar, type ActiveSection } from "./components/Sidebar";
import { ThemeToggle } from "./components/ThemeToggle";
import { ChatPage } from "./components/Chat/ChatPage";
import type { EmotionTurn } from "./types/emotion";
import { useTheme } from "./hooks/useTheme";
const MoodTrackerPage = lazy(() =>
  import("./components/MoodTracker/MoodTrackerPage").then((module) => ({
    default: module.MoodTrackerPage,
  })),
);
export default function App() {
  const [active, setActive] = useState<ActiveSection>("chat");
  const [turns, setTurns] = useState<EmotionTurn[]>([]);
  const upsertTurn = (turn: EmotionTurn) =>
    setTurns((all) => all.some((item) => item.id === turn.id)
      ? all.map((item) => (item.id === turn.id ? turn : item))
      : [...all, turn]);
  const { dark, toggle } = useTheme();
  return (
    <div className="app-shell">
      <Sidebar active={active} onChange={setActive} />
      <main className="main">
        <header className="page-header">
          <div>
            <div className="eyebrow header-eyebrow">
              YOUR SPACE TO FEEL HEARD
            </div>
            <h1>{active === "chat" ? "Chat" : "Mood Tracker"}</h1>
            <p>
              {active === "chat"
                ? "Talk freely. I’m here to understand and support you."
                : "See what the Emotion Reasoner understood, how it chose to respond, and what the neuron steering changed."}
            </p>
          </div>
          <ThemeToggle dark={dark} toggle={toggle} />
        </header>
        <ChatPage
          active={active === "chat"}
          onEmotionTurn={upsertTurn}
        />
        {active === "mood" && (
          <Suspense
            fallback={
              <div className="mood-page" role="status">
                Loading mood preview…
              </div>
            }
          >
            <MoodTrackerPage turns={turns} />
          </Suspense>
        )}
      </main>
    </div>
  );
}
