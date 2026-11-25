import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Sidebar, type ActiveSection } from "./components/Sidebar";
import { ThemeToggle } from "./components/ThemeToggle";
import { ChatPage, modelLabel } from "./components/Chat/ChatPage";
import { ConversationList } from "./components/ConversationList";
import { AccountDialog, ConsentDialog } from "./components/PrivacyDialogs";
import type { EmotionTurn } from "./types/emotion";
import type { Conversation, Me } from "./api/conversations";
import { deleteConversation, getMe, giveConsent, listConversations, updateConversation } from "./api/conversations";
import { useTheme } from "./hooks/useTheme";
const MoodTrackerPage = lazy(() =>
  import("./components/MoodTracker/MoodTrackerPage").then((module) => ({
    default: module.MoodTrackerPage,
  })),
);
const LABELS: Record<string, string> = { llama: "Llama 3.1-8B", llama2: "Llama-2-13B", emollama: "EmoLLaMA-7B", sarvam: "Sarvam-30B" };

export default function App() {
  const [active, setActive] = useState<ActiveSection>("chat");
  const [turns, setTurns] = useState<EmotionTurn[]>([]);
  const [me, setMe] = useState<Me | null>(null);
  const [persistence, setPersistence] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [accountOpen, setAccountOpen] = useState(false);
  const upsertTurn = useCallback((turn: EmotionTurn) =>
    setTurns((all) => all.some((item) => item.id === turn.id)
      ? all.map((item) => (item.id === turn.id ? turn : item))
      : [...all, turn]), []);
  const resetTurns = useCallback(() => setTurns([]), []);
  const { dark, toggle } = useTheme();

  const refresh = useCallback(async () => {
    const next = await getMe();
    setMe(next);
    setPersistence(true);
    if (!next.consent.required) setConversations(await listConversations());
    else setConversations([]);
    return next;
  }, []);

  useEffect(() => {
    refresh().catch(() => setPersistence(false));
  }, [refresh]);

  const onConversation = useCallback((change: Partial<Conversation> & { id: string }, created?: boolean) => {
    setConversations((all) => {
      const exists = all.some((item) => item.id === change.id);
      const merged = exists
        ? all.map((item) => (item.id === change.id ? { ...item, ...change } : item))
        : [change as Conversation, ...all];
      return [...merged].sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0));
    });
    if (created) setActiveId(change.id);
  }, []);

  async function rename(id: string, title: string) {
    const updated = await updateConversation(id, { title });
    onConversation(updated);
  }
  async function remove(id: string) {
    await deleteConversation(id);
    setConversations((all) => all.filter((item) => item.id !== id));
    if (activeId === id) setActiveId(null);
  }
  async function accountChanged() {
    const next = await refresh();
    if (activeId && !(await listConversations()).some((item) => item.id === activeId)) setActiveId(null);
    if (next.consent.required) setAccountOpen(false);
  }
  const closeAccount = useCallback(() => setAccountOpen(false), []);
  const label = (key: string | null) => (key ? LABELS[key] ?? modelLabel(key) : "No model yet");
  const account = me ? (me.user.anonymous ? "Anonymous · privacy" : `${me.user.username} · privacy`) : undefined;
  return (
    <div className="app-shell">
      <Sidebar active={active} onChange={setActive} account={persistence ? account : undefined} onAccount={() => setAccountOpen(true)}>
        {persistence && me && !me.consent.required && (
          <ConversationList
            conversations={conversations}
            activeId={activeId}
            busyId={conversations.find((item) => item.generating)?.id ?? null}
            onSelect={(id) => { setActiveId(id); setActive("chat"); }}
            onNew={() => { setActiveId(null); setActive("chat"); }}
            onRename={rename}
            onDelete={remove}
            modelLabel={label}
          />
        )}
      </Sidebar>
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
                : "How detected emotions evolve, what steering changed in each reply, which neurons were shifted, and what the experiments support."}
            </p>
          </div>
          <ThemeToggle dark={dark} toggle={toggle} />
        </header>
        <ChatPage
          active={active === "chat"}
          onEmotionTurn={upsertTurn}
          onResetTurns={resetTurns}
          conversationId={activeId}
          onConversation={onConversation}
          storeHistory={me?.user.store_history ?? false}
          persistence={persistence && !!me && !me.consent.required}
        />
        {active === "mood" && (
          <Suspense
            fallback={
              <div className="mood-page" role="status">
                Loading mood preview…
              </div>
            }
          >
            <MoodTrackerPage turns={turns} onTurn={upsertTurn} />
          </Suspense>
        )}
      </main>
      {me && me.consent.required && (
        <ConsentDialog me={me} onDecide={async (store, days) => { setMe(await giveConsent(store, days)); await refresh(); }} />
      )}
      {me && accountOpen && <AccountDialog me={me} onClose={closeAccount} onChanged={accountChanged} />}
    </div>
  );
}
