import { useCallback, useEffect, useState } from "react";
import { Info } from "lucide-react";
import { RefreshCw } from "lucide-react";
import { fetchEmotionTrace, fetchModelEvaluation, fetchResearch, sendFeedback, setResearchCompare } from "../../api/chat";
import type { ConversationSummary, EmotionTurn, ModelEvaluation, ResearchOverview, Trajectory } from "../../types/emotion";
import { CausalSection } from "./CausalSection";
import { ImpactSection } from "./ImpactSection";
import { ModelSection } from "./ModelSection";
import { NeuronSection } from "./NeuronSection";
import { OverviewSection, type DashboardTab } from "./OverviewSection";
import { TimelineSection } from "./TimelineSection";
import { Empty, Failure, Loading, fixed, label } from "./shared";

const TABS: [DashboardTab, string][] = [
  ["overview", "Overview"], ["timeline", "Emotion Timeline"], ["comparison", "Response Comparison"],
  ["neurons", "Neuron Explorer"], ["causal", "Causal Ablations"], ["models", "Model Evaluation"],
];
const PER_TURN: DashboardTab[] = ["comparison", "neurons"];

export function MoodTrackerPage({ turns, onTurn }: { turns: EmotionTurn[]; onTurn: (turn: EmotionTurn) => void }) {
  const [refreshing, setRefreshing] = useState(false);
  const [refreshed, setRefreshed] = useState(0);
  const [picked, setPicked] = useState<string | null>(null);
  const [tab, setTab] = useState<DashboardTab>("overview");
  const [research, setResearch] = useState<ResearchOverview | null>(null);
  const [researchState, setResearchState] = useState<"loading" | "ready" | "none">("loading");
  const [evaluation, setEvaluation] = useState<ModelEvaluation | null>(null);
  const [evaluationError, setEvaluationError] = useState("");
  const [toggling, setToggling] = useState(false);
  // Trajectory and summary after a rating was added (newer than the ones attached to the last turn).
  const [rated, setRated] = useState<{ conversation: string; turns: number; trajectory: Trajectory; summary: ConversationSummary } | null>(null);
  const latest = [...turns].reverse().find((item) => item.trajectory);
  const ready = turns.filter((item) => item.status === "ready").length;
  const analysing = turns.some((item) => item.status === "analysing");

  // Saved results only: opening the dashboard never runs a model.
  useEffect(() => {
    const controller = new AbortController();
    fetchResearch(controller.signal)
      .then((data) => { setResearch(data); setResearchState(data ? "ready" : "none"); })
      .catch(() => { if (!controller.signal.aborted) setResearchState("none"); });
    return () => controller.abort();
  }, [ready, refreshed]);
  const loadEvaluation = useCallback((signal?: AbortSignal) => {
    setEvaluationError("");
    fetchModelEvaluation(signal).then(setEvaluation).catch((cause) => {
      if (!signal?.aborted) setEvaluationError(cause instanceof Error ? cause.message : "Unknown error");
    });
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    loadEvaluation(controller.signal);
    return () => controller.abort();
  }, [loadEvaluation]);

  const turn = turns.find((item) => item.id === picked) ?? turns[turns.length - 1];
  const turnNumber = turn ? turns.indexOf(turn) + 1 : 0;
  const override = rated && latest && rated.conversation === latest.conversationId && rated.turns === (latest.trajectory?.turns.length ?? 0) ? rated : null;
  const trajectory = override?.trajectory ?? latest?.trajectory;
  const summary = override?.summary ?? latest?.summary;
  const conversation = latest?.conversationId;

  const feedback = (turnIndex: number, rating: { mood_rating?: number; helpfulness?: number }) => {
    if (!conversation) return;
    void sendFeedback(conversation, turnIndex, rating).then((view) => {
      if (view) setRated({ conversation, turns: view.trajectory.turns.length, trajectory: view.trajectory, summary: view.summary });
    });
  };
  const toggleCompare = (enabled: boolean) => {
    setToggling(true);
    void setResearchCompare(enabled)
      .then((settings) => { if (settings) setResearch((current) => (current ? { ...current, settings } : current)); })
      .finally(() => setToggling(false));
  };
  // Re-read every turn and the saved experiment files (offline judging and new reports appear without a new chat).
  const refresh = () => {
    setRefreshing(true);
    const jobs = turns.filter((item) => item.turnId && item.status === "ready").map((item) =>
      fetchEmotionTrace(item.turnId as string).then((trace) => {
        if (trace) onTurn({ ...item, ...trace, conversationId: trace.conversation_id, turnIndex: trace.turn_index });
      }).catch(() => undefined));
    void Promise.all(jobs).then(() => { setRated(null); setRefreshed((value) => value + 1); loadEvaluation(); })
      .finally(() => setRefreshing(false));
  };
  const needsTurn = (
    <Empty title="No conversation yet">
      Send a chat message first. This section shows real measurements of a conversation turn; nothing is shown until one exists.
    </Empty>
  );
  const noRecord = (
    <Empty title="No emotion record yet">
      The analysis of the turns sent so far is unavailable. Send another message or use Refresh data.
    </Empty>
  );

  return (
    <section className="mood-page" aria-label="Emotion and neuron steering analytics">
      <div className="demo-notice">
        <Info size={19} />
        <div>
          <strong>Emotion &amp; neuron steering analytics</strong>
          <p>
            Every number is labelled by where it comes from: <em>measured</em> inside the model, <em>classifier-estimated</em>,{" "}
            <em>LLM-judged</em> in saved experiments, <em>user-reported</em>, or <em>not available</em>. Changes in the user's detected
            emotion are observations, not effects of the replies. This is a research view, not a clinical assessment.
          </p>
        </div>
        <span className="badge">LIVE</span>
      </div>

      <div className="tabs" role="tablist" aria-label="Dashboard sections">
        {TABS.map(([id, name]) => (
          <button key={id} role="tab" id={`tab-${id}`} aria-selected={tab === id} aria-controls="mood-panel"
            className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
            {name}
          </button>
        ))}
        <button type="button" className="ghost-button refresh" onClick={refresh} disabled={refreshing}
          title="Re-read the turn records, offline judge results and saved experiment reports">
          <RefreshCw size={13} aria-hidden="true" /> {refreshing ? "Refreshing…" : "Refresh data"}
        </button>
      </div>

      {PER_TURN.includes(tab) && turn && (
        <>
          <div className="turn-picker" role="group" aria-label="Conversation turn">
            {turns.map((item, index) => (
              <button key={item.id} aria-pressed={item.id === turn.id} className={item.id === turn.id ? "active" : ""} onClick={() => setPicked(item.id)}>
                Turn {index + 1}
                <span>{item.plan?.user_emotion ?? item.preview?.emotion ?? "…"}</span>
                {item.decision && <span className={`dot ${item.decision.status}`} title={label(item.decision.status)} />}
              </button>
            ))}
          </div>
          {tab === "neurons" && <p className="turn-quote">“{turn.userText}”</p>}
        </>
      )}

      <div id="mood-panel" role="tabpanel" aria-labelledby={`tab-${tab}`} data-tab={tab}>
        {PER_TURN.includes(tab) && turn && !turn.plan && (
          <article className="mood-card">
            <p className="muted" role="status">
              {turn.status === "analysing" ? "Loading this turn's analysis…" : "The analysis for this turn is unavailable."}
              {turn.preview?.steeringEmotion && ` Steering applied: ${label(turn.preview.steeringEmotion)}, α ${fixed(turn.preview.alpha)}.`}
            </p>
          </article>
        )}
        {tab === "overview" && (
          <OverviewSection summary={summary} trajectory={trajectory} research={research} models={evaluation} onNavigate={setTab} />
        )}
        {tab === "timeline" && (trajectory
          ? <TimelineSection trajectory={trajectory} canRate={Boolean(conversation)} onFeedback={feedback} />
          : analysing ? <Loading what="the conversation timeline" /> : turns.length ? noRecord : needsTurn)}
        {tab === "comparison" && (turn
          ? turn.plan && <ImpactSection turn={turn} turnNumber={turnNumber} trajectory={trajectory} research={research} onToggleCompare={toggleCompare} toggling={toggling} />
          : needsTurn)}
        {tab === "neurons" && (turn
          ? turn.plan && <NeuronSection turn={turn} research={research} />
          : researchState === "loading" ? <Loading what="the model's panels" />
            : research ? <NeuronSection turn={{ id: "none", status: "ready", userText: "" }} research={research} /> : needsTurn)}
        {tab === "causal" && (researchState === "loading"
          ? <Loading what="the causal experiments" />
          : <CausalSection research={research} turn={turn ?? { id: "none", status: "ready", userText: "" }} />)}
        {tab === "models" && (evaluation
          ? <ModelSection evaluation={evaluation} />
          : evaluationError ? <Failure what="The saved model results" error={evaluationError} onRetry={() => loadEvaluation()} />
            : <Loading what="the saved model results" />)}
      </div>
      <p className="mood-footnote">
        A space for reflection, not a clinical assessment. MindMate is an AI, not a therapist.
      </p>
    </section>
  );
}
