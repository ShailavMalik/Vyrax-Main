import { useCallback, useEffect, useRef, useState } from "react";
import { motion as Motion } from "framer-motion";
import { useEmotionState } from "./hooks/useEmotionState";
import CameraFeed from "./components/CameraFeed";
import ConfidenceBar from "./components/ConfidenceBar";
import EmotionPanel from "./components/EmotionPanel";
import TransitionPanel from "./components/TransitionPanel";
import MiniGraph from "./components/MiniGraph";
import ModeToggle from "./components/ModeToggle";
import InsightsPanel from "./components/InsightsPanel";
import EmotionGraph from "./components/EmotionGraph";
import SessionSummary from "./components/SessionSummary";

const API_URL = "http://localhost:8000/emotion";
const NORMAL_POLL_INTERVAL = 350;

function App() {
  const {
    currentEmotion,
    previousEmotion,
    confidence,
    emotionHistory,
    hasFace,
    error,
    modelTelemetry,
    sessionStartTime,
    updateEmotionData,
    setError,
    clearHistory,
  } = useEmotionState();
  const [connected, setConnected] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const pollIntervalRef = useRef(null);

  // Fetch emotion data from backend in normal mode
  const fetchEmotionNormal = useCallback(async () => {
    try {
      const response = await fetch(API_URL);
      if (!response.ok) {
        throw new Error(`Fetch failed with status ${response.status}`);
      }

      const payload = await response.json();
      setConnected(true);
      updateEmotionData(payload);
    } catch (error) {
      setConnected(false);
      if (cameraEnabled) {
        setError("Connection lost");
      }
    }
  }, [updateEmotionData, cameraEnabled, setError]);

  // Start normal polling loop
  useEffect(() => {
    if (!cameraEnabled) {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
      setError("");
      setConnected(true);
      return;
    }

    fetchEmotionNormal();
    pollIntervalRef.current = setInterval(
      fetchEmotionNormal,
      NORMAL_POLL_INTERVAL,
    );

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [cameraEnabled, fetchEmotionNormal, setError]);

  const handleClearSession = () => {
    clearHistory();
  };

  return (
    <div className="relative min-h-screen overflow-x-hidden px-3 py-3 text-slate-100 sm:px-5 lg:px-6 lg:py-4">
      <div className="aurora-bg pointer-events-none absolute inset-0" />
      <div className="scanline-overlay" />

      <Motion.div
        className="pointer-events-none absolute -left-16 top-24 h-52 w-52 rounded-full bg-cyan-400/20 blur-3xl"
        animate={{ y: [0, -16, 0], opacity: [0.45, 0.75, 0.45] }}
        transition={{ duration: 7.2, repeat: Infinity, ease: "easeInOut" }}
      />
      <Motion.div
        className="pointer-events-none absolute -right-12 bottom-16 h-60 w-60 rounded-full bg-violet-500/20 blur-3xl"
        animate={{ y: [0, 12, 0], opacity: [0.4, 0.72, 0.4] }}
        transition={{ duration: 8.5, repeat: Infinity, ease: "easeInOut" }}
      />

      <Motion.main
        className="relative z-10 mx-auto flex max-w-[1720px] flex-col gap-4"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.65, ease: "easeOut" }}>
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="font-body text-[11px] uppercase tracking-[0.5em] text-cyan-200/70">
              Vyra-X Neural Console
            </p>
            <h1 className="cyber-title mt-1 font-heading text-2xl uppercase sm:text-3xl lg:text-4xl">
              AI Emotion Cockpit
            </h1>
            <p className="mt-1 text-[10px] uppercase tracking-[0.28em] text-cyan-100/55">
              Real-time affect telemetry
            </p>
          </div>
          <ModeToggle
            cameraEnabled={cameraEnabled}
            onToggleCamera={setCameraEnabled}
          />
        </header>

        <section className="grid flex-1 min-h-0 gap-4 lg:grid-cols-[1.5fr_1fr]">
          <div className="min-h-0">
            <CameraFeed
              cameraEnabled={cameraEnabled}
              hasFace={hasFace}
              currentEmotion={currentEmotion}
              confidence={confidence}
              cameraError={!cameraEnabled ? "" : error}
              modelTelemetry={modelTelemetry}
            />
          </div>

          <div className="flex min-h-0 flex-col gap-2.5 lg:gap-3">
            <EmotionPanel
              emotion={currentEmotion}
              cameraEnabled={cameraEnabled}
              hasFace={hasFace}
            />

            <ConfidenceBar confidence={confidence} compact />

            <TransitionPanel
              previousEmotion={previousEmotion}
              currentEmotion={currentEmotion}
            />

            <MiniGraph emotionHistory={emotionHistory} />

            <InsightsPanel
              emotionHistory={emotionHistory}
              compact
              connected={connected}
              ledEmotion={currentEmotion}
            />
          </div>
        </section>

        <section className="grid gap-4 pb-6 pt-1 lg:grid-cols-[1.25fr_0.75fr]">
          <EmotionGraph emotionHistory={emotionHistory} />
          <SessionSummary
            emotionHistory={emotionHistory}
            sessionStartTime={sessionStartTime}
            onClearSession={handleClearSession}
          />
        </section>
      </Motion.main>
    </div>
  );
}

export default App;
