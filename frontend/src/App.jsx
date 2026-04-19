import { useCallback, useEffect, useRef, useState } from "react";
import { motion as Motion } from "framer-motion";
import { useEmotionState } from "./hooks/useEmotionState";
import { generateDemoData, buildDemoDebugData } from "./utils/emotionUtils";
import CameraFeed from "./components/CameraFeed";
import ConfidenceBar from "./components/ConfidenceBar";
import EmotionPanel from "./components/EmotionPanel";
import TransitionPanel from "./components/TransitionPanel";
import MiniGraph from "./components/MiniGraph";
import ModeToggle from "./components/ModeToggle";
import DebugPanel from "./components/DebugPanel";
import EmotionGraph from "./components/EmotionGraph";
import InsightsPanel from "./components/InsightsPanel";
import SessionSummary from "./components/SessionSummary";

const API_URL = "http://localhost:8000/emotion";
const NORMAL_POLL_INTERVAL = 350;
const DEMO_POLL_INTERVAL = 450;

function App() {
  const {
    currentEmotion,
    previousEmotion,
    confidence,
    emotionHistory,
    hasFace,
    error,
    isDemo,
    sessionStartTime,
    updateEmotionData,
    setDemoMode,
    setError,
    clearHistory,
  } = useEmotionState();
  const [connected, setConnected] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [demoDebug, setDemoDebug] = useState(() => buildDemoDebugData("neutral", 0));
  const demoSequenceIndexRef = useRef(0);
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

  // Generate demo data in demo mode
  const fetchEmotionDemo = useCallback(() => {
    try {
      const demoData = generateDemoData(demoSequenceIndexRef.current);
      demoSequenceIndexRef.current = (demoSequenceIndexRef.current + 1) % 8; // Demo sequence cycles every 8 items
      setConnected(true);
      updateEmotionData(demoData);
      setDemoDebug(buildDemoDebugData(demoData.emotion, demoData.confidence));
    } catch (error) {
      setError("Demo mode error");
    }
  }, [updateEmotionData, setError]);

  // Start polling with the appropriate interval
  useEffect(() => {
    if (!cameraEnabled && !isDemo) {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
      setError("");
      setConnected(true);
      return;
    }

    if (isDemo) {
      fetchEmotionDemo();
      pollIntervalRef.current = setInterval(
        fetchEmotionDemo,
        DEMO_POLL_INTERVAL,
      );
    } else {
      fetchEmotionNormal();
      pollIntervalRef.current = setInterval(
        fetchEmotionNormal,
        NORMAL_POLL_INTERVAL,
      );
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [cameraEnabled, isDemo, fetchEmotionNormal, fetchEmotionDemo, setError]);

  const handleToggleDemoMode = (newMode) => {
    setDemoMode(newMode);
  };

  const handleClearSession = () => {
    clearHistory();
  };

  const isDemoMode = isDemo;

  return (
    <div className="relative min-h-screen overflow-x-hidden px-3 py-3 text-slate-100 sm:px-5 lg:px-6 lg:py-4">
      <div className="aurora-bg pointer-events-none absolute inset-0" />

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
            <p className="font-body text-xs uppercase tracking-[0.45em] text-cyan-200/75">
              Vyra-X Neural Console
            </p>
            <h1 className="mt-1 font-heading text-2xl uppercase tracking-[0.12em] text-slate-100 sm:text-3xl lg:text-4xl">
              AI Emotion Cockpit
            </h1>
          </div>
          <ModeToggle
            demoMode={isDemoMode}
            cameraEnabled={cameraEnabled}
            onToggleMode={handleToggleDemoMode}
            onToggleCamera={setCameraEnabled}
          />
        </header>

        <section className="grid h-[calc(100vh-140px)] min-h-[620px] gap-4 overflow-hidden lg:grid-cols-[1.5fr_1fr]">
          <div className="h-full min-h-0">
            <CameraFeed
              cameraEnabled={cameraEnabled}
              hasFace={hasFace}
              currentEmotion={currentEmotion}
              confidence={confidence}
              cameraError={!cameraEnabled ? "" : error}
            />
          </div>

          <div className="flex h-full min-h-0 flex-col gap-3 overflow-hidden">
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
            />

            {isDemoMode && <DebugPanel debug={demoDebug} />}
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
