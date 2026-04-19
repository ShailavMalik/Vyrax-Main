import { AnimatePresence, motion as Motion } from "framer-motion";
import { getEmotionColor } from "../utils/emotionUtils";

const emotionMap = {
  happy: { label: "HAPPY", icon: "😊" },
  sad: { label: "SAD", icon: "💙" },
  angry: { label: "ANGRY", icon: "😤" },
  neutral: { label: "NEUTRAL", icon: "😐" },
  confused: { label: "CONFUSED", icon: "🤔" },
  surprised: { label: "SURPRISED", icon: "😲" },
  fear: { label: "FEAR", icon: "😨" },
  disgust: { label: "DISGUST", icon: "😒" },
};

function EmotionPanel({ emotion, cameraEnabled, hasFace }) {
  const emotionKey = (emotion || "neutral").toLowerCase();

  const isCameraOff = !cameraEnabled;
  const isDetectingFace = cameraEnabled && hasFace !== true;

  const displayKey =
    isCameraOff ? "camera-off"
    : isDetectingFace ? "detecting-face"
    : emotionKey;

  const display =
    isCameraOff ? { label: "CAMERA OFF", icon: "📷" }
    : isDetectingFace ? { label: "DETECTING FACE...", icon: "🔎" }
    : emotionMap[emotionKey] || emotionMap.neutral;

  const glowColor =
    isCameraOff ? "#94a3b8"
    : isDetectingFace ? "#60a5fa"
    : getEmotionColor(emotionKey);

  return (
    <section className="glass-panel relative overflow-hidden rounded-2xl border border-cyan-200/25 p-4 lg:p-5">
      <Motion.div
        className="pointer-events-none absolute inset-0 opacity-25 blur-2xl"
        style={{ backgroundColor: glowColor }}
        animate={{ opacity: [0.15, 0.3, 0.15] }}
        transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
      />

      <div className="relative z-10">
        <p className="text-[10px] uppercase tracking-[0.28em] text-cyan-200/75">
          Current Emotion
        </p>
        <AnimatePresence mode="wait">
          <Motion.div
            key={displayKey}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
            className="mt-2 flex items-end gap-3">
            <span className="text-3xl lg:text-4xl">{display.icon}</span>
            <h2
              className="font-heading text-3xl leading-none tracking-[0.12em] lg:text-4xl"
              style={{
                color: glowColor,
                textShadow: `0 0 18px ${glowColor}90`,
              }}>
              {display.label}
            </h2>
          </Motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}

export default EmotionPanel;
