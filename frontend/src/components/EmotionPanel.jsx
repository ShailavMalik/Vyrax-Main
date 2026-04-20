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
    <section className="glass-panel panel-chrome relative flex-none overflow-hidden rounded-2xl border border-cyan-200/25 p-3 min-h-[96px] lg:min-h-[108px] lg:p-4">
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
            className="mt-1.5 flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-1 pb-0.5">
            <span className="text-2xl leading-none lg:text-3xl">
              {display.icon}
            </span>
            <h2
              className="min-w-0 max-w-full break-words pr-1 font-heading text-[clamp(1.05rem,1.9vw,1.7rem)] leading-[1.15] tracking-[0.07em] sm:text-[clamp(1.2rem,2vw,1.95rem)]"
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
