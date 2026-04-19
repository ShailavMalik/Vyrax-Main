import { AnimatePresence, motion as Motion } from "framer-motion";
import { getEmotionColor } from "../utils/emotionUtils";

const emotionMap = {
  happy: {
    label: "HAPPY 😊",
    color: "text-yellow-300",
    feedback: "Positive energy detected",
  },
  sad: {
    label: "SAD 💙",
    color: "text-sky-300",
    feedback: "Showing signs of dissatisfaction",
  },
  angry: {
    label: "ANGRY 😤",
    color: "text-rose-300",
    feedback: "Take a deep breath",
  },
  surprise: {
    label: "SURPRISE 😲",
    color: "text-cyan-300",
    feedback: "High engagement detected",
  },
  neutral: {
    label: "NEUTRAL 😐",
    color: "text-cyan-300",
    feedback: "Focused and steady",
  },
  confused: {
    label: "CONFUSED 🤔",
    color: "text-purple-300",
    feedback: "User appears uncertain",
  },
  fear: {
    label: "FEAR 😨",
    color: "text-orange-300",
    feedback: "Heightened stress levels",
  },
  disgust: {
    label: "DISGUST 😒",
    color: "text-lime-300",
    feedback: "Strong negative reaction",
  },
};

function EmotionDisplay({ emotion, confidence }) {
  const emotionKey = (emotion || "neutral").toLowerCase();
  const config = emotionMap[emotionKey] || emotionMap.neutral;
  const glowColor = getEmotionColor(emotionKey);

  return (
    <section className="glass-panel relative overflow-hidden rounded-3xl border border-blue-200/25 p-6 sm:p-8">
      {/* Emotion-based glow background */}
      <Motion.div
        className="pointer-events-none absolute -inset-4 rounded-3xl opacity-20 blur-2xl"
        style={{ backgroundColor: glowColor }}
        animate={{
          opacity: [0.15, 0.25, 0.15],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      <div className="relative z-10">
        <p className="text-xs uppercase tracking-[0.38em] text-cyan-200/70">
          Current Emotion
        </p>

        <AnimatePresence mode="wait">
          <Motion.h2
            key={emotionKey}
            initial={{ opacity: 0, scale: 0.84, y: 18 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 1.08, y: -18 }}
            transition={{ duration: 0.35, ease: "easeOut" }}
            className={`mt-4 text-4xl font-black tracking-[0.15em] sm:text-5xl ${config.color}`}
            style={{
              textShadow: `0 0 20px ${glowColor}80`,
            }}>
            {config.label}
          </Motion.h2>
        </AnimatePresence>

        {/* Confidence percentage */}
        <Motion.div
          className="mt-3 flex items-center gap-2"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.15, duration: 0.3 }}>
          <span className="text-xs text-slate-400">Confidence:</span>
          <span className="text-lg font-bold text-blue-300">
            {Math.round(confidence * 100)}%
          </span>
        </Motion.div>

        <Motion.p
          key={`${emotionKey}-feedback`}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          className="mt-5 text-base text-slate-200/85">
          {config.feedback}
        </Motion.p>
      </div>
    </section>
  );
}

export default EmotionDisplay;
