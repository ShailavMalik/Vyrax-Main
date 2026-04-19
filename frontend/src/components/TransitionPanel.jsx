import { motion as Motion } from "framer-motion";

function TransitionPanel({ previousEmotion, currentEmotion }) {
  const fromEmotion = (previousEmotion || "neutral").toUpperCase();
  const toEmotion = (currentEmotion || "neutral").toUpperCase();

  return (
    <section className="glass-panel rounded-2xl border border-slate-200/20 p-4 lg:p-5">
      <p className="text-[10px] uppercase tracking-[0.26em] text-slate-300/75">Transition</p>
      <div className="mt-2 flex items-center gap-2 text-sm font-semibold tracking-[0.1em] text-slate-200 lg:text-base">
        <span className="truncate">{fromEmotion}</span>
        <Motion.span
          className="text-cyan-300"
          animate={{ x: [0, 4, 0], opacity: [0.8, 1, 0.8] }}
          transition={{ duration: 1.2, repeat: Infinity }}>
          →
        </Motion.span>
        <span className="truncate text-cyan-200">{toEmotion}</span>
      </div>
    </section>
  );
}

export default TransitionPanel;
