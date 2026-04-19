function TransitionIndicator({ previousEmotion, currentEmotion }) {
  const fromEmotion =
    previousEmotion ? previousEmotion.toUpperCase() : "UNKNOWN";
  const toEmotion = currentEmotion ? currentEmotion.toUpperCase() : "NEUTRAL";

  return (
    <section className="glass-panel rounded-3xl border border-slate-200/20 p-6">
      <p className="text-xs uppercase tracking-[0.3em] text-slate-300/65">
        Transition
      </p>
      <p className="mt-3 font-heading text-xl tracking-[0.12em] text-slate-100/90 sm:text-2xl">
        {fromEmotion} <span className="text-cyan-300">→</span> {toEmotion}
      </p>
    </section>
  );
}

export default TransitionIndicator;
