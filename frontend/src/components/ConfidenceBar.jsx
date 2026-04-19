import { motion as Motion } from "framer-motion";

function ConfidenceBar({ confidence, compact = false }) {
  const clamped =
    Number.isFinite(confidence) ? Math.max(0, Math.min(confidence, 1)) : 0;
  const percentage = Math.round(clamped * 100);

  return (
    <section
      className={`glass-panel rounded-2xl border border-indigo-200/25 ${compact ? "p-4" : "p-6"}`}>
      <div className={`flex items-center justify-between ${compact ? "mb-2" : "mb-3"}`}>
        <p className={`uppercase text-indigo-200/70 ${compact ? "text-[10px] tracking-[0.26em]" : "text-xs tracking-[0.34em]"}`}>
          Confidence
        </p>
        <p className={`${compact ? "text-base" : "text-lg"} font-semibold text-indigo-100`}>
          {percentage}%
        </p>
      </div>

      <div className={`${compact ? "h-3" : "h-4"} overflow-hidden rounded-full border border-indigo-300/25 bg-slate-900/80`}>
        <Motion.div
          className="h-full origin-left rounded-full bg-gradient-to-r from-cyan-300 via-blue-400 to-violet-400 shadow-[0_0_20px_rgba(96,165,250,0.7)]"
          initial={false}
          animate={{ scaleX: clamped }}
          transition={{ duration: 0.45, ease: "easeOut" }}
        />
      </div>
    </section>
  );
}

export default ConfidenceBar;
