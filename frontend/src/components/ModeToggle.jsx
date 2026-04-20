import { motion as Motion } from "framer-motion";

function PillButton({ active, onClick, label, compact = false }) {
  return (
    <Motion.button
      type="button"
      whileHover={{ y: -1, scale: 1.01 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`rounded-full border font-semibold uppercase transition-all ${compact ? "px-2.5 py-1 text-[9px] tracking-[0.13em] lg:text-[10px]" : "px-3 py-1.5 text-[10px] tracking-[0.16em] lg:text-xs"} ${
        active ?
          "border-cyan-300/60 bg-gradient-to-r from-cyan-400/18 to-blue-400/18 text-cyan-50 shadow-[0_0_18px_rgba(34,211,238,0.42)]"
        : "border-slate-300/20 bg-slate-900/35 text-slate-300 hover:border-cyan-200/35"
      }`}>
      {label}
    </Motion.button>
  );
}

function ModeToggle({ cameraEnabled, onToggleCamera }) {
  return (
    <section className="glass-panel panel-chrome rounded-2xl border border-cyan-200/20 p-2.5 lg:p-3">
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center gap-2 rounded-full border border-cyan-300/55 bg-cyan-300/15 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-100 lg:text-xs">
          <span className="h-1.5 w-1.5 rounded-full bg-cyan-200 shadow-[0_0_10px_rgba(125,211,252,0.95)]" />
          Live
        </span>
        <PillButton
          active={cameraEnabled}
          onClick={() => onToggleCamera(!cameraEnabled)}
          label={cameraEnabled ? "Cam On" : "Cam Off"}
          compact
        />
      </div>
    </section>
  );
}

export default ModeToggle;
