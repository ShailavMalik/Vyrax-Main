import { motion as Motion } from "framer-motion";

function PillButton({ active, onClick, label }) {
  return (
    <Motion.button
      type="button"
      whileHover={{ y: -1 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`rounded-full border px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] transition-all lg:text-xs ${
        active
          ? "border-cyan-300/60 bg-cyan-300/15 text-cyan-100 shadow-[0_0_14px_rgba(34,211,238,0.45)]"
          : "border-slate-300/20 bg-slate-900/35 text-slate-300 hover:border-slate-300/40"
      }`}>
      {label}
    </Motion.button>
  );
}

function ModeToggle({ demoMode, cameraEnabled, onToggleMode, onToggleCamera }) {
  return (
    <section className="glass-panel rounded-2xl border border-cyan-200/20 p-3 lg:p-4">
      <div className="flex flex-wrap gap-2">
        <PillButton
          active={!demoMode}
          onClick={() => onToggleMode(false)}
          label="Normal"
        />
        <PillButton
          active={demoMode}
          onClick={() => onToggleMode(true)}
          label="Demo"
        />
        <PillButton
          active={cameraEnabled}
          onClick={() => onToggleCamera(!cameraEnabled)}
          label={cameraEnabled ? "Camera On" : "Camera Off"}
        />
      </div>
    </section>
  );
}

export default ModeToggle;
