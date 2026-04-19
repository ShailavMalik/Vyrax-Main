import { motion as Motion } from "framer-motion";

function ToggleButton({ active, label, onClick }) {
  return (
    <Motion.button
      type="button"
      whileHover={{ y: -2, scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`rounded-full border px-4 py-2 text-sm font-semibold uppercase tracking-[0.14em] transition-all duration-300 ${
        active ?
          "border-cyan-300/60 bg-cyan-300/15 text-cyan-100 shadow-[0_0_18px_rgba(34,211,238,0.65)]"
        : "border-slate-400/25 bg-slate-900/30 text-slate-300 hover:border-slate-300/45 hover:text-slate-100"
      }`}>
      {label}
    </Motion.button>
  );
}

function DemoModeToggle({ demoMode, onToggle }) {
  return (
    <section className="glass-panel rounded-3xl border border-cyan-200/20 p-5">
      <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/70">
        Operating Mode
      </p>
      <div className="mt-4 flex flex-wrap gap-3">
        <ToggleButton
          active={!demoMode}
          label="Normal Mode"
          onClick={() => onToggle(false)}
        />
        <ToggleButton
          active={demoMode}
          label="Demo Mode"
          onClick={() => onToggle(true)}
        />
      </div>
    </section>
  );
}

export default DemoModeToggle;
