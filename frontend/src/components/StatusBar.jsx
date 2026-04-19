import { motion as Motion } from "framer-motion";

function StatusBar({ ledStatus, connected }) {
  return (
    <section className="glass-panel rounded-3xl border border-emerald-200/20 p-6">
      <div className="flex items-center justify-between gap-4 text-sm uppercase tracking-[0.24em] text-slate-200/80">
        <div className="flex items-center gap-2">
          <Motion.span
            className={`h-3 w-3 rounded-full ${
              ledStatus ?
                "bg-emerald-400 shadow-[0_0_18px_rgba(74,222,128,0.95)]"
              : "bg-slate-500 shadow-none"
            }`}
            animate={ledStatus ? { scale: [1, 1.2, 1] } : { scale: 1 }}
            transition={{ repeat: ledStatus ? Infinity : 0, duration: 1.35 }}
          />
          <span>{ledStatus ? "LED ACTIVE" : "LED INACTIVE"}</span>
        </div>

        <span className={connected ? "text-cyan-300" : "text-rose-300"}>
          {connected ? "LINK STABLE" : "OFFLINE"}
        </span>
      </div>
    </section>
  );
}

export default StatusBar;
