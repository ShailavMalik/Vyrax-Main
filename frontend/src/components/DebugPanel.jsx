function DebugPanel({ debug }) {
  const pipeline = debug?.pipelineSteps || [];

  return (
    <section className="glass-panel rounded-2xl border border-violet-300/30 bg-violet-500/10 p-4 lg:p-5">
      <p className="text-[10px] uppercase tracking-[0.28em] text-violet-200/85">Model Debug Panel</p>

      <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-slate-300">
        <div className="rounded-lg border border-violet-200/20 bg-slate-900/45 p-2">
          <p className="text-slate-400">Model</p>
          <p className="mt-1 font-semibold text-violet-100">{debug.modelName}</p>
        </div>
        <div className="rounded-lg border border-violet-200/20 bg-slate-900/45 p-2">
          <p className="text-slate-400">Version</p>
          <p className="mt-1 font-semibold text-violet-100">{debug.version}</p>
        </div>
        <div className="rounded-lg border border-violet-200/20 bg-slate-900/45 p-2">
          <p className="text-slate-400">FPS</p>
          <p className="mt-1 font-semibold text-violet-100">{debug.fps}</p>
        </div>
        <div className="rounded-lg border border-violet-200/20 bg-slate-900/45 p-2">
          <p className="text-slate-400">Detection</p>
          <p className="mt-1 font-semibold text-violet-100">{Math.round(debug.detectionConfidence * 100)}%</p>
        </div>
      </div>

      <div className="mt-3 rounded-lg border border-violet-200/20 bg-slate-900/45 p-3">
        <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Pipeline</p>
        <p className="mt-1 text-xs text-slate-200">{pipeline.join(" -> ")}</p>
      </div>

      <div className="mt-2 rounded-lg border border-violet-200/20 bg-slate-900/45 p-3">
        <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Decision</p>
        <p className="mt-1 text-xs text-slate-200">{debug.decisionExplanation}</p>
      </div>
    </section>
  );
}

export default DebugPanel;
