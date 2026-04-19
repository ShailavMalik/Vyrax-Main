import { motion as Motion } from "framer-motion";

function CameraView({ demoMode, cameraReady, cameraError }) {
  const statusText =
    cameraError ? cameraError
    : cameraReady ? "Camera stream live"
    : "Waiting for camera stream...";

  return (
    <Motion.section
      className={`glass-panel relative overflow-hidden rounded-3xl border p-4 transition-all duration-500 ${
        demoMode ?
          "border-violet-300/60 shadow-[0_0_45px_rgba(167,139,250,0.45)]"
        : "border-cyan-400/30 shadow-[0_0_28px_rgba(34,211,238,0.25)]"
      }`}
      whileHover={{ scale: 1.01, y: -4 }}
      transition={{ type: "spring", stiffness: 180, damping: 20 }}>
      <div className="mb-3 flex items-center justify-between text-xs uppercase tracking-[0.3em] text-cyan-200/75">
        <span>Live Optic Feed</span>
        <div className="flex items-center gap-2">
          {demoMode && (
            <span className="flex items-center gap-1 rounded-full bg-violet-500/20 px-2 py-1 text-violet-300">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-400" />
              <span className="text-[9px]">DEMO</span>
            </span>
          )}
          <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.85)]" />
        </div>
      </div>

      <div className="relative overflow-hidden rounded-2xl border border-cyan-200/20 bg-slate-950/40">
        <img
          src="http://localhost:8000/video_feed"
          alt="Live camera feed"
          className="h-[340px] w-full object-cover sm:h-[420px] lg:h-[560px]"
        />
        <div className="absolute left-4 top-4 rounded-full border border-cyan-200/30 bg-slate-950/75 px-3 py-1 text-[11px] uppercase tracking-[0.28em] text-cyan-100 backdrop-blur-sm">
          {statusText}
        </div>
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.25),transparent_58%)]" />
      </div>
    </Motion.section>
  );
}

export default CameraView;
