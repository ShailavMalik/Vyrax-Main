import { useEffect, useState } from "react";
import { motion as Motion } from "framer-motion";

function CameraFeed({ cameraEnabled, hasFace, currentEmotion, confidence, cameraError }) {
  const emotionLabel = (currentEmotion || "neutral").toUpperCase();
  const [streamErrored, setStreamErrored] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    if (!cameraEnabled) {
      setStreamErrored(false);
      return;
    }

    if (!streamErrored) {
      return;
    }

    const retry = window.setTimeout(() => {
      setReloadKey((prev) => prev + 1);
      setStreamErrored(false);
    }, 2000);

    return () => {
      window.clearTimeout(retry);
    };
  }, [cameraEnabled, streamErrored]);

  return (
    <section className="glass-panel relative h-full overflow-hidden rounded-3xl border border-cyan-300/25 p-3 lg:p-4">
      <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-[0.28em] text-cyan-200/75 lg:mb-3">
        <span>Camera Feed</span>
        <span className={`h-2 w-2 rounded-full ${cameraEnabled ? "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.85)]" : "bg-slate-500"}`} />
      </div>

      <div className="relative h-[42vh] min-h-[260px] overflow-hidden rounded-2xl border border-cyan-100/20 bg-slate-950/50 lg:h-[56vh]">
        {cameraEnabled ? (
          <>
            {!streamErrored && (
              <img
                src={`http://localhost:8000/video_feed?reload=${reloadKey}`}
                alt="Live camera feed"
                className="h-full w-full object-cover"
                onError={() => setStreamErrored(true)}
                onLoad={() => setStreamErrored(false)}
              />
            )}

            {streamErrored && (
              <div className="flex h-full items-center justify-center bg-gradient-to-br from-slate-900 to-slate-950 text-xs uppercase tracking-[0.2em] text-slate-400">
                Reconnecting camera feed...
              </div>
            )}

            {hasFace === true && (
              <Motion.div
                key={emotionLabel}
                initial={{ opacity: 0.35, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.25 }}
                className="absolute right-3 top-3 rounded-full border border-cyan-300/35 bg-slate-950/75 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-100 backdrop-blur-sm lg:text-xs">
                {emotionLabel} · {Math.round(confidence * 100)}%
              </Motion.div>
            )}

            {hasFace === false && (
              <div className="absolute right-3 top-3 rounded-full border border-amber-300/35 bg-amber-950/45 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-amber-200 backdrop-blur-sm lg:text-xs">
                No Face Detected
              </div>
            )}
          </>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
            <p className="text-sm uppercase tracking-[0.3em] text-slate-300">Camera Disabled</p>
            <p className="text-xs text-slate-400">Turn camera on to resume live detection.</p>
          </div>
        )}

        {!!cameraError && cameraEnabled && (
          <div className="absolute bottom-3 left-3 rounded-full border border-rose-300/35 bg-rose-950/55 px-3 py-1 text-[10px] uppercase tracking-[0.15em] text-rose-200">
            {cameraError}
          </div>
        )}
      </div>
    </section>
  );
}

export default CameraFeed;
