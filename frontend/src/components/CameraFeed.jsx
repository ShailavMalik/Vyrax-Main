import { useEffect, useState } from "react";
import { motion as Motion } from "framer-motion";

function CameraFeed({
  cameraEnabled,
  hasFace,
  currentEmotion,
  confidence,
  cameraError,
  modelTelemetry,
}) {
  const emotionLabel = (currentEmotion || "neutral").toUpperCase();
  const [streamErrored, setStreamErrored] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  const fpsDisplay =
    Number.isFinite(modelTelemetry?.fps) ? modelTelemetry.fps.toFixed(1) : "--";
  const decisionSource = (modelTelemetry?.decisionSource || "N/A").replaceAll(
    "_",
    " ",
  );
  const geometryReason = (modelTelemetry?.geometryReason || "N/A").replaceAll(
    "_",
    " ",
  );
  const snapshotTakenAt = Number(modelTelemetry?.snapshotTakenAt);
  const snapshotTaken =
    Boolean(modelTelemetry?.snapshotTaken) && Number.isFinite(snapshotTakenAt) ?
      Date.now() - snapshotTakenAt < 1800
    : false;
  const snapshotFlashKey = Number.isFinite(snapshotTakenAt) ? snapshotTakenAt : 0;
  const snapshotReason = (modelTelemetry?.snapshotReason || "").replaceAll(
    "_",
    " ",
  );
  const topScoreEmotion = modelTelemetry?.topScoreEmotion || "N/A";
  const topScoreValue =
    Number.isFinite(modelTelemetry?.topScoreValue) ?
      `${Math.round(modelTelemetry.topScoreValue * 100)}%`
    : "--";
  const triggerSummary =
    (
      Array.isArray(modelTelemetry?.ruleTriggers) &&
      modelTelemetry.ruleTriggers.length > 0
    ) ?
      modelTelemetry.ruleTriggers.join(" | ")
    : "No active rule triggers";

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
    <section className="glass-panel panel-chrome relative flex h-full min-h-0 flex-col overflow-hidden rounded-3xl border border-cyan-300/25 p-3 lg:p-4">
      <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-[0.28em] text-cyan-200/75 lg:mb-3">
        <span>Camera Feed</span>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[9px] tracking-[0.18em] ${snapshotTaken ? "border-amber-300/60 bg-amber-500/20 text-amber-100" : "border-slate-500/40 bg-slate-800/45 text-slate-300"}`}>
            <span
              className={`h-1.5 w-1.5 rounded-full ${snapshotTaken ? "bg-amber-300" : "bg-slate-500"}`}
            />
            SNAP
          </span>
          <span
            className={`h-2 w-2 rounded-full ${cameraEnabled ? "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.85)]" : "bg-slate-500"}`}
          />
        </div>
      </div>

      <div className="relative h-[42vh] min-h-[260px] overflow-hidden rounded-2xl border border-cyan-100/20 bg-slate-950/50 lg:h-[56vh]">
        {cameraEnabled ?
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

            {snapshotTaken && (
              <Motion.div
                key={`snapshot-flash-${snapshotFlashKey}`}
                initial={{ opacity: 0.56 }}
                animate={{ opacity: [0.56, 0.24, 0.08, 0] }}
                transition={{ duration: 0.55, ease: "easeOut" }}
                className="pointer-events-none absolute inset-0 bg-cyan-100"
              />
            )}

            {snapshotTaken && (
              <Motion.div
                key={`snapshot-dim-${snapshotFlashKey}`}
                initial={{ opacity: 0.34 }}
                animate={{ opacity: [0.34, 0.2, 0] }}
                transition={{ duration: 0.9, ease: "easeOut" }}
                className="pointer-events-none absolute inset-0 bg-slate-950"
              />
            )}

            {snapshotTaken && (
              <Motion.div
                initial={{ opacity: 0, scale: 0.8, y: -8 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ duration: 0.18 }}
                className="absolute left-3 top-3 flex items-center gap-2 rounded-full border border-amber-300/45 bg-amber-950/70 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-amber-100 backdrop-blur-sm lg:text-xs">
                <span className="h-2 w-2 rounded-full bg-amber-300 shadow-[0_0_10px_rgba(252,211,77,0.95)]" />
                <svg
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                  className="h-3.5 w-3.5 text-amber-100"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round">
                  <path d="M4 8h3l1.6-2h6.8L17 8h3a1 1 0 0 1 1 1v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a1 1 0 0 1 1-1z" />
                  <circle cx="12" cy="13" r="3.2" />
                </svg>
                <span>Captured</span>
                {!!snapshotReason && (
                  <span className="text-amber-200/80">{snapshotReason}</span>
                )}
              </Motion.div>
            )}

            {hasFace === false && (
              <div className="absolute right-3 top-3 rounded-full border border-amber-300/35 bg-amber-950/45 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-amber-200 backdrop-blur-sm lg:text-xs">
                No Face Detected
              </div>
            )}
          </>
        : <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
            <p className="text-sm uppercase tracking-[0.3em] text-slate-300">
              Camera Disabled
            </p>
            <p className="text-xs text-slate-400">
              Turn camera on to resume live detection.
            </p>
          </div>
        }

        {!!cameraError && cameraEnabled && (
          <div className="absolute bottom-3 left-3 rounded-full border border-rose-300/35 bg-rose-950/55 px-3 py-1 text-[10px] uppercase tracking-[0.15em] text-rose-200">
            {cameraError}
          </div>
        )}
      </div>

      <div className="mt-3 grid flex-1 min-h-[112px] grid-cols-1 gap-2 text-[11px] sm:grid-cols-2 lg:text-xs">
        <div className="rounded-xl border border-cyan-200/20 bg-slate-900/55 p-2.5">
          <p className="text-[10px] uppercase tracking-[0.18em] text-cyan-200/70">
            Pipeline FPS
          </p>
          <p className="mt-1 font-heading text-base leading-tight tracking-[0.05em] text-cyan-100 tabular-nums sm:text-lg">
            {fpsDisplay}
          </p>
        </div>

        <div className="rounded-xl border border-cyan-200/20 bg-slate-900/55 p-2.5">
          <p className="text-[10px] uppercase tracking-[0.18em] text-cyan-200/70">
            Decision Source
          </p>
          <p className="mt-1 break-words font-semibold uppercase tracking-[0.06em] text-cyan-50">
            {decisionSource}
          </p>
        </div>

        <div className="rounded-xl border border-cyan-200/20 bg-slate-900/55 p-2.5">
          <p className="text-[10px] uppercase tracking-[0.18em] text-cyan-200/70">
            Top Model Vote
          </p>
          <p className="mt-1 break-words font-semibold uppercase tracking-[0.06em] text-cyan-50">
            {String(topScoreEmotion).replaceAll("_", " ")} {topScoreValue}
          </p>
        </div>

        <div className="rounded-xl border border-cyan-200/20 bg-slate-900/55 p-2.5">
          <p className="text-[10px] uppercase tracking-[0.18em] text-cyan-200/70">
            Geometry Reason
          </p>
          <p className="mt-1 break-words font-semibold uppercase tracking-[0.06em] text-cyan-50">
            {geometryReason}
          </p>
        </div>

        <div className="col-span-2 rounded-xl border border-cyan-200/20 bg-slate-900/55 p-2.5">
          <p className="text-[10px] uppercase tracking-[0.18em] text-cyan-200/70">
            Rule Triggers
          </p>
          <p className="mt-1 break-words text-[11px] text-slate-200 lg:text-xs">
            {triggerSummary}
          </p>
        </div>
      </div>
    </section>
  );
}

export default CameraFeed;
