import { motion as Motion } from "framer-motion";
import { analyzeEmotionPattern } from "../utils/emotionUtils";

function SessionSummary({ emotionHistory, sessionStartTime, onClearSession }) {
  const analysis = analyzeEmotionPattern(emotionHistory);

  const sessionDuration = Math.round((Date.now() - sessionStartTime) / 1000);
  const minutes = Math.floor(sessionDuration / 60);
  const seconds = sessionDuration % 60;

  const exportReport = () => {
    const report = {
      timestamp: new Date().toISOString(),
      sessionDuration: `${minutes}m ${seconds}s`,
      totalEmotionsDetected: emotionHistory.length,
      dominantEmotion: analysis.dominantEmotion,
      averageConfidence: analysis.averageConfidence,
      emotionDistribution: analysis.emotionCounts,
      stability: analysis.stability,
      emotionHistory,
    };

    const dataStr = JSON.stringify(report, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `emotion-report-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <Motion.section
      className="glass-panel rounded-3xl border border-emerald-200/25 p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}>
      <p className="text-xs uppercase tracking-[0.38em] text-cyan-200/70">
        📊 Session Summary
      </p>

      <Motion.div
        className="mt-5 space-y-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.5 }}>
        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-300/30 bg-emerald-500/10 p-4">
            <p className="text-xs text-slate-400">Session Duration</p>
            <p className="mt-2 text-xl font-bold text-emerald-400">
              {minutes}m {seconds}s
            </p>
          </div>

          <div className="rounded-lg border border-emerald-300/30 bg-emerald-500/10 p-4">
            <p className="text-xs text-slate-400">Emotions Detected</p>
            <p className="mt-2 text-xl font-bold text-emerald-400">
              {emotionHistory.length}
            </p>
          </div>

          <div className="rounded-lg border border-emerald-300/30 bg-emerald-500/10 p-4">
            <p className="text-xs text-slate-400">Avg Confidence</p>
            <p className="mt-2 text-xl font-bold text-emerald-400">
              {Math.round(analysis.averageConfidence * 100)}%
            </p>
          </div>

          <div className="rounded-lg border border-emerald-300/30 bg-emerald-500/10 p-4">
            <p className="text-xs text-slate-400">Most Frequent</p>
            <p className="mt-2 text-xl font-bold text-emerald-400 capitalize">
              {analysis.dominantEmotion}
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <Motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={exportReport}
            className="flex-1 rounded-full border border-emerald-400/50 bg-emerald-500/20 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-emerald-300 transition-all hover:border-emerald-300 hover:bg-emerald-500/30 hover:shadow-[0_0_15px_rgba(16,185,129,0.3)]">
            📥 Export Report
          </Motion.button>

          <Motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onClearSession}
            className="flex-1 rounded-full border border-slate-400/30 bg-slate-500/10 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-300 transition-all hover:border-slate-300 hover:bg-slate-500/20">
            🔄 Clear Session
          </Motion.button>
        </div>
      </Motion.div>
    </Motion.section>
  );
}

export default SessionSummary;
