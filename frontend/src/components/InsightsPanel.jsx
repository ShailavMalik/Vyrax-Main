import { motion as Motion } from "framer-motion";
import { analyzeEmotionPattern, getEmotionColor } from "../utils/emotionUtils";

function InsightsPanel({ emotionHistory, compact = false, connected = true }) {
  const analysis = analyzeEmotionPattern(emotionHistory);

  const getStabilityIcon = (stability) => {
    switch (stability) {
      case "stable":
        return "✓";
      case "moderate":
        return "~";
      case "fluctuating":
        return "⟷";
      default:
        return "?";
    }
  };

  const getStabilityColor = (stability) => {
    switch (stability) {
      case "stable":
        return "text-emerald-400";
      case "moderate":
        return "text-yellow-400";
      case "fluctuating":
        return "text-rose-400";
      default:
        return "text-slate-400";
    }
  };

  return (
    <Motion.section
      className={`glass-panel rounded-2xl border border-purple-200/25 ${compact ? "p-4" : "p-6"}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}>
      <p className={`${compact ? "text-[10px] tracking-[0.26em]" : "text-xs tracking-[0.38em]"} uppercase text-cyan-200/70`}>
        AI Insights
      </p>

      <Motion.div
        className={`${compact ? "mt-3 space-y-2" : "mt-5 space-y-4"}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.5 }}>
        <div className="rounded-lg border border-purple-300/30 bg-purple-500/10 p-4">
          <p className="text-sm leading-relaxed text-slate-200">
            {analysis.insight}
          </p>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <div className="rounded-lg border border-slate-300/20 bg-slate-800/50 p-2.5">
            <p className="text-[10px] text-slate-400">Dominant</p>
            <p className="mt-1 text-sm font-bold" style={{ color: getEmotionColor(analysis.dominantEmotion) }}>
              {analysis.dominantEmotion}
            </p>
          </div>

          <div className="rounded-lg border border-slate-300/20 bg-slate-800/50 p-2.5">
            <p className="text-[10px] text-slate-400">Stability</p>
            <p className={`mt-1 text-sm font-bold ${getStabilityColor(analysis.stability)}`}>
              {getStabilityIcon(analysis.stability)} {analysis.stability}
            </p>
          </div>

          <div className="rounded-lg border border-slate-300/20 bg-slate-800/50 p-2.5">
            <p className="text-[10px] text-slate-400">System</p>
            <p className={`mt-1 text-sm font-bold ${connected ? "text-emerald-300" : "text-rose-300"}`}>
              {connected ? "online" : "offline"}
            </p>
          </div>
        </div>

        {!compact && (
          <div>
            <p className="mb-3 text-xs text-slate-400">Emotion Breakdown</p>
            <div className="space-y-2">
              {Object.entries(analysis.emotionCounts)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 5)
                .map(([emotion, count]) => {
                  const percentage = Math.round(
                    (count / emotionHistory.length) * 100,
                  );
                  return (
                    <div key={emotion} className="flex items-center justify-between">
                      <span className="text-xs capitalize text-slate-400">{emotion}</span>
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-24 rounded-full bg-slate-700">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${percentage}%`,
                              backgroundColor: getEmotionColor(emotion),
                            }}
                          />
                        </div>
                        <span className="w-8 text-right text-xs font-semibold text-slate-300">
                          {percentage}%
                        </span>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        )}
      </Motion.div>
    </Motion.section>
  );
}

export default InsightsPanel;
