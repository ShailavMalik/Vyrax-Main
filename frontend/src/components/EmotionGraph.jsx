import { motion as Motion } from "framer-motion";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

function EmotionGraph({ emotionHistory }) {
  if (emotionHistory.length === 0) {
    return (
      <section className="glass-panel rounded-3xl border border-blue-200/25 p-6">
        <p className="text-xs uppercase tracking-[0.38em] text-cyan-200/70">
          Emotion Analytics
        </p>
        <div className="mt-6 flex h-64 items-center justify-center">
          <p className="text-slate-400">Waiting for data...</p>
        </div>
      </section>
    );
  }

  // Prepare data for line chart
  const chartData = emotionHistory.map((entry, idx) => ({
    time: idx,
    confidence: Math.round(entry.confidence * 100),
    emotion: entry.emotion,
  }));

  // Count emotions for bar chart
  const emotionCounts = {};
  emotionHistory.forEach((entry) => {
    emotionCounts[entry.emotion] = (emotionCounts[entry.emotion] || 0) + 1;
  });

  const barData = Object.entries(emotionCounts)
    .map(([emotion, count]) => ({
      emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      count,
    }))
    .sort((a, b) => b.count - a.count);

  const dominantEmotion = barData[0];

  return (
    <section className="glass-panel rounded-3xl border border-blue-200/25 p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-xs uppercase tracking-[0.38em] text-cyan-200/70">
          Emotion Analytics
        </p>
        {dominantEmotion && (
          <div className="text-right">
            <p className="text-xs text-slate-400">Most Prevalent</p>
            <p className="text-sm font-bold text-blue-300">
              {dominantEmotion.emotion} ({dominantEmotion.count})
            </p>
          </div>
        )}
      </div>

      <Motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="space-y-4">
        {/* Confidence Trend */}
        <div>
          <p className="mb-2 text-xs text-slate-400">Confidence Trend</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(148,163,184,0.1)"
              />
              <XAxis
                stroke="rgba(148,163,184,0.3)"
                style={{ fontSize: "12px" }}
                tick={false}
              />
              <YAxis
                stroke="rgba(148,163,184,0.3)"
                style={{ fontSize: "12px" }}
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgba(15,23,42,0.9)",
                  border: "1px solid rgba(34,211,238,0.3)",
                  borderRadius: "8px",
                }}
                formatter={(value) => `${value}%`}
              />
              <Line
                type="monotone"
                dataKey="confidence"
                stroke="#00CCFF"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Emotion Distribution */}
        <div>
          <p className="mb-2 text-xs text-slate-400">Emotion Distribution</p>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={barData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(148,163,184,0.1)"
              />
              <XAxis
                dataKey="emotion"
                stroke="rgba(148,163,184,0.3)"
                style={{ fontSize: "11px" }}
              />
              <YAxis
                stroke="rgba(148,163,184,0.3)"
                style={{ fontSize: "12px" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgba(15,23,42,0.9)",
                  border: "1px solid rgba(34,211,238,0.3)",
                  borderRadius: "8px",
                }}
              />
              <Bar dataKey="count" fill="#00CCFF" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Motion.div>
    </section>
  );
}

export default EmotionGraph;
