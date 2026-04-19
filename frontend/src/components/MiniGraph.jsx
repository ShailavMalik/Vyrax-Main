import { ResponsiveContainer, LineChart, Line, Tooltip } from "recharts";

function MiniGraph({ emotionHistory }) {
  const now = Date.now();
  const windowStart = now - 20000;

  const recent = emotionHistory
    .filter((entry) => entry.timestamp >= windowStart)
    .slice(-40)
    .map((entry, idx) => ({
      t: idx,
      confidence: Math.round((entry.confidence || 0) * 100),
    }));

  return (
    <section className="glass-panel rounded-2xl border border-cyan-200/20 p-4 lg:p-5">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-[10px] uppercase tracking-[0.26em] text-cyan-200/75">Mini Trend</p>
        <span className="text-[10px] uppercase tracking-[0.2em] text-slate-400">20s</span>
      </div>

      <div className="h-20 lg:h-24">
        {recent.length > 1 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={recent}>
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgba(15,23,42,0.92)",
                  border: "1px solid rgba(34,211,238,0.25)",
                  borderRadius: "8px",
                  padding: "4px 8px",
                  fontSize: "11px",
                }}
                formatter={(value) => [`${value}%`, "Confidence"]}
                labelFormatter={() => "Live"}
              />
              <Line
                type="monotone"
                dataKey="confidence"
                stroke="#22d3ee"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-400">Collecting live points...</div>
        )}
      </div>
    </section>
  );
}

export default MiniGraph;
