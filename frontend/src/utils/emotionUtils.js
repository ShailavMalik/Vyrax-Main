// Demo mode data generator
const DEMO_SEQUENCE = [
  "happy",
  "neutral",
  "confused",
  "surprised",
  "happy",
  "sad",
  "neutral",
  "happy",
];

const EMOTION_CONFIDENCE_RANGE = {
  happy: [0.75, 0.95],
  neutral: [0.65, 0.85],
  confused: [0.6, 0.8],
  surprised: [0.7, 0.9],
  sad: [0.7, 0.88],
  angry: [0.75, 0.92],
  fear: [0.68, 0.86],
};

export function generateDemoData(sequenceIndex) {
  const emotion = DEMO_SEQUENCE[sequenceIndex % DEMO_SEQUENCE.length];
  const [min, max] = EMOTION_CONFIDENCE_RANGE[emotion] || [0.6, 0.8];
  const confidence = min + Math.random() * (max - min);

  return {
    emotion,
    confidence: Math.round(confidence * 100) / 100,
    cameraReady: true,
    error: "",
    timestamp: Date.now(),
  };
}

export function buildDemoDebugData(emotion, confidence) {
  const fps = 24 + Math.floor(Math.random() * 8);
  const normalizedEmotion = (emotion || "neutral").toLowerCase();

  return {
    modelName: "VyraX EmotionNet",
    version: "v3.2.1-demo",
    fps,
    detectionConfidence: confidence || 0,
    pipelineSteps: [
      "Capture Frame",
      "Face Detect",
      "Landmarks",
      "Emotion Classifier",
      "Temporal Smoothing",
    ],
    decisionExplanation: `Dominant signal is ${normalizedEmotion} with ${(confidence * 100).toFixed(0)}% confidence after temporal smoothing.`,
  };
}

// Analyze emotion patterns for insights
export function analyzeEmotionPattern(emotionHistory) {
  if (emotionHistory.length === 0) {
    return {
      dominantEmotion: "neutral",
      stability: "stable",
      averageConfidence: 0,
      emotionCounts: {},
      insight: "Waiting for data...",
    };
  }

  // Count emotions
  const emotionCounts = {};
  let totalConfidence = 0;

  emotionHistory.forEach((entry) => {
    emotionCounts[entry.emotion] = (emotionCounts[entry.emotion] || 0) + 1;
    totalConfidence += entry.confidence;
  });

  const dominantEmotion = Object.keys(emotionCounts).reduce((a, b) =>
    emotionCounts[a] > emotionCounts[b] ? a : b,
  );

  const averageConfidence = totalConfidence / emotionHistory.length;

  // Analyze stability (look at last 10-15 entries)
  const recentEntries = emotionHistory.slice(-15);
  const uniqueEmotions = new Set(recentEntries.map((e) => e.emotion)).size;
  let changeFrequency = 0;

  for (let i = 1; i < recentEntries.length; i++) {
    if (recentEntries[i].emotion !== recentEntries[i - 1].emotion) {
      changeFrequency++;
    }
  }

  let stability = "stable";
  if (changeFrequency > recentEntries.length * 0.4) {
    stability = "fluctuating";
  } else if (changeFrequency > recentEntries.length * 0.2) {
    stability = "moderate";
  }

  // Generate insight
  let insight = "";
  const dominantPercent = Math.round(
    (emotionCounts[dominantEmotion] / emotionHistory.length) * 100,
  );

  if (dominantEmotion === "neutral") {
    insight = `User is ${dominantPercent}% neutral and focused.`;
  } else if (dominantEmotion === "happy") {
    insight = `Positive energy detected (${dominantPercent}% happy).`;
  } else if (dominantEmotion === "sad") {
    insight = `User showing signs of dissatisfaction (${dominantPercent}% sad).`;
  } else if (dominantEmotion === "confused") {
    insight = `User appears uncertain or confused (${dominantPercent}%).`;
  } else if (dominantEmotion === "surprised") {
    insight = `High engagement with surprising content (${dominantPercent}%).`;
  }

  if (stability === "fluctuating") {
    insight += " Emotional state is rapidly changing.";
  } else if (stability === "stable") {
    insight += " Emotional stability: High.";
  }

  return {
    dominantEmotion,
    stability,
    averageConfidence: Math.round(averageConfidence * 100) / 100,
    emotionCounts,
    insight,
  };
}

// Get emotion color for glow effect
export function getEmotionColor(emotion) {
  const colorMap = {
    happy: "#FFD700", // yellow gold
    sad: "#0099FF", // sky blue
    angry: "#FF3333", // red
    surprised: "#00FFCC", // cyan
    neutral: "#00CCFF", // bright cyan
    confused: "#FF99FF", // magenta
    fear: "#FF6600", // orange
    disgust: "#66FF00", // lime
  };
  return colorMap[emotion] || "#00CCFF";
}
