import { useCallback, useReducer } from "react";

function normalizeStatus(rawStatus) {
  return String(rawStatus || "")
    .trim()
    .toLowerCase()
    .replace(/[_\s]+/g, "-");
}

function normalizeEmotionLabel(rawEmotion) {
  if (!rawEmotion || typeof rawEmotion !== "string") {
    return "uncertain";
  }

  const value = rawEmotion.trim().toLowerCase();
  if (value === "uncertain" || value === "no_face" || value === "none") {
    return "uncertain";
  }

  const aliases = {
    happiness: "happy",
    sadness: "sad",
    anger: "angry",
    surprise: "surprised",
  };

  if (aliases[value]) {
    return aliases[value];
  }

  return value;
}

function normalizeConfidence(rawValue) {
  const numeric = Number(rawValue);
  if (!Number.isFinite(numeric)) {
    return 0;
  }

  // Backend confidence is currently returned as percentage (0..100).
  if (numeric > 1) {
    return Math.max(0, Math.min(numeric / 100, 1));
  }

  return Math.max(0, Math.min(numeric, 1));
}

function getTopScore(scores) {
  if (!scores || typeof scores !== "object") {
    return { label: null, value: null };
  }

  let bestLabel = null;
  let bestValue = -Infinity;

  Object.entries(scores).forEach(([label, value]) => {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > bestValue) {
      bestLabel = label;
      bestValue = numeric;
    }
  });

  if (!bestLabel || !Number.isFinite(bestValue)) {
    return { label: null, value: null };
  }

  return { label: bestLabel, value: bestValue };
}

function extractEmotionPayload(newData) {
  const status = normalizeStatus(newData?.status);
  const hasFace =
    newData?.face_detected === true ? true
    : newData?.face_detected === false ? false
    : newData?.face_box ? true
    : newData?.box ? true
    : status === "no-face" ? false
    : undefined;

  const emotion = normalizeEmotionLabel(
    newData?.emotion ?? newData?.final_emotion ?? newData?.smoothed_emotion,
  );

  const confidence = normalizeConfidence(
    newData?.confidence ??
      newData?.final_confidence ??
      newData?.smoothed_confidence ??
      0,
  );

  const fpsValue = Number(newData?.fps);
  const fps = Number.isFinite(fpsValue) ? fpsValue : null;
  const topScore = getTopScore(newData?.scores ?? newData?.all_scores);
  const snapshotTaken = Boolean(
    newData?.snapshot_taken ?? newData?.snapshotTaken,
  );
  const snapshotTakenAtValue = Number(
    newData?.snapshot_taken_at ?? newData?.snapshotTakenAt,
  );
  const snapshotTakenAt =
    Number.isFinite(snapshotTakenAtValue) ? snapshotTakenAtValue : null;

  const topScoreValue = Number(topScore.value);
  const normalizedTopScoreValue =
    Number.isFinite(topScoreValue) ? normalizeConfidence(topScoreValue) : null;

  return {
    emotion,
    confidence,
    status,
    hasFace,
    cameraReady: newData?.cameraReady ?? newData?.camera_ready,
    error: newData?.error || "",
    timestamp: Date.now(),
    modelTelemetry: {
      fps,
      snapshotTaken,
      snapshotTakenAt,
      snapshotReason: newData?.snapshot_reason || newData?.snapshotReason || "",
      snapshotUrl: newData?.snapshot_url || newData?.snapshotUrl || "",
      decisionSource: newData?.decision_source || "N/A",
      geometryReason: newData?.geometry_reason || "N/A",
      ruleTriggers:
        Array.isArray(newData?.rule_triggers) ? newData.rule_triggers : [],
      topScoreEmotion: topScore.label,
      topScoreValue: normalizedTopScoreValue,
    },
  };
}

// Centralized state management for emotion data
const initialState = {
  currentEmotion: "neutral",
  previousEmotion: "neutral",
  confidence: 0,
  emotionHistory: [],
  timestamp: null,
  status: "",
  hasFace: undefined,
  cameraReady: false,
  error: "",
  modelTelemetry: {
    fps: null,
    snapshotTaken: false,
    snapshotTakenAt: null,
    snapshotReason: "",
    snapshotUrl: "",
    decisionSource: "N/A",
    geometryReason: "N/A",
    ruleTriggers: [],
    topScoreEmotion: null,
    topScoreValue: null,
  },
  isDemo: false,
  sessionStartTime: Date.now(),
};

const emotionReducer = (state, action) => {
  switch (action.type) {
    case "UPDATE_EMOTION": {
      const now = action.payload.timestamp || Date.now();
      const nextHasFace =
        action.payload.hasFace !== undefined ?
          action.payload.hasFace
        : state.hasFace;
      const shouldUpdateEmotion = nextHasFace !== false;
      const nextEmotion =
        shouldUpdateEmotion ? action.payload.emotion : state.currentEmotion;
      const nextConfidence =
        shouldUpdateEmotion ? action.payload.confidence : state.confidence;
      const nextHistory =
        shouldUpdateEmotion ?
          [
            ...state.emotionHistory,
            {
              emotion: nextEmotion,
              confidence: nextConfidence,
              timestamp: action.payload.timestamp || Date.now(),
            },
          ].slice(-50)
        : state.emotionHistory;

      return {
        ...state,
        previousEmotion:
          shouldUpdateEmotion ? state.currentEmotion : state.previousEmotion,
        currentEmotion: nextEmotion,
        confidence: nextConfidence,
        timestamp: now,
        status: action.payload.status || state.status,
        hasFace: nextHasFace,
        cameraReady:
          action.payload.cameraReady !== undefined ?
            action.payload.cameraReady
          : state.cameraReady,
        error: action.payload.error || "",
        modelTelemetry: (() => {
          const nextTelemetry =
            action.payload.modelTelemetry || state.modelTelemetry;
          return {
            ...nextTelemetry,
            fps:
              Number.isFinite(nextTelemetry?.fps) ?
                nextTelemetry.fps
              : (state.modelTelemetry?.fps ?? null),
            snapshotTakenAt:
              Number.isFinite(nextTelemetry?.snapshotTakenAt) ?
                nextTelemetry.snapshotTakenAt
              : (state.modelTelemetry?.snapshotTakenAt ?? null),
            snapshotTaken:
              typeof nextTelemetry?.snapshotTaken === "boolean" ?
                nextTelemetry.snapshotTaken
              : (state.modelTelemetry?.snapshotTaken ?? false),
          };
        })(),
        emotionHistory: nextHistory,
      };
    }

    case "SET_DEMO_MODE":
      return {
        ...state,
        isDemo: action.payload,
      };

    case "SET_CAMERA_READY":
      return {
        ...state,
        cameraReady: action.payload,
      };

    case "SET_ERROR":
      return {
        ...state,
        error: action.payload,
      };

    case "CLEAR_HISTORY":
      return {
        ...state,
        emotionHistory: [],
        hasFace: undefined,
        status: "",
        sessionStartTime: Date.now(),
      };

    default:
      return state;
  }
};

export function useEmotionState() {
  const [state, dispatch] = useReducer(emotionReducer, initialState);

  const updateEmotionData = useCallback((newData) => {
    dispatch({
      type: "UPDATE_EMOTION",
      payload: extractEmotionPayload(newData),
    });
  }, []);

  const setDemoMode = useCallback((isDemo) => {
    dispatch({
      type: "SET_DEMO_MODE",
      payload: isDemo,
    });
  }, []);

  const setCameraReady = useCallback((ready) => {
    dispatch({
      type: "SET_CAMERA_READY",
      payload: ready,
    });
  }, []);

  const setError = useCallback((error) => {
    dispatch({
      type: "SET_ERROR",
      payload: error,
    });
  }, []);

  const clearHistory = useCallback(() => {
    dispatch({
      type: "CLEAR_HISTORY",
    });
  }, []);

  return {
    ...state,
    updateEmotionData,
    setDemoMode,
    setCameraReady,
    setError,
    clearHistory,
  };
}
