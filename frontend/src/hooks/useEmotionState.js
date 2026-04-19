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

  return value;
}

function extractEmotionPayload(newData) {
  const status = normalizeStatus(newData?.status);
  const hasFace =
    newData?.box ? true
    : status === "no-face" ? false
    : undefined;

  const emotion = normalizeEmotionLabel(
    newData?.emotion ?? newData?.final_emotion ?? newData?.smoothed_emotion,
  );

  const confidenceValue =
    newData?.confidence ??
    newData?.final_confidence ??
    newData?.smoothed_confidence ??
    0;
  const confidence =
    Number.isFinite(Number(confidenceValue)) ? Number(confidenceValue) : 0;

  return {
    emotion,
    confidence,
    status,
    hasFace,
    cameraReady: newData?.cameraReady ?? newData?.camera_ready,
    error: newData?.error || "",
    timestamp: Date.now(),
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
  isDemo: false,
  sessionStartTime: Date.now(),
};

const emotionReducer = (state, action) => {
  switch (action.type) {
    case "UPDATE_EMOTION": {
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
        timestamp: action.payload.timestamp || Date.now(),
        status: action.payload.status || state.status,
        hasFace: nextHasFace,
        cameraReady:
          action.payload.cameraReady !== undefined ?
            action.payload.cameraReady
          : state.cameraReady,
        error: action.payload.error || "",
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
