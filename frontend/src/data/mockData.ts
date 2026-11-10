import type { EmotionControllerState } from "../types/emotion";
export const demoController: EmotionControllerState = {
  detectedEmotion: {
    primaryEmotion: "Anger",
    intensity: 0.82,
    valence: -0.72,
    arousal: 0.86,
  },
  responseDirection: "Calm & Supportive",
  responseStrength: 0.76,
  internalParameters: {
    calmness: 0.78,
    empathy: 0.72,
    supportiveness: 0.68,
    assertiveness: 0.24,
  },
};
export const moodHistory = [-0.2, -0.35, -0.72, -0.55, -0.3, 0.05].map(
  (score, i) => ({ turn: i + 1, score }),
);
