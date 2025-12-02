import cv2
import numpy as np
import mediapipe as mp
import joblib

class StaticRecognizer:
    def __init__(self):
        self.model = joblib.load("model/static_model_xgb.pkl")
        self.encoder = joblib.load("model/label_encoder.pkl")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
        pts -= pts[0]  # normalize to wrist
        return pts.flatten()

    def predict(self, frame):
        feat = self.extract(frame)
        if feat is None:
            return None

        pred_idx = self.model.predict([feat])[0]
        return self.encoder.inverse_transform([pred_idx])[0]
