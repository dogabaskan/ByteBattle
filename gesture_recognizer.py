from collections import deque
import numpy as np
from hand_tracker import HandTracker
import cv2
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


class DynamicTracker:
    def __init__(self, window=8):
        self.window = window
        self.positions = deque(maxlen=window)
        self.areas = deque(maxlen=window)

    def update(self, pts):
        x = pts[0, 0]  # wrist
        y = pts[0, 1]

        area = np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])

        self.positions.append((x, y))
        self.areas.append(area)

    def swipe(self):
        if len(self.positions) < self.window:
            return None
        x0 = self.positions[0][0]
        x1 = self.positions[-1][0]
        dx = x1 - x0

        TH = 0.15
        if dx > TH:  return "swipe_right"
        if dx < -TH: return "swipe_left"
        return None

    def scroll(self):
        if len(self.positions) < self.window:
            return None
        y0 = self.positions[0][1]
        y1 = self.positions[-1][1]
        dy = y1 - y0

        TH = 0.12
        if dy > TH:  return "scroll_down"
        if dy < -TH: return "scroll_up"
        return None

    def push(self):
        if len(self.areas) < self.window:
            return None
        a0 = self.areas[0]
        a1 = self.areas[-1]
        if a0 > 0 and a1 > a0 * 1.4:
            return "push"
        return None

class GestureRecognizer:
    def __init__(self):
        self.static = StaticRecognizer()
        self.dynamic = DynamicTracker()
        self.hand_tracker = HandTracker()

    def process(self, frame):
        # Track hands
        result = self.hand_tracker.process(frame)
        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0]
        pts = self.hand_tracker.extract(lm)

        # Update dynamic tracker
        self.dynamic.update(pts)

        # FIRST: detect dynamic gestures
        for method in [self.dynamic.swipe, self.dynamic.scroll, self.dynamic.push]:
            g = method()
            if g:
                return g

        # SECOND: static gestures
        static_gesture = self.static.predict(frame)
        return static_gesture