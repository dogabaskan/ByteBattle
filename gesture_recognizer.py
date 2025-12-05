from collections import deque
import numpy as np
import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
from hand_tracker import HandTracker



class DynamicTracker:
    def __init__(self, window=8):
        self.window = window
        self.positions = deque(maxlen=window)
        self.areas = deque(maxlen=window)

    def reset(self):
        self.positions.clear()
        self.areas.clear()

    def update(self, pts):
        x = pts[0, 0]
        y = pts[0, 1]
        area = np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])

        self.positions.append((x, y))
        self.areas.append(area)

    # Swipe Detection
    def swipe(self):
        if len(self.positions) < self.window: return None
        x0 = self.positions[0][0]
        x1 = self.positions[-1][0]
        dx = x1 - x0
        TH = 0.12
        if dx > TH: return "swipe_right"
        if dx < -TH: return "swipe_left"
        return None

    # Scroll Detection
    def scroll(self):
        if len(self.positions) < self.window: return None
        y0 = self.positions[0][1]
        y1 = self.positions[-1][1]
        dy = y1 - y0
        TH = 0.12
        if dy > TH: return "scroll_down"
        if dy < -TH: return "scroll_up"
        return None

    # Push Detection (toward camera)
    def push(self):
        if len(self.areas) < self.window: return None
        a0 = self.areas[0]
        a1 = self.areas[-1]
        if a1 > a0 * 1.35:
            return "push"
        return None

    # Pull Detection (away from camera)
    def pull(self):
        if len(self.areas) < self.window: return None
        a0 = self.areas[0]
        a1 = self.areas[-1]
        if a1 < a0 * 0.75:
            return "pull"
        return None
class StaticRecognizer:
    def __init__(self):
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

        # Compute finger curl (heuristic)
        # If curled = fist
        tips = [8, 12, 16, 20]
        curl = 0
        for t in tips:
            if lm.landmark[t].y > lm.landmark[t-2].y:
                curl += 1

        if curl >= 3:
            return "fist"
        return None

class GestureRecognizer:
    def __init__(self):
        self.static = StaticRecognizer()
        self.dynamic = DynamicTracker()
        self.hand_tracker = HandTracker()

    def reset(self):
        self.dynamic.reset()

    def process(self, frame):
        # Step 1 — detect hand
        result = self.hand_tracker.process(frame)
        if not result.multi_hand_landmarks:
            self.reset()
            return None

        lm = result.multi_hand_landmarks[0]
        pts = self.hand_tracker.extract(lm)

        # Step 2 — dynamic detection
        self.dynamic.update(pts)

        for method in [self.dynamic.swipe, self.dynamic.scroll,
                       self.dynamic.push, self.dynamic.pull]:
            g = method()
            if g:
                return g

        # Step 3 — check FIST (reset)
        static = self.static.extract(frame)
        if static == "fist":
            self.reset()
            return "reset"

        return None
