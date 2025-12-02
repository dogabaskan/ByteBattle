import cv2
from features import FeatureExtractor

fx = FeatureExtractor()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    features, lm = fx.get_features(frame)

    if lm is not None:
        fx.tracker.draw(frame, lm)

    cv2.imshow("Live Features", frame)

    if features is not None:
        print(features.shape)   # should be (63,)

    if cv2.waitKey(1) & 0xFF == 27:
        break
