import cv2
from gesture_recognizer import GestureRecognizer

gr = GestureRecognizer()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gesture = gr.process(frame)
    if gesture:
        cv2.putText(frame, gesture, (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    cv2.imshow("Dynamic Gestures", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
