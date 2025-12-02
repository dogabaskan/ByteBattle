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

    cv2.imshow("Gesture Control Preview", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
