import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    results = detector.detect(mp_image)

    gesture = "No Hand"

    if results.hand_landmarks:

        hand = results.hand_landmarks[0]

        # Finger tips
        tips = [8, 12, 16, 20]

        # Finger joints
        pips = [6, 10, 14, 18]

        fingers_up = 0

        for tip, pip in zip(tips, pips):

            if hand[tip].y < hand[pip].y:
                fingers_up += 1

        # Thumb
        if abs(hand[4].x - hand[3].x) > 0.03:
            fingers_up += 1

        if fingers_up >= 4:
            gesture = "OPEN HAND"

        elif fingers_up <= 1:
            gesture = "FIST"

        else:
            gesture = "OTHER"

        # Draw hand points
        for landmark in hand:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    cv2.putText(
        frame,
        gesture,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()