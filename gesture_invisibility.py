import cv2
import numpy as np
import mediapipe as mp

from cvzone.SelfiSegmentationModule import SelfiSegmentation
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- HAND DETECTION ----------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# ---------- BODY SEGMENTATION ----------
segmentor = SelfiSegmentation()

cap = cv2.VideoCapture(0)
background = None

while True:
    success, frame = cap.read()
    if not success:
        print("Camera not found")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- gesture ----
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)
    gesture = "NO HAND"

    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        fingers_up = sum(1 for t, p in zip(tips, pips) if hand[t].y < hand[p].y)
        if abs(hand[4].x - hand[3].x) > 0.03:
            fingers_up += 1

        if fingers_up >= 4:
            gesture = "OPEN HAND"
        elif fingers_up <= 1:
            gesture = "FIST"
        else:
            gesture = "OTHER"

    # ---- keys ----
    key = cv2.waitKey(1) & 0xFF
    if key == ord("b"):
        background = frame.copy()
        print("Background captured!")

    # ---- invisibility ----
    result = frame.copy()

    if background is not None and gesture == "OPEN HAND":
        magenta = (255, 0, 255)
        cut = segmentor.removeBG(frame, magenta, cutThreshold=0.5)

        # person = every pixel that is NOT magenta
        mask = (~np.all(cut == magenta, axis=2)).astype(np.uint8)

        # grow the mask a little and soften the edges
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
        mask = mask[:, :, None]

        # person pixels -> saved background
        result = (background * mask + frame * (1 - mask)).astype(np.uint8)

    cv2.putText(result, gesture, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Gesture Invisibility", result)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()