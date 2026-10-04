import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0)

segmentor = SelfiSegmentation()

background = None

while True:

    success, frame = cap.read()

    if not success:
        print("Camera not found")
        break

    frame = cv2.flip(frame, 1)

    # If background has been captured
    if background is not None:

        result = segmentor.removeBG(
            frame,
            background,
            cutThreshold=0.5
        )

    else:

        result = frame

    # Show result
    cv2.imshow("Invisibility Mode", result)

    # Read keyboard AFTER showing window
    key = cv2.waitKey(1) & 0xFF

    # Press B to capture background
    if key == ord("b"):

        background = frame.copy()

        print("Background captured!")

    # Press Q to close
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()