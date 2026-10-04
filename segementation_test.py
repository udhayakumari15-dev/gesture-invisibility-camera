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

    # Capture background
    if background is None:
        result = frame

    else:
        result = segmentor.removeBG(
            frame,
            background,
            cutThreshold=0.3
        )

    cv2.imshow("Segmentation Test", result)

    key = cv2.waitKey(1) & 0xFF

    # Capture background
    if key == ord("b"):
        background = frame.copy()
        print("Background captured!")

    # Quit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()