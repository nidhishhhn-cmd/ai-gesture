"""Simple check that OpenCV can open and display the webcam."""
import cv2

CAMERA_INDEX = 0
camera = cv2.VideoCapture(CAMERA_INDEX)
try:
    if not camera.isOpened():
        print("Could not open webcam. Check CAMERA_INDEX and camera permissions.")
    else:
        print("Webcam opened. Press Q in the video window to quit.")
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break
            cv2.imshow("Camera Test", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()
