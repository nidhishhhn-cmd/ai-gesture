"""Control PowerPoint slides with left and right hand swipes.

Start the PowerPoint slideshow and click its window so it receives keyboard
input. Then run this program and make a clear sideways swipe in front of the
webcam. Press Q in the webcam window to stop safely.
"""

from collections import deque
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import pyautogui


# Gesture settings: change these if the gesture is too sensitive or too difficult.
CAMERA_INDEX = 0
HISTORY_LENGTH = 30  # About one second of webcam movement at a normal frame rate.
MIN_SWIPE_SAMPLES = 5  # Do not decide from only one or two frames.
SWIPE_THRESHOLD = 0.12  # Normalized horizontal movement: 0.0 to 1.0
MOVEMENT_DEADBAND = 0.003  # Ignore tiny frame-to-frame landmark jitter.
COOLDOWN_SECONDS = 1.0  # Minimum time between slide changes
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Lines joining the 21 hand landmark dots. Each pair is (start landmark, end landmark).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def get_model_path():
    """Download MediaPipe's official pretrained hand model only if it is missing."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    print("Downloading MediaPipe's hand landmark model (first run only)...")
    try:
        urlretrieve(MODEL_URL, MODEL_PATH)
    except URLError as error:
        raise RuntimeError(
            "Could not download the hand model. Check your internet connection "
            "and run the program again."
        ) from error
    return MODEL_PATH


def create_hand_detector():
    """Create MediaPipe's ready-made hand landmark detector for webcam video."""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(get_model_path())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def draw_hand_landmarks(frame, landmarks):
    """Draw the 21 detected hand points and simple lines between them."""
    height, width = frame.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def detect_gesture(wrist_history):
    """Return a swipe, total movement, and tracking direction.

    The oldest and newest values in the long history measure total horizontal
    movement. Small changes are ignored while checking that the movement has a
    mostly consistent direction, which prevents normal hand jitter from firing.
    """
    if len(wrist_history) < 2:
        return None, 0.0, "Tracking hand..."

    movement = wrist_history[-1] - wrist_history[0]
    if movement > MOVEMENT_DEADBAND:
        direction = "Tracking right"
    elif movement < -MOVEMENT_DEADBAND:
        direction = "Tracking left"
    else:
        direction = "Hold hand still, then swipe"

    if len(wrist_history) < MIN_SWIPE_SAMPLES:
        return None, movement, direction

    # Look only at meaningful changes. This lets a hand pause briefly during a
    # swipe, but rejects movement whose direction repeatedly changes.
    frame_changes = [
        later - earlier
        for earlier, later in zip(wrist_history, list(wrist_history)[1:])
        if abs(later - earlier) >= MOVEMENT_DEADBAND
    ]
    if len(frame_changes) < MIN_SWIPE_SAMPLES - 1:
        return None, movement, direction

    right_steps = sum(change > 0 for change in frame_changes)
    left_steps = sum(change < 0 for change in frame_changes)
    total_steps = len(frame_changes)
    mostly_right = right_steps / total_steps >= 0.70
    mostly_left = left_steps / total_steps >= 0.70

    if movement >= SWIPE_THRESHOLD and mostly_right:
        return "Right swipe", movement, "RIGHT SWIPE DETECTED"
    if movement <= -SWIPE_THRESHOLD and mostly_left:
        return "Left swipe", movement, "LEFT SWIPE DETECTED"
    return None, movement, direction


def perform_action(gesture):
    """Send one keyboard key to the currently focused app, such as PowerPoint."""
    if gesture == "Right swipe":
        pyautogui.press("right")  # Next PowerPoint slide
    elif gesture == "Left swipe":
        pyautogui.press("left")   # Previous PowerPoint slide


def draw_text(frame, text, position, color=(255, 255, 255)):
    """Draw readable text with a dark outline on the camera image."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                color, 2, cv2.LINE_AA)


def main():
    # PyAutoGUI's top-left-corner failsafe remains enabled for an additional escape.
    pyautogui.PAUSE = 0.05

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}. Try closing other camera apps.")
        return

    try:
        hand_detector = create_hand_detector()
    except RuntimeError as error:
        camera.release()
        print(error)
        return
    wrist_history = deque(maxlen=HISTORY_LENGTH)
    last_action_time = 0.0
    status = "Show one hand to the camera"
    debug_wrist_x = 0.0
    debug_movement = 0.0

    print("Gesture controller started. Press Q in the webcam window to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Camera frame could not be read. Stopping safely.")
                break

            # Flip first so camera movement feels natural, like a mirror.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.monotonic() * 1000)
            results = hand_detector.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                draw_hand_landmarks(frame, hand_landmarks)

                # Landmark 0 is the wrist. Its x position is normalized from 0 to 1.
                wrist_x = hand_landmarks[0].x
                wrist_history.append(wrist_x)
                gesture, debug_movement, status = detect_gesture(wrist_history)
                debug_wrist_x = wrist_x
                now = time.monotonic()

                if gesture:
                    if now - last_action_time >= COOLDOWN_SECONDS:
                        perform_action(gesture)
                        last_action_time = now
                        status = f"{gesture.upper()} DETECTED"
                    else:
                        status = "Cooldown: wait before next swipe"
                    # A detected movement is consumed even during cooldown. This
                    # prevents one long hand movement from changing several slides.
                    wrist_history.clear()
            else:
                # A new hand appearance should not combine with old positions.
                wrist_history.clear()
                status = "No hand detected"
                debug_wrist_x = 0.0
                debug_movement = 0.0

            draw_text(frame, f"Status: {status}", (20, 35), (0, 255, 0))
            draw_text(frame, f"Wrist X: {debug_wrist_x:.2f}", (20, 65))
            draw_text(frame, f"Movement: {debug_movement:+.2f}", (20, 92))
            draw_text(frame, f"History: {len(wrist_history)}/{HISTORY_LENGTH}", (20, 119))
            draw_text(frame, "Swipe Left = Previous | Swipe Right = Next", (20, 146))
            draw_text(frame, "Q = Quit", (20, 173))
            cv2.imshow("Smart Gesture Presentation Control", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    except KeyboardInterrupt:
        print("Stopped with Ctrl+C.")
    finally:
        # This runs even if an unexpected error occurs inside the loop.
        hand_detector.close()
        camera.release()
        cv2.destroyAllWindows()
        print("Gesture controller stopped.")


if __name__ == "__main__":
    main()
