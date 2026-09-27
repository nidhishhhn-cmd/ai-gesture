"""Control PowerPoint slides with left and right hand swipes.

Start the PowerPoint slideshow and click its window so it receives keyboard
input. Then run this program and make a clear sideways swipe in front of the
webcam. Press Q in the webcam window to stop safely.

Improvements:
- Detects LEFT and RIGHT hand separately using MediaPipe's handedness output.
- Tracks both hands simultaneously with independent swipe histories.
- On-screen label shows which hand(s) are visible with distinct colours.
- Higher confidence thresholds and tighter gesture logic for accuracy.
- Corrects the left/right label inversion caused by the mirror flip.
- Counts extended fingers (0-10) per hand with a large on-screen display.
"""

from collections import deque
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
import pyautogui


# Gesture settings: change these if the gesture is too sensitive or too difficult.
CAMERA_INDEX = 0
HISTORY_LENGTH = 40        # ~1.3 s at 30 fps – longer window = more reliable
MIN_SWIPE_SAMPLES = 8      # Require at least this many frames before deciding
SWIPE_THRESHOLD = 0.14     # Normalised horizontal travel needed (0.0 – 1.0)
MOVEMENT_DEADBAND = 0.004  # Ignore per-frame jitter smaller than this
COOLDOWN_SECONDS = 1.2     # Minimum time between slide changes
DIRECTION_PURITY = 0.75    # Fraction of steps that must be in ONE direction
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Colour per hand: Right → green-ish, Left → cyan-ish
HAND_COLORS = {
    "Right": (0, 220, 80),
    "Left":  (0, 200, 255),
}

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
    """Create a MediaPipe HandLandmarker configured for live video."""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(get_model_path())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,                         # Track BOTH hands to tell L from R
        min_hand_detection_confidence=0.75,  # Higher = fewer false positives
        min_hand_presence_confidence=0.75,
        min_tracking_confidence=0.75,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def count_fingers(landmarks, hand_label):
    """Count how many fingers are extended (0-5) for one hand.

    Landmark indices used:
      Thumb  : tip=4,  ip_joint=3   (compare X, direction depends on hand side)
      Index  : tip=8,  pip=6        (tip Y < pip Y  → extended)
      Middle : tip=12, pip=10
      Ring   : tip=16, pip=14
      Pinky  : tip=20, pip=18
    """
    count = 0

    # --- Thumb ---
    # After the mirror flip the image is like a mirror, so:
    #   Right hand thumb extends to the LEFT  (tip.x < ip.x)
    #   Left  hand thumb extends to the RIGHT (tip.x > ip.x)
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    if hand_label == "Right":
        if thumb_tip.x < thumb_ip.x:
            count += 1
    else:  # Left hand
        if thumb_tip.x > thumb_ip.x:
            count += 1

    # --- Four fingers (tip Y above pip Y means extended) ---
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip_idx, pip_idx in finger_pairs:
        if landmarks[tip_idx].y < landmarks[pip_idx].y:
            count += 1

    return count


def draw_hand_landmarks(frame, landmarks, hand_label, finger_count):
    """Draw the 21 hand points, connecting lines, and finger count bubble."""
    color = HAND_COLORS.get(hand_label, (0, 255, 0))
    height, width = frame.shape[:2]
    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 2)
    for point in points:
        cv2.circle(frame, point, 5, (255, 255, 255), -1)
        cv2.circle(frame, point, 3, color, -1)

    # --- Hand label at wrist ---
    wrist_pt = points[0]
    draw_text(frame, f"{hand_label} hand",
              (wrist_pt[0] - 40, wrist_pt[1] + 25),
              color)

    # --- Large finger-count bubble above the wrist ---
    bubble_center = (wrist_pt[0], wrist_pt[1] - 55)
    cv2.circle(frame, bubble_center, 32, (30, 30, 30), -1)       # dark fill
    cv2.circle(frame, bubble_center, 32, color, 2)                # coloured ring
    count_str = str(finger_count)
    text_size  = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
    text_x = bubble_center[0] - text_size[0] // 2
    text_y = bubble_center[1] + text_size[1] // 2
    cv2.putText(frame, count_str, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)


def detect_swipe(wrist_history):
    """Analyse wrist X-position history; return (gesture|None, movement, hint).

    The oldest and newest values in the history measure total horizontal
    movement. Small per-frame jitter is filtered out, and only gestures whose
    direction is consistent (>= DIRECTION_PURITY fraction of steps) are fired.
    """
    if len(wrist_history) < 2:
        return None, 0.0, "Tracking…"

    movement = wrist_history[-1] - wrist_history[0]
    if movement > MOVEMENT_DEADBAND:
        direction = "Moving right →"
    elif movement < -MOVEMENT_DEADBAND:
        direction = "Moving left ←"
    else:
        direction = "Hold still, then swipe"

    if len(wrist_history) < MIN_SWIPE_SAMPLES:
        return None, movement, direction

    # Only count frames where the hand actually moved (filter jitter).
    frame_changes = [
        later - earlier
        for earlier, later in zip(wrist_history, list(wrist_history)[1:])
        if abs(later - earlier) >= MOVEMENT_DEADBAND
    ]
    if len(frame_changes) < MIN_SWIPE_SAMPLES - 1:
        return None, movement, direction

    total_steps  = len(frame_changes)
    right_steps  = sum(c > 0 for c in frame_changes)
    left_steps   = sum(c < 0 for c in frame_changes)
    mostly_right = right_steps / total_steps >= DIRECTION_PURITY
    mostly_left  = left_steps  / total_steps >= DIRECTION_PURITY

    if movement >= SWIPE_THRESHOLD and mostly_right:
        return "Right swipe", movement, "RIGHT SWIPE ✓"
    if movement <= -SWIPE_THRESHOLD and mostly_left:
        return "Left swipe", movement, "LEFT SWIPE ✓"
    return None, movement, direction


def perform_action(gesture, hand_label):
    """Send one keyboard key to the focused app based on swipe direction.

    Swipe right (either hand) → next slide.
    Swipe left  (either hand) → previous slide.
    The hand_label is used only for the on-screen status message.
    """
    if gesture == "Right swipe":
        pyautogui.press("right")   # Next PowerPoint slide
    elif gesture == "Left swipe":
        pyautogui.press("left")    # Previous PowerPoint slide


def draw_text(frame, text, position, color=(255, 255, 255)):
    """Draw readable text with a dark outline on the camera image."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                color, 2, cv2.LINE_AA)


def main():
    # PyAutoGUI's top-left-corner failsafe remains enabled for safety.
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

    # Separate wrist history per hand so they never interfere with each other.
    histories = {
        "Right": deque(maxlen=HISTORY_LENGTH),
        "Left":  deque(maxlen=HISTORY_LENGTH),
    }
    finger_counts = {"Right": 0, "Left": 0}
    last_action_time = 0.0
    status = "Show a hand to the camera"

    print("Gesture controller started. Press Q in the webcam window to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Camera frame could not be read. Stopping safely.")
                break

            # Flip so the view is mirror-like (feels natural to the user).
            frame = cv2.flip(frame, 1)
            rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.monotonic() * 1000)
            results = hand_detector.detect_for_video(mp_image, timestamp_ms)

            detected_hands = set()

            if results.hand_landmarks:
                for idx, hand_landmarks in enumerate(results.hand_landmarks):
                    # MediaPipe reports handedness for the *original* (un-flipped)
                    # image, so after cv2.flip(frame, 1) the labels are swapped.
                    # We invert them here to match what the user sees on screen.
                    raw_label  = results.handedness[idx][0].display_name  # "Left" or "Right"
                    hand_label = "Left" if raw_label == "Right" else "Right"

                    detected_hands.add(hand_label)
                    finger_count = count_fingers(hand_landmarks, hand_label)
                    draw_hand_landmarks(frame, hand_landmarks, hand_label, finger_count)

                    wrist_x = hand_landmarks[0].x
                    histories[hand_label].append(wrist_x)
                    # Store finger count so we can show it in the HUD
                    finger_counts[hand_label] = finger_count

                    gesture, movement, hint = detect_swipe(histories[hand_label])
                    now = time.monotonic()

                    if gesture and (now - last_action_time >= COOLDOWN_SECONDS):
                        perform_action(gesture, hand_label)
                        last_action_time = now
                        status = f"{hand_label.upper()} HAND – {gesture.upper()}"
                        # Clear ALL histories so one long sweep doesn't fire twice.
                        for h in histories.values():
                            h.clear()
                    elif gesture:
                        status = "Cooldown – wait before next swipe"
                    else:
                        status = f"{', '.join(detected_hands)} hand – {hint}"

            else:
                # No hands visible – reset everything.
                for h in histories.values():
                    h.clear()
                finger_counts = {"Right": 0, "Left": 0}
                status = "No hand detected – show a hand to the camera"

            # ------------------------------------------------------------------
            # HUD overlay
            # ------------------------------------------------------------------
            r_len  = len(histories["Right"])
            l_len  = len(histories["Left"])
            r_cnt  = finger_counts["Right"]
            l_cnt  = finger_counts["Left"]
            total  = r_cnt + l_cnt
            overlay_lines = [
                (f"Status : {status}",
                 (20, 38),  (0, 255, 120)),
                (f"Right fingers : {r_cnt}   Left fingers : {l_cnt}   Total : {total}",
                 (20, 68),  (255, 220, 80)),
                ("RIGHT hand  swipe right/left  →  Next / Prev slide",
                 (20, 98),  HAND_COLORS["Right"]),
                ("LEFT  hand  swipe right/left  →  Next / Prev slide",
                 (20, 125), HAND_COLORS["Left"]),
                (f"Right history : {r_len}/{HISTORY_LENGTH}   Left history : {l_len}/{HISTORY_LENGTH}",
                 (20, 155), (160, 160, 160)),
                ("Swipe Right = Next slide   |   Swipe Left = Prev slide",
                 (20, 182), (200, 200, 200)),
                ("Q = Quit",
                 (20, 209), (200, 200, 200)),
            ]
            for text, pos, color in overlay_lines:
                draw_text(frame, text, pos, color)

            cv2.imshow("Smart Gesture Control – Left/Right Hand Aware", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    except KeyboardInterrupt:
        print("Stopped with Ctrl+C.")
    finally:
        # Runs even if an unexpected error occurs inside the loop.
        hand_detector.close()
        camera.release()
        cv2.destroyAllWindows()
        print("Gesture controller stopped.")


if __name__ == "__main__":
    main()
