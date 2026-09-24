# Smart Gesture-Controlled Media & Presentation System

This beginner project uses a normal webcam, OpenCV and MediaPipe's pretrained hand detector. It tracks the wrist landmark and uses simple movement rules. It does not need a dataset or model training. A right swipe sends the Right Arrow key; a left swipe sends the Left Arrow key.

## Requirements

Use Python 3.10 or 3.11 for the smoothest installation experience with MediaPipe. Install Python from python.org and enable **Add Python to PATH** in its installer. In VS Code, open this project folder and choose **Terminal > New Terminal**.

## Create and activate a virtual environment (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell says script execution is disabled, run this once in that terminal and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The terminal prompt should show `(.venv)`. In VS Code, select this interpreter with **Python: Select Interpreter** if prompted.

## Install packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Try the project

Check the camera first:

```powershell
python camera_test.py
```

Show your hand to test landmark detection:

```powershell
python hand_test.py
```

Run the complete application:

```powershell
python main.py
```

A webcam window shows the camera image, hand skeleton when detected, status, and gesture instructions. Press **Q** while that window is selected to stop safely. You can also press Ctrl+C in the terminal.

## Use with PowerPoint

1. Open PowerPoint and start the slide show (F5, or Shift+F5 for the current slide).
2. Keep the slide show active and visible. Start `main.py` from the VS Code terminal.
3. Position yourself so your hand is visible to the webcam, then swipe horizontally across the camera view.
4. A swipe right presses the Right Arrow key (next slide); a swipe left presses the Left Arrow key (previous slide). The one-second cooldown prevents repeated slide changes from one movement.
5. To stop, select the webcam window and press Q. The webcam and windows are released when the program exits.

The window mirrors the camera view, so move your hand from the left side of the displayed frame to the right for a right swipe. If PowerPoint does not respond, click the slide show once to make it active, then try again. PyAutoGUI sends keys to whichever application is active.

## Common problems

- **`py` or `python` is not recognized:** Install Python from python.org, enable PATH in setup, reopen VS Code, and try `py --version`.
- **`No module named ...`:** Activate `.venv` and run `python -m pip install -r requirements.txt` again. Check that VS Code selected `.venv` as its interpreter.
- **PowerShell blocks activation:** Use the temporary `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` command above; it applies only to this terminal session.
- **Camera cannot open / black window:** Close other camera apps, check Windows Settings > Privacy & security > Camera permissions, and try changing `CAMERA_INDEX` from `0` to `1` in the Python file.
- **Hand is not detected:** Improve lighting, keep the whole hand in view, move closer, and avoid a busy background.
- **Swipe does not register:** Make a faster, wider movement across the frame. Adjust `SWIPE_THRESHOLD` near the top of `main.py` (smaller is more sensitive); the wrist history and cooldown can also be adjusted there.
- **Wrong slide changes or no PowerPoint response:** Make sure the slide show is the active window. The app sends standard arrow keys to the active application.
- **MediaPipe installation fails:** Confirm Python is 3.10 or 3.11, then retry the install commands inside the activated virtual environment.

## Checklist

- [ ] requirements installed
- [ ] camera working
- [ ] MediaPipe detecting hand
- [ ] swipe left working
- [ ] swipe right working
- [ ] PowerPoint previous slide working
- [ ] PowerPoint next slide working
