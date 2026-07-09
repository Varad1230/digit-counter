# Digit Counter — Hand Gesture Recognition

This project uses computer vision to track hand gestures via webcam and recognize digits (0-9) in real-time, based on which fingers are extended.

## Features
- Real-time hand detection and landmark tracking using MediaPipe
- Finger state analysis (extended/folded) from landmark positions
- Digit recognition (0-9) from finger configurations
- On-screen display of finger count, recognized digit, per-finger status, and FPS
- Screenshot capture (`s` key)

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

## Installation
1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the main script:
```
python main.py
```

On Windows, you can also use:
```
run.bat
```

- Show your hand to the webcam
- The application detects your hand, counts extended fingers, and recognizes the corresponding digit (0-9)
- Press `q` to quit the application
- Press `s` to save a screenshot to `assets/`

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for the hand gesture reference for each digit.

## Project Structure
```
digit-counter/
├── main.py                  # Application entry point
├── requirements.txt
├── run.bat                  # Windows convenience launcher
├── INSTRUCTIONS.md          # Gesture reference for digits 0-9
├── assets/                  # Saved screenshots
└── src/
    ├── hand_tracker.py      # MediaPipe hand detection/tracking
    ├── finger_counter.py    # Finger extended/folded state analysis
    └── digit_recognizer.py  # Maps finger states to digits 0-9
```
