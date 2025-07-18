# Finger Counter using Hand Tracking

This project uses computer vision to track hand gestures and count fingers (0-9) in real-time.

## Features
- Real-time hand detection using MediaPipe
- Finger counting using landmark position analysis
- Visual display of finger count with hand landmarks

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
python src/finger_counter.py
```

- Show your hand to the webcam
- The application will detect your hand and count the number of extended fingers
- Press 'q' to quit the application 