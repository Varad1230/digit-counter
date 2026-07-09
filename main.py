"""
Finger Counter using Hand Tracking

This application uses computer vision to track hand gestures and count digits (0-9)
based on finger positions.

Usage:
    python main.py

Controls:
    Press 'q' to quit
    Press 's' to save a screenshot
"""

import cv2
import os
import time
import sys

# Add src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from finger_counter import FingerCounter
from hand_tracker import HandTracker


def save_screenshot(frame, output_dir='assets'):
    """Save a screenshot of the current frame"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"screenshot_{timestamp}.jpg")

    cv2.imwrite(filename, frame)
    print(f"Screenshot saved: {filename}")

    return filename


def main():
    print("=" * 50)
    print("Finger Counter using Hand Tracking")
    print("=" * 50)
    print("Controls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save a screenshot")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counter = FingerCounter()
    tracker = HandTracker()

    prev_time = 0
    curr_time = 0

    print("Starting hand detection. Move your hand in front of the camera...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)

        # Track hands using MediaPipe
        frame, results = tracker.track_hands(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Count fingers using actual hand landmarks
                finger_status = counter.count_fingers(
                    hand_landmarks, frame.shape[1], frame.shape[0]
                )

                # Recognize digit from finger configuration
                digit = counter.digit_recognizer.recognize_digit(tuple(finger_status))
                digit_name = counter.digit_recognizer.get_digit_name(digit)

                # Display finger count and recognized digit
                cv2.putText(
                    frame,
                    f"Finger Count: {sum(finger_status)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    f"Digit: {digit_name}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                # Draw finger status
                for i, status in enumerate(finger_status):
                    finger_name = ["Thumb", "Index", "Middle", "Ring", "Pinky"][i]
                    status_text = "Extended" if status == 1 else "Folded"
                    cv2.putText(
                        frame,
                        f"{finger_name}: {status_text}",
                        (390, 30 + (i * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )
        else:
            # No hand detected - don't run digit recognition at all
            cv2.putText(
                frame,
                "No hand detected",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Display controls
        cv2.putText(
            frame,
            "Press 'q' to quit, 's' to screenshot",
            (10, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Display the frame
        cv2.imshow('Finger Counter', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            saved_file = save_screenshot(frame)
            print(f"Screenshot saved to {saved_file}")

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()