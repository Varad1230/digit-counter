import numpy as np
from digit_recognizer import DigitRecognizer


class FingerCounter:
    def __init__(self):
        # Finger status (0: folded, 1: extended)
        self.fingers = [0, 0, 0, 0, 0]  # Thumb, Index, Middle, Ring, Pinky

        # Initialize digit recognizer
        self.digit_recognizer = DigitRecognizer()

    def count_fingers(self, hand_landmarks, frame_width=640, frame_height=480):
        """
        Analyze hand landmarks to determine finger states.

        Args:
            hand_landmarks: A single hand's landmarks from MediaPipe
                            (results.multi_hand_landmarks[i])
            frame_width: Width of the frame (for converting normalized coords)
            frame_height: Height of the frame (for converting normalized coords)

        Returns:
            self.fingers: [thumb, index, middle, ring, pinky] -> 1 = extended, 0 = folded
        """
        points = []
        for landmark in hand_landmarks.landmark:
            points.append((landmark.x * frame_width, landmark.y * frame_height))

        # Thumb: check the angle at the IP joint (landmarks 4, 3, 2)
        thumb_angle = self._calculate_angle(points[4], points[3], points[2])
        self.fingers[0] = 1 if thumb_angle > 150 else 0

        # Other fingers: tip is "above" (smaller y) the PIP joint when extended
        self.fingers[1] = 1 if points[8][1] < points[6][1] else 0    # Index
        self.fingers[2] = 1 if points[12][1] < points[10][1] else 0  # Middle
        self.fingers[3] = 1 if points[16][1] < points[14][1] else 0  # Ring
        self.fingers[4] = 1 if points[20][1] < points[18][1] else 0  # Pinky

        return self.fingers

    def _calculate_angle(self, a, b, c):
        """
        Calculate the angle (in degrees) at point b, formed by a-b-c.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # avoid NaN from float error
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)