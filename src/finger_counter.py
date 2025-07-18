import cv2
import mediapipe as mp
import numpy as np
import math
from digit_recognizer import DigitRecognizer

class FingerCounter:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Finger status (0: folded, 1: extended)
        self.fingers = [0, 0, 0, 0, 0]  # Thumb, Index, Middle, Ring, Pinky
        
        # Initialize digit recognizer
        self.digit_recognizer = DigitRecognizer()
        
    def count_fingers(self, hand_landmarks):
        """
        Analyze hand landmarks to determine finger states
        Returns the finger status array [thumb, index, middle, ring, pinky]
        """
        # Get hand landmarks
        points = []
        for landmark in hand_landmarks.landmark:
            points.append((int(landmark.x * 640), int(landmark.y * 480)))
        
        # Check if thumb is extended (using angle)
        thumb_angle = self._calculate_angle(points[4], points[3], points[2])
        self.fingers[0] = 1 if thumb_angle > 150 else 0
        
        # Check if other fingers are extended
        # Extended finger has y-coordinate of fingertip lower than pip (middle finger joint)
        # (Assuming hand is upright)
        if points[8][1] < points[6][1]:  # Index finger
            self.fingers[1] = 1
        else:
            self.fingers[1] = 0
            
        if points[12][1] < points[10][1]:  # Middle finger
            self.fingers[2] = 1
        else:
            self.fingers[2] = 0
            
        if points[16][1] < points[14][1]:  # Ring finger
            self.fingers[3] = 1
        else:
            self.fingers[3] = 0
            
        if points[20][1] < points[18][1]:  # Pinky finger
            self.fingers[4] = 1
        else:
            self.fingers[4] = 0
            
        return self.fingers
    
    def _calculate_angle(self, a, b, c):
        """
        Calculate angle between three points
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize finger counter
    counter = FingerCounter()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break
        
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get hand landmarks
        results = counter.hands.process(rgb_frame)
        
        # Draw hand landmarks and count fingers
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                counter.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    counter.mp_hands.HAND_CONNECTIONS,
                    counter.mp_drawing_styles.get_default_hand_landmarks_style(),
                    counter.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Count fingers and get their status
                finger_status = counter.count_fingers(hand_landmarks)
                
                # Recognize digit from finger configuration
                digit = counter.digit_recognizer.recognize_digit(finger_status)
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
                
                # Draw bounding box around hand
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])
                
                x_min, y_min = int(x_min * frame.shape[1]), int(y_min * frame.shape[0])
                x_max, y_max = int(x_max * frame.shape[1]), int(y_max * frame.shape[0])
                
                cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (0, 255, 0), 2)
        else:
            # No hand detected
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
        
        # Display the frame
        cv2.imshow('Finger Counter', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 