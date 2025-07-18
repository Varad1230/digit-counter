import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Allow tracking multiple hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def track_hands(self, frame):
        """
        Track hands in the given frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            frame: Processed frame with hand landmarks drawn
            results: MediaPipe hand processing results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Draw hand label
                wrist = hand_landmarks.landmark[0]
                x_wrist = int(wrist.x * frame.shape[1])
                y_wrist = int(wrist.y * frame.shape[0])
                
                cv2.putText(
                    frame,
                    "Hand",
                    (x_wrist - 20, y_wrist - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Draw bounding box
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])
                
                x_min, y_min = int(x_min * frame.shape[1]), int(y_min * frame.shape[0])
                x_max, y_max = int(x_max * frame.shape[1]), int(y_max * frame.shape[0])
                
                cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (0, 255, 0), 2)
        
        return frame, results

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize hand tracker
    tracker = HandTracker()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break
        
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Track hands
        frame, results = tracker.track_hands(frame)
        
        # Add instructions
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Display the frame
        cv2.imshow('Hand Tracker', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 