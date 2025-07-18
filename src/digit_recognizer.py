import numpy as np

class DigitRecognizer:
    def __init__(self):
        # Initialize digit mapping for 0-9
        # Each entry maps a specific finger configuration to a digit
        # Format: [thumb, index, middle, ring, pinky]
        self.digit_mapping = {
            # 0: All fingers closed (fist)
            (0, 0, 0, 0, 0): 0,
            
            # 1: Only index finger extended
            (0, 1, 0, 0, 0): 1,
            
            # 2: Index and middle fingers extended (V sign)
            (0, 1, 1, 0, 0): 2,
            
            # 3: Thumb, index, and middle fingers extended
            (1, 1, 1, 0, 0): 3,
            
            # 4: All fingers except thumb extended
            (0, 1, 1, 1, 1): 4,
            
            # 5: All five fingers extended (open hand)
            (1, 1, 1, 1, 1): 5,
            
            # 6: Thumb, pinky and index extended (rock sign)
            (1, 1, 0, 0, 1): 6,
            
            # 7: Thumb, index, middle and ring extended
            (1, 1, 1, 1, 0): 7,
            
            # 8: Thumb, middle, ring and pinky extended
            (1, 0, 1, 1, 1): 8,
            
            # 9: Thumb and pinky extended (hang loose sign)
            (1, 0, 0, 0, 1): 9
        }
        
        # Default value for unrecognized gestures
        self.default_digit = -1
    
    def recognize_digit(self, finger_status):
        """
        Recognize digit from finger configuration
        
        Args:
            finger_status: List of 5 values indicating each finger's state (0: folded, 1: extended)
                           [thumb, index, middle, ring, pinky]
        
        Returns:
            int: Recognized digit (0-9) or -1 if unrecognized
        """
        # Convert finger status to tuple for dictionary lookup
        finger_tuple = tuple(finger_status)
        
        # Return the corresponding digit or default value
        return self.digit_mapping.get(finger_tuple, self.default_digit)
    
    def get_digit_name(self, digit):
        """
        Get the name of the digit
        
        Args:
            digit: Integer digit (0-9) or -1
            
        Returns:
            str: Name of the digit or "Unknown"
        """
        if digit == -1:
            return "Unknown"
        return str(digit) 