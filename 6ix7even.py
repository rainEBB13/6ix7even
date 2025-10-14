import cv2
import numpy as np
from collections import deque
import time
import os
import subprocess
import random

class SimpleMotionDetector:
    def __init__(self):
        # Motion tracking
        self.prev_frame = None
        self.motion_history = deque(maxlen=30)
        
        # Detection state
        self.last_detection_time = 0
        self.cooldown = 0  # seconds between detections (reduced for vine boom)
        self.motion_threshold = 1500  # Adjust sensitivity
        
        # For tracking motion patterns
        self.high_motion_count = 0
        self.motion_peaks = deque(maxlen=5)
        
        # Check for vine boom sound
        self.sound_file = "vine_boom.mp3"
        self.get_out_file = "get_out.mp3"

        if os.path.exists(self.sound_file):
            print("✅ Vine boom sound loaded!")
        else:
            print("⚠️  Warning: vine_boom.mp3 not found. Please add it to the directory.")

        if os.path.exists(self.get_out_file):
             print("✅ Get out sound loaded!")
        else:
            print("⚠️  Warning: get_out.mp3 not found.")

        
    def detect_motion(self, frame):
        """Detect motion in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0, frame
        
        # Calculate difference
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate total motion
        total_motion = 0
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Filter small movements
                continue
            total_motion += cv2.contourArea(contour)
            
            # Draw contours for visualization
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        self.prev_frame = gray
        
        return total_motion, frame
    
    def detect_67_gesture(self):
        """Detect repeated raising motion (67 gesture)"""
        if len(self.motion_history) < 20:
            return False
        
        current_time = time.time()
        if current_time - self.last_detection_time < self.cooldown:
            return False
        
        # Look for oscillating pattern (raising hands repeatedly)
        motion_list = list(self.motion_history)
        
        # Count peaks (moments of high motion)
        peaks = 0
        for i in range(1, len(motion_list) - 1):
            if (motion_list[i] > motion_list[i-1] and 
                motion_list[i] > motion_list[i+1] and 
                motion_list[i] > self.motion_threshold):
                peaks += 1
        
        # Check for sustained high motion (both hands moving)
        avg_motion = np.mean(motion_list[-15:])
        
        # Detect gesture: multiple peaks + sustained motion
        if peaks >= 2 and avg_motion > self.motion_threshold * 0.7:
            self.last_detection_time = current_time
            return True
        
        return False
    
    def play_vine_boom(self):
        """Play the vine boom sound effect using macOS afplay"""
        if os.path.exists(self.sound_file):
            # Use afplay (built into macOS) to play sound in background
            subprocess.Popen(['afplay', self.sound_file])
            print("\n💥 VINE BOOM! 💥\n")
        else:
            print("\n⚠️  Vine boom sound not found!\n")
    
    def play_get_out(self):
        """Play the Get out sound effect using macOS"""
        if os.path.exists(self.get_out_file):  # CHANGE THIS from self.sound_file
            subprocess.Popen(['afplay', self.get_out_file])  # CHANGE THIS too
            print("\n🚪 GET OUT! 🚪\n")
        else:
            print("\n⚠️ Get out not found!\n")
    
    def play_random_sound(self):
        """Randomly play either vine boom or get out"""
        if random.choice([True, False]):
            self.play_vine_boom()
        else:
            self.play_get_out()

    def run(self):
        """Main loop to run the detector"""
        print("\n🔍 Attempting to open camera...")
        
        # Try different camera indices
        cap = None
        for i in range(5):
            print(f"   Trying camera index {i}...")
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    print(f"   ✅ Camera {i} opened successfully!")
                    cap = test_cap
                    break
                else:
                    test_cap.release()
            else:
                test_cap.release()
        
        if cap is None or not cap.isOpened():
            print("\n❌ Error: Could not open any camera")
            print("\nTroubleshooting tips:")
            print("1. Make sure no other app is using your camera")
            print("2. Check System Preferences > Security & Privacy > Camera")
            print("3. Grant camera access to Terminal/VSCode")
            print("4. Try restarting your computer")
            return
        
        print("\n" + "="*60)
        print("🎯 67 MOTION DETECTOR - VINE BOOM EDITION!")
        print("="*60)
        print("\n📹 Instructions:")
        print("   • Raise both hands up and down repeatedly")
        print("   • Make big, energetic movements")
        print("   • Keep your whole upper body visible")
        print("   • Get ready for the BOOM! 💥")
        print("\n⌨️  Controls:")
        print("   • Press 'q' to quit")
        print("   • Press 's' to adjust sensitivity")
        print("\n" + "="*60 + "\n")
        
        sensitivity_mode = 0  # 0=normal, 1=high, 2=low
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect motion
            motion_amount, processed_frame = self.detect_motion(frame)
            self.motion_history.append(motion_amount)
            
            # Create display
            display = processed_frame.copy()
            h, w = display.shape[:2]
            
            # Draw motion bar
            bar_height = int((motion_amount / 5000) * 100)
            bar_height = min(bar_height, 100)
            cv2.rectangle(display, (w-50, h-20), (w-10, h-20-bar_height), (0, 255, 0), -1)
            cv2.rectangle(display, (w-50, h-120), (w-10, h-20), (255, 255, 255), 2)
            
            # Display info
            cv2.putText(display, "67 MOTION DETECTOR", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            motion_text = f"Motion: {int(motion_amount)}"
            cv2.putText(display, motion_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            sens_text = ["Normal", "High", "Low"][sensitivity_mode]
            cv2.putText(display, f"Sensitivity: {sens_text}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Check for gesture
            if self.detect_67_gesture():
                cv2.putText(display, "67 DETECTED!", (w//2 - 150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.imshow('67 Motion Detector', display)
                cv2.waitKey(200)  # Brief pause to show detection
                
                self.play_random_sound()  # TO THIS
            
            cv2.imshow('67 Motion Detector', display)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Cycle sensitivity
                sensitivity_mode = (sensitivity_mode + 1) % 3
                if sensitivity_mode == 0:
                    self.motion_threshold = 1500
                elif sensitivity_mode == 1:
                    self.motion_threshold = 1000
                else:
                    self.motion_threshold = 2000
                print(f"Sensitivity changed to: {['Normal', 'High', 'Low'][sensitivity_mode]}")
        
        print("\n👋 Shutting down...\n")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = SimpleMotionDetector()
    detector.run()

    # add a stop function

