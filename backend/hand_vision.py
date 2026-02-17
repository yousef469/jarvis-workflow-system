"""
JARVIS Hand Vision System - Offline Hand Detection & Gesture Recognition
=========================================================================
Iron Man-style hand tracking with gesture-based image capture.

Features:
- Real-time hand tracking (MediaPipe)
- Gesture recognition (catch, point, thumbs up)
- Image capture on gesture
- 100% OFFLINE - no internet needed

Usage:
    from hand_vision import HandVision
    
    vision = HandVision()
    vision.start()  # Opens camera, tracks hands
    # When "catch" gesture detected → captures image
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass
from enum import Enum

# MediaPipe for hand detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class Gesture(Enum):
    """Recognized gestures"""
    NONE = "none"
    OPEN_HAND = "open_hand"
    CLOSED_FIST = "closed_fist"
    CATCH = "catch"           # Open → Closed quickly
    POINT = "point"           # Index finger extended
    THUMBS_UP = "thumbs_up"
    PEACE = "peace"           # Two fingers


@dataclass
class HandState:
    """Current state of detected hand"""
    detected: bool = False
    landmarks: list = None
    gesture: Gesture = Gesture.NONE
    confidence: float = 0.0
    position: tuple = (0, 0)  # Center of hand
    timestamp: float = 0.0


class GestureEngine:
    """
    Recognizes gestures from hand landmarks.
    Pure Python logic - no heavy AI needed.
    """
    
    def __init__(self):
        self.history: List[Dict] = []
        self.history_max = 30  # ~1 second at 30fps
        self.catch_threshold = 0.3  # seconds for catch gesture
        
    def analyze(self, landmarks) -> Gesture:
        """Analyze landmarks and return detected gesture"""
        if not landmarks:
            return Gesture.NONE
        
        # Get finger states
        fingers = self._get_finger_states(landmarks)
        
        # Record history
        now = time.time()
        self.history.append({
            "time": now,
            "fingers": fingers,
            "open": sum(fingers) >= 4
        })
        
        # Trim history
        if len(self.history) > self.history_max:
            self.history.pop(0)
        
        # Check for CATCH gesture (open → closed quickly)
        if self._detect_catch():
            return Gesture.CATCH
        
        # Check static gestures
        thumb, index, middle, ring, pinky = fingers
        
        # Closed fist - all fingers down
        if sum(fingers) == 0:
            return Gesture.CLOSED_FIST
        
        # Open hand - all fingers up
        if sum(fingers) >= 4:
            return Gesture.OPEN_HAND
        
        # Point - only index up
        if index and not middle and not ring and not pinky:
            return Gesture.POINT
        
        # Thumbs up - only thumb up
        if thumb and not index and not middle and not ring and not pinky:
            return Gesture.THUMBS_UP
        
        # Peace - index and middle up
        if index and middle and not ring and not pinky:
            return Gesture.PEACE
        
        return Gesture.NONE
    
    def _get_finger_states(self, landmarks) -> List[bool]:
        """
        Determine which fingers are extended.
        Returns [thumb, index, middle, ring, pinky]
        """
        # Landmark indices for fingertips and bases
        # Thumb: 4 (tip), 2 (base)
        # Index: 8 (tip), 6 (base)
        # Middle: 12 (tip), 10 (base)
        # Ring: 16 (tip), 14 (base)
        # Pinky: 20 (tip), 18 (base)
        
        tips = [4, 8, 12, 16, 20]
        bases = [2, 6, 10, 14, 18]
        
        fingers = []
        
        for i, (tip, base) in enumerate(zip(tips, bases)):
            tip_y = landmarks[tip].y
            base_y = landmarks[base].y
            
            # Thumb uses x-axis (horizontal)
            if i == 0:
                tip_x = landmarks[tip].x
                base_x = landmarks[base].x
                # Thumb extended if tip is far from palm
                fingers.append(abs(tip_x - base_x) > 0.05)
            else:
                # Other fingers: extended if tip is above base (lower y)
                fingers.append(tip_y < base_y - 0.02)
        
        return fingers
    
    def _detect_catch(self) -> bool:
        """
        Detect catch gesture: hand goes from open to closed quickly.
        """
        if len(self.history) < 10:
            return False
        
        now = time.time()
        
        # Look for open hand in recent history
        open_time = None
        for entry in reversed(self.history):
            if now - entry["time"] > self.catch_threshold:
                break
            if entry["open"]:
                open_time = entry["time"]
                break
        
        if not open_time:
            return False
        
        # Check if hand is now closed
        recent = self.history[-3:]  # Last few frames
        closed_count = sum(1 for e in recent if not e["open"])
        
        if closed_count >= 2:
            # Clear history to prevent repeat detection
            self.history.clear()
            return True
        
        return False
    
    def reset(self):
        """Reset gesture history"""
        self.history.clear()


class HandVision:
    """
    Main hand vision system.
    Opens camera, tracks hands, recognizes gestures, captures images.
    
    Iron Man Mode: Display an image overlay that you can "catch" and send!
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        capture_dir: str = "./captured_images",
        on_capture: Optional[Callable] = None,
        on_gesture: Optional[Callable] = None,
        overlay_image: Optional[str] = None,  # Image to display and "catch"
    ):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe required. Run: pip install mediapipe")
        
        self.camera_id = camera_id
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(exist_ok=True)
        
        # Callbacks
        self.on_capture = on_capture  # Called when image captured
        self.on_gesture = on_gesture  # Called on any gesture
        
        # Overlay image (Iron Man mode)
        self.overlay_image_path = overlay_image
        self.overlay_image = None
        self.overlay_position = [100, 100]  # x, y position
        self.overlay_size = (200, 200)
        self.overlay_grabbed = False
        self.overlay_sent = False
        
        # Load overlay image if provided
        if overlay_image and Path(overlay_image).exists():
            img = cv2.imread(overlay_image)
            if img is not None:
                # Resize to overlay size
                self.overlay_image = cv2.resize(img, self.overlay_size)
                print(f"[HandVision] Overlay image loaded: {overlay_image}")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        
        # Gesture engine
        self.gesture_engine = GestureEngine()
        
        # State
        self.running = False
        self.cap = None
        self.current_state = HandState()
        self.last_capture_time = 0
        self.capture_cooldown = 2.0  # Seconds between captures
        
        # Threading
        self._thread = None
        self._lock = threading.Lock()
    
    def start(self, show_window: bool = True):
        """Start hand tracking (blocking if show_window=True)"""
        self.running = True
        
        if show_window:
            self._run_with_display()
        else:
            self._thread = threading.Thread(target=self._run_headless)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self):
        """Stop hand tracking"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def _run_with_display(self):
        """Run with OpenCV window display"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print("[HandVision] Error: Could not open camera")
            return
        
        print("[HandVision] Camera opened. Press 'Q' to quit.")
        print("[HandVision] Make a CATCH gesture (open hand → close) to capture!")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame, state = self._process_frame(frame)
            
            # Draw UI
            frame = self._draw_ui(frame, state)
            
            # Show
            cv2.imshow("JARVIS Hand Vision", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop()
    
    def _run_headless(self):
        """Run without display (background mode)"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print("[HandVision] Error: Could not open camera")
            return
        
        print("[HandVision] Running in background mode...")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            _, state = self._process_frame(frame)
            
            time.sleep(0.03)  # ~30 FPS
        
        self.cap.release()
    
    def _process_frame(self, frame) -> tuple:
        """Process a single frame for hand detection"""
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        state = HandState(timestamp=time.time())
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )
            
            # Get gesture
            gesture = self.gesture_engine.analyze(hand_landmarks.landmark)
            
            # Calculate hand center
            cx = int(np.mean([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            cy = int(np.mean([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            
            state.detected = True
            state.landmarks = hand_landmarks.landmark
            state.gesture = gesture
            state.position = (cx, cy)
            
            # Handle gesture callbacks
            if gesture != Gesture.NONE and self.on_gesture:
                self.on_gesture(gesture, state)
            
            # Handle CATCH gesture → capture image
            if gesture == Gesture.CATCH:
                self._handle_catch(frame)
        
        with self._lock:
            self.current_state = state
        
        return frame, state
    
    def _handle_catch(self, frame):
        """Handle catch gesture - capture image or send overlay"""
        now = time.time()
        
        # Cooldown check
        if now - self.last_capture_time < self.capture_cooldown:
            return
        
        self.last_capture_time = now
        
        # IRON MAN MODE: If overlay image exists, send it instead of capturing
        if self.overlay_image is not None and self.overlay_image_path and not self.overlay_sent:
            print(f"[HandVision] 🚀 SENDING OVERLAY: {self.overlay_image_path}")
            self.overlay_sent = True
            
            # Callback with the overlay image path
            if self.on_capture:
                self.on_capture(self.overlay_image_path)
            
            # Auto-close after 2 seconds
            def auto_close():
                time.sleep(2)
                self.running = False
            threading.Thread(target=auto_close, daemon=True).start()
            return
        
        # Normal mode: Capture from camera
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = self.capture_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"[HandVision] 📸 CAPTURED: {filepath}")
        
        # Callback
        if self.on_capture:
            self.on_capture(str(filepath))
    
    def _draw_ui(self, frame, state: HandState):
        """Draw UI overlay on frame with Iron Man-style image overlay"""
        h, w = frame.shape[:2]
        
        # Status bar background
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        
        # Title
        title = "JARVIS HAND VISION"
        if self.overlay_image is not None:
            title += " - IRON MAN MODE"
        cv2.putText(frame, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Hand status
        if state.detected:
            status = f"Hand: DETECTED | Gesture: {state.gesture.value}"
            color = (0, 255, 0)
        else:
            status = "Hand: NOT DETECTED"
            color = (0, 0, 255)
        
        cv2.putText(frame, status, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw overlay image (Iron Man hologram style)
        if self.overlay_image is not None and not self.overlay_sent:
            ox, oy = self.overlay_position
            oh, ow = self.overlay_image.shape[:2]
            
            # Keep within bounds
            ox = max(0, min(ox, w - ow))
            oy = max(60, min(oy, h - oh - 30))
            
            # If hand detected and near image, move image toward hand
            if state.detected and not self.overlay_grabbed:
                hx, hy = state.position
                dist = ((hx - ox - ow//2)**2 + (hy - oy - oh//2)**2)**0.5
                
                # If hand is close, highlight the image
                if dist < 150:
                    # Draw glow effect
                    cv2.rectangle(frame, (ox-5, oy-5), (ox+ow+5, oy+oh+5), (0, 255, 255), 3)
            
            # Draw the overlay image with transparency effect
            try:
                # Add cyan border (hologram style)
                cv2.rectangle(frame, (ox-2, oy-2), (ox+ow+2, oy+oh+2), (255, 255, 0), 2)
                
                # Blend overlay onto frame
                alpha = 0.85
                roi = frame[oy:oy+oh, ox:ox+ow]
                blended = cv2.addWeighted(self.overlay_image, alpha, roi, 1-alpha, 0)
                frame[oy:oy+oh, ox:ox+ow] = blended
                
                # Label
                cv2.putText(frame, "CATCH to send!", (ox, oy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            except:
                pass
        
        # Show "SENT!" message
        if self.overlay_sent:
            cv2.putText(frame, "IMAGE SENT!", (w//2 - 100, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Instructions
        if self.overlay_image is not None:
            instr = "CATCH the image to send it! | Q to quit"
        else:
            instr = "Make CATCH gesture to capture | Q to quit"
        cv2.putText(frame, instr, (10, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Gesture indicator
        if state.gesture == Gesture.CATCH and not self.overlay_sent:
            cv2.putText(frame, "CATCH!", (w//2 - 50, h//2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        
        return frame
    
    def get_state(self) -> HandState:
        """Get current hand state (thread-safe)"""
        with self._lock:
            return self.current_state
    
    def capture_now(self) -> Optional[str]:
        """Manually trigger capture"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"manual_{timestamp}.jpg"
        filepath = self.capture_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"[HandVision] 📸 Manual capture: {filepath}")
        
        return str(filepath)


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🖐️ JARVIS Hand Vision System")
    print("="*60)
    print("\nControls:")
    print("  - Show open hand, then close to CATCH")
    print("  - Press Q to quit")
    print("="*60)
    
    def on_capture(filepath):
        print(f"\n🎯 IMAGE CAPTURED: {filepath}\n")
    
    def on_gesture(gesture, state):
        if gesture != Gesture.NONE:
            print(f"[Gesture] {gesture.value}")
    
    vision = HandVision(
        on_capture=on_capture,
        on_gesture=on_gesture,
    )
    
    vision.start(show_window=True)
