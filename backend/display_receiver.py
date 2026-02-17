"""
JARVIS Display Receiver - Auto-Display Images on Screen
========================================================
Run this on the receiving device (Laptop B).
When an image is received, it automatically displays fullscreen.

Usage:
    python display_receiver.py
    
    # Or with pairing
    python display_receiver.py --pair "pairing_data_json"
"""

import os
import sys
import json
import time
import threading
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from device_transfer import DeviceManager, RECEIVED_DIR

# Try to import display libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    import tkinter as tk
    from PIL import ImageTk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False


class FullscreenDisplay:
    """Display images fullscreen"""
    
    def __init__(self):
        self.current_image = None
        self.window = None
        self._display_thread = None
        
    def show_image_cv2(self, image_path: Path, duration: float = 0):
        """Show image fullscreen using OpenCV"""
        if not CV2_AVAILABLE:
            print("[Display] OpenCV not available")
            return
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[Display] Cannot load: {image_path}")
            return
        
        # Create fullscreen window
        cv2.namedWindow("JARVIS Display", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("JARVIS Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Resize to screen
        screen_width = 1920  # Default, will be overridden
        screen_height = 1080
        
        try:
            import ctypes
            user32 = ctypes.windll.user32
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
        except:
            pass
        
        # Scale image to fit screen
        h, w = img.shape[:2]
        scale = min(screen_width / w, screen_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Center on black background
        canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        x_offset = (screen_width - new_w) // 2
        y_offset = (screen_height - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
        
        cv2.imshow("JARVIS Display", canvas)
        
        if duration > 0:
            cv2.waitKey(int(duration * 1000))
            cv2.destroyAllWindows()
        else:
            # Wait for any key
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def show_image_tk(self, image_path: Path, duration: float = 0):
        """Show image fullscreen using Tkinter"""
        if not TK_AVAILABLE:
            print("[Display] Tkinter/PIL not available")
            return
        
        def display():
            root = tk.Tk()
            root.attributes('-fullscreen', True)
            root.configure(background='black')
            
            # Load and resize image
            img = Image.open(image_path)
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            
            # Scale to fit
            img.thumbnail((screen_width, screen_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            
            label = tk.Label(root, image=photo, bg='black')
            label.image = photo  # Keep reference
            label.pack(expand=True)
            
            # Close on click or key
            root.bind('<Button-1>', lambda e: root.destroy())
            root.bind('<Key>', lambda e: root.destroy())
            root.bind('<Escape>', lambda e: root.destroy())
            
            if duration > 0:
                root.after(int(duration * 1000), root.destroy)
            
            root.mainloop()
        
        # Run in thread to not block
        self._display_thread = threading.Thread(target=display)
        self._display_thread.start()
    
    def show_image(self, image_path: Path, duration: float = 0):
        """Show image using best available method"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"[Display] Image not found: {image_path}")
            return
        
        print(f"[Display] 🖼️ Showing: {image_path.name}")
        
        # Try OpenCV first (better fullscreen)
        if CV2_AVAILABLE:
            self.show_image_cv2(image_path, duration)
        elif TK_AVAILABLE:
            self.show_image_tk(image_path, duration)
        else:
            # Fallback: open with system viewer
            print("[Display] Using system viewer")
            if os.name == 'nt':
                os.startfile(image_path)
            else:
                os.system(f'xdg-open "{image_path}"')


class DisplayReceiver:
    """
    Main receiver class.
    Listens for images and displays them automatically.
    """
    
    def __init__(self, device_name: str = "JARVIS_Display"):
        self.device_name = device_name
        self.display = FullscreenDisplay()
        self.dm = None
        
    def on_image_received(self, filepath: Path, sender_ip: str):
        """Called when image is received"""
        print(f"\n{'='*60}")
        print(f"📥 IMAGE RECEIVED!")
        print(f"   From: {sender_ip}")
        print(f"   File: {filepath}")
        print(f"{'='*60}\n")
        
        # Auto-display
        self.display.show_image(filepath)
    
    def start(self):
        """Start receiver"""
        print("\n" + "="*60)
        print("🖥️ JARVIS DISPLAY RECEIVER")
        print("="*60)
        
        self.dm = DeviceManager(
            device_name=self.device_name,
            on_receive=self.on_image_received,
        )
        
        print(f"\nDevice: {self.device_name}")
        print(f"IP: {self.dm.server.port}")
        print(f"Listening on port: {self.dm.server.port}")
        print("\nWaiting for images...")
        print("Press Ctrl+C to exit.\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
    
    def pair_with(self, pairing_data: str):
        """Pair with another device using pairing data"""
        try:
            data = json.loads(pairing_data)
            self.dm.add_from_pairing_data(data)
            print(f"✓ Paired with: {data.get('name')}")
        except Exception as e:
            print(f"Error pairing: {e}")
    
    def generate_pairing(self):
        """Generate pairing QR for this device"""
        if self.dm is None:
            self.dm = DeviceManager(device_name=self.device_name)
        return self.dm.pair_device()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import numpy as np  # For CV2 canvas
    
    receiver = DisplayReceiver()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--pair" and len(sys.argv) > 2:
            # Pair with provided data
            receiver.dm = DeviceManager(device_name=receiver.device_name)
            receiver.pair_with(sys.argv[2])
        elif sys.argv[1] == "--generate":
            # Generate pairing QR
            receiver.generate_pairing()
            input("\nPress Enter to start receiver...")
    
    receiver.start()
