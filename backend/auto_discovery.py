"""
JARVIS Auto Device Discovery - Zero Config
===========================================
Automatically finds devices on local network without manual IP setup.

Features:
- UDP broadcast discovery (finds all JARVIS receivers)
- Auto-pairing (no QR needed for same network)
- Works on hotspot or WiFi
- 100% OFFLINE

How it works:
1. JARVIS broadcasts "JARVIS_DISCOVER" on UDP port 5759
2. Any device running the receiver responds with its info
3. JARVIS adds them to trusted devices automatically
"""

import socket
import json
import threading
import time
from typing import List, Dict, Optional

DISCOVERY_PORT = 5759
TRANSFER_PORT = 5757
BROADCAST_INTERVAL = 5  # seconds


class DeviceDiscovery:
    """
    Auto-discover JARVIS receivers on local network.
    No manual IP configuration needed!
    """
    
    def __init__(self):
        self.discovered_devices: Dict[str, dict] = {}
        self.running = False
        self._thread = None
        self.my_ip = self._get_local_ip()
        self.device_name = "JARVIS_Main"
    
    def _get_local_ip(self) -> str:
        """Get this device's local IP"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def _get_broadcast_address(self) -> str:
        """Get broadcast address for local network"""
        ip_parts = self.my_ip.split('.')
        return f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.255"
    
    def discover_once(self, timeout: float = 2.0) -> List[dict]:
        """
        Send one discovery broadcast and wait for responses.
        Returns list of discovered devices.
        """
        devices = []
        
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)
        
        # Broadcast discovery message
        message = json.dumps({
            "type": "JARVIS_DISCOVER",
            "sender": self.device_name,
            "ip": self.my_ip,
        }).encode()
        
        try:
            broadcast_addr = self._get_broadcast_address()
            sock.sendto(message, (broadcast_addr, DISCOVERY_PORT))
            print(f"[Discovery] Broadcast sent to {broadcast_addr}:{DISCOVERY_PORT}")
            
            # Listen for responses
            start = time.time()
            while time.time() - start < timeout:
                try:
                    data, addr = sock.recvfrom(1024)
                    response = json.loads(data.decode())
                    
                    if response.get("type") == "JARVIS_RECEIVER":
                        device = {
                            "name": response.get("name", "Unknown"),
                            "ip": addr[0],
                            "port": response.get("port", TRANSFER_PORT),
                            "device_type": response.get("device_type", "unknown"),
                        }
                        
                        # Don't add ourselves
                        if addr[0] != self.my_ip:
                            devices.append(device)
                            self.discovered_devices[addr[0]] = device
                            print(f"[Discovery] Found: {device['name']} ({addr[0]}) - {device['device_type']}")
                            
                except socket.timeout:
                    break
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"[Discovery] Error: {e}")
        finally:
            sock.close()
        
        return devices
    
    def start_continuous(self):
        """Start continuous discovery in background"""
        self.running = True
        self._thread = threading.Thread(target=self._discovery_loop, daemon=True)
        self._thread.start()
        print("[Discovery] Continuous discovery started")
    
    def stop(self):
        """Stop continuous discovery"""
        self.running = False
    
    def _discovery_loop(self):
        """Background discovery loop"""
        while self.running:
            self.discover_once(timeout=1.0)
            time.sleep(BROADCAST_INTERVAL)
    
    def get_online_devices(self) -> List[dict]:
        """Get list of currently discovered devices"""
        return list(self.discovered_devices.values())


class ReceiverBeacon:
    """
    Run this on receiving devices (laptop, phone, etc.)
    Responds to JARVIS discovery broadcasts.
    """
    
    def __init__(self, device_name: str = "Receiver", device_type: str = "laptop"):
        self.device_name = device_name
        self.device_type = device_type  # "laptop", "phone", "tablet"
        self.running = False
        self.my_ip = self._get_local_ip()
    
    def _get_local_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def start(self):
        """Start listening for discovery broadcasts"""
        self.running = True
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', DISCOVERY_PORT))
        sock.settimeout(1.0)
        
        print(f"\n{'='*60}")
        print(f"📡 JARVIS Receiver Beacon Active")
        print(f"{'='*60}")
        print(f"  Device: {self.device_name}")
        print(f"  Type: {self.device_type}")
        print(f"  IP: {self.my_ip}")
        print(f"  Listening on port: {DISCOVERY_PORT}")
        print(f"{'='*60}")
        print("  Waiting for JARVIS to discover me...")
        print(f"{'='*60}\n")
        
        while self.running:
            try:
                data, addr = sock.recvfrom(1024)
                message = json.loads(data.decode())
                
                if message.get("type") == "JARVIS_DISCOVER":
                    print(f"[Beacon] Discovery from {addr[0]} ({message.get('sender', 'unknown')})")
                    
                    # Send response
                    response = json.dumps({
                        "type": "JARVIS_RECEIVER",
                        "name": self.device_name,
                        "port": TRANSFER_PORT,
                        "device_type": self.device_type,
                    }).encode()
                    
                    sock.sendto(response, addr)
                    print(f"[Beacon] Responded to {addr[0]}")
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Beacon] Error: {e}")
        
        sock.close()
    
    def stop(self):
        self.running = False


# =============================================================================
# SIMPLE IMAGE RECEIVER (for any device)
# =============================================================================

class SimpleImageReceiver:
    """
    Simple TCP image receiver that works on any device.
    - Receives images from JARVIS
    - Saves to local folder
    - Auto-opens image (optional)
    """
    
    def __init__(self, save_dir: str = "./received_images", auto_open: bool = True):
        self.save_dir = save_dir
        self.auto_open = auto_open
        self.running = False
        
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def start(self):
        """Start receiving images"""
        import struct
        
        self.running = True
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', TRANSFER_PORT))
        server.listen(5)
        server.settimeout(1.0)
        
        print(f"[Receiver] Listening for images on port {TRANSFER_PORT}")
        
        while self.running:
            try:
                client, addr = server.accept()
                print(f"[Receiver] Connection from {addr[0]}")
                
                # Receive header length
                header_len_data = client.recv(4)
                if len(header_len_data) < 4:
                    client.close()
                    continue
                    
                header_len = struct.unpack('!I', header_len_data)[0]
                
                # Receive header
                header_json = client.recv(header_len).decode()
                header = json.loads(header_json)
                
                filename = header.get("filename", "image.jpg")
                size = header.get("size", 0)
                
                print(f"[Receiver] Receiving: {filename} ({size} bytes)")
                
                # Receive image data
                image_data = b''
                while len(image_data) < size:
                    chunk = client.recv(min(8192, size - len(image_data)))
                    if not chunk:
                        break
                    image_data += chunk
                
                # Save image
                import os
                filepath = os.path.join(self.save_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                
                # Send confirmation
                client.send(b"OK")
                client.close()
                
                print(f"[Receiver] ✓ Saved: {filepath}")
                
                # Auto-open
                if self.auto_open:
                    self._open_image(filepath)
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Receiver] Error: {e}")
        
        server.close()
    
    def _open_image(self, filepath: str):
        """Open image with default viewer"""
        import platform
        import subprocess
        import os
        
        try:
            if platform.system() == "Windows":
                os.startfile(filepath)
            elif platform.system() == "Darwin":
                subprocess.run(["open", filepath])
            else:
                subprocess.run(["xdg-open", filepath])
            print(f"[Receiver] Opened: {filepath}")
        except Exception as e:
            print(f"[Receiver] Could not open: {e}")
    
    def stop(self):
        self.running = False


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("📡 JARVIS Auto Discovery System")
    print("="*60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "--discover":
            # Discover devices on network
            discovery = DeviceDiscovery()
            print("\nSearching for JARVIS receivers...")
            devices = discovery.discover_once(timeout=3.0)
            print(f"\nFound {len(devices)} device(s)")
            for d in devices:
                print(f"  - {d['name']} ({d['ip']}) - {d['device_type']}")
        
        elif mode == "--receiver":
            # Run as receiver (on second device)
            name = sys.argv[2] if len(sys.argv) > 2 else "MyDevice"
            dtype = sys.argv[3] if len(sys.argv) > 3 else "laptop"
            
            # Start beacon and receiver in parallel
            beacon = ReceiverBeacon(device_name=name, device_type=dtype)
            receiver = SimpleImageReceiver(auto_open=True)
            
            # Run beacon in background
            beacon_thread = threading.Thread(target=beacon.start, daemon=True)
            beacon_thread.start()
            
            # Run receiver in foreground
            print("\nPress Ctrl+C to stop\n")
            try:
                receiver.start()
            except KeyboardInterrupt:
                beacon.stop()
                receiver.stop()
                print("\nStopped.")
        
        else:
            print(f"Unknown mode: {mode}")
    
    else:
        print("\nUsage:")
        print("  python auto_discovery.py --discover")
        print("      Search for receivers on network")
        print("")
        print("  python auto_discovery.py --receiver [name] [type]")
        print("      Run as receiver (on second device)")
        print("      Example: python auto_discovery.py --receiver MyPhone phone")
        print("")
        print("For Android: Use the companion app or Termux with this script")
