"""
JARVIS Device Transfer System - Offline Image Transfer
=======================================================
Transfer images between devices on local network.

Features:
- QR-based device pairing (one-time setup)
- Automatic device discovery on local network
- Secure file transfer over TCP
- Trusted device memory (persistent)
- 100% OFFLINE - works on hotspot or local WiFi

Usage:
    # On sender (JARVIS main device):
    from device_transfer import DeviceTransfer
    transfer = DeviceTransfer()
    transfer.send_image("path/to/image.jpg")
    
    # On receiver (display device):
    python device_transfer.py --receiver
"""

import socket
import json
import os
import time
import threading
import subprocess
import platform
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
import struct

# QR code generation
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

TRANSFER_PORT = 5757
DISCOVERY_PORT = 5758
TRUSTED_DEVICES_FILE = Path("./trusted_devices.json")
RECEIVED_IMAGES_DIR = Path("./received_images")
RECEIVED_IMAGES_DIR.mkdir(exist_ok=True)


@dataclass
class TrustedDevice:
    """A trusted/paired device"""
    name: str
    ip: str
    secret_key: str
    paired_at: str
    last_seen: str = ""
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# =============================================================================
# TRUSTED DEVICE MEMORY
# =============================================================================

class TrustedDeviceMemory:
    """
    Persistent storage for trusted devices.
    Devices only need to be paired once via QR code.
    """
    
    def __init__(self, filepath: Path = TRUSTED_DEVICES_FILE):
        self.filepath = filepath
        self.devices: Dict[str, TrustedDevice] = {}
        self._load()
    
    def _load(self):
        """Load trusted devices from file"""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    for name, d in data.items():
                        self.devices[name] = TrustedDevice.from_dict(d)
                print(f"[TrustedDevices] Loaded {len(self.devices)} trusted devices")
            except Exception as e:
                print(f"[TrustedDevices] Error loading: {e}")
    
    def _save(self):
        """Save trusted devices to file"""
        try:
            data = {name: dev.to_dict() for name, dev in self.devices.items()}
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[TrustedDevices] Error saving: {e}")
    
    def add(self, device: TrustedDevice):
        """Add a trusted device"""
        self.devices[device.name] = device
        self._save()
        print(f"[TrustedDevices] Added: {device.name} ({device.ip})")
    
    def remove(self, name: str):
        """Remove a trusted device"""
        if name in self.devices:
            del self.devices[name]
            self._save()
            print(f"[TrustedDevices] Removed: {name}")
    
    def get(self, name: str) -> Optional[TrustedDevice]:
        """Get a trusted device by name"""
        return self.devices.get(name)
    
    def get_by_ip(self, ip: str) -> Optional[TrustedDevice]:
        """Get a trusted device by IP"""
        for dev in self.devices.values():
            if dev.ip == ip:
                return dev
        return None
    
    def update_ip(self, name: str, new_ip: str):
        """Update device IP (for DHCP changes)"""
        if name in self.devices:
            self.devices[name].ip = new_ip
            self.devices[name].last_seen = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save()
    
    def list_all(self) -> List[TrustedDevice]:
        """List all trusted devices"""
        return list(self.devices.values())
    
    def is_trusted(self, ip: str, secret: str) -> bool:
        """Check if device is trusted"""
        dev = self.get_by_ip(ip)
        if dev and dev.secret_key == secret:
            return True
        return False


# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def get_local_ip() -> str:
    """Get this device's local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def get_subnet() -> str:
    """Get local subnet (e.g., 192.168.1)"""
    ip = get_local_ip()
    parts = ip.split('.')
    return '.'.join(parts[:3])


def scan_network() -> List[str]:
    """
    Scan local network for active devices.
    Uses ARP table for speed.
    """
    active_ips = []
    
    try:
        # Use ARP table (fast)
        if platform.system() == "Windows":
            output = subprocess.check_output("arp -a", shell=True).decode()
            for line in output.split('\n'):
                if 'dynamic' in line.lower():
                    parts = line.split()
                    if len(parts) >= 1:
                        ip = parts[0]
                        if ip.startswith(get_subnet()):
                            active_ips.append(ip)
        else:
            output = subprocess.check_output("arp -a", shell=True).decode()
            for line in output.split('\n'):
                if '(' in line and ')' in line:
                    ip = line.split('(')[1].split(')')[0]
                    if ip.startswith(get_subnet()):
                        active_ips.append(ip)
    except Exception as e:
        print(f"[Network] ARP scan error: {e}")
    
    return active_ips


def check_device_online(ip: str, port: int = TRANSFER_PORT, timeout: float = 1.0) -> bool:
    """Check if a device is online and listening"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False


# =============================================================================
# QR PAIRING
# =============================================================================

def generate_pairing_qr(device_name: str = "JARVIS") -> tuple:
    """
    Generate QR code for device pairing.
    Returns (qr_image_path, secret_key)
    """
    if not QR_AVAILABLE:
        print("[Pairing] qrcode module not available")
        return None, None
    
    # Generate secret key
    secret_key = secrets.token_hex(16)
    
    # Pairing data
    pairing_data = {
        "name": device_name,
        "ip": get_local_ip(),
        "port": TRANSFER_PORT,
        "secret": secret_key,
    }
    
    # Create QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(json.dumps(pairing_data))
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save QR image
    qr_path = Path(f"./pairing_qr_{device_name}.png")
    img.save(qr_path)
    
    print(f"\n{'='*60}")
    print("📱 DEVICE PAIRING QR CODE GENERATED")
    print(f"{'='*60}")
    print(f"  Device: {device_name}")
    print(f"  IP: {get_local_ip()}")
    print(f"  Port: {TRANSFER_PORT}")
    print(f"  QR saved to: {qr_path}")
    print(f"{'='*60}")
    print("  Scan this QR with the other device to pair!")
    print(f"{'='*60}\n")
    
    return str(qr_path), secret_key


def parse_pairing_qr(qr_data: str) -> Optional[TrustedDevice]:
    """Parse QR code data and create TrustedDevice"""
    try:
        data = json.loads(qr_data)
        return TrustedDevice(
            name=data["name"],
            ip=data["ip"],
            secret_key=data["secret"],
            paired_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as e:
        print(f"[Pairing] Invalid QR data: {e}")
        return None


# =============================================================================
# FILE TRANSFER
# =============================================================================

class DeviceTransfer:
    """
    Main device transfer system.
    Handles sending and receiving images between paired devices.
    """
    
    def __init__(self):
        self.memory = TrustedDeviceMemory()
        self.local_ip = get_local_ip()
        self.running = False
        self._receiver_thread = None
    
    def pair_device(self, name: str = "JARVIS") -> str:
        """Generate pairing QR code"""
        qr_path, secret = generate_pairing_qr(name)
        
        # Store our own secret for verification
        self_device = TrustedDevice(
            name=f"{name}_SELF",
            ip=self.local_ip,
            secret_key=secret,
            paired_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.memory.add(self_device)
        
        return qr_path
    
    def add_trusted_device(self, qr_data: str) -> bool:
        """Add device from scanned QR data"""
        device = parse_pairing_qr(qr_data)
        if device:
            self.memory.add(device)
            return True
        return False
    
    def add_trusted_device_manual(self, name: str, ip: str, secret: str) -> bool:
        """Manually add a trusted device"""
        device = TrustedDevice(
            name=name,
            ip=ip,
            secret_key=secret,
            paired_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.memory.add(device)
        return True
    
    def find_online_devices(self) -> List[TrustedDevice]:
        """Find which trusted devices are currently online"""
        online = []
        
        for device in self.memory.list_all():
            if "_SELF" in device.name:
                continue
            
            if check_device_online(device.ip):
                device.last_seen = time.strftime("%Y-%m-%d %H:%M:%S")
                online.append(device)
                print(f"[Discovery] ✓ {device.name} ({device.ip}) - ONLINE")
            else:
                print(f"[Discovery] ✗ {device.name} ({device.ip}) - offline")
        
        return online
    
    def send_image(self, image_path: str, device_name: str = None) -> bool:
        """
        Send image to a trusted device.
        If device_name is None, sends to first online device.
        """
        # Find target device
        if device_name:
            device = self.memory.get(device_name)
            if not device:
                print(f"[Transfer] Device '{device_name}' not found in trusted devices")
                return False
        else:
            # Find first online device
            online = self.find_online_devices()
            if not online:
                print("[Transfer] No trusted devices online")
                return False
            device = online[0]
        
        # Check if online
        if not check_device_online(device.ip):
            print(f"[Transfer] Device {device.name} is offline")
            return False
        
        # Read image
        if not os.path.exists(image_path):
            print(f"[Transfer] Image not found: {image_path}")
            return False
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        filename = os.path.basename(image_path)
        
        # Prepare header
        header = {
            "filename": filename,
            "size": len(image_data),
            "secret": device.secret_key,
            "sender": self.local_ip,
        }
        header_json = json.dumps(header).encode()
        
        try:
            # Connect
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((device.ip, TRANSFER_PORT))
            
            # Send header length + header
            sock.send(struct.pack('!I', len(header_json)))
            sock.send(header_json)
            
            # Send image data
            sock.sendall(image_data)
            
            # Wait for confirmation
            response = sock.recv(1024).decode()
            sock.close()
            
            if response == "OK":
                print(f"[Transfer] ✓ Image sent to {device.name} ({device.ip})")
                return True
            else:
                print(f"[Transfer] ✗ Transfer failed: {response}")
                return False
                
        except Exception as e:
            print(f"[Transfer] Error: {e}")
            return False
    
    def start_receiver(self, on_receive: callable = None, auto_display: bool = True):
        """
        Start receiver server to accept incoming images.
        
        Args:
            on_receive: Callback function(filepath) when image received
            auto_display: Automatically open received images
        """
        self.running = True
        self._on_receive = on_receive
        self._auto_display = auto_display
        
        self._receiver_thread = threading.Thread(target=self._receiver_loop)
        self._receiver_thread.daemon = True
        self._receiver_thread.start()
        
        print(f"\n{'='*60}")
        print("📥 JARVIS RECEIVER STARTED")
        print(f"{'='*60}")
        print(f"  Listening on: {self.local_ip}:{TRANSFER_PORT}")
        print(f"  Save directory: {RECEIVED_IMAGES_DIR}")
        print(f"  Auto-display: {auto_display}")
        print(f"{'='*60}\n")
    
    def stop_receiver(self):
        """Stop receiver server"""
        self.running = False
    
    def _receiver_loop(self):
        """Main receiver loop"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', TRANSFER_PORT))
        server.listen(5)
        server.settimeout(1)
        
        while self.running:
            try:
                client, addr = server.accept()
                threading.Thread(
                    target=self._handle_client,
                    args=(client, addr),
                    daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Receiver] Error: {e}")
        
        server.close()
    
    def _handle_client(self, client: socket.socket, addr: tuple):
        """Handle incoming connection"""
        try:
            # Receive header length
            header_len_data = client.recv(4)
            header_len = struct.unpack('!I', header_len_data)[0]
            
            # Receive header
            header_json = client.recv(header_len).decode()
            header = json.loads(header_json)
            
            filename = header["filename"]
            size = header["size"]
            secret = header["secret"]
            sender = header.get("sender", addr[0])
            
            # Verify trust (check if secret matches any trusted device)
            trusted = False
            for dev in self.memory.list_all():
                if dev.secret_key == secret:
                    trusted = True
                    break
            
            if not trusted:
                print(f"[Receiver] ✗ Untrusted device: {addr[0]}")
                client.send(b"UNTRUSTED")
                client.close()
                return
            
            # Receive image data
            image_data = b''
            while len(image_data) < size:
                chunk = client.recv(min(8192, size - len(image_data)))
                if not chunk:
                    break
                image_data += chunk
            
            # Save image
            filepath = RECEIVED_IMAGES_DIR / filename
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            # Send confirmation
            client.send(b"OK")
            client.close()
            
            print(f"[Receiver] ✓ Image received: {filepath}")
            
            # Callback
            if self._on_receive:
                self._on_receive(str(filepath))
            
            # Auto-display
            if self._auto_display:
                self._display_image(str(filepath))
                
        except Exception as e:
            print(f"[Receiver] Error handling client: {e}")
            try:
                client.send(b"ERROR")
                client.close()
            except:
                pass
    
    def _display_image(self, filepath: str):
        """Open image with default viewer"""
        try:
            if platform.system() == "Windows":
                os.startfile(filepath)
            elif platform.system() == "Darwin":
                subprocess.run(["open", filepath])
            else:
                subprocess.run(["xdg-open", filepath])
            print(f"[Display] Opened: {filepath}")
        except Exception as e:
            print(f"[Display] Error: {e}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("📡 JARVIS Device Transfer System")
    print("="*60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "--receiver":
            # Start as receiver
            transfer = DeviceTransfer()
            transfer.start_receiver(auto_display=True)
            
            print("Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                transfer.stop_receiver()
                print("\nReceiver stopped.")
        
        elif mode == "--pair":
            # Generate pairing QR
            transfer = DeviceTransfer()
            name = sys.argv[2] if len(sys.argv) > 2 else "JARVIS"
            transfer.pair_device(name)
        
        elif mode == "--send":
            # Send image
            if len(sys.argv) < 3:
                print("Usage: --send <image_path> [device_name]")
                sys.exit(1)
            
            transfer = DeviceTransfer()
            image_path = sys.argv[2]
            device_name = sys.argv[3] if len(sys.argv) > 3 else None
            
            success = transfer.send_image(image_path, device_name)
            sys.exit(0 if success else 1)
        
        elif mode == "--scan":
            # Scan for devices
            transfer = DeviceTransfer()
            print("\nScanning for online trusted devices...")
            online = transfer.find_online_devices()
            print(f"\nFound {len(online)} online devices")
        
        else:
            print(f"Unknown mode: {mode}")
            print("\nUsage:")
            print("  --receiver     Start as image receiver")
            print("  --pair [name]  Generate pairing QR code")
            print("  --send <path>  Send image to trusted device")
            print("  --scan         Scan for online devices")
    
    else:
        print("\nUsage:")
        print("  python device_transfer.py --receiver")
        print("  python device_transfer.py --pair [device_name]")
        print("  python device_transfer.py --send <image_path> [device_name]")
        print("  python device_transfer.py --scan")
        print("\nExample workflow:")
        print("  1. On JARVIS device: python device_transfer.py --pair")
        print("  2. Scan QR with other device")
        print("  3. On other device: python device_transfer.py --receiver")
        print("  4. On JARVIS: python device_transfer.py --send image.jpg")
