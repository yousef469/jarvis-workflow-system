"""
JARVIS Web Receiver Server
==========================
Hosts a web page that phones/tablets can open to receive images.

NO APP NEEDED - just open a webpage on your phone!

How it works:
1. JARVIS runs this web server
2. Phone opens http://JARVIS_IP:8080 in browser
3. When you catch an image, it appears on phone instantly via WebSocket
"""

import asyncio
import websockets
import http.server
import socketserver
import threading
import base64
import os
import json
from pathlib import Path

# Configuration
WEB_PORT = 8080
WS_PORT = 8081

# Connected clients
connected_clients = set()

# HTML page for phone
RECEIVER_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>JARVIS Receiver</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #000;
            color: #0ff;
            font-family: 'Segoe UI', Arial, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #0ff;
        }
        .header h1 { font-size: 24px; font-weight: 300; }
        .header .subtitle { font-size: 12px; color: #666; margin-top: 5px; }
        .status {
            padding: 15px;
            text-align: center;
            font-size: 16px;
            background: #111;
        }
        .status.connected { color: #0f0; }
        .status.waiting { color: #ff0; }
        .status.error { color: #f00; }
        .image-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: #0a0a0a;
        }
        .image-container img {
            max-width: 100%;
            max-height: 100%;
            border: 3px solid #0ff;
            border-radius: 10px;
            box-shadow: 0 0 30px rgba(0,255,255,0.3);
        }
        .placeholder {
            text-align: center;
            color: #333;
        }
        .placeholder .icon { font-size: 100px; margin-bottom: 20px; }
        .placeholder p { margin: 10px 0; }
        .footer {
            background: #111;
            padding: 15px;
            text-align: center;
            font-size: 14px;
            color: #666;
            border-top: 1px solid #333;
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 JARVIS Receiver</h1>
        <div class="subtitle">Iron Man Style Image Transfer</div>
    </div>
    
    <div class="status waiting pulse" id="status">
        ⏳ Connecting to JARVIS...
    </div>
    
    <div class="image-container" id="imageContainer">
        <div class="placeholder">
            <div class="icon">📡</div>
            <p>Waiting for image from JARVIS...</p>
            <p style="font-size: 12px; color: #444;">Say "Hand" and make a catch gesture</p>
        </div>
    </div>
    
    <div class="footer">
        <p>Connected to JARVIS • Images appear automatically</p>
    </div>

    <script>
        const WS_PORT = """ + str(WS_PORT) + """;
        let ws = null;
        let imageCount = 0;
        
        function updateStatus(msg, type) {
            const status = document.getElementById('status');
            status.textContent = msg;
            status.className = 'status ' + type;
            if (type === 'waiting') status.classList.add('pulse');
        }
        
        function showImage(base64Data) {
            imageCount++;
            const container = document.getElementById('imageContainer');
            container.innerHTML = '<img src="data:image/png;base64,' + base64Data + '" alt="Image ' + imageCount + '">';
            updateStatus('✓ Image #' + imageCount + ' received!', 'connected');
            
            // Vibrate if supported
            if (navigator.vibrate) navigator.vibrate(200);
        }
        
        function connect() {
            const host = window.location.hostname;
            const wsUrl = 'ws://' + host + ':' + WS_PORT;
            
            updateStatus('⏳ Connecting to ' + host + '...', 'waiting');
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                updateStatus('✓ Connected to JARVIS!', 'connected');
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'image') {
                        showImage(data.data);
                    } else if (data.type === 'status') {
                        updateStatus(data.message, 'connected');
                    }
                } catch (e) {
                    console.error('Parse error:', e);
                }
            };
            
            ws.onclose = () => {
                updateStatus('⚠ Disconnected - Reconnecting...', 'error');
                setTimeout(connect, 3000);
            };
            
            ws.onerror = (err) => {
                updateStatus('✗ Connection error', 'error');
            };
        }
        
        // Start connection
        connect();
        
        // Keep screen on
        if ('wakeLock' in navigator) {
            navigator.wakeLock.request('screen').catch(() => {});
        }
    </script>
</body>
</html>
"""


class WebHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the receiver HTML page"""
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(RECEIVER_HTML.encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass  # Suppress logs


async def websocket_handler(websocket, path):
    """Handle WebSocket connections from phones"""
    connected_clients.add(websocket)
    client_ip = websocket.remote_address[0]
    print(f"[WebReceiver] 📱 Phone connected: {client_ip}")
    
    try:
        # Send welcome message
        await websocket.send(json.dumps({
            "type": "status",
            "message": "Connected! Waiting for images..."
        }))
        
        # Keep connection alive
        async for message in websocket:
            pass  # We don't expect messages from client
            
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[WebReceiver] 📱 Phone disconnected: {client_ip}")


async def send_image_to_all(image_path: str):
    """Send image to all connected phones"""
    if not connected_clients:
        print("[WebReceiver] No phones connected")
        return False
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    
    message = json.dumps({
        "type": "image",
        "data": image_data,
        "filename": os.path.basename(image_path)
    })
    
    # Send to all connected clients
    sent_count = 0
    for client in connected_clients.copy():
        try:
            await client.send(message)
            sent_count += 1
        except:
            connected_clients.discard(client)
    
    print(f"[WebReceiver] ✓ Image sent to {sent_count} phone(s)")
    return sent_count > 0


def send_image_sync(image_path: str) -> bool:
    """Synchronous wrapper for sending images"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(send_image_to_all(image_path))


class WebReceiverServer:
    """Main server class"""
    
    def __init__(self):
        self.running = False
        self.web_thread = None
        self.ws_thread = None
        self.local_ip = self._get_local_ip()
    
    def _get_local_ip(self) -> str:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def start(self):
        """Start web and websocket servers"""
        self.running = True
        
        # Start HTTP server
        self.web_thread = threading.Thread(target=self._run_web_server, daemon=True)
        self.web_thread.start()
        
        # Start WebSocket server
        self.ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self.ws_thread.start()
        
        print(f"\n{'='*60}")
        print("📱 JARVIS Web Receiver Started!")
        print(f"{'='*60}")
        print(f"  On your phone, open this URL:")
        print(f"  👉 http://{self.local_ip}:{WEB_PORT}")
        print(f"{'='*60}")
        print(f"  Make sure phone is on same WiFi/hotspot")
        print(f"{'='*60}\n")
    
    def _run_web_server(self):
        """Run HTTP server"""
        with socketserver.TCPServer(("", WEB_PORT), WebHandler) as httpd:
            httpd.serve_forever()
    
    def _run_ws_server(self):
        """Run WebSocket server"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        start_server = websockets.serve(websocket_handler, "0.0.0.0", WS_PORT)
        loop.run_until_complete(start_server)
        loop.run_forever()
    
    def send_image(self, image_path: str) -> bool:
        """Send image to all connected phones"""
        return send_image_sync(image_path)
    
    def get_connected_count(self) -> int:
        """Get number of connected phones"""
        return len(connected_clients)


# Global instance
_server = None

def get_web_receiver() -> WebReceiverServer:
    global _server
    if _server is None:
        _server = WebReceiverServer()
    return _server

def start_web_receiver():
    """Start the web receiver server"""
    server = get_web_receiver()
    server.start()
    return server

def send_to_phones(image_path: str) -> bool:
    """Send image to all connected phones"""
    server = get_web_receiver()
    return server.send_image(image_path)


# CLI
if __name__ == "__main__":
    import sys
    
    server = WebReceiverServer()
    server.start()
    
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            cmd = input("Enter image path to send (or 'q' to quit): ").strip()
            if cmd.lower() == 'q':
                break
            if os.path.exists(cmd):
                server.send_image(cmd)
            else:
                print(f"File not found: {cmd}")
    except KeyboardInterrupt:
        print("\nStopped.")
