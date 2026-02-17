import sys
import asyncio
import time
import threading

# Global references to be set by server.py
_sio = None
_loop = None

# Thread-local storage to prevent recursion (Redirector -> Print -> Redirector)
_local = threading.local()

# Buffered logs for early startup (before socket is connected)
_log_buffer = []
_buffer_lock = threading.Lock()

def set_logger_refs(sio, loop):
    global _sio, _loop
    _sio = sio
    _loop = loop
    # Drain buffer when refs are set
    with _buffer_lock:
        while _log_buffer:
            msg, level, timestamp = _log_buffer.pop(0)
            emit_sys_log(msg, level, timestamp)

def emit_sys_log(msg: str, level: str = "INFO", timestamp: float = None):
    """
    Broadcaster for system logs to the frontend UI ONLY.
    Redirection to terminal is handled by TerminalRedirector.
    """
    if not msg: return
    
    # Prevent recursion
    if getattr(_local, 'is_logging', False):
        return

    ts = timestamp or time.time()
    lvl = level.upper()

    try:
        global _sio, _loop
        if _sio and _loop:
            _loop.call_soon_threadsafe(
                lambda: asyncio.create_task(_sio.emit('sys_log', {
                    "msg": msg.strip(), 
                    "level": lvl, 
                    "timestamp": ts
                }))
            )
        else:
            # Buffer logs if socket not ready
            with _buffer_lock:
                if len(_log_buffer) < 1000: # Safety cap
                    _log_buffer.append((msg, level, ts))
    except:
        pass

class TerminalRedirector:
    """
    Bulletproof stream proxy that silences terminal noise 
    while preserving all logs for the UI.
    """
    def __init__(self, original_stream, level="INFO"):
        self._stream = original_stream
        self.level = level
        # Explicitly set core attributes to satisfy uvicorn/logging
        self.encoding = getattr(original_stream, 'encoding', 'utf-8')
        self.errors = getattr(original_stream, 'errors', 'strict')

    def __getattr__(self, name):
        """Proxy all other calls (fileno, isatty, flush, etc.) to the real stream."""
        return getattr(self._stream, name)

    def isatty(self):
        """Uvicorn/Logging often check this to decide on coloring."""
        try:
            return self._stream.isatty()
        except:
            return False

    def write(self, message):
        if not message:
            return

        ms = message.strip()
        
        # 1. PHYSICAL TERMINAL FILTER (Clean but Informative)
        msg_lower = message.lower()
        is_triggered = "wake word detected" in msg_lower or "triggered" in msg_lower
        is_listening = "listening for 'hey jarvis'" in msg_lower or "listening for command" in msg_lower or "listening..." in msg_lower
        is_dot = message == "."
        
        # Core status tags we want to see
        is_core_status = "[" in message and ("]" in message) and any(tag in message for tag in ["WakeWord", "STT", "Piper", "Dispatcher", "Interaction", "API", "ImageGen", "Brain"])
        
        if is_triggered or is_listening or is_dot or is_core_status:
            try:
                if is_triggered:
                    self._stream.write("\n[Jarvis] ✨ Triggered\n")
                elif is_listening:
                    self._stream.write("[Jarvis] 🎤 Listening")
                else:
                    self._stream.write(message)
                self._stream.flush()
            except:
                pass
        
        # 2. UI LOG STREAMING ( Full Visibility )
        # No changes here, UI still sees everything.
        if getattr(_local, 'is_logging', False):
            return

        # 2. UI LOG STREAMING ( Full Visibility )
        # Everything, including the "noise", goes to the UI.
        if getattr(_local, 'is_logging', False):
            return

        if not ms:
            return

        _local.is_logging = True
        try:
            global _sio, _loop
            if _sio and _loop:
                _loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(_sio.emit('sys_log', {
                        "msg": message.strip(), # Keep full message for UI
                        "level": self.level,
                        "timestamp": time.time()
                    }))
                )
            else:
                # Buffer for early startup
                with _buffer_lock:
                    if len(_log_buffer) < 1000:
                        _log_buffer.append((message.strip(), self.level, time.time()))
        except:
            pass
        finally:
            _local.is_logging = False

    def flush(self):
        try: self._stream.flush()
        except: pass

def inject_terminal_hook():
    """Forces the terminal to be quiet while keeping the UI live."""
    # Ensure we use the real base streams for redirection
    sys.stdout = TerminalRedirector(sys.__stdout__, "INFO")
    sys.stderr = TerminalRedirector(sys.__stderr__, "ERROR")
    
    # Also override any existing logging handlers to use our new streams
    try:
        import logging
        for handler in logging.root.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(sys.stdout)
    except:
        pass

    # confirm to the user via base stream that J is silent now
    sys.__stdout__.write("[Jarvis] Master Bridge Active. Terminal silenced.\n")
    sys.__stdout__.flush()
