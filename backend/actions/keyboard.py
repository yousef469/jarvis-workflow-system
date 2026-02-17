"""Keyboard and Mouse Actions"""
import time

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


# Key name mapping
KEY_MAP = {
    "enter": "enter",
    "return": "enter",
    "tab": "tab",
    "space": "space",
    "backspace": "backspace",
    "delete": "delete",
    "escape": "escape",
    "esc": "escape",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "home": "home",
    "end": "end",
    "pageup": "pageup",
    "pagedown": "pagedown",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "win": "win",
    "windows": "win",
    "cmd": "win",
    "command": "win",
    "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4",
    "f5": "f5", "f6": "f6", "f7": "f7", "f8": "f8",
    "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
}


def type_text(text: str, interval: float = 0.02) -> dict:
    """Type text character by character"""
    if not PYAUTOGUI_AVAILABLE:
        raise Exception("pyautogui not installed")
    
    pyautogui.typewrite(text, interval=interval)
    return {"typed": text}


def press_key(key: str) -> dict:
    """Press a single key"""
    if not PYAUTOGUI_AVAILABLE:
        raise Exception("pyautogui not installed")
    
    key_lower = key.lower()
    mapped_key = KEY_MAP.get(key_lower, key_lower)
    
    pyautogui.press(mapped_key)
    return {"pressed": key}


def hotkey(*keys) -> dict:
    """Press a keyboard shortcut (e.g., Ctrl+C)"""
    if not PYAUTOGUI_AVAILABLE:
        raise Exception("pyautogui not installed")
    
    # Handle both hotkey(["ctrl", "c"]) and hotkey("ctrl", "c")
    if len(keys) == 1 and isinstance(keys[0], list):
        keys = keys[0]
    
    # Map key names
    mapped_keys = [KEY_MAP.get(k.lower(), k.lower()) for k in keys]
    
    pyautogui.hotkey(*mapped_keys)
    return {"hotkey": list(keys)}


def click(x: int = None, y: int = None, button: str = "left") -> dict:
    """Click at position (or current position if not specified)"""
    if not PYAUTOGUI_AVAILABLE:
        raise Exception("pyautogui not installed")
    
    if x is not None and y is not None:
        pyautogui.click(x, y, button=button)
        return {"clicked": {"x": x, "y": y, "button": button}}
    else:
        pyautogui.click(button=button)
        pos = pyautogui.position()
        return {"clicked": {"x": pos.x, "y": pos.y, "button": button}}


def double_click(x: int = None, y: int = None) -> dict:
    """Double click at position"""
    if not PYAUTOGUI_AVAILABLE:
        raise Exception("pyautogui not installed")
    
    if x is not None and y is not None:
        pyautogui.doubleClick(x, y)
    else:
        pyautogui.doubleClick()
    
    return {"double_clicked": True}


def right_click(x: int = None, y: int = None) -> dict:
    """Right click at position"""
    return click(x, y, button="right")


def scroll(amount: int) -> dict:
    """Scroll up (positive) or down (negative)"""
    if not PYAUTOGUI_AVAILABLE:
        raise Exception("pyautogui not installed")
    
    pyautogui.scroll(amount)
    return {"scrolled": amount}


def move_mouse(x: int, y: int, duration: float = 0.2) -> dict:
    """Move mouse to position"""
    if not PYAUTOGUI_AVAILABLE:
        raise Exception("pyautogui not installed")
    
    pyautogui.moveTo(x, y, duration=duration)
    return {"moved_to": {"x": x, "y": y}}
