"""Media Control Actions"""
import subprocess
import platform

try:
    from pynput.keyboard import Key, Controller
    keyboard = Controller()
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


def play_music(playlist: str = None) -> dict:
    """Play music - sends media play key"""
    if PYNPUT_AVAILABLE:
        keyboard.press(Key.media_play_pause)
        keyboard.release(Key.media_play_pause)
    else:
        # Fallback: use nircmd on Windows
        if platform.system() == "Windows":
            subprocess.run("nircmd.exe sendkeypress 0xB3", shell=True, capture_output=True)
    
    return {"action": "play", "playlist": playlist}


def pause_music() -> dict:
    """Pause music - sends media pause key"""
    if PYNPUT_AVAILABLE:
        keyboard.press(Key.media_play_pause)
        keyboard.release(Key.media_play_pause)
    
    return {"action": "pause"}


def next_track() -> dict:
    """Skip to next track"""
    if PYNPUT_AVAILABLE:
        keyboard.press(Key.media_next)
        keyboard.release(Key.media_next)
    
    return {"action": "next"}


def prev_track() -> dict:
    """Go to previous track"""
    if PYNPUT_AVAILABLE:
        keyboard.press(Key.media_previous)
        keyboard.release(Key.media_previous)
    
    return {"action": "previous"}


def set_volume(level: int) -> dict:
    """Set system volume (0-100)"""
    level = max(0, min(100, level))
    
    if platform.system() == "Windows":
        # Use nircmd or PowerShell
        try:
            # PowerShell method
            ps_cmd = f'''
            $obj = New-Object -ComObject WScript.Shell
            1..50 | ForEach-Object {{ $obj.SendKeys([char]174) }}
            1..{level // 2} | ForEach-Object {{ $obj.SendKeys([char]175) }}
            '''
            subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
        except:
            pass
    
    return {"volume": level}


def mute() -> dict:
    """Mute/unmute system volume"""
    if PYNPUT_AVAILABLE:
        keyboard.press(Key.media_volume_mute)
        keyboard.release(Key.media_volume_mute)
    
    return {"action": "mute_toggle"}
