"""App Control Actions"""
import subprocess
import os
import platform

# App name to executable/app mapping
if platform.system() == "Windows":
    APP_MAP = {
        "chrome": "chrome",
        "google chrome": "chrome",
        "firefox": "firefox",
        "edge": "msedge",
        "microsoft edge": "msedge",
        "brave": "brave",
        "notepad": "notepad",
        "vscode": "code",
        "vs code": "code",
        "visual studio code": "code",
        "sublime": "subl",
        "explorer": "explorer",
        "file explorer": "explorer",
        "terminal": "wt",
        "cmd": "cmd",
        "powershell": "powershell",
        "spotify": "spotify",
        "vlc": "vlc",
        "discord": "discord",
        "slack": "slack",
        "teams": "teams",
        "calculator": "calc",
        "paint": "mspaint",
        "word": "winword",
        "excel": "excel",
        "powerpoint": "powerpnt",
    }
else:
    # macOS / Linux Mappings
    APP_MAP = {
        # Browsers
        "chrome": "Google Chrome",
        "google chrome": "Google Chrome",
        "firefox": "Firefox",
        "safari": "Safari",
        "arc": "Arc",
        
        # Editors
        "vscode": "Visual Studio Code",
        "vs code": "Visual Studio Code",
        "visual studio code": "Visual Studio Code",
        "textedit": "TextEdit",
        "notes": "Notes",
        
        # System
        "finder": "Finder",
        "terminal": "Terminal",
        "iterm": "iTerm",
        "activity monitor": "Activity Monitor",
        "system settings": "System Settings",
        
        # Media
        "spotify": "Spotify",
        "music": "Music",
        "vlc": "VLC",
        
        # Communication
        "discord": "Discord",
        "slack": "Slack",
        "whatsapp": "WhatsApp",
        "telegram": "Telegram",
        
        # Other
        "calculator": "Calculator",
        "calendar": "Calendar",
        "reminders": "Reminders",
    }


def open_app(name: str) -> dict:
    """Open an application by name"""
    name_lower = name.lower().strip()
    
    # Get executable name
    exe = APP_MAP.get(name_lower, name_lower)
    
    try:
        if platform.system() == "Windows":
            subprocess.Popen(f"start {exe}", shell=True)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", "-a", exe])
        else:
            subprocess.Popen([exe])
        
        return {"opened": name, "executable": exe}
    except Exception as e:
        raise Exception(f"Failed to open {name}: {e}")


def close_app(name: str) -> dict:
    """Close an application by name"""
    name_lower = name.lower().strip()
    exe = APP_MAP.get(name_lower, name_lower)
    
    try:
        if platform.system() == "Windows":
            subprocess.run(f"taskkill /IM {exe}.exe /F", shell=True, capture_output=True)
        elif platform.system() == "Darwin":
            # Graceful quit via AppleScript
            subprocess.run(['osascript', '-e', f'tell application "{exe}" to quit'], capture_output=True)
        else:
            subprocess.run(["pkill", "-f", exe], capture_output=True)
        
        return {"closed": name}
    except Exception as e:
        raise Exception(f"Failed to close {name}: {e}")


def focus_app(name: str) -> dict:
    """Bring an application window to focus"""
    name_lower = name.lower().strip()
    exe = APP_MAP.get(name_lower, name_lower)
    
    try:
        if platform.system() == "Darwin":
            subprocess.run(['osascript', '-e', f'tell application "{exe}" to activate'], capture_output=True)
            return {"focused": name, "method": "applescript"}
            
        import pygetwindow as gw
        windows = gw.getWindowsWithTitle(name)
        
        if windows:
            win = windows[0]
            win.activate()
            return {"focused": name, "title": win.title}
        else:
            return open_app(name)
            
    except Exception as e:
        # Fallback: just try to open it
        return open_app(name)
