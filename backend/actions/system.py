"""System Control Actions"""
import subprocess
import platform
import os
from datetime import datetime


def shutdown(delay: int = 0) -> dict:
    """Shutdown the computer"""
    if platform.system() == "Windows":
        if delay > 0:
            subprocess.run(f"shutdown /s /t {delay}", shell=True)
        else:
            subprocess.run("shutdown /s /t 0", shell=True)
    else:
        subprocess.run(["shutdown", "-h", f"+{delay // 60}"])
    
    return {"action": "shutdown", "delay": delay}


def restart(delay: int = 0) -> dict:
    """Restart the computer"""
    if platform.system() == "Windows":
        subprocess.run(f"shutdown /r /t {delay}", shell=True)
    else:
        subprocess.run(["shutdown", "-r", f"+{delay // 60}"])
    
    return {"action": "restart", "delay": delay}


def sleep_pc() -> dict:
    """Put computer to sleep"""
    if platform.system() == "Windows":
        subprocess.run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True)
    else:
        subprocess.run(["pmset", "sleepnow"])
    
    return {"action": "sleep"}


def lock_pc() -> dict:
    """Lock the computer"""
    if platform.system() == "Windows":
        subprocess.run("rundll32.exe user32.dll,LockWorkStation", shell=True)
    else:
        subprocess.run(["pmset", "displaysleepnow"])
    
    return {"action": "lock"}


def screenshot(save_path: str = None) -> dict:
    """Take a screenshot"""
    try:
        import pyautogui
        from PIL import Image
        
        # Generate filename if not provided
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"screenshot_{timestamp}.png"
        
        # Take screenshot
        img = pyautogui.screenshot()
        img.save(save_path)
        
        return {"action": "screenshot", "path": save_path}
        
    except ImportError:
        # Fallback: use Windows Snipping Tool
        if platform.system() == "Windows":
            subprocess.run("snippingtool /clip", shell=True)
            return {"action": "screenshot", "method": "snipping_tool"}
        raise Exception("Screenshot requires pyautogui: pip install pyautogui")


def cancel_shutdown() -> dict:
    """Cancel a pending shutdown"""
    if platform.system() == "Windows":
        subprocess.run("shutdown /a", shell=True)
    else:
        subprocess.run(["shutdown", "-c"])
    
    return {"action": "cancel_shutdown"}
