"""Browser Actions"""
import webbrowser
import urllib.parse
import subprocess
import os
import platform

# Try to import pyautogui for same-tab navigation
try:
    import pyautogui
    import time as pytime
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

# Find Chrome path
CHROME_PATH = None
if platform.system() == "Windows":
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
    ]
elif platform.system() == "Darwin":
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chrome.app/Contents/MacOS/Chrome",
    ]
else:
    chrome_paths = ["google-chrome", "chrome"]

for path in chrome_paths:
    if os.path.exists(path) or platform.system() != "Windows" and subprocess.run(["which", path], capture_output=True).returncode == 0:
        CHROME_PATH = path
        break

def open_url(url: str, new_tab: bool = True) -> dict:
    """Open a URL in Chrome (not default browser)"""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    if CHROME_PATH:
        try:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", "-a", "Google Chrome", url])
            else:
                subprocess.Popen([CHROME_PATH, url])
            return {"opened": url, "browser": "chrome"}
        except:
            pass
    
    # Fallback to default browser
    if new_tab:
        webbrowser.open_new_tab(url)
    else:
        webbrowser.open(url)
    return {"opened": url, "browser": "default"}


def search_in_current_tab(url: str) -> dict:
    """Navigate to URL in current tab using keyboard shortcut"""
    if PYAUTOGUI_AVAILABLE:
        try:
            # macOS uses Command+L, Windows uses Ctrl+L
            modifier = 'command' if platform.system() == 'Darwin' else 'ctrl'
            pyautogui.hotkey(modifier, 'l')
            pytime.sleep(0.1)
            pyautogui.typewrite(url, interval=0.01)
            pyautogui.press('enter')
            return {"navigated": url, "same_tab": True}
        except:
            pass
    # Fallback to new tab
    webbrowser.open(url)
    return {"navigated": url, "same_tab": False}


def search_google(query: str, new_tab: bool = True) -> dict:
    """Search Google in Chrome"""
    encoded = urllib.parse.quote(query)
    url = f"https://www.google.com/search?q={encoded}"
    
    return open_url(url, new_tab=new_tab)


def search_youtube(query: str, new_tab: bool = True) -> dict:
    """Search YouTube in Chrome"""
    encoded = urllib.parse.quote(query)
    url = f"https://www.youtube.com/results?search_query={encoded}"
    
    return open_url(url, new_tab=new_tab)


def search_github(query: str) -> dict:
    """Search GitHub for a query"""
    encoded = urllib.parse.quote(query)
    url = f"https://github.com/search?q={encoded}"
    webbrowser.open(url)
    return {"searched": query, "engine": "github", "url": url}


def search_stackoverflow(query: str) -> dict:
    """Search Stack Overflow for a query"""
    encoded = urllib.parse.quote(query)
    url = f"https://stackoverflow.com/search?q={encoded}"
    webbrowser.open(url)
    return {"searched": query, "engine": "stackoverflow", "url": url}
