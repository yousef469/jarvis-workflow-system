import pyautogui
import time
import os
import platform
import subprocess

# Common App Aliases for "Open [App]" commands
APP_ALIASES = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "brave": "Brave Browser",
    "code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "terminal": "Terminal",
    "finder": "Finder",
    "safari": "Safari",
    "notes": "Notes",
    "messages": "Messages",
    "spotify": "Spotify",
    "discord": "Discord",
    "slack": "Slack",
    "obs": "OBS",
    "vlc": "VLC",
    "calculator": "Calculator",
    "calendar": "Calendar",
    "photos": "Photos",
    "settings": "System Settings"
}

try:
    from ahk import AHK
    HAS_AHK_LIB = True
except ImportError:
    HAS_AHK_LIB = False

class JarvisAutomation:
    def __init__(self):
        """
        Initializes the best automation engine for the current OS.
        Windows: AHK (Primary), PyAutoGUI (Fallback)
        macOS: AppleScript (Primary), PyAutoGUI (Fallback)
        """
        self.os_type = platform.system()
        self.ahk = None
        
        if self.os_type == "Windows" and HAS_AHK_LIB:
            # Explicitly set path for ahk[binary] in virtualenv
            ahk_path = os.path.join(os.getcwd(), "jarvis_env", "Scripts", "AutoHotkey.exe")
            try:
                if os.path.exists(ahk_path):
                    self.ahk = AHK(executable_path=ahk_path)
                else:
                    self.ahk = AHK()
                print("[Automation V2] ✅ AHK Engine initialized (Windows PRIMARY).")
            except Exception as e:
                print(f"[Automation V2] ⚠️ AHK initialization failed: {e}. PyAutoGUI fallback mode.")
        elif self.os_type == "Darwin":
            print("[Automation V2] ✅ macOS AppleScript Engine initialized (PRIMARY).")
        else:
            print(f"[Automation V2] ℹ️ Running on {self.os_type}. Using PyAutoGUI/Native commands.")

    def run_applescript(self, script: str):
        """Helper to run AppleScript commands on macOS."""
        if self.os_type != "Darwin":
            return None
        try:
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            print(f"[Automation V2] AppleScript error: {e}")
            return None

    def type_text(self, text: str):
        """Types text into the active window."""
        print(f"[Automation V2] Typing: {text[:20]}...")
        
        if self.os_type == "Darwin":
            # Escape quotes for AppleScript
            safe_text = text.replace('"', '\\"')
            script = f'tell application "System Events" to keystroke "{safe_text}"'
            self.run_applescript(script)
            return

        if self.ahk:
            try:
                self.ahk.type(text)
                return
            except Exception as e:
                print(f"[Automation V2] AHK type failed: {e}, falling back to PyAutoGUI")
        
        pyautogui.write(text, interval=0.01)

    def press_key(self, key: str):
        """Presses a specific key (e.g., 'enter', 'esc')."""
        print(f"[Automation V2] Pressing: {key}")
        
        if self.os_type == "Darwin":
            # Map key names to AppleScript key codes/names
            key_map = {
                "enter": "return",
                "esc": "escape",
                "tab": "tab",
                "space": "space",
                "up": "up arrow",
                "down": "down arrow",
                "left": "left arrow",
                "right": "right arrow"
            }
            as_key = key_map.get(key.lower(), key)
            script = f'tell application "System Events" to key code (get code of "{as_key}")' # Complex, simpler:
            script = f'tell application "System Events" to key code (key code of "{as_key}")' # Actually:
            if key.lower() in ["enter", "return"]:
                script = 'tell application "System Events" to key code 36'
            elif key.lower() == "esc":
                script = 'tell application "System Events" to key code 53'
            else:
                script = f'tell application "System Events" to keystroke "{key}"' # Fallback to keystroke
            
            self.run_applescript(script)
            return

        if self.ahk:
            try:
                self.ahk.key_press(key)
                return
            except Exception as e:
                print(f"[Automation V2] AHK key_press failed: {e}, falling back to PyAutoGUI")
        
        pyautogui.press(key)

    def click_at(self, x: int, y: int):
        """Clicks at specific screen coordinates."""
        print(f"[Automation V2] Clicking at: {x}, {y}")
        # Click is best handled by PyAutoGUI on both platforms for simplicity
        pyautogui.click(x, y)

    def hotkey(self, *keys):
        """Press a hotkey combination."""
        print(f"[Automation V2] Hotkey: {keys}")
        
        if self.os_type == "Darwin":
            # Map common names to AppleScript modifiers
            # command, control, option, shift
            key_list = [k.lower() for k in keys]
            main_key = key_list[-1]
            modifiers = key_list[:-1]
            
            mod_map = {
                "ctrl": "control down",
                "alt": "option down",
                "option": "option down",
                "shift": "shift down",
                "win": "command down",
                "command": "command down",
                "cmd": "command down"
            }
            
            as_mods = [mod_map.get(m, m) for m in modifiers]
            if as_mods:
                mods_str = " using {" + ", ".join(as_mods) + "}"
            else:
                mods_str = ""
            
            script = f'tell application "System Events" to keystroke "{main_key}"{mods_str}'
            self.run_applescript(script)
            return

        if self.ahk:
            try:
                ahk_keys = ""
                for k in keys:
                    if k.lower() == "ctrl": ahk_keys += "^"
                    elif k.lower() == "alt": ahk_keys += "!"
                    elif k.lower() == "shift": ahk_keys += "+"
                    elif k.lower() == "win": ahk_keys += "#"
                    else: ahk_keys += k
                self.ahk.send_input(ahk_keys)
                return
            except Exception as e:
                print(f"[Automation V2] AHK hotkey failed: {e}, falling back to PyAutoGUI")
        
        pyautogui.hotkey(*keys)

    def open_app(self, app_name: str):
        """Opens an application."""
        # Resolve alias (e.g., "chrome" -> "Google Chrome")
        real_name = APP_ALIASES.get(app_name.lower(), app_name)
        print(f"[Automation V2] Opening: {real_name} (requested: {app_name})")
        
        if self.os_type == "Darwin":
            subprocess.Popen(["open", "-a", real_name])
        elif self.os_type == "Windows":
            try:
                os.startfile(real_name)
            except:
                subprocess.Popen(f"start {real_name}", shell=True)
        else:
            subprocess.Popen([real_name])

    def find_window(self, title_query: str):
        """Finds and focus a window by title."""
        if self.os_type == "Darwin":
            # Use AppleScript to activate app by name
            script = f'tell application "{title_query}" to activate'
            self.run_applescript(script)
            return True
            
        if not self.ahk:
            return None
        
        try:
            win = self.ahk.find_window(title=title_query)
            if win:
                win.activate()
                return win
            return None
        except Exception as e:
            print(f"[Automation V2] AHK window search failed: {e}")
            return None

# Singleton instance
automation_v2 = JarvisAutomation()
