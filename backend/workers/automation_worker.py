"""Automation Worker V24 - System Control and App Automation"""

import asyncio
from typing import Dict, Any


class AutomationWorker:
    """Handles system automation tasks (open apps, type text, volume, etc.)"""
    
    def __init__(self):
        self.name = "automation"
    
    async def execute(self, command: str) -> Dict[str, Any]:
        """
        Execute automation task.
        Command format: "action:target" (e.g., "open_app:Safari", "volume:up")
        """
        print(f"[AutomationWorker] ⚙️ Executing: {command}")
        
        try:
            from automation_engine_v2 import automation_v2
            import pyautogui
            
            # Parse command
            if ":" in command:
                action, target = command.split(":", 1)
            else:
                # Smart parsing for V24.5 cognitive outputs
                cmd_lower = command.lower().strip()
                if cmd_lower.startswith("open "):
                    action = "open_app"
                    target = command[5:].strip()
                elif cmd_lower.startswith("close "):
                    action = "close_app"
                    target = command[6:].strip()
                elif cmd_lower.startswith("launch "):
                    action = "open_app"
                    target = command[7:].strip()
                else:
                    action = cmd_lower
                    target = ""
            
            action = action.lower().strip()
            target = target.strip()
            
            result_msg = ""
            
            # App Control
            if action == "open_app":
                await asyncio.to_thread(automation_v2.open_app, target)
                result_msg = f"Opened {target}"
            
            elif action == "type_text" or action == "type":
                await asyncio.to_thread(automation_v2.type_text, target)
                result_msg = f"Typed text"
            
            elif action == "press_key" or action == "press":
                await asyncio.to_thread(automation_v2.press_key, target)
                result_msg = f"Pressed {target}"
            
            # Volume Control
            elif action == "volume_up" or (action == "volume" and target == "up"):
                await asyncio.to_thread(pyautogui.press, "volumeup")
                result_msg = "Volume increased"
            
            elif action == "volume_down" or (action == "volume" and target == "down"):
                await asyncio.to_thread(pyautogui.press, "volumedown")
                result_msg = "Volume decreased"
            
            elif action == "mute":
                await asyncio.to_thread(pyautogui.press, "volumemute")
                result_msg = "Volume muted/unmuted"
            
            # Media Control
            elif action == "play_pause" or action == "media_play":
                await asyncio.to_thread(pyautogui.press, "playpause")
                result_msg = "Media playback toggled"
            
            elif action == "next_track" or action == "media_next":
                await asyncio.to_thread(pyautogui.press, "nexttrack")
                result_msg = "Skipped to next track"
            
            elif action == "prev_track" or action == "media_prev":
                await asyncio.to_thread(pyautogui.press, "prevtrack")
                result_msg = "Went to previous track"
            
            # Window Control
            elif action == "close_window":
                await asyncio.to_thread(pyautogui.hotkey, "command", "w")
                result_msg = "Closed active window"
            
            elif action == "minimize":
                await asyncio.to_thread(pyautogui.hotkey, "command", "m")
                result_msg = "Minimized window"
            
            elif action == "fullscreen":
                await asyncio.to_thread(pyautogui.hotkey, "ctrl", "command", "f")
                result_msg = "Toggled fullscreen"
            
            # Screenshot
            elif action == "screenshot":
                from workers.vision_worker import vision_worker
                ss_result = await vision_worker.take_screenshot()
                result_msg = f"Screenshot taken: {ss_result.get('path', 'saved')}"
            
            # Scroll
            elif action == "scroll_up":
                await asyncio.to_thread(pyautogui.scroll, 5)
                result_msg = "Scrolled up"
            
            elif action == "scroll_down":
                await asyncio.to_thread(pyautogui.scroll, -5)
                result_msg = "Scrolled down"
            
            # Hotkey
            elif action == "hotkey":
                keys = target.split("+")
                await asyncio.to_thread(automation_v2.hotkey, *keys)
                result_msg = f"Pressed hotkey: {target}"
            
            else:
                return {
                    "source": "automation",
                    "command": command,
                    "error": f"Unknown action: {action}",
                    "summary": f"I don't know how to perform: {action}",
                    "status": "error"
                }
            
            return {
                "source": "automation",
                "command": command,
                "action": action,
                "target": target,
                "summary": result_msg,
                "status": "success"
            }
            
        except Exception as e:
            print(f"[AutomationWorker] ❌ Error: {e}")
            return {
                "source": "automation",
                "command": command,
                "error": str(e),
                "summary": f"Automation failed: {str(e)[:100]}",
                "status": "error"
            }


# Singleton instance
automation_worker = AutomationWorker()
