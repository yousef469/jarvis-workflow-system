"""Action Registry - All available Atlas actions"""
from typing import Dict, Callable, Any
from .apps import open_app, close_app, focus_app
from .browser import open_url, search_google, search_youtube
from .media import play_music, pause_music, next_track, prev_track, set_volume, mute
from .system import shutdown, restart, sleep_pc, lock_pc, screenshot
from .keyboard import type_text, press_key, hotkey, click, scroll

# Action registry
ACTIONS: Dict[str, Callable] = {
    # App control
    "open_app": open_app,
    "close_app": close_app,
    "focus_app": focus_app,
    
    # Browser
    "open_url": open_url,
    "search_google": search_google,
    "search_youtube": search_youtube,
    
    # Media
    "play_music": play_music,
    "pause_music": pause_music,
    "next_track": next_track,
    "prev_track": prev_track,
    "volume": set_volume,
    "set_volume": set_volume,
    "mute": mute,
    
    # System
    "shutdown": shutdown,
    "restart": restart,
    "sleep": sleep_pc,
    "lock": lock_pc,
    "screenshot": screenshot,
    
    # Keyboard/Mouse
    "type_text": type_text,
    "press_key": press_key,
    "hotkey": hotkey,
    "click": click,
    "scroll": scroll,
}


def execute_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single action and return result"""
    action_type = action.get("type")
    
    if not action_type:
        return {"success": False, "error": "No action type specified"}
    
    if action_type not in ACTIONS:
        return {"success": False, "error": f"Unknown action: {action_type}"}
    
    try:
        # Get action function
        func = ACTIONS[action_type]
        
        # Extract parameters (everything except 'type')
        params = {k: v for k, v in action.items() if k != "type"}
        
        # Execute
        result = func(**params)
        
        return {"success": True, "action": action_type, "result": result}
        
    except Exception as e:
        return {"success": False, "action": action_type, "error": str(e)}


def execute_actions(actions: list) -> list:
    """Execute multiple actions and return results"""
    results = []
    for action in actions:
        result = execute_action(action)
        results.append(result)
    return results
