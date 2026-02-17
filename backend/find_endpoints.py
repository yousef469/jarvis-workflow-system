import os

SERVER_PATH = r"c:\Users\Yousef\projects\live ai asistant\jarvis-system\backend\server.py"

with open(SERVER_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

found = False
for i, line in enumerate(lines):
    if "api_vision_describe" in line:
        print(f"Found 'api_vision_describe' on line {i+1}: {line.strip()}")
        found = True

if not found:
    print("NOT FOUND.")
