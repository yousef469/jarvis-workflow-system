import os

SERVER_PATH = r"c:\Users\Yousef\projects\live ai asistant\jarvis-system\backend\server.py"

with open(SERVER_PATH, "r", encoding="utf-8") as f:
    content = f.read()

target = '        temp_path = "temp_gen_source.png"'
replacement = '        temp_path = os.path.abspath("temp_gen_source.png")'

if target in content:
    new_content = content.replace(target, replacement)
    with open(SERVER_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Fixed temp_path to absolute.")
else:
    print("Could not find target line.")
