import os

SERVER_PATH = r"c:\Users\Yousef\projects\live ai asistant\jarvis-system\backend\server.py"

with open(SERVER_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the main block
main_idx = -1
for i, line in enumerate(lines):
    if 'if __name__ == "__main__":' in line:
        main_idx = i
        break

if main_idx == -1:
    print("Could not find main block!")
    exit(1)

# Find the injection start
# We look for the marker we added
injection_marker = "# VISION TO IMAGE GENERATION"
injection_idx = -1
for i, line in enumerate(lines):
    # Only look AFTER main_idx to confirm it is misplaced
    if i > main_idx and injection_marker in line:
        injection_idx = i
        break

if injection_idx == -1:
    print("Could not find injected block after main!")
    exit(1)

print(f"Main block at {main_idx}")
print(f"Injection block at {injection_idx}")

# Extract parts
# 1. Everything before main
part1 = lines[:main_idx]
# 2. The main block (from main_idx up to injection_idx)
part_main = lines[main_idx:injection_idx]
# 3. The injected code (from injection_idx to end)
part_injection = lines[injection_idx:]

# New order: Part1 + PartInjection + PartMain
new_content = part1 + ["\n\n"] + part_injection + ["\n\n"] + part_main

with open(SERVER_PATH, "w", encoding="utf-8") as f:
    f.writelines(new_content)

print(f"Fixed server.py order. Moved {len(part_injection)} lines up.")
