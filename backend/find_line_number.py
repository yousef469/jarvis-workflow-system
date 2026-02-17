
filename = r"c:\Users\Yousef\projects\live ai asistant\jarvis-system\backend\server.py"
search_term = "api_vision_describe"

with open(filename, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if search_term in line:
            print(f"FOUND_LINE:{i+1}")
            exit()
print("NOT_FOUND")
