import sounddevice as sd

print("--- AVAILABLE MICROPHONES ---")
devices = sd.query_devices()
print(devices)

print("\n--- RECOMMENDED ---")
# Try to find the default input
try:
    default_input = sd.query_devices(kind='input')
    print(f"Default Mic: {default_input['name']}")
    print(f"ID: {default_input['index']}")
except:
    print("Could not detect default mic.")
