"""Quick TTS test"""
import pyttsx3

print("Testing TTS...")
engine = pyttsx3.init()
engine.setProperty('rate', 175)

# List voices
voices = engine.getProperty('voices')
print(f"Found {len(voices)} voices:")
for i, voice in enumerate(voices):
    print(f"  {i}: {voice.name}")

# Speak
engine.say("Yes Yousef? How can I help you?")
engine.runAndWait()
print("Done!")
