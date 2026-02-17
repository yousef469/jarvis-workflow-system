"""Test microphone and speech recognition"""
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer
from pathlib import Path

SAMPLE_RATE = 16000
MIC_INDEX = 2  # Internal mic

print("Loading Vosk model...")
model = Model(str(Path("models/vosk-model-en-us-0.22")))
rec = KaldiRecognizer(model, SAMPLE_RATE)
rec.SetWords(True)

print(f"\n🎤 Using Mic ID: {MIC_INDEX}")
print("Speak now! Say 'Jarvis' or 'Computer' or anything...")
print("Press Ctrl+C to stop\n")

def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    
    audio_bytes = bytes(indata)
    
    if rec.AcceptWaveform(audio_bytes):
        result = json.loads(rec.Result())
        text = result.get("text", "")
        if text:
            print(f"✅ HEARD: '{text}'")
    else:
        partial = json.loads(rec.PartialResult())
        text = partial.get("partial", "")
        if text and len(text) > 2:
            print(f"   partial: '{text}'", end='\r')

try:
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        device=MIC_INDEX,
        dtype='int16',
        channels=1,
        callback=callback
    ):
        print("Listening...")
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    print(f"Error: {e}")
