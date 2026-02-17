"""
JARVIS Voice Setup Helper
Helps you set up a voice sample for cloning
"""

import os
from pathlib import Path

print("=" * 60)
print("JARVIS Voice Cloning Setup")
print("=" * 60)

print("""
To clone JARVIS's voice, you need a WAV audio sample.

OPTIONS:

1. FIND A JARVIS CLIP:
   - Search YouTube for "JARVIS voice lines Iron Man"
   - Download a clean clip (6-15 seconds, no music)
   - Convert to WAV format
   - Save as: jarvis_sample.wav

2. USE YOUR OWN VOICE:
   - Record yourself speaking for 10-15 seconds
   - Save as WAV format
   - Name it: jarvis_sample.wav

3. DOWNLOAD FROM FREESOUND:
   - Visit freesound.org
   - Search for "British male voice" or "AI assistant voice"
   - Download a clean sample

REQUIREMENTS FOR GOOD CLONING:
- WAV format (not MP3)
- 6-15 seconds of speech
- Clear audio, no background noise
- Single speaker only

Once you have the file, place it here:
""")

project_dir = Path(__file__).parent.parent.parent
sample_path = project_dir / "jarvis_sample.wav"
print(f"  {sample_path}")

# Check if sample exists
if sample_path.exists():
    print(f"\n✓ Voice sample found: {sample_path}")
    print("  You can now run jarvis_voice.py to test!")
else:
    print(f"\n✗ No voice sample found yet")
    print("  Add jarvis_sample.wav to use voice cloning")

# Also check in backend folder
backend_sample = Path(__file__).parent / "jarvis_sample.wav"
if backend_sample.exists():
    print(f"\n✓ Voice sample found: {backend_sample}")

print("\n" + "=" * 60)
print("QUICK TEST (without voice cloning):")
print("=" * 60)
print("""
Run this to test with default voice:
  python jarvis_voice.py

The XTTS model will use its default voice until you add a sample.
""")
