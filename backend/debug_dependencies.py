import sys
import os
import platform

print("="*50)
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Platform: {platform.platform()}")
print("="*50)

print("\nChecking 'piper'...")
try:
    import piper
    print(f"✅ piper imported successfully from: {os.path.dirname(piper.__file__)}")
except ImportError as e:
    print(f"❌ piper import failed: {e}")
except Exception as e:
    print(f"❌ piper error: {e}")

print("\nChecking 'openwakeword'...")
try:
    import openwakeword
    print(f"✅ openwakeword imported successfully from: {os.path.dirname(openwakeword.__file__)}")
except ImportError as e:
    print(f"❌ openwakeword import failed: {e}")
except Exception as e:
    print(f"❌ openwakeword error: {e}")

print("="*50)
