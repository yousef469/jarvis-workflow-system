import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from image_generator import process_image_request

import traceback

def test_gen():
    prompt = "a futuristic cyberpunk city with neon lights and rain"
    print(f"Testing image generation for: {prompt}")
    try:
        result = process_image_request(prompt)
        print(f"Result: {result}")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_gen()
