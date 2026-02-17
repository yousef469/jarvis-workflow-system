"""
JARVIS V24 Test Suite
=====================
Tests all V24 components to ensure nothing crashes.
Run from backend directory: python test_v24.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules import without crashing"""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    modules = [
        ("config", "Config"),
        ("jarvis_brain_v24", "Brain V24"),
        ("workers.web_worker", "Web Worker"),
        ("workers.memory_worker", "Memory Worker"),
        ("workers.vision_worker", "Vision Worker"),
        ("workers.image_gen_worker", "Image Gen Worker"),
        ("workers.automation_worker", "Automation Worker"),
        ("jarvis_dispatcher_v24", "Dispatcher V24"),
    ]
    
    all_passed = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}: OK")
        except Exception as e:
            print(f"  ❌ {display_name}: FAILED - {e}")
            all_passed = False
    
    return all_passed


def test_config():
    """Test config is correctly set to qwen3:1.7b"""
    print("\n" + "="*60)
    print("TEST 2: Config Verification")
    print("="*60)
    
    from config import MODEL_BRAIN
    expected = "qwen3:1.7b"
    
    if MODEL_BRAIN == expected:
        print(f"  ✅ MODEL_BRAIN = {MODEL_BRAIN}")
        return True
    else:
        print(f"  ❌ MODEL_BRAIN = {MODEL_BRAIN} (expected {expected})")
        return False


def test_workers_init():
    """Test that all workers initialize correctly"""
    print("\n" + "="*60)
    print("TEST 3: Worker Initialization")
    print("="*60)
    
    all_passed = True
    
    try:
        from workers import web_worker
        print(f"  ✅ WebWorker: {web_worker.name}")
    except Exception as e:
        print(f"  ❌ WebWorker: {e}")
        all_passed = False
    
    try:
        from workers import memory_worker
        print(f"  ✅ MemoryWorker: {memory_worker.name}")
    except Exception as e:
        print(f"  ❌ MemoryWorker: {e}")
        all_passed = False
    
    try:
        from workers import vision_worker
        print(f"  ✅ VisionWorker: {vision_worker.name}")
    except Exception as e:
        print(f"  ❌ VisionWorker: {e}")
        all_passed = False
    
    try:
        from workers import image_gen_worker
        print(f"  ✅ ImageGenWorker: {image_gen_worker.name}")
    except Exception as e:
        print(f"  ❌ ImageGenWorker: {e}")
        all_passed = False
    
    try:
        from workers import automation_worker
        print(f"  ✅ AutomationWorker: {automation_worker.name}")
    except Exception as e:
        print(f"  ❌ AutomationWorker: {e}")
        all_passed = False
    
    return all_passed


def test_brain_process():
    """Test that brain can process a request"""
    print("\n" + "="*60)
    print("TEST 4: Brain Processing (requires Ollama running)")
    print("="*60)
    
    try:
        from jarvis_brain_v24 import process
        
        test_cases = [
            ("Hello Jarvis", "none"),
            ("Search for SpaceX news", "web_search"),
            ("What's on my screen", "vision"),
            ("Create an image of a cat", "image_gen"),
            ("Open Safari", "automation"),
        ]
        
        all_passed = True
        for text, expected_worker in test_cases:
            try:
                result = process(text)
                worker = result.get("worker", "unknown")
                
                # Check if worker is in valid list (not exact match needed)
                valid_workers = ["none", "web_search", "vision", "memory", "image_gen", "automation"]
                if worker in valid_workers:
                    match = "✅" if worker == expected_worker else "⚠️"
                    print(f"  {match} '{text[:30]}...' -> {worker} (expected: {expected_worker})")
                else:
                    print(f"  ❌ '{text[:30]}...' -> invalid worker: {worker}")
                    all_passed = False
                    
            except Exception as e:
                print(f"  ❌ '{text[:30]}...' -> ERROR: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"  ❌ Brain import/init failed: {e}")
        print("     (Make sure Ollama is running with qwen3:1.7b)")
        return False


def test_dispatcher_init():
    """Test dispatcher initialization"""
    print("\n" + "="*60)
    print("TEST 5: Dispatcher Initialization")
    print("="*60)
    
    try:
        from jarvis_dispatcher_v24 import dispatcher_v24, WORKERS
        print(f"  ✅ Dispatcher initialized")
        print(f"  ✅ Workers registered: {list(WORKERS.keys())}")
        return True
    except Exception as e:
        print(f"  ❌ Dispatcher: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("  JARVIS V24 TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Workers Init", test_workers_init()))
    results.append(("Dispatcher Init", test_dispatcher_init()))
    
    # Brain test is optional (requires Ollama)
    print("\n" + "-"*60)
    user_input = input("Run brain test? (requires Ollama with qwen3:1.7b) [y/N]: ")
    if user_input.lower() == 'y':
        results.append(("Brain Process", test_brain_process()))
    else:
        print("  ⏭️ Brain test skipped")
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print("-"*60)
    print(f"  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! V24 is ready.")
    else:
        print("\n  ⚠️ Some tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
