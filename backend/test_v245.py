"""
JARVIS V24.5 Test Suite
=======================
Tests the memory-first cognitive pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that V24.5 modules import correctly."""
    print("\n" + "="*60)
    print("TEST 1: V24.5 Module Imports")
    print("="*60)
    
    try:
        from jarvis_brain_v245 import brain_v245
        print(f"  ✅ Brain V24.5 imported")
    except Exception as e:
        print(f"  ❌ Brain V24.5: {e}")
        return False
    
    try:
        from jarvis_dispatcher_v245 import dispatcher_v245
        print(f"  ✅ Dispatcher V24.5 imported")
    except Exception as e:
        print(f"  ❌ Dispatcher V24.5: {e}")
        return False
    
    return True


def test_memory_collections():
    """Test that all memory collections exist."""
    print("\n" + "="*60)
    print("TEST 2: Memory Collections")
    print("="*60)
    
    from jarvis_brain_v245 import brain_v245
    
    collections = ['facts', 'assets', 'lessons', 'preferences', 'mistakes', 'actions']
    all_ok = True
    
    for name in collections:
        coll = getattr(brain_v245, name, None)
        if coll is not None:
            print(f"  ✅ {name}: ready")
        else:
            print(f"  ❌ {name}: missing")
            all_ok = False
    
    return all_ok


def test_memory_fetch():
    """Test Step 0: Memory pre-fetch."""
    print("\n" + "="*60)
    print("TEST 3: Memory Pre-fetch (Step 0)")
    print("="*60)
    
    from jarvis_brain_v245 import brain_v245
    
    context = brain_v245.step_0_memory_fetch("test query for memory")
    
    required_keys = ['relevant_facts', 'past_lessons', 'preferences', 'past_mistakes', 'recent_actions']
    all_ok = True
    
    for key in required_keys:
        if key in context:
            print(f"  ✅ {key}: {type(context[key]).__name__}")
        else:
            print(f"  ❌ {key}: missing")
            all_ok = False
    
    return all_ok


def test_cognitive_planning():
    """Test Step 1: Memory-informed planning."""
    print("\n" + "="*60)
    print("TEST 4: Cognitive Planning (Step 1)")
    print("="*60)
    
    from jarvis_brain_v245 import brain_v245
    
    test_cases = [
        ("Hello Jarvis", "none"),
        ("Search for Tesla news", "web_search"),
        ("What's on my screen", "vision"),
        ("Generate an image of a dog", "image_gen"),
        ("Open Notes", "automation"),
    ]
    
    passed = 0
    for text, expected in test_cases:
        try:
            memory_context = brain_v245.step_0_memory_fetch(text)
            result = brain_v245.step_1_plan(text, memory_context)
            worker = result.get("worker", "unknown")
            reasoning = result.get("reasoning", "")[:30]
            
            if worker == expected:
                print(f"  ✅ '{text[:25]}...' -> {worker}")
                passed += 1
            else:
                print(f"  ⚠️ '{text[:25]}...' -> {worker} (expected: {expected})")
                print(f"     Reasoning: {reasoning}")
        except Exception as e:
            print(f"  ❌ '{text[:25]}...' -> ERROR: {e}")
    
    print(f"\n  Planning accuracy: {passed}/{len(test_cases)}")
    return passed >= 4  # Allow 1 miss


def test_memory_update():
    """Test Step 4: Memory update (learning)."""
    print("\n" + "="*60)
    print("TEST 5: Memory Update (Step 4)")
    print("="*60)
    
    from jarvis_brain_v245 import brain_v245
    import time
    
    # Add a test lesson
    test_goal = f"Test action at {int(time.time())}"
    test_result = {"status": "success", "summary": "Test completed"}
    test_review = {"success": True, "quality": "good", "lesson": "Test lesson: always verify"}
    
    try:
        brain_v245.step_4_update_memory(test_goal, "test", test_result, test_review)
        print("  ✅ Memory update completed")
        
        # Verify it was stored
        if brain_v245.actions:
            results = brain_v245.actions.query(query_texts=[test_goal], n_results=1)
            if results["documents"] and results["documents"][0]:
                print(f"  ✅ Action logged: {results['documents'][0][0][:50]}")
            else:
                print("  ⚠️ Action may not have been stored")
        
        return True
    except Exception as e:
        print(f"  ❌ Memory update failed: {e}")
        return False


def test_preference_fact():
    """Test adding preferences and facts."""
    print("\n" + "="*60)
    print("TEST 6: Preferences & Facts")
    print("="*60)
    
    from jarvis_brain_v245 import brain_v245
    
    try:
        brain_v245.add_preference("User prefers Safari browser")
        print("  ✅ Added preference")
        
        brain_v245.add_fact("User's name is Sir")
        print("  ✅ Added fact")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("  JARVIS V24.5 COGNITIVE BRAIN TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Memory Collections", test_memory_collections()))
    results.append(("Memory Fetch", test_memory_fetch()))
    results.append(("Preferences & Facts", test_preference_fact()))
    results.append(("Memory Update", test_memory_update()))
    
    # Optional: Brain planning test
    print("\n" + "-"*60)
    user_input = input("Run cognitive planning test? (requires Ollama) [y/N]: ")
    if user_input.lower() == 'y':
        results.append(("Cognitive Planning", test_cognitive_planning()))
    
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
        print("\n  🎉 V24.5 Cognitive Brain is ready!")
    else:
        print("\n  ⚠️ Some tests failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
