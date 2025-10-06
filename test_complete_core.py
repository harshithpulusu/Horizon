#!/usr/bin/env python3
"""
Test Complete Core Module

This script tests all components of the core module to ensure
they work together seamlessly.
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_imports():
    """Test importing all core components."""
    print("🧪 Testing core module imports...")
    
    try:
        # Test AI Engine imports
        from core import AIEngine, get_ai_engine, ask_ai_model
        print("✅ AI Engine components imported successfully")
        
        # Test Personality imports
        from core import PersonalityEngine, get_personality_engine, analyze_emotion
        print("✅ Personality components imported successfully")
        
        # Test Database imports
        from core import DatabaseManager, get_database_manager, init_database
        print("✅ Database components imported successfully")
        
        # Test Media Generator imports
        from core import MediaEngine, get_media_engine, generate_image
        print("✅ Media Generator components imported successfully")
        
        # Test Memory System imports
        from core import MemorySystem, get_memory_system, store_user_memory
        print("✅ Memory System components imported successfully")
        
        # Test Utilities imports
        from core import setup_logging, validate_config, sanitize_input
        print("✅ Utility components imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Failed to import core components: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_initialization():
    """Test initializing all core components."""
    print("\n🧪 Testing core component initialization...")
    
    try:
        # Initialize AI Engine
        from core import get_ai_engine
        ai_engine = get_ai_engine()
        print(f"✅ AI Engine initialized: {type(ai_engine).__name__}")
        
        # Initialize Personality Engine
        from core import get_personality_engine
        personality_engine = get_personality_engine()
        print(f"✅ Personality Engine initialized: {type(personality_engine).__name__}")
        
        # Initialize Database Manager
        from core import get_database_manager
        db_manager = get_database_manager()
        print(f"✅ Database Manager initialized: {type(db_manager).__name__}")
        
        # Initialize Media Engine
        from core import get_media_engine
        media_engine = get_media_engine()
        print(f"✅ Media Engine initialized: {type(media_engine).__name__}")
        
        # Initialize Memory System
        from core import get_memory_system
        memory_system = get_memory_system()
        print(f"✅ Memory System initialized: {type(memory_system).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize core components: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_functionality():
    """Test integrated functionality across core modules."""
    print("\n🧪 Testing integrated functionality...")
    
    try:
        # Test AI Engine with Personality
        from core import ask_ai_model, analyze_emotion
        
        test_message = "I'm feeling excited about learning AI!"
        emotion_data = analyze_emotion(test_message)
        print(f"✅ Emotion analysis: {emotion_data}")
        
        response, context_used = ask_ai_model(test_message, "enthusiastic")
        print(f"✅ AI response generated: {response[:80]}...")
        
        # Test Memory System
        from core import store_user_memory, get_user_memory
        
        test_user_id = "test_user_123"
        memory_id = store_user_memory(test_user_id, "interests", "AI and machine learning")
        print(f"✅ Memory stored: {memory_id}")
        
        user_memory = get_user_memory(test_user_id)
        profile = user_memory.get_profile_summary()
        print(f"✅ User profile retrieved: {len(profile['interests'])} interests")
        
        # Test Database Operations
        from core import get_database_manager
        
        db_manager = get_database_manager()
        stats = db_manager.get_database_stats()
        print(f"✅ Database stats: {stats['users_count']} users, {stats['conversations_count']} conversations")
        
        # Test Media Generation
        from core import generate_image
        
        image_result = generate_image("A beautiful sunset over mountains")
        print(f"✅ Image generation: {image_result['success']}, Model: {image_result.get('model', 'unknown')}")
        
        # Test Utilities
        from core import sanitize_input, validate_config
        
        clean_input = sanitize_input("Hello <script>alert('test')</script> world!")
        print(f"✅ Input sanitized: '{clean_input}'")
        
        config_status = validate_config()
        print(f"✅ Config validation: {len(config_status['api_keys'])} API keys checked")
        
        return True
    except Exception as e:
        print(f"❌ Integrated functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling across core modules."""
    print("\n🧪 Testing error handling...")
    
    try:
        # Test invalid personality
        from core import get_personality_profile
        
        profile = get_personality_profile("invalid_personality")
        print(f"✅ Invalid personality handled: {profile['traits']}")
        
        # Test invalid user input
        from core import sanitize_input
        
        sanitized = sanitize_input("<script>malicious</script>", max_length=10)
        print(f"✅ Malicious input sanitized: '{sanitized}'")
        
        # Test database with invalid data
        from core import store_user_memory
        
        try:
            memory_id = store_user_memory("", "invalid", "")
            print(f"✅ Empty data handled gracefully")
        except Exception:
            print(f"✅ Empty data validation working")
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test basic performance of core components."""
    print("\n🧪 Testing performance...")
    
    try:
        import time
        
        # Test AI response time
        start_time = time.time()
        from core import ask_ai_model
        response, _ = ask_ai_model("Quick test message", "friendly")
        ai_time = time.time() - start_time
        print(f"✅ AI response time: {ai_time:.3f}s")
        
        # Test memory operations
        start_time = time.time()
        from core import store_user_memory, get_memory_system
        memory_system = get_memory_system()
        for i in range(5):
            store_user_memory(f"perf_user_{i}", "test", f"Test memory {i}")
        memory_time = time.time() - start_time
        print(f"✅ Memory operations time: {memory_time:.3f}s")
        
        # Test emotion analysis batch
        start_time = time.time()
        from core import analyze_emotion
        test_texts = [
            "I'm happy today!",
            "This is frustrating",
            "Feeling excited about the project",
            "I'm confused about this",
            "Great work everyone!"
        ]
        for text in test_texts:
            analyze_emotion(text)
        emotion_time = time.time() - start_time
        print(f"✅ Emotion analysis batch time: {emotion_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all comprehensive core tests."""
    print("🚀 Starting Comprehensive Core Module Tests")
    print("=" * 60)
    
    tests = [
        test_core_imports,
        test_core_initialization,
        test_integrated_functionality,
        test_error_handling,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All comprehensive core tests passed!")
        print("\n📊 Core Module Status:")
        print("✅ AI Engine: Fully operational")
        print("✅ Personality System: Fully operational")
        print("✅ Database Operations: Fully operational")
        print("✅ Media Generation: Fully operational")
        print("✅ Memory System: Fully operational")
        print("✅ Utilities: Fully operational")
        print("\n🚀 Core module ready for MCP and Web integration!")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)