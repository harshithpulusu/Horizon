#!/usr/bin/env python3
"""
Test Core Module Integration

This script tests the integration between the extracted core AI engine
and the existing web application functionality.
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ai_engine_import():
    """Test importing AI engine from core module."""
    print("🧪 Testing AI engine import...")
    
    try:
        from core import AIEngine, get_ai_engine, ask_ai_model, generate_fallback_response
        print("✅ Successfully imported AI engine components from core module")
        return True
    except Exception as e:
        print(f"❌ Failed to import AI engine: {e}")
        return False

def test_ai_engine_initialization():
    """Test AI engine initialization."""
    print("\n🧪 Testing AI engine initialization...")
    
    try:
        from core import get_ai_engine
        engine = get_ai_engine()
        print(f"✅ AI engine initialized: {type(engine).__name__}")
        
        # Check if APIs are available
        if engine.ai_model_available:
            print("✅ OpenAI ChatGPT API available")
        else:
            print("⚠️ OpenAI ChatGPT API not available (fallback mode)")
            
        if engine.gemini_configured:
            print("✅ Google Gemini API available")
        else:
            print("⚠️ Google Gemini API not available")
            
        return True
    except Exception as e:
        print(f"❌ Failed to initialize AI engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_responses():
    """Test fallback response generation."""
    print("\n🧪 Testing fallback response generation...")
    
    try:
        from core import generate_fallback_response
        
        test_cases = [
            ("What is artificial intelligence?", "friendly"),
            ("How do I learn programming?", "professional"),
            ("Tell me about science", "enthusiastic"),
            ("What's technology?", "casual"),
            ("Help me understand learning", "zen")
        ]
        
        for user_input, personality in test_cases:
            response = generate_fallback_response(user_input, personality)
            print(f"✅ {personality.capitalize()} response: {response[:80]}...")
            
        return True
    except Exception as e:
        print(f"❌ Failed to generate fallback responses: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_model_function():
    """Test main AI model function."""
    print("\n🧪 Testing main AI model function...")
    
    try:
        from core import ask_ai_model
        
        # Test with a simple question
        user_input = "What is the meaning of life?"
        personality = "friendly"
        
        response, context_used = ask_ai_model(user_input, personality)
        
        print(f"✅ AI response generated: {response[:100]}...")
        print(f"✅ Context used: {context_used}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test AI model function: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_personality_handling():
    """Test different personality types."""
    print("\n🧪 Testing personality handling...")
    
    try:
        from core import generate_fallback_response
        
        personalities = [
            "friendly", "professional", "casual", "enthusiastic", 
            "witty", "sarcastic", "zen", "scientist", "pirate", 
            "shakespearean", "valley_girl", "cowboy", "robot"
        ]
        
        user_input = "Hello there!"
        
        for personality in personalities:
            response = generate_fallback_response(user_input, personality)
            print(f"✅ {personality}: {response[:60]}...")
            
        return True
    except Exception as e:
        print(f"❌ Failed to test personality handling: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all core integration tests."""
    print("🚀 Starting Core Module Integration Tests")
    print("=" * 50)
    
    tests = [
        test_ai_engine_import,
        test_ai_engine_initialization,
        test_fallback_responses,
        test_ai_model_function,
        test_personality_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core integration tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)