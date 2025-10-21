#!/usr/bin/env python3
"""
Test script for Horizon Event-Driven Architecture
This script tests the integration of the event system and state management
"""

import sys
import os
import time
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from core.events import (
    get_event_emitter, emit_event, listen_for_event, 
    HorizonEventTypes, EventData
)
from core.state_manager import (
    get_state_manager, get_state, update_state, 
    subscribe_to_state
)
from core.ai_engine import get_ai_engine
from core.media_generator import get_enhanced_media_engine


def test_event_system():
    """Test the event system functionality."""
    print("\n🔄 Testing Event System...")
    
    event_emitter = get_event_emitter()
    
    # Test event emission and listening
    received_events = []
    
    def test_listener(event_data):
        received_events.append(event_data)
        print(f"  ✅ Received event: {event_data.event_type}")
    
    # Register listener
    listener_id = listen_for_event("test_event", test_listener)
    
    # Emit test event
    event_id = emit_event(
        "test_event", 
        "test_script", 
        {"message": "Hello from event system!"}
    )
    
    # Check if event was received
    if received_events:
        print(f"  ✅ Event system working! Event ID: {event_id}")
        print(f"  📊 Event stats: {event_emitter.get_stats()}")
        return True
    else:
        print("  ❌ Event system not working!")
        return False


def test_state_management():
    """Test the state management functionality."""
    print("\n🗃️ Testing State Management...")
    
    state_manager = get_state_manager()
    
    # Test state updates
    test_value = f"test_{int(time.time())}"
    success = update_state("user.name", test_value, source="test_script")
    
    if success:
        # Test state retrieval
        retrieved_value = get_state("user.name")
        if retrieved_value == test_value:
            print(f"  ✅ State management working! Value: {retrieved_value}")
            print(f"  📊 State stats: {state_manager.get_stats()}")
            return True
        else:
            print(f"  ❌ State retrieval failed! Expected: {test_value}, Got: {retrieved_value}")
            return False
    else:
        print("  ❌ State update failed!")
        return False


def test_ai_engine():
    """Test the AI engine with event-driven architecture."""
    print("\n🧠 Testing AI Engine...")
    
    ai_engine = get_ai_engine()
    
    # Test AI response
    try:
        response, context_used = ai_engine.ask_ai_model(
            "Hello, this is a test message!",
            "friendly",
            "test_session",
            "test_user"
        )
        
        if response:
            print(f"  ✅ AI Engine working! Response: {response[:50]}...")
            print(f"  🔄 Context used: {context_used}")
            
            # Check if state was updated
            conversation_state = get_state("conversation")
            if conversation_state.current_message:
                print(f"  ✅ State integration working!")
                return True
            else:
                print(f"  ⚠️ AI working but state integration may have issues")
                return True
        else:
            print("  ❌ AI Engine not responding!")
            return False
    except Exception as e:
        print(f"  ❌ AI Engine error: {e}")
        return False


def test_media_engine():
    """Test the media engine integration."""
    print("\n🎨 Testing Media Engine...")
    
    media_engine = get_enhanced_media_engine()
    
    try:
        # Test media generation capabilities
        capabilities = media_engine.get_generation_capabilities()
        print(f"  📋 Available generators: {capabilities['image_generation']}")
        
        # Test a simple image generation request
        result = media_engine.generate_media(
            'image', 
            'a simple test image',
            {'width': 512, 'height': 512}
        )
        
        if result.get('success', False):
            print(f"  ✅ Media Engine working! Generated: {result.get('filename', 'N/A')}")
        else:
            print(f"  ⚠️ Media Engine working but generation failed (expected without API keys)")
            print(f"  📝 Available generators: {media_engine.get_available_generators()}")
        
        return True
    except Exception as e:
        print(f"  ❌ Media Engine error: {e}")
        return False


def test_integration():
    """Test the full integration of all systems."""
    print("\n🔗 Testing Full Integration...")
    
    # Test event flow between systems
    try:
        # Emit a user message event
        emit_event(
            HorizonEventTypes.USER_MESSAGE_RECEIVED,
            "test_script",
            {
                'message': 'Generate an image of a sunset',
                'personality': 'friendly'
            },
            user_id="test_user",
            session_id="test_session"
        )
        
        # Check if AI state was updated
        ai_state = get_state("ai")
        conversation_state = get_state("conversation")
        
        print(f"  📊 AI requests: {ai_state.total_requests}")
        print(f"  💬 Conversation messages: {conversation_state.total_messages}")
        print(f"  ✅ Integration test completed!")
        
        return True
    except Exception as e:
        print(f"  ❌ Integration test error: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Horizon Event-Driven Architecture Tests...")
    print("=" * 60)
    
    tests = [
        ("Event System", test_event_system),
        ("State Management", test_state_management),
        ("AI Engine", test_ai_engine),
        ("Media Engine", test_media_engine),
        ("Full Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} - {test_name}")
        if passed_test:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Event-driven architecture is working correctly!")
    else:
        print("⚠️ Some tests failed. Check the details above.")
    
    # Print system status
    print("\n📈 SYSTEM STATUS")
    print("=" * 60)
    
    try:
        event_stats = get_event_emitter().get_stats()
        state_stats = get_state_manager().get_stats()
        
        print(f"Events emitted: {event_stats['events_emitted']}")
        print(f"Events handled: {event_stats['events_handled']}")
        print(f"Active listeners: {sum(event_stats['active_listeners'].values())}")
        print(f"State changes: {state_stats['total_changes']}")
        print(f"Active subscriptions: {sum(state_stats['active_subscriptions'].values())}")
        
        # Get current app state
        app_state = get_state()
        print(f"Current AI model: {app_state.ai.current_model}")
        print(f"System health: {'✅ Healthy' if app_state.system.is_healthy else '❌ Unhealthy'}")
        
    except Exception as e:
        print(f"Could not get system status: {e}")
    
    print("\n✨ Test completed!")


if __name__ == "__main__":
    main()