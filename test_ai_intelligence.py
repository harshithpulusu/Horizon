#!/usr/bin/env python3
"""
Test script for AI Personality & Intelligence features
Demonstrates the new capabilities added to Horizon AI Assistant
"""

import requests
import json
import time

def test_ai_intelligence():
    """Test the new AI intelligence features"""
    base_url = "http://localhost:8080"
    
    print("🤖 HORIZON AI INTELLIGENCE TEST")
    print("=" * 50)
    
    # Test data
    test_messages = [
        {
            "input": "Hi! I'm feeling excited about this new AI features!",
            "personality": "enthusiastic",
            "user_id": "test_user_123"
        },
        {
            "input": "My name is Alice and I work as a software developer",
            "personality": "friendly", 
            "user_id": "test_user_123"
        },
        {
            "input": "I'm feeling a bit worried about the future of AI",
            "personality": "zen",
            "user_id": "test_user_123"
        },
        {
            "input": "Can you help me with some programming questions?",
            "personality": "scientist",
            "user_id": "test_user_123"
        }
    ]
    
    session_id = None
    
    # Test conversation with AI intelligence
    print("\n🎯 Testing AI Conversation with Intelligence Features")
    print("-" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n📝 Test Message {i}:")
        print(f"Input: {message['input']}")
        print(f"Personality: {message['personality']}")
        
        # Add session_id to subsequent messages
        if session_id:
            message['session_id'] = session_id
        
        try:
            # Send message to AI
            response = requests.post(f"{base_url}/api/process", 
                                   json=message, 
                                   timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                session_id = data.get('session_id')
                
                print(f"✅ Response: {data.get('response', 'No response')[:100]}...")
                print(f"🎭 Emotion Detected: {data.get('emotion_detected', 'unknown')}")
                print(f"😊 Sentiment Score: {data.get('sentiment_score', 0):.2f}")
                print(f"🧠 AI Source: {data.get('ai_source', 'unknown')}")
                print(f"📊 Context Used: {data.get('context_used', False)}")
                
                if data.get('ai_insights'):
                    insights = data['ai_insights']
                    print(f"🔍 AI Insights: {len(insights)} insights available")
                
            else:
                print(f"❌ Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection Error: {e}")
            print("💡 Make sure the server is running with: python app.py")
            return
        
        time.sleep(1)  # Small delay between requests
    
    # Test AI insights endpoint
    print(f"\n📊 Testing AI Insights (Session: {session_id})")
    print("-" * 50)
    
    try:
        response = requests.get(f"{base_url}/api/ai-insights", 
                              params={
                                  'session_id': session_id,
                                  'user_id': 'test_user_123'
                              })
        
        if response.status_code == 200:
            insights = response.json()
            print("✅ AI Insights Retrieved:")
            print(f"   📈 Session Stats: {insights.get('session_insights', {})}")
            print(f"   🧠 Memory Count: {insights.get('user_memories', {}).get('count', 0)}")
            print(f"   🎭 Personalities Used: {len(insights.get('personality_stats', []))}")
            print(f"   😊 Recent Emotions: {len(insights.get('recent_emotions', []))}")
        else:
            print(f"❌ Insights Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Insights Error: {e}")
    
    # Test personalities endpoint
    print(f"\n🎭 Testing Personalities Endpoint")
    print("-" * 50)
    
    try:
        response = requests.get(f"{base_url}/api/personalities")
        
        if response.status_code == 200:
            data = response.json()
            personalities = data.get('personalities', [])
            print(f"✅ Found {len(personalities)} personalities:")
            
            for p in personalities[:5]:  # Show first 5
                print(f"   • {p['name']}: {p['description'][:50]}...")
                print(f"     Usage: {p['usage_count']}, Rating: {p.get('rating', 'N/A')}")
        else:
            print(f"❌ Personalities Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Personalities Error: {e}")
    
    # Test memory endpoint
    print(f"\n🧠 Testing Memory Endpoint")
    print("-" * 50)
    
    try:
        response = requests.get(f"{base_url}/api/memory", 
                              params={'user_id': 'test_user_123'})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ User Memories: {data.get('count', 0)} items")
            
            memories = data.get('memories', [])
            for memory in memories[:3]:  # Show first 3
                if len(memory) >= 3:
                    print(f"   • {memory[1]}: {memory[2]}")
        else:
            print(f"❌ Memory Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Memory Error: {e}")
    
    print("\n🎉 AI Intelligence Test Complete!")
    print("=" * 50)
    print("\n✨ New Features Available:")
    print("🧠 Memory System: Remembers user preferences and info")
    print("🎭 Multiple Personalities: 13+ different AI characters") 
    print("😊 Emotion Detection: Analyzes user mood and responds appropriately")
    print("📚 Learning System: AI gets smarter with each interaction")
    print("📊 AI Insights: Detailed analytics and intelligence data")
    print("🔄 Context Awareness: Maintains conversation flow and memory")

if __name__ == "__main__":
    test_ai_intelligence()
