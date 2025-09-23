#!/usr/bin/env python3
"""
Test script for AI Personality Ecosystem and Cross-Platform Sync features
"""

import urllib.request
import urllib.parse
import json
import time

# Base URL for the application
BASE_URL = "http://127.0.0.1:8080"

def make_request(url, method="GET", data=None):
    """Make HTTP request using urllib"""
    try:
        if data:
            data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header('Content-Type', 'application/json')
        else:
            req = urllib.request.Request(url, method=method)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.getcode(), json.loads(response.read().decode())
    except Exception as e:
        return None, str(e)

def test_personalities_api():
    """Test the personalities API endpoint"""
    print("🧪 Testing AI Personality Ecosystem...")
    
    status, data = make_request(f"{BASE_URL}/api/personalities")
    print(f"GET /api/personalities - Status: {status}")
    
    if status == 200:
        print(f"✅ Success: Found {len(data.get('personalities', []))} personalities")
        for personality in data.get('personalities', [])[:3]:  # Show first 3
            print(f"   - {personality.get('name', 'Unknown')} ({personality.get('type', 'Unknown')})")
    else:
        print(f"❌ Failed: {data}")

def test_personality_switching():
    """Test personality switching functionality"""
    print("\n🔄 Testing Personality Switching...")
    
    switch_data = {"personality_id": "artist"}
    status, data = make_request(f"{BASE_URL}/api/personalities/switch", "POST", switch_data)
    print(f"POST /api/personalities/switch - Status: {status}")
    
    if status == 200:
        print(f"✅ Successfully switched to: {data.get('personality', {}).get('name', 'Unknown')}")
    else:
        print(f"❌ Failed to switch: {data}")

def test_sync_status():
    """Test cross-platform sync status"""
    print("\n🔄 Testing Cross-Platform Sync...")
    
    status, data = make_request(f"{BASE_URL}/api/sync/status")
    print(f"GET /api/sync/status - Status: {status}")
    
    if status == 200:
        print(f"✅ Sync status: {data.get('sync_status', {}).get('status', 'Unknown')}")
        print(f"   Device count: {data.get('sync_status', {}).get('device_count', 0)}")
    else:
        print(f"❌ Failed to get sync status: {data}")

def test_chat_with_personality():
    """Test chatting with a specific personality"""
    print("\n💬 Testing Chat with Personality...")
    
    chat_data = {
        "message": "Switch to Artist AI and help me create a logo design",
        "session_id": "test_session"
    }
    status, data = make_request(f"{BASE_URL}/api/chat", "POST", chat_data)
    print(f"POST /api/chat - Status: {status}")
    
    if status == 200:
        print(f"✅ Chat response received")
        print(f"   Response type: {data.get('type', 'Unknown')}")
        if 'response' in data:
            print(f"   Response preview: {data['response'][:100]}...")
    else:
        print(f"❌ Failed to chat: {data}")

def main():
    """Run all tests"""
    print("🚀 Starting AI Personality Ecosystem & Cross-Platform Sync Tests\n")
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    # Run tests
    test_personalities_api()
    test_personality_switching()
    test_sync_status()
    test_chat_with_personality()
    
    print("\n✨ Test completed!")

if __name__ == "__main__":
    main()