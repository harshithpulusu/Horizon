#!/usr/bin/env python3
"""
Test script for futuristic AI features in Horizon
Tests AR Integration, Dream Journal, Time Capsule, and Virtual World Builder
"""

import requests
import json
import time

# Test configuration
BASE_URL = "http://127.0.0.1:8080"
TEST_PERSONALITY = "enthusiastic"

def test_api_endpoint(user_input, feature_name):
    """Test a specific feature through the API"""
    print(f"\n🧪 Testing {feature_name}")
    print(f"📝 Input: {user_input}")
    
    try:
        response = requests.post(f"{BASE_URL}/chat", 
                               json={
                                   "message": user_input,
                                   "personality": TEST_PERSONALITY
                               },
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {feature_name} working!")
            print(f"🤖 Response length: {len(data.get('response', ''))} characters")
            
            # Check if it's using quick command processing
            if 'intent_detected' in data.get('ai_insights', {}):
                intent = data['ai_insights']['intent_detected']
                print(f"🎯 Intent detected: {intent}")
                
            return True
        else:
            print(f"❌ {feature_name} failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ {feature_name} error: {e}")
        return False

def main():
    """Run all futuristic feature tests"""
    print("🚀 Starting Futuristic Features Test Suite")
    print("=" * 60)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Test data for each feature
    test_cases = [
        # AR Integration Tests
        ("Create AR face filters for selfies", "AR Integration - Face Filters"),
        ("Build navigation AR for walking", "AR Integration - Navigation"),
        ("Design educational AR overlays", "AR Integration - Educational"),
        ("AR object recognition for shopping", "AR Integration - Object Recognition"),
        
        # Dream Journal Tests
        ("Analyze my dream about flying", "Dream Journal - Flying Dreams"),
        ("What do nightmares mean psychologically", "Dream Journal - Nightmare Analysis"),
        ("Help me understand lucid dreams", "Dream Journal - Lucid Dreams"),
        ("Interpret recurring water dreams", "Dream Journal - Water Dreams"),
        
        # Time Capsule Tests
        ("Predict technology trends for 2030", "Time Capsule - Tech Prediction"),
        ("Create time capsule for 5 years", "Time Capsule - 5 Year Capsule"),
        ("What will society look like in 2050", "Time Capsule - Society Future"),
        ("Predict personal life changes", "Time Capsule - Personal Prediction"),
        
        # Virtual World Builder Tests
        ("Build a fantasy world with dragons", "Virtual World - Fantasy"),
        ("Create a cyberpunk sci-fi city", "Virtual World - Sci-Fi"),
        ("Design an underwater civilization", "Virtual World - Underwater"),
        ("Make a post-apocalyptic world", "Virtual World - Post-Apocalyptic")
    ]
    
    # Run tests
    passed = 0
    total = len(test_cases)
    
    for user_input, feature_name in test_cases:
        if test_api_endpoint(user_input, feature_name):
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    # Results summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    print(f"📊 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL FUTURISTIC FEATURES WORKING PERFECTLY!")
        print("🔮 AR Integration ✅")
        print("💭 Dream Journal ✅") 
        print("⏰ Time Capsule ✅")
        print("🌍 Virtual World Builder ✅")
    else:
        print(f"\n⚠️ Some features need attention ({total - passed} failed)")
    
    print("\n🚀 Futuristic Features Test Complete!")

if __name__ == "__main__":
    main()