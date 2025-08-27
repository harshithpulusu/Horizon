#!/usr/bin/env python3
"""
Test script to demonstrate enhanced NLP capabilities with spaCy
"""

import requests
import json

def test_nlp_understanding():
    """Test the enhanced NLP understanding"""
    base_url = "http://127.0.0.1:8000"
    
    # Test cases that should be better understood with spaCy
    test_cases = [
        # Timer tests with natural language
        "Hey Horizon, set a timer for 5 minutes please",
        "Can you set timer 10 mins",
        "Timer for 30 seconds",
        "countdown 2 minutes",
        "Set alarm for 1 hour",
        
        # Math with natural expressions
        "What's 15 times 7?",
        "Calculate 100 divided by 4",
        "Solve 25 plus 75",
        "What is 50 minus 23?",
        
        # Reminders with complex sentences
        "Remind me to call mom tomorrow",
        "Don't forget to buy groceries",
        "Remember to take medication",
        "Set reminder: meeting at 3pm",
        
        # Time requests
        "What time is it right now?",
        "Current time please",
        "Tell me the time",
        
        # Jokes with variations
        "Make me laugh",
        "Tell me something funny",
        "I need a good joke",
        
        # Complex sentences
        "Hey Horizon, can you please set a timer for 15 minutes so I can take a break?",
        "I'd like to know what 45 multiplied by 12 equals please",
        "Could you remind me to water the plants this evening?"
    ]
    
    print("🧠 Testing Enhanced NLP Understanding with spaCy")
    print("=" * 60)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Testing: '{test_input}'")
        
        try:
            response = requests.post(f"{base_url}/api/process", 
                                   json={"message": test_input, "personality": "friendly"},
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                intent = data.get('intent', 'unknown')
                confidence = data.get('confidence', 0)
                result = data.get('response', 'No response')
                
                print(f"    ✅ Intent: {intent} (confidence: {confidence:.3f})")
                print(f"    📝 Response: {result[:100]}{'...' if len(result) > 100 else ''}")
            else:
                print(f"    ❌ Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Connection error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 NLP Enhancement Test Complete!")
    print("\nKey improvements with spaCy:")
    print("• Better entity extraction (numbers, time units)")
    print("• More flexible pattern matching")
    print("• Enhanced confidence scoring")
    print("• Natural language understanding")

if __name__ == "__main__":
    test_nlp_understanding()
