#!/usr/bin/env python3
"""
Test script for the Smart AI Assistant
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import ask_ai_model, AI_MODEL_AVAILABLE, SMART_RESPONSES

def test_smart_ai():
    print("🧪 Testing Smart AI Assistant")
    print(f"AI Model Available: {AI_MODEL_AVAILABLE}")
    print()
    
    # Test different question types and personalities
    test_cases = [
        ("What is artificial intelligence?", "friendly"),
        ("How do I learn programming?", "professional"),
        ("Why is the sky blue?", "enthusiastic"),
        ("When should I start learning AI?", "casual"),
        ("Where can I find good resources?", "friendly"),
        ("Tell me about music", "enthusiastic"),
        ("What about food recommendations?", "casual"),
        ("I love technology", "professional"),
        ("Movie suggestions please", "friendly"),
        ("Random conversation", "enthusiastic")
    ]
    
    for i, (question, personality) in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"Question: {question}")
        print(f"Personality: {personality}")
        
        response = ask_ai_model(question, personality)
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    test_smart_ai()
