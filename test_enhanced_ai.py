#!/usr/bin/env python3
"""
Test script for the Enhanced AI Assistant with Hugging Face
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced functions
from app_enhanced import enhanced_ask_ai_model, AI_MODEL_AVAILABLE, analyze_text_context

def test_enhanced_ai():
    print("🧪 Testing Enhanced AI Assistant with Hugging Face Tokenization")
    print(f"AI Model Available: {AI_MODEL_AVAILABLE}")
    print()
    
    # Test complex AI and tech questions
    test_cases = [
        ("What is artificial intelligence and how does it work?", "enthusiastic"),
        ("How can I learn machine learning and deep learning?", "professional"),
        ("Why is programming such an important skill today?", "friendly"),
        ("Tell me about the future of technology", "casual"),
        ("What makes science so fascinating?", "enthusiastic"),
        ("How do neural networks actually learn?", "professional"),
        ("What's the difference between AI and machine learning?", "friendly"),
        ("I want to become a software developer", "casual"),
        ("Explain quantum computing to me", "enthusiastic"),
        ("What programming languages should I learn?", "professional")
    ]
    
    for i, (question, personality) in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"Question: {question}")
        print(f"Personality: {personality}")
        
        # Analyze the text first
        analysis = analyze_text_context(question)
        print(f"Analysis: Complexity={analysis['complexity']}, Topic={analysis.get('topic_category', 'general')}, Tokens={analysis.get('token_count', 0)}")
        
        response = enhanced_ask_ai_model(question, personality)
        print(f"Response: {response}")
        print("-" * 60)

if __name__ == "__main__":
    test_enhanced_ai()
