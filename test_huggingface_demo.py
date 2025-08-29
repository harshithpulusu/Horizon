#!/usr/bin/env python3
"""
Demonstration of Hugging Face Enhanced AI vs Basic AI
"""

import requests
import json

def test_ai_comparison():
    print("🧪 Testing Enhanced AI with Hugging Face Transformers")
    print("=" * 60)
    
    # Test complex questions that showcase the enhanced capabilities
    test_questions = [
        {
            "input": "What is artificial intelligence and how does machine learning work?",
            "personality": "professional",
            "description": "Complex AI/ML Question"
        },
        {
            "input": "Explain quantum computing in simple terms",
            "personality": "friendly", 
            "description": "Technical Topic Explanation"
        },
        {
            "input": "How can I become a better programmer?",
            "personality": "enthusiastic",
            "description": "Programming Advice"
        },
        {
            "input": "Tell me about the future of technology",
            "personality": "casual",
            "description": "Technology Discussion"
        }
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question['description']}")
        print(f"Question: {question['input']}")
        print(f"Personality: {question['personality']}")
        print("-" * 40)
        
        try:
            response = requests.post(
                "http://127.0.0.1:8080/api/process",
                headers={"Content-Type": "application/json"},
                json=question,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Enhanced AI Response:")
                print(f"   {data['response']}")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Server not running on port 8080")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Key Features of Enhanced AI:")
    print("✅ Hugging Face tokenization for better text understanding")
    print("✅ Advanced topic detection (AI, programming, science, etc.)")
    print("✅ Context-aware responses based on text complexity")
    print("✅ Personality-adapted responses")
    print("✅ Rich knowledge base for technical topics")
    print("✅ Intelligent conversation flow")

if __name__ == "__main__":
    test_ai_comparison()
