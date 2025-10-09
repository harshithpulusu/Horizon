#!/usr/bin/env python3
"""
Simple test to verify ChatGPT API is working
"""

import sys
import os
sys.path.append('.')

from config import Config
from openai import OpenAI

# Test ChatGPT API connection
def test_chatgpt():
    print("🧪 Testing ChatGPT API connection...")
    
    try:
        # Initialize OpenAI client
        api_key = getattr(Config, 'OPENAI_API_KEY', None)
        if not api_key:
            print("❌ No OpenAI API key found in config")
            return False
            
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized")
        
        # Test a simple API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Say hello and confirm you're working!"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content.strip()
        print(f"✅ ChatGPT Response: {ai_response}")
        return True
        
    except Exception as e:
        print(f"❌ ChatGPT API Error: {e}")
        return False

if __name__ == "__main__":
    test_chatgpt()