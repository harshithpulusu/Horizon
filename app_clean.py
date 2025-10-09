#!/usr/bin/env python3
"""
Horizon AI Assistant - Clean Working Version
Streamlined ChatGPT integration with essential features
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import sqlite3
import json
import re
import os
import time
from openai import OpenAI
from config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=getattr(Config, 'OPENAI_API_KEY', None))
    AI_MODEL_AVAILABLE = True
    print("✅ ChatGPT API connected successfully")
except Exception as e:
    client = None
    AI_MODEL_AVAILABLE = False
    print(f"⚠️ ChatGPT API initialization failed: {e}")

# Create directories
os.makedirs('static/generated_images', exist_ok=True)

def init_db():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_input TEXT,
                ai_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'anonymous',
                intent TEXT,
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️ Database initialization failed: {e}")

def recognize_intent(text):
    """Simple intent recognition"""
    text_lower = text.lower()
    
    # Basic patterns
    if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return 'greeting'
    elif any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
        return 'goodbye'
    elif 'time' in text_lower or 'what time is it' in text_lower:
        return 'time'
    elif 'date' in text_lower or 'what day' in text_lower:
        return 'date'
    elif any(op in text_lower for op in ['+', '-', '*', '/', 'calculate', 'math']):
        return 'math'
    elif 'generate' in text_lower and 'image' in text_lower:
        return 'image_generation'
    else:
        return 'general'

def handle_greeting():
    """Handle greeting messages"""
    greetings = [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Hey! I'm here to assist you.",
        "Good to see you! What would you like to know?"
    ]
    import random
    return random.choice(greetings)

def handle_time():
    """Handle time requests"""
    current_time = datetime.now().strftime("%I:%M %p")
    return f"The current time is {current_time}"

def handle_date():
    """Handle date requests"""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    return f"Today is {current_date}"

def handle_math(user_input):
    """Handle basic math operations"""
    try:
        # Extract numbers and operations
        expression = re.sub(r'[^0-9+\-*/().\s]', '', user_input)
        if expression.strip():
            result = eval(expression.strip())
            return f"The answer is: {result}"
        else:
            return "I couldn't understand the math problem. Please try again with a clear expression like '5 + 3' or '10 * 2'."
    except Exception as e:
        return "I had trouble calculating that. Please make sure your math expression is valid."

def handle_image_generation(user_input):
    """Handle image generation using DALL-E"""
    if not AI_MODEL_AVAILABLE or not client:
        return "🎨 I'd love to generate images for you! However, I need an OpenAI API key to access DALL-E. Please check your configuration."
    
    try:
        # Extract image description
        prompt = user_input.replace('generate', '').replace('create', '').replace('make', '').replace('image', '').replace('picture', '').strip()
        if not prompt:
            prompt = "a beautiful landscape"
        
        print(f"🎨 Generating image with DALL-E: {prompt}")
        
        # Generate image using DALL-E
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        return f"🎨 Image generated successfully! Here's your image: {image_url}"
        
    except Exception as e:
        print(f"Image generation error: {e}")
        return f"🎨 I had trouble generating that image. Error: {str(e)}"

def get_chatgpt_response(user_input, personality='friendly'):
    """Get response from ChatGPT"""
    if not AI_MODEL_AVAILABLE or not client:
        return "I'm having trouble connecting to ChatGPT right now. Please try again later."
    
    try:
        # Personality prompts
        personality_prompts = {
            'friendly': "You are Horizon, a warm and friendly AI assistant. Be helpful, encouraging, and use a conversational tone.",
            'professional': "You are Horizon, a professional AI assistant. Use formal language and provide structured, accurate responses.",
            'casual': "You are Horizon, a casual and laid-back AI assistant. Use relaxed language and be conversational.",
            'enthusiastic': "You are Horizon, an enthusiastic AI assistant! Be energetic and excited to help!"
        }
        
        system_prompt = personality_prompts.get(personality, personality_prompts['friendly'])
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            temperature=0.8
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"ChatGPT API error: {e}")
        return f"I'm having trouble processing that request. Error: {str(e)}"

def save_conversation(session_id, user_input, ai_response, intent, user_id='anonymous'):
    """Save conversation to database"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (session_id, user_input, ai_response, intent, user_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, user_input, ai_response, intent, user_id))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving conversation: {e}")

def process_user_input(user_input, personality='friendly', session_id=None, user_id='anonymous'):
    """Process user input and return appropriate response"""
    if not user_input or not user_input.strip():
        return "I didn't quite catch that. Could you please say something?", session_id
    
    # Initialize database
    init_db()
    
    # Generate session ID if needed
    if not session_id:
        session_id = f"session_{int(time.time())}"
    
    # Recognize intent
    intent = recognize_intent(user_input)
    
    # Handle specific intents locally (quick responses)
    if intent == 'greeting':
        response = handle_greeting()
    elif intent == 'time':
        response = handle_time()
    elif intent == 'date':
        response = handle_date()
    elif intent == 'math':
        response = handle_math(user_input)
    elif intent == 'image_generation':
        response = handle_image_generation(user_input)
    else:
        # Use ChatGPT for general conversation
        response = get_chatgpt_response(user_input, personality)
    
    # Save conversation
    save_conversation(session_id, user_input, response, intent, user_id)
    
    return response, session_id

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('simple_chat.html')

@app.route('/simple')
def simple():
    return render_template('simple_chat.html')

@app.route('/api/process', methods=['POST'])
def api_process():
    """Main chat API endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({'error': 'No input provided'}), 400
        
        user_input = data.get('input', '').strip()
        personality = data.get('personality', 'friendly')
        session_id = data.get('session_id')
        user_id = data.get('user_id', 'anonymous')
        
        if not user_input:
            return jsonify({'error': 'Empty input provided'}), 400
        
        # Process the input
        start_time = time.time()
        response, session_id = process_user_input(user_input, personality, session_id, user_id)
        response_time = round(time.time() - start_time, 2)
        
        # Determine processing type
        intent = recognize_intent(user_input)
        is_quick = intent in ['greeting', 'time', 'date', 'math']
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'personality': personality,
            'session_id': session_id,
            'intent': intent,
            'is_quick_command': is_quick,
            'response_time': f"{response_time}s",
            'ai_source': 'local' if is_quick else 'chatgpt',
            'status': 'success'
        })
        
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'chatgpt_available': AI_MODEL_AVAILABLE
    })

if __name__ == '__main__':
    print("🚀 Starting Horizon AI Assistant (Clean Version)...")
    print("✅ Database initialization...")
    init_db()
    print("🌐 Server starting on http://0.0.0.0:8080...")
    print("📱 Local access: http://127.0.0.1:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)