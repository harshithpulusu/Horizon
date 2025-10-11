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
    elif any(phrase in text_lower for phrase in [
        'generate image', 'create image', 'make image', 'generate picture', 'create picture', 
        'make picture', 'draw', 'create an image', 'generate an image', 'make an image'
    ]):
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
        # Extract image description more thoroughly
        text_lower = user_input.lower()
        prompt = user_input
        
        # Remove trigger words to extract the actual description
        for word in ['generate', 'create', 'make', 'draw', 'image', 'picture', 'an', 'a', 'of']:
            prompt = re.sub(r'\b' + word + r'\b', '', prompt, flags=re.IGNORECASE)
        
        prompt = re.sub(r'\s+', ' ', prompt).strip()  # Clean up extra spaces
        
        if not prompt or len(prompt) < 3:
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
        return f"🎨 I've created your image! Here it is: <img src='{image_url}' alt='{prompt}' style='max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;' /><br>📝 Prompt: {prompt}"
        
    except Exception as e:
        print(f"Image generation error: {e}")
        return f"🎨 I had trouble generating that image. Error: {str(e)}"

def get_chatgpt_response(user_input, personality='friendly'):
    """Get response from ChatGPT"""
    if not AI_MODEL_AVAILABLE or not client:
        return "I'm having trouble connecting to ChatGPT right now. Please try again later."
    
    try:
        # Enhanced personality prompts with more character
        personality_prompts = {
            'friendly': "You are Horizon, a warm and friendly AI assistant. Be helpful, encouraging, and use a conversational tone. Use emojis occasionally 😊 and phrases like 'I'd be happy to help!' and 'That's a great question!'",
            'professional': "You are Horizon, a professional AI assistant. Use formal language, structured responses, and business terminology. Begin responses with phrases like 'I shall assist you with that matter' or 'Allow me to provide you with accurate information.' Maintain corporate formality.",
            'casual': "You are Horizon, a super chill and laid-back AI assistant. Use casual slang like 'Hey there!', 'No worries!', 'Cool!', 'Awesome!', and 'For sure!' Keep things relaxed and conversational like talking to a friend.",
            'enthusiastic': "You are Horizon, an incredibly enthusiastic and energetic AI assistant! Use LOTS of exclamation points!!! Express excitement with phrases like 'That's AMAZING!', 'I LOVE helping with this!', and 'This is fantastic!' Use emojis liberally! 🚀✨🎉",
            'wise': "You are Horizon, a wise and thoughtful AI assistant. Speak with depth and consideration, offering profound insights. Use phrases like 'In my understanding...' and 'One might consider...' Share knowledge with wisdom and patience.",
            'creative': "You are Horizon, a highly creative and imaginative AI assistant! Think outside the box, use vivid descriptions, and approach problems with artistic flair. Use colorful language and creative metaphors to make responses engaging and inspiring! 🎨✨"
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

@app.route('/api/history/<session_id>')
def get_conversation_history(session_id):
    """Get conversation history for a session"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_input, ai_response, timestamp, intent
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''', (session_id,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'user_input': row[0],
                'ai_response': row[1],
                'timestamp': row[2],
                'intent': row[3]
            })
        
        conn.close()
        
        return jsonify({
            'session_id': session_id,
            'history': history,
            'count': len(history),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/personalities')
def get_personalities():
    """Get available personality types"""
    personalities = {
        'friendly': {
            'name': 'Friendly',
            'description': 'Warm, encouraging, and conversational with emojis',
            'example': 'I\'d be happy to help! 😊 That\'s a great question!'
        },
        'professional': {
            'name': 'Professional', 
            'description': 'Formal, structured, and business-oriented',
            'example': 'I shall assist you with that matter in a professional capacity.'
        },
        'casual': {
            'name': 'Casual',
            'description': 'Relaxed, laid-back, and friendly like talking to a buddy',
            'example': 'Hey there! No worries, I got you covered. That\'s awesome!'
        },
        'enthusiastic': {
            'name': 'Enthusiastic',
            'description': 'High-energy, excited, and lots of exclamation points!',
            'example': 'That\'s AMAZING! I LOVE helping with this! 🚀✨🎉'
        },
        'wise': {
            'name': 'Wise',
            'description': 'Thoughtful, deep, and philosophical responses',
            'example': 'In my understanding, one might consider the profound implications...'
        },
        'creative': {
            'name': 'Creative',
            'description': 'Imaginative, artistic, and colorful language',
            'example': 'Let\'s paint this conversation with vibrant ideas! 🎨✨'
        }
    }
    
    return jsonify({
        'personalities': personalities,
        'default': 'friendly',
        'count': len(personalities)
    })

if __name__ == '__main__':
    print("🚀 Starting Horizon AI Assistant (Clean Version)...")
    print("✅ Database initialization...")
    init_db()
    print("🌐 Server starting on http://0.0.0.0:8080...")
    print("📱 Local access: http://127.0.0.1:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)