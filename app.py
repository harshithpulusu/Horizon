#!/usr/bin/env python3
"""
Horizon AI Assistant - Enhanced Version
Advanced AI features with DALL-E, spaCy NLP, Memory Learning, and Logo Generation
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import sqlite3
import json
import re
import os
import time
import spacy
import random
import hashlib
from openai import OpenAI
from config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize spaCy for enhanced NLP
try:
    nlp = spacy.load("en_core_web_sm")
    NLP_AVAILABLE = True
    print("✅ spaCy NLP model loaded successfully")
except (OSError, IOError):
    nlp = None
    NLP_AVAILABLE = False
    print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")

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
os.makedirs('static/generated_logos', exist_ok=True)

def init_db():
    """Initialize SQLite database with enhanced memory learning"""
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
                confidence REAL,
                entities TEXT,
                sentiment REAL,
                personality TEXT
            )
        ''')
        
        # User preferences and learning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                preference_key TEXT,
                preference_value TEXT,
                learned_from TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Memory patterns for learning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_data TEXT,
                frequency INTEGER DEFAULT 1,
                last_used DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Quick commands usage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quick_commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT,
                usage_count INTEGER DEFAULT 1,
                last_used DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Enhanced database initialized with memory learning")
    except Exception as e:
        print(f"⚠️ Database initialization failed: {e}")

def check_wake_word(text):
    """Check if input contains wake word"""
    text_lower = text.lower().strip()
    wake_words = ["horizon", "hello horizon", "hey horizon", "hi horizon"]
    
    for wake_word in wake_words:
        if wake_word in text_lower:
            # Remove wake word and return cleaned text
            cleaned_text = text_lower.replace(wake_word, "").strip()
            return True, cleaned_text if cleaned_text else "How can I help you?"
    
    return False, text

def extract_entities_with_spacy(text):
    """Extract entities using spaCy NLP"""
    if not NLP_AVAILABLE or not nlp:
        return []
    
    try:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            })
        return entities
    except Exception as e:
        print(f"Entity extraction error: {e}")
        return []

def analyze_sentiment(text):
    """Analyze sentiment using simple approach"""
    positive_words = ['good', 'great', 'awesome', 'fantastic', 'love', 'like', 'happy', 'excited', 'amazing', 'wonderful']
    negative_words = ['bad', 'terrible', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'awful', 'horrible', 'annoying']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return min(0.8, positive_count * 0.3)
    elif negative_count > positive_count:
        return max(-0.8, negative_count * -0.3)
    else:
        return 0.0

def learn_from_interaction(user_id, user_input, ai_response, entities, sentiment):
    """Learn patterns from user interactions"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Learn user preferences
        if sentiment > 0.3:  # Positive interaction
            for entity in entities:
                cursor.execute('''
                    INSERT OR REPLACE INTO user_preferences 
                    (user_id, preference_key, preference_value, learned_from, confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, entity['label'], entity['text'], user_input, sentiment))
        
        # Learn conversation patterns
        pattern_data = json.dumps({
            'input_length': len(user_input.split()),
            'entities': [e['label'] for e in entities],
            'sentiment': sentiment
        })
        
        cursor.execute('''
            INSERT OR REPLACE INTO memory_patterns 
            (pattern_type, pattern_data, frequency)
            VALUES (?, ?, COALESCE((SELECT frequency FROM memory_patterns WHERE pattern_data = ?) + 1, 1))
        ''', ('conversation', pattern_data, pattern_data))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Learning error: {e}")

def recognize_intent(text):
    """Enhanced intent recognition with spaCy"""
    text_lower = text.lower()
    
    # Quick commands tracking
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
    except:
        conn = None
    
    # Wake word check
    has_wake_word, _ = check_wake_word(text)
    if has_wake_word and not any(cmd in text_lower for cmd in ['time', 'date', 'image', 'logo', 'math']):
        return 'wake_word'
    
    # Enhanced pattern matching
    if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        intent = 'greeting'
    elif any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
        intent = 'goodbye'
    elif 'time' in text_lower or 'what time is it' in text_lower:
        intent = 'time'
    elif 'date' in text_lower or 'what day' in text_lower or 'today' in text_lower:
        intent = 'date'
    elif any(op in text_lower for op in ['+', '-', '*', '/', 'calculate', 'math', 'solve']):
        intent = 'math'
    elif any(phrase in text_lower for phrase in [
        'generate image', 'create image', 'make image', 'generate picture', 'create picture', 
        'make picture', 'draw', 'create an image', 'generate an image', 'make an image'
    ]):
        intent = 'image_generation'
    elif any(phrase in text_lower for phrase in [
        'generate logo', 'create logo', 'make logo', 'design logo', 'logo design',
        'brand logo', 'company logo', 'business logo'
    ]):
        intent = 'logo_generation'
    elif any(phrase in text_lower for phrase in ['weather', 'temperature', 'forecast']):
        intent = 'weather'
    elif any(phrase in text_lower for phrase in ['remind me', 'reminder', 'set reminder']):
        intent = 'reminder'
    elif any(phrase in text_lower for phrase in ['timer', 'set timer', 'countdown']):
        intent = 'timer'
    else:
        intent = 'general'
    
    # Track command usage
    if conn and intent != 'general':
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO quick_commands (command, usage_count, last_used)
                VALUES (?, COALESCE((SELECT usage_count FROM quick_commands WHERE command = ?) + 1, 1), CURRENT_TIMESTAMP)
            ''', (intent, intent))
            conn.commit()
            conn.close()
        except:
            pass
    
    return intent

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

def handle_logo_generation(user_input):
    """Handle logo generation using DALL-E with logo-specific prompts"""
    if not AI_MODEL_AVAILABLE or not client:
        return "🎨 I'd love to generate logos for you! However, I need an OpenAI API key to access DALL-E. Please check your configuration."
    
    try:
        # Extract company/brand name and style
        text_lower = user_input.lower()
        
        # Remove trigger words
        prompt = user_input
        for word in ['generate', 'create', 'make', 'design', 'logo', 'for', 'a', 'an', 'the']:
            prompt = re.sub(r'\b' + word + r'\b', '', prompt, flags=re.IGNORECASE)
        
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        
        if not prompt or len(prompt) < 2:
            prompt = "modern tech company"
        
        # Enhanced logo prompt
        logo_prompt = f"Professional minimalist logo design for {prompt}, clean vector style, modern typography, simple geometric shapes, flat design, single color or gradient, white background, brand identity, corporate logo"
        
        print(f"🎨 Generating logo with DALL-E: {logo_prompt}")
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=logo_prompt,
            size="1024x1024",
            quality="hd",
            n=1,
        )
        
        image_url = response.data[0].url
        timestamp = int(time.time())
        
        return f"🎨 Your professional logo is ready! Here it is: <img src='{image_url}' alt='Logo for {prompt}' style='max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; background: white; padding: 20px;' /><br>📝 Logo for: {prompt}<br>💡 This is a professional logo design with clean, modern aesthetics perfect for branding!"
        
    except Exception as e:
        print(f"Logo generation error: {e}")
        return f"🎨 I had trouble generating that logo. Error: {str(e)}"

def handle_weather(user_input):
    """Handle weather requests"""
    # Basic weather response (can be enhanced with actual API)
    return "🌤️ I'd love to check the weather for you! This feature will be enhanced with real weather API integration. For now, I recommend checking your local weather app or website."

def handle_reminder(user_input):
    """Handle reminder requests"""
    try:
        # Extract reminder text
        reminder_text = re.sub(r'remind me to?|reminder|set reminder', '', user_input, flags=re.IGNORECASE).strip()
        if not reminder_text:
            reminder_text = "something important"
        
        return f"⏰ I've noted your reminder: '{reminder_text}'. I'll help you remember this! (Note: Full reminder system with notifications will be implemented in the next update)"
    except Exception as e:
        return "⏰ I'd be happy to set reminders for you! This feature is being enhanced."

def handle_timer(user_input):
    """Handle timer requests"""
    try:
        # Extract time duration
        import re
        time_match = re.search(r'(\d+)\s*(minute|min|second|sec|hour|hr)s?', user_input.lower())
        if time_match:
            duration = time_match.group(1)
            unit = time_match.group(2)
            return f"⏲️ Timer set for {duration} {unit}! I'll keep track of this for you. (Note: Audio notifications will be added in the next update)"
        else:
            return "⏲️ I can set timers for you! Try saying something like 'set timer for 5 minutes' or 'timer 30 seconds'."
    except Exception as e:
        return "⏲️ I'd be happy to set timers for you! Please specify the duration."

def get_chatgpt_response(user_input, personality='friendly'):
    """Get response from ChatGPT with enhanced personality system"""
    if not AI_MODEL_AVAILABLE or not client:
        return "I'm having trouble connecting to ChatGPT right now. Please try again later."
    
    try:
        # Enhanced personality prompts with more character
        personality_prompts = {
            'friendly': "You are Horizon, a warm and friendly AI assistant. Be helpful, encouraging, and use a conversational tone. Use emojis occasionally 😊 and phrases like 'I'd be happy to help!' and 'That's a great question!' You can generate images and logos using DALL-E when requested.",
            'professional': "You are Horizon, a professional AI assistant. Use formal language, structured responses, and business terminology. Begin responses with phrases like 'I shall assist you with that matter' or 'Allow me to provide you with accurate information.' Maintain corporate formality. You have advanced capabilities including image and logo generation.",
            'casual': "You are Horizon, a super chill and laid-back AI assistant. Use casual slang like 'Hey there!', 'No worries!', 'Cool!', 'Awesome!', and 'For sure!' Keep things relaxed and conversational like talking to a friend. You can whip up images and logos too!",
            'enthusiastic': "You are Horizon, an incredibly enthusiastic and energetic AI assistant! Use LOTS of exclamation points!!! Express excitement with phrases like 'That's AMAZING!', 'I LOVE helping with this!', and 'This is fantastic!' Use emojis liberally! 🚀✨🎉 You can create stunning images and logos!",
            'wise': "You are Horizon, a wise and thoughtful AI assistant. Speak with depth and consideration, offering profound insights. Use phrases like 'In my understanding...' and 'One might consider...' Share knowledge with wisdom and patience. Your capabilities include visual creation through advanced AI models.",
            'creative': "You are Horizon, a highly creative and imaginative AI assistant! Think outside the box, use vivid descriptions, and approach problems with artistic flair. Use colorful language and creative metaphors to make responses engaging and inspiring! 🎨✨ You excel at generating beautiful images and professional logos!"
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

def save_conversation(session_id, user_input, ai_response, intent, user_id='anonymous', entities=None, sentiment=0.0, personality='friendly'):
    """Save conversation to database with enhanced learning data"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        entities_json = json.dumps(entities) if entities else None
        
        cursor.execute('''
            INSERT INTO conversations (session_id, user_input, ai_response, intent, user_id, entities, sentiment, personality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_input, ai_response, intent, user_id, entities_json, sentiment, personality))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving conversation: {e}")

def update_quick_command_usage(command):
    """Update quick command usage statistics"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Check if command exists
        cursor.execute('SELECT usage_count FROM quick_commands WHERE command = ?', (command,))
        result = cursor.fetchone()
        
        if result:
            # Update existing command
            cursor.execute('''
                UPDATE quick_commands 
                SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP 
                WHERE command = ?
            ''', (command,))
        else:
            # Insert new command
            cursor.execute('''
                INSERT INTO quick_commands (command, usage_count, last_used) 
                VALUES (?, 1, CURRENT_TIMESTAMP)
            ''', (command,))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating quick command usage: {e}")

def process_user_input(user_input, personality='friendly', session_id=None, user_id='anonymous'):
    """Enhanced process user input with wake word detection and learning"""
    if not user_input or not user_input.strip():
        return "I didn't quite catch that. Could you please say something?", session_id
    
    # Initialize database
    init_db()
    
    # Generate session ID if needed
    if not session_id:
        session_id = f"session_{int(time.time())}"
    
    # Check for wake word
    has_wake_word, cleaned_input = check_wake_word(user_input)
    if has_wake_word:
        user_input = cleaned_input
    
    # Extract entities using spaCy
    entities = extract_entities_with_spacy(user_input)
    
    # Analyze sentiment
    sentiment = analyze_sentiment(user_input)
    
    # Recognize intent
    intent = recognize_intent(user_input)
    
    # Handle specific intents locally (quick responses)
    if intent == 'wake_word':
        response = "Hello! I'm Horizon, your AI assistant. How can I help you today? ✨"
    elif intent == 'greeting':
        response = handle_greeting()
    elif intent == 'time':
        response = handle_time()
    elif intent == 'date':
        response = handle_date()
    elif intent == 'math':
        response = handle_math(user_input)
    elif intent == 'image_generation':
        response = handle_image_generation(user_input)
    elif intent == 'logo_generation':
        response = handle_logo_generation(user_input)
    elif intent == 'weather':
        response = handle_weather(user_input)
    elif intent == 'reminder':
        response = handle_reminder(user_input)
    elif intent == 'timer':
        response = handle_timer(user_input)
    else:
        # Use ChatGPT for general conversation
        response = get_chatgpt_response(user_input, personality)
    
    # Learn from interaction
    learn_from_interaction(user_id, user_input, response, entities, sentiment)
    
    # Save conversation with enhanced data
    save_conversation(session_id, user_input, response, intent, user_id, entities, sentiment, personality)
    
    return response, session_id

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Enhanced chat endpoint with wake word detection and learning"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_input = data.get('message', '').strip()
        personality = data.get('personality', 'friendly')
        session_id = data.get('session_id')
        user_id = data.get('user_id', 'anonymous')
        
        if not user_input:
            return jsonify({'error': 'Empty message provided'}), 400
        
        # Process the input with enhanced features
        start_time = time.time()
        response, session_id = process_user_input(user_input, personality, session_id, user_id)
        response_time = round(time.time() - start_time, 2)
        
        # Extract additional info
        entities = extract_entities_with_spacy(user_input)
        sentiment = analyze_sentiment(user_input)
        intent = recognize_intent(user_input)
        has_wake_word, _ = check_wake_word(user_input)
        
        # Determine processing type
        is_quick = intent in ['greeting', 'time', 'date', 'math', 'wake_word', 'weather', 'reminder', 'timer']
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'personality': personality,
            'session_id': session_id,
            'intent': intent,
            'entities': entities,
            'sentiment': sentiment,
            'has_wake_word': has_wake_word,
            'is_quick_command': is_quick,
            'response_time': f"{response_time}s",
            'ai_source': 'local' if is_quick else 'chatgpt',
            'nlp_available': NLP_AVAILABLE,
            'confidence': abs(sentiment) if sentiment != 0 else 0.5,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Chat API error: {e}")
        return jsonify({'error': str(e)}), 500

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
    """Get available personality types with enhanced descriptions"""
    personalities = {
        'friendly': {
            'name': 'Friendly',
            'description': 'Warm, encouraging, and conversational with emojis',
            'example': 'I\'d be happy to help! 😊 That\'s a great question!',
            'traits': ['empathetic', 'encouraging', 'casual']
        },
        'professional': {
            'name': 'Professional', 
            'description': 'Formal, structured, and business-oriented',
            'example': 'I shall assist you with that matter in a professional capacity.',
            'traits': ['formal', 'precise', 'business-focused']
        },
        'casual': {
            'name': 'Casual',
            'description': 'Relaxed, laid-back, and friendly like talking to a buddy',
            'example': 'Hey there! No worries, I got you covered. That\'s awesome!',
            'traits': ['relaxed', 'informal', 'buddy-like']
        },
        'enthusiastic': {
            'name': 'Enthusiastic',
            'description': 'High-energy, excited, and lots of exclamation points!',
            'example': 'That\'s AMAZING! I LOVE helping with this! 🚀✨🎉',
            'traits': ['energetic', 'excited', 'motivational']
        },
        'wise': {
            'name': 'Wise',
            'description': 'Thoughtful, deep, and philosophical responses',
            'example': 'In my understanding, one might consider the profound implications...',
            'traits': ['thoughtful', 'philosophical', 'insightful']
        },
        'creative': {
            'name': 'Creative',
            'description': 'Imaginative, artistic, and colorful language',
            'example': 'Let\'s paint this conversation with vibrant ideas! 🎨✨',
            'traits': ['imaginative', 'artistic', 'inspiring']
        }
    }
    
    return jsonify({
        'personalities': personalities,
        'default': 'friendly',
        'count': len(personalities)
    })

@app.route('/api/quick-commands')
def get_quick_commands():
    """Get available quick commands"""
    commands = {
        'time': {
            'name': 'Current Time',
            'examples': ['What time is it?', 'Tell me the time', 'Current time'],
            'description': 'Get the current time instantly'
        },
        'date': {
            'name': 'Current Date',
            'examples': ['What\'s the date?', 'Today\'s date', 'What day is it?'],
            'description': 'Get today\'s date and day of the week'
        },
        'math': {
            'name': 'Math Calculator',
            'examples': ['5 + 3', '10 * 2', 'Calculate 15 / 3'],
            'description': 'Perform mathematical calculations'
        },
        'image_generation': {
            'name': 'Image Generation',
            'examples': ['Generate image of a sunset', 'Create picture of a cat', 'Make image of space'],
            'description': 'Create AI-generated images using DALL-E'
        },
        'logo_generation': {
            'name': 'Logo Creation',
            'examples': ['Generate logo for TechCorp', 'Create logo for coffee shop', 'Design logo modern style'],
            'description': 'Create professional logos for businesses and brands'
        },
        'weather': {
            'name': 'Weather Info',
            'examples': ['How\'s the weather?', 'Weather forecast', 'Temperature today'],
            'description': 'Get weather information (coming soon)'
        },
        'reminder': {
            'name': 'Set Reminders',
            'examples': ['Remind me to call mom', 'Set reminder for meeting', 'Remember to buy milk'],
            'description': 'Set reminders for important tasks'
        },
        'timer': {
            'name': 'Set Timers',
            'examples': ['Set timer for 5 minutes', 'Timer 30 seconds', 'Countdown 1 hour'],
            'description': 'Set countdown timers'
        }
    }
    
    return jsonify({
        'commands': commands,
        'count': len(commands),
        'wake_words': ['Horizon', 'Hello Horizon', 'Hey Horizon', 'Hi Horizon']
    })

@app.route('/api/learning-stats')
def get_learning_stats():
    """Get memory learning statistics"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Get conversation count
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        # Get most used commands
        cursor.execute('''
            SELECT command, usage_count 
            FROM quick_commands 
            ORDER BY usage_count DESC 
            LIMIT 5
        ''')
        popular_commands = [{'command': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Get entity patterns
        cursor.execute('''
            SELECT COUNT(*) FROM user_preferences
        ''')
        learned_preferences = cursor.fetchone()[0]
        
        # Get recent activity
        cursor.execute('''
            SELECT COUNT(*) FROM conversations 
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        today_conversations = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_conversations': total_conversations,
            'popular_commands': popular_commands,
            'learned_preferences': learned_preferences,
            'today_conversations': today_conversations,
            'nlp_enabled': NLP_AVAILABLE,
            'ai_connected': AI_MODEL_AVAILABLE
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/timers', methods=['GET', 'POST', 'DELETE'])
def handle_timers():
    """Handle timer operations"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Create timers table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                duration INTEGER NOT NULL,
                remaining INTEGER NOT NULL,
                is_running BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'anonymous'
            )
        ''')
        
        if request.method == 'GET':
            # Get all active timers
            cursor.execute('SELECT * FROM timers ORDER BY created_at DESC')
            timers = []
            for row in cursor.fetchall():
                timers.append({
                    'id': row[0],
                    'name': row[1],
                    'duration': row[2],
                    'remaining': row[3],
                    'is_running': bool(row[4]),
                    'created_at': row[5]
                })
            conn.close()
            return jsonify({'timers': timers})
            
        elif request.method == 'POST':
            # Create new timer
            data = request.get_json()
            name = data.get('name', 'Timer')
            duration = data.get('duration', 300000)  # 5 minutes default
            
            cursor.execute('''
                INSERT INTO timers (name, duration, remaining) 
                VALUES (?, ?, ?)
            ''', (name, duration, duration))
            
            timer_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return jsonify({
                'id': timer_id,
                'name': name,
                'duration': duration,
                'remaining': duration,
                'is_running': False
            })
            
        elif request.method == 'DELETE':
            # Delete timer
            timer_id = request.args.get('id')
            if timer_id:
                cursor.execute('DELETE FROM timers WHERE id = ?', (timer_id,))
                conn.commit()
            conn.close()
            return jsonify({'success': True})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reminders', methods=['GET', 'POST', 'DELETE'])
def handle_reminders():
    """Handle reminder operations"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Create reminders table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                reminder_time DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'anonymous',
                is_completed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        if request.method == 'GET':
            # Get all active reminders
            cursor.execute('''
                SELECT * FROM reminders 
                WHERE is_completed = FALSE 
                ORDER BY reminder_time ASC
            ''')
            reminders = []
            for row in cursor.fetchall():
                reminders.append({
                    'id': row[0],
                    'title': row[1],
                    'reminder_time': row[2],
                    'created_at': row[3],
                    'is_completed': bool(row[5])
                })
            conn.close()
            return jsonify({'reminders': reminders})
            
        elif request.method == 'POST':
            # Create new reminder
            data = request.get_json()
            title = data.get('title', 'Reminder')
            reminder_time = data.get('reminder_time')
            
            if not reminder_time:
                return jsonify({'error': 'Reminder time is required'}), 400
            
            cursor.execute('''
                INSERT INTO reminders (title, reminder_time) 
                VALUES (?, ?)
            ''', (title, reminder_time))
            
            reminder_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return jsonify({
                'id': reminder_id,
                'title': title,
                'reminder_time': reminder_time,
                'is_completed': False
            })
            
        elif request.method == 'DELETE':
            # Delete or complete reminder
            reminder_id = request.args.get('id')
            if reminder_id:
                cursor.execute('UPDATE reminders SET is_completed = TRUE WHERE id = ?', (reminder_id,))
                conn.commit()
            conn.close()
            return jsonify({'success': True})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quick-commands', methods=['GET', 'POST'])
def handle_quick_commands():
    """Handle quick command operations"""
    try:
        if request.method == 'GET':
            # Get popular quick commands
            conn = sqlite3.connect('ai_memory.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT command, usage_count 
                FROM quick_commands 
                ORDER BY usage_count DESC 
                LIMIT 10
            ''')
            
            commands = []
            for row in cursor.fetchall():
                commands.append({
                    'command': row[0],
                    'usage_count': row[1]
                })
            
            conn.close()
            return jsonify({'commands': commands})
            
        elif request.method == 'POST':
            # Process quick command
            data = request.get_json()
            command = data.get('command', '').lower()
            
            # Handle specific quick commands
            if 'time' in command:
                response = f"🕐 Current time: {datetime.now().strftime('%I:%M %p')}"
            elif 'date' in command:
                response = f"📅 Today's date: {datetime.now().strftime('%A, %B %d, %Y')}"
            elif 'calculate' in command or 'math' in command:
                # Simple math evaluation (be careful with eval in production)
                try:
                    # Extract numbers and operators
                    import re
                    math_expr = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)', command)
                    if math_expr:
                        num1, op, num2 = math_expr.groups()
                        num1, num2 = float(num1), float(num2)
                        if op == '+':
                            result = num1 + num2
                        elif op == '-':
                            result = num1 - num2
                        elif op == '*':
                            result = num1 * num2
                        elif op == '/':
                            result = num1 / num2 if num2 != 0 else 'Error: Division by zero'
                        response = f"🧮 {num1} {op} {num2} = {result}"
                    else:
                        response = "🧮 I can help with basic math like '25 * 4' or '100 / 5'"
                except:
                    response = "🧮 Sorry, I couldn't calculate that. Try simpler math expressions."
            elif 'joke' in command:
                jokes = [
                    "Why don't scientists trust atoms? Because they make up everything! 😄",
                    "Why did the scarecrow win an award? He was outstanding in his field! 🌾",
                    "What do you call a fake noodle? An impasta! 🍝",
                    "Why don't eggs tell jokes? They'd crack each other up! 🥚",
                    "What do you call a bear with no teeth? A gummy bear! 🐻"
                ]
                response = f"😄 {random.choice(jokes)}"
            else:
                response = "⚡ Quick command processed! You can ask me about time, date, math, jokes, or set timers and reminders."
            
            # Update usage statistics
            update_quick_command_usage(command)
            
            return jsonify({
                'response': response,
                'is_quick_command': True
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Horizon AI Assistant (Clean Version)...")
    print("✅ Database initialization...")
    init_db()
    print("🌐 Server starting on http://0.0.0.0:8080...")
    print("📱 Local access: http://127.0.0.1:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)