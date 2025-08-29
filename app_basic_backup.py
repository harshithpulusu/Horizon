from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import sqlite3
import json
import re
import random
import time
import threading
import os
from config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Smart Pattern-Based AI system
try:
    print("🧠 Initializing Smart AI Assistant...")
    
    # Advanced response patterns for intelligent conversation
    SMART_RESPONSES = {
        'questions': {
            'what': [
                "That's a great question! Based on what I know, {}",
                "Let me think about that. {}",
                "From my understanding, {}"
            ],
            'how': [
                "Here's how you can approach that: {}",
                "The best way to do that would be: {}",
                "Let me explain the process: {}"
            ],
            'why': [
                "The reason for that is: {}",
                "That happens because: {}",
                "The explanation is: {}"
            ],
            'when': [
                "The timing for that would be: {}",
                "That typically happens: {}",
                "You should expect: {}"
            ],
            'where': [
                "You can find that: {}",
                "The location would be: {}",
                "That's typically located: {}"
            ]
        },
        'topics': {
            'weather': [
                "Weather is fascinating! I wish I could check current conditions for you.",
                "I'd love to help with weather info, but I don't have access to current data.",
                "Weather patterns are interesting! You might want to check a weather app."
            ],
            'food': [
                "Food is one of life's great pleasures! What kind of cuisine are you thinking about?",
                "I love talking about food! Are you looking for recipe ideas or restaurant suggestions?",
                "Cooking can be so rewarding! What are you in the mood for?"
            ],
            'technology': [
                "Technology is evolving so rapidly these days! What aspect interests you?",
                "I find tech developments fascinating! What would you like to discuss?",
                "Technology shapes our world in amazing ways! Tell me more about your interest."
            ],
            'music': [
                "Music is universal! What genre or artist are you interested in?",
                "I appreciate good music! What kind of sounds move you?",
                "Music has such power to connect us! What's playing in your heart?"
            ],
            'movies': [
                "Cinema is an incredible art form! What kind of movies do you enjoy?",
                "I love discussing films! Any particular genre or recent releases?",
                "Movies can transport us to different worlds! What's caught your attention?"
            ]
        },
        'general': [
            "That's an interesting perspective! Tell me more about what you think.",
            "I appreciate you sharing that with me. What's your take on it?",
            "That sounds important to you. Can you elaborate?",
            "I'm here to listen and help however I can. What else would you like to explore?",
            "Thank you for bringing that up. What aspects are most important to you?",
            "I find that topic fascinating. What drew your interest to it?",
            "That's worth discussing! What's your experience with that?",
            "I'm curious about your thoughts on this. What do you think?",
            "That's a valuable point. How do you see it affecting things?",
            "I appreciate your insight. What would you like to know more about?"
        ]
    }
    
    AI_MODEL_AVAILABLE = True
    print("✅ Smart AI Assistant ready - Pattern-based conversational system loaded")
    
except Exception as e:
    AI_MODEL_AVAILABLE = False
    print(f"❌ Failed to initialize AI system: {e}")
    print("⚠️  Running with basic responses only")

# Global variables
timers = {}
reminders = []

# Database setup
def init_db():
    """Initialize the SQLite database for conversation storage"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_input TEXT,
                ai_response TEXT,
                personality TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

def save_conversation(user_input, ai_response, personality):
    """Save conversation to database"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (timestamp, user_input, ai_response, personality)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), user_input, ai_response, personality))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving conversation: {e}")

# Intent recognition patterns
INTENT_PATTERNS = {
    'greeting': [
        r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
        r'\bhow are you\b',
        r'\bwhat\'s up\b'
    ],
    'time': [
        r'\b(time|clock)\b',
        r'\bwhat time is it\b',
        r'\bcurrent time\b'
    ],
    'date': [
        r'\b(date|today)\b',
        r'\bwhat day is it\b',
        r'\bwhat\'s the date\b'
    ],
    'joke': [
        r'\b(joke|funny|humor)\b',
        r'\btell me a joke\b',
        r'\bmake me laugh\b'
    ],
    'math': [
        r'\bcalculate\b',
        r'\bmath\b',
        r'\bcompute\b',
        r'\d+\s*[\+\-\*\/]\s*\d+',
        r'\bwhat is \d+',
        r'\bsolve\b'
    ],
    'timer': [
        r'\bset timer\b',
        r'\btimer for\b',
        r'\balarm\b',
        r'\bremind me in\b'
    ],
    'goodbye': [
        r'\b(bye|goodbye|see you|farewell)\b',
        r'\btalk to you later\b'
    ]
}

def recognize_intent(text):
    """Recognize user intent from text using pattern matching"""
    text_lower = text.lower()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent
    
    return 'general'

# Skill handlers
def handle_greeting(personality):
    """Handle greeting intents"""
    greetings = {
        'friendly': ["Hello! How can I help you today?", "Hi there! What can I do for you?"],
        'professional': ["Good day. How may I assist you?", "Hello. What can I help you with?"],
        'casual': ["Hey! What's up?", "Hi! How's it going?"],
        'enthusiastic': ["Hello! I'm so excited to help you today!", "Hi there! Ready to have some fun?"]
    }
    return random.choice(greetings.get(personality, greetings['friendly']))

def handle_time():
    """Handle time requests"""
    current_time = datetime.now().strftime("%I:%M %p")
    return f"The current time is {current_time}."

def handle_date():
    """Handle date requests"""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    return f"Today is {current_date}."

def handle_joke(personality):
    """Handle joke requests"""
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why did the math book look so sad? Because it had too many problems!",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "What do you call a bear with no teeth? A gummy bear!",
        "Why did the coffee file a police report? It got mugged!"
    ]
    
    if personality == 'enthusiastic':
        return "Oh, I love jokes! " + random.choice(jokes) + " 😄"
    elif personality == 'professional':
        return "Here's a light-hearted joke: " + random.choice(jokes)
    else:
        return random.choice(jokes)

def handle_math(text):
    """Handle basic math calculations"""
    try:
        # Extract math expressions
        if '+' in text:
            nums = re.findall(r'\d+', text)
            if len(nums) >= 2:
                result = int(nums[0]) + int(nums[1])
                return f"{nums[0]} + {nums[1]} = {result}"
        elif '-' in text:
            nums = re.findall(r'\d+', text)
            if len(nums) >= 2:
                result = int(nums[0]) - int(nums[1])
                return f"{nums[0]} - {nums[1]} = {result}"
        elif '*' in text or 'times' in text:
            nums = re.findall(r'\d+', text)
            if len(nums) >= 2:
                result = int(nums[0]) * int(nums[1])
                return f"{nums[0]} × {nums[1]} = {result}"
        elif '/' in text or 'divided' in text:
            nums = re.findall(r'\d+', text)
            if len(nums) >= 2 and int(nums[1]) != 0:
                result = int(nums[0]) / int(nums[1])
                return f"{nums[0]} ÷ {nums[1]} = {result}"
        
        return "I can help with basic math! Try asking me something like '5 + 3' or 'what is 10 times 2'."
    except Exception as e:
        return "Sorry, I couldn't calculate that. Try a simpler math problem!"

def handle_timer(text):
    """Handle timer requests"""
    try:
        # Extract time from text
        time_match = re.search(r'(\d+)\s*(minute|minutes|min|second|seconds|sec|hour|hours)', text.lower())
        if time_match:
            duration = int(time_match.group(1))
            unit = time_match.group(2)
            
            # Convert to seconds
            if 'hour' in unit:
                seconds = duration * 3600
            elif 'minute' in unit or 'min' in unit:
                seconds = duration * 60
            else:
                seconds = duration
            
            timer_id = f"timer_{int(time.time())}"
            timers[timer_id] = {
                'start_time': time.time(),
                'duration': seconds,
                'description': f"{duration} {unit}"
            }
            
            return f"Timer set for {duration} {unit}! I'll let you know when it's done."
        else:
            return "Please specify a time, like 'set timer for 5 minutes' or 'timer for 30 seconds'."
    except Exception as e:
        return "Sorry, I couldn't set that timer. Try something like 'set timer for 5 minutes'."

def handle_goodbye(personality):
    """Handle goodbye intents"""
    goodbyes = {
        'friendly': ["Goodbye! Have a great day!", "See you later! Take care!"],
        'professional': ["Goodbye. Have a productive day.", "Thank you. Until next time."],
        'casual': ["See ya!", "Catch you later!"],
        'enthusiastic': ["Bye bye! It was so fun talking with you!", "See you soon! Have an amazing day!"]
    }
    return random.choice(goodbyes.get(personality, goodbyes['friendly']))

def ask_ai_model(user_input, personality):
    """Use smart pattern-based AI for general questions and complex interactions"""
    if not AI_MODEL_AVAILABLE:
        return "I'd love to help with that, but my AI system isn't available right now. I can help with time, date, jokes, math, and timers though!"
    
    try:
        user_lower = user_input.lower()
        
        # Determine question type
        question_type = None
        for qtype in ['what', 'how', 'why', 'when', 'where']:
            if user_lower.startswith(qtype):
                question_type = qtype
                break
        
        # Check for topic keywords
        topic_responses = []
        for topic, responses in SMART_RESPONSES['topics'].items():
            if topic in user_lower:
                topic_responses.extend(responses)
        
        # Personality-based response selection
        if personality == 'enthusiastic':
            enthusiasm_phrases = ["That's amazing! ", "How exciting! ", "I love this topic! ", "Fantastic question! "]
            prefix = random.choice(enthusiasm_phrases)
        elif personality == 'professional':
            prefix = "I understand your inquiry. "
        elif personality == 'casual':
            casual_phrases = ["Cool question! ", "Interesting! ", "Good point! ", "Nice! "]
            prefix = random.choice(casual_phrases)
        else:  # friendly
            prefix = ""
        
        # Generate response based on question type and topics
        if topic_responses:
            response = random.choice(topic_responses)
        elif question_type and question_type in SMART_RESPONSES['questions']:
            # Create contextual response for question types
            context_responses = {
                'what': "it depends on the specific context and circumstances",
                'how': "there are several approaches you could take",
                'why': "there are multiple factors that contribute to this",
                'when': "the timing can vary based on different factors",
                'where': "the location or source would depend on what you're looking for"
            }
            template = random.choice(SMART_RESPONSES['questions'][question_type])
            context = context_responses.get(question_type, "that's a complex topic")
            response = template.format(context)
        else:
            # Use general intelligent responses
            response = random.choice(SMART_RESPONSES['general'])
        
        # Add personality prefix
        final_response = prefix + response
        
        # Ensure response isn't too long
        if len(final_response) > 200:
            final_response = final_response[:200] + "..."
            
        return final_response
        
    except Exception as e:
        print(f"AI model error: {e}")
        fallback_responses = [
            "That's an interesting question! Let me think about that.",
            "I understand what you're asking. Can you tell me more?",
            "That's a good point. What would you like to know specifically?",
            "I'm here to help! Could you rephrase that question?"
        ]
        return random.choice(fallback_responses)

def process_user_input(user_input, personality='friendly'):
    """Process user input and return appropriate response"""
    if not user_input or not user_input.strip():
        return "I didn't quite catch that. Could you please say something?"
    
    # Recognize intent
    intent = recognize_intent(user_input)
    
    # Handle specific intents
    if intent == 'greeting':
        response = handle_greeting(personality)
    elif intent == 'time':
        response = handle_time()
    elif intent == 'date':
        response = handle_date()
    elif intent == 'joke':
        response = handle_joke(personality)
    elif intent == 'math':
        response = handle_math(user_input)
    elif intent == 'timer':
        response = handle_timer(user_input)
    elif intent == 'goodbye':
        response = handle_goodbye(personality)
    else:
        # Use AI model for general questions
        response = ask_ai_model(user_input, personality)
    
    # Save conversation
    save_conversation(user_input, response, personality)
    
    return response

# Routes
@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Horizon AI Assistant',
        'ai_model_available': AI_MODEL_AVAILABLE
    })

@app.route('/api/process', methods=['POST'])
def process_message():
    """Process incoming messages"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_input = data.get('input', '').strip()
        personality = data.get('personality', 'friendly')
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # Process the input
        response = process_user_input(user_input, personality)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'personality': personality
        })
        
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/timers')
def get_timers():
    """Get active timers"""
    active_timers = []
    current_time = time.time()
    
    for timer_id, timer_info in timers.items():
        elapsed = current_time - timer_info['start_time']
        remaining = timer_info['duration'] - elapsed
        
        if remaining > 0:
            active_timers.append({
                'id': timer_id,
                'description': timer_info['description'],
                'remaining': int(remaining)
            })
        else:
            # Timer finished, remove it
            timers.pop(timer_id, None)
    
    return jsonify({'active_timers': active_timers})

if __name__ == '__main__':
    print("🚀 Starting Horizon AI Assistant...")
    
    # Initialize database
    init_db()
    
    print("✅ Intent recognition loaded")
    print("🌐 Server starting on http://127.0.0.1:8080...")
    
    app.run(host='127.0.0.1', port=8080, debug=True)
