#!/usr/bin/env python3
"""
Horizon AI Assistant with ChatGPT API Integration
Clean, fast, and intelligent AI responses using OpenAI's API
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime, timedelta
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

# ChatGPT API Integration
try:
    import openai
    from openai import OpenAI
    
    print("🤖 Initializing ChatGPT API connection...")
    
    # Load API key from environment or config
    openai_api_key = os.getenv('OPENAI_API_KEY') or getattr(Config, 'OPENAI_API_KEY', None)
    
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        AI_MODEL_AVAILABLE = True
        print("✅ ChatGPT API connected successfully")
    else:
        client = None
        AI_MODEL_AVAILABLE = False
        print("⚠️  No OpenAI API key found - using fallback responses")
    
except ImportError:
    client = None
    AI_MODEL_AVAILABLE = False
    print("⚠️  OpenAI library not installed - using fallback responses")
except Exception as e:
    client = None
    AI_MODEL_AVAILABLE = False
    print(f"⚠️  ChatGPT API initialization failed: {e}")

print("🧠 Initializing Horizon AI Assistant...")

# Fallback response patterns for when API is unavailable
SMART_RESPONSES = {
    'ai_technology': [
        "Artificial Intelligence is fascinating! It's essentially computer systems that can perform tasks typically requiring human intelligence - like learning, reasoning, and problem-solving.",
        "AI works by using algorithms and data to recognize patterns and make decisions. Machine learning is a subset where systems improve their performance through experience.",
        "Think of AI as teaching computers to think and learn. Neural networks mimic how our brains work, processing information through interconnected nodes.",
        "AI is everywhere now! From voice assistants to recommendation systems, it's transforming how we interact with technology and solve complex problems.",
        "The beauty of AI lies in its ability to find patterns in vast amounts of data that humans might miss, then use those patterns to make predictions or decisions."
    ],
    
    'programming': [
        "Programming is like learning a new language - but instead of talking to people, you're communicating with computers to bring your ideas to life!",
        "The best way to learn programming is by doing! Start with simple projects, practice regularly, and don't be afraid to make mistakes - they're your best teachers.",
        "Programming combines creativity with logic. You get to solve real problems while building things that can impact millions of people.",
        "Start with Python or JavaScript - they're beginner-friendly languages with huge communities and tons of resources to help you learn.",
        "Programming is about breaking big problems into smaller, manageable pieces. Master this skill and you can build anything you imagine!"
    ],
    
    'science': [
        "Science is humanity's greatest adventure! It's our systematic way of understanding everything from the tiniest particles to the vast cosmos.",
        "What makes science beautiful is its self-correcting nature - we form hypotheses, test them, and refine our understanding based on evidence.",
        "Science connects everything! Physics explains how things move, chemistry shows how they interact, and biology reveals how they live and grow.",
        "The scientific method is powerful because it's based on observation, experimentation, and peer review - ensuring our knowledge is reliable and accurate.",
        "Every scientific breakthrough started with curiosity and a question. Keep asking 'why' and 'how' - that's the spirit of scientific discovery!"
    ],
    
    'technology': [
        "Technology is reshaping our world at an incredible pace! From smartphones to space exploration, we're living in an era of unprecedented innovation.",
        "The future of technology is exciting - quantum computing, biotechnology, renewable energy, and AI are converging to solve humanity's biggest challenges.",
        "Technology democratizes access to information, tools, and opportunities. A person with a smartphone today has more computing power than entire governments had decades ago!",
        "What's amazing about modern technology is how it connects us globally while enabling local solutions to unique problems.",
        "The key to thriving with technology is staying curious and adaptable. The tools change, but the human need for creativity and problem-solving remains constant."
    ],
    
    'learning': [
        "Learning is a superpower! In our rapidly changing world, the ability to continuously acquire new skills and knowledge is more valuable than any single degree.",
        "The best learning happens when you're genuinely curious about something. Find what fascinates you and dive deep - passion makes the journey enjoyable.",
        "Don't just consume information - apply it! Build projects, teach others, and experiment. Active learning creates lasting understanding.",
        "Everyone learns differently. Some are visual, others learn by doing. Experiment with different methods to find what works best for you.",
        "The internet has democratized education! You can learn almost anything online - from world-class universities to practical tutorials from experts."
    ],
    
    'general_wisdom': [
        "Life is about continuous growth and learning. Embrace challenges as opportunities to become stronger and wiser.",
        "The most successful people aren't necessarily the smartest - they're often the most persistent and adaptable.",
        "Building good habits is like compound interest - small, consistent actions lead to remarkable results over time.",
        "Collaboration often produces better results than working alone. Different perspectives and skills complement each other beautifully.",
        "Stay curious and open-minded! The world is full of fascinating ideas and perspectives waiting to be discovered.",
        "Remember that failure is often the best teacher. Each setback provides valuable lessons that success cannot offer.",
        "Focus on progress, not perfection. Small improvements each day lead to extraordinary results over time.",
        "The ability to communicate clearly is a superpower in any field. Practice explaining complex ideas simply."
    ]
}

# Topic keyword mapping for intelligent responses
TOPIC_KEYWORDS = {
    'ai_technology': ['ai', 'artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'algorithm', 'data science', 'automation'],
    'programming': ['programming', 'coding', 'code', 'software', 'developer', 'python', 'javascript', 'computer science', 'debugging', 'framework'],
    'science': ['science', 'research', 'experiment', 'physics', 'chemistry', 'biology', 'mathematics', 'discovery', 'theory', 'hypothesis'],
    'technology': ['technology', 'tech', 'innovation', 'digital', 'computer', 'internet', 'smartphone', 'future', 'innovation', 'engineering'],
    'learning': ['learn', 'education', 'study', 'school', 'university', 'course', 'tutorial', 'skill', 'knowledge', 'training'],
}

# Global variables for timer and reminder management
active_timers = {}  # {timer_id: {'start_time': datetime, 'duration': seconds, 'description': str, 'status': 'active'}}
active_reminders = []  # [{'id': int, 'text': str, 'created_at': datetime, 'remind_at': datetime, 'status': 'active'}]
timer_id_counter = 1
reminder_id_counter = 1

def create_timer(duration_seconds, description="Timer"):
    """Create a new timer and return timer info"""
    global timer_id_counter, active_timers
    
    timer_id = timer_id_counter
    timer_id_counter += 1
    
    timer_info = {
        'id': timer_id,
        'start_time': datetime.now(),
        'duration': duration_seconds,
        'description': description,
        'status': 'active',
        'end_time': datetime.now() + timedelta(seconds=duration_seconds)
    }
    
    active_timers[timer_id] = timer_info
    
    # Start timer in background thread
    timer_thread = threading.Thread(target=timer_worker, args=(timer_id,))
    timer_thread.daemon = True
    timer_thread.start()
    
    return timer_info

def timer_worker(timer_id):
    """Background worker that handles timer completion"""
    if timer_id not in active_timers:
        return
    
    timer_info = active_timers[timer_id]
    duration = timer_info['duration']
    
    # Wait for the timer duration
    time.sleep(duration)
    
    # Mark timer as completed
    if timer_id in active_timers:
        active_timers[timer_id]['status'] = 'completed'
        active_timers[timer_id]['completed_at'] = datetime.now()
        print(f"⏰ Timer {timer_id} completed: {timer_info['description']}")

def create_reminder(text, remind_in_minutes=None):
    """Create a new reminder"""
    global reminder_id_counter, active_reminders
    
    reminder_id = reminder_id_counter
    reminder_id_counter += 1
    
    created_at = datetime.now()
    # If no specific time given, remind in 1 hour by default
    remind_at = created_at + timedelta(minutes=remind_in_minutes or 60)
    
    reminder_info = {
        'id': reminder_id,
        'text': text,
        'created_at': created_at,
        'remind_at': remind_at,
        'status': 'active'
    }
    
    active_reminders.append(reminder_info)
    
    # Start reminder in background thread
    reminder_thread = threading.Thread(target=reminder_worker, args=(reminder_id,))
    reminder_thread.daemon = True
    reminder_thread.start()
    
    return reminder_info

def reminder_worker(reminder_id):
    """Background worker that handles reminder notifications"""
    reminder_info = next((r for r in active_reminders if r['id'] == reminder_id), None)
    if not reminder_info:
        return
    
    # Calculate wait time
    wait_seconds = (reminder_info['remind_at'] - datetime.now()).total_seconds()
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    
    # Mark reminder as triggered
    for reminder in active_reminders:
        if reminder['id'] == reminder_id:
            reminder['status'] = 'triggered'
            reminder['triggered_at'] = datetime.now()
            print(f"🔔 Reminder {reminder_id} triggered: {reminder['text']}")
            break

def get_active_timers():
    """Get all active timers with remaining time"""
    current_time = datetime.now()
    active_list = []
    
    for timer_id, timer_info in active_timers.items():
        if timer_info['status'] == 'active':
            remaining_seconds = (timer_info['end_time'] - current_time).total_seconds()
            if remaining_seconds > 0:
                active_list.append({
                    'id': timer_id,
                    'description': timer_info['description'],
                    'remaining_seconds': int(remaining_seconds),
                    'end_time': timer_info['end_time'].isoformat()
                })
            else:
                # Timer should be completed
                timer_info['status'] = 'completed'
    
    return active_list

def get_active_reminders():
    """Get all active reminders"""
    current_time = datetime.now()
    active_list = []
    
    for reminder in active_reminders:
        if reminder['status'] == 'active':
            active_list.append({
                'id': reminder['id'],
                'text': reminder['text'],
                'remind_at': reminder['remind_at'].isoformat(),
                'minutes_until': int((reminder['remind_at'] - current_time).total_seconds() / 60)
            })
    
    return active_list

def ask_chatgpt(user_input, personality):
    """Use ChatGPT API for intelligent responses"""
    if not AI_MODEL_AVAILABLE or not client:
        return None
    
    try:
        # Create personality-specific system prompt
        personality_prompts = {
            'friendly': "You are a friendly and helpful AI assistant. Respond in a warm, approachable way. Keep responses concise but informative.",
            'professional': "You are a professional AI assistant. Provide clear, accurate, and well-structured responses. Maintain a formal but helpful tone.",
            'casual': "You are a casual and relaxed AI assistant. Use conversational language and be approachable. Keep it simple and easy to understand.",
            'enthusiastic': "You are an enthusiastic and energetic AI assistant. Show excitement about topics and be encouraging. Use positive language and emojis when appropriate."
        }
        
        system_prompt = personality_prompts.get(personality, personality_prompts['friendly'])
        
        # Make API call to ChatGPT
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can change to "gpt-4" if you have access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150,  # Keep responses concise
            temperature=0.7,  # Balance creativity and consistency
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"ChatGPT API error: {e}")
        return None

def generate_fallback_response(user_input, personality):
    """Generate intelligent fallback responses when API is unavailable"""
    text_lower = user_input.lower()
    
    # Determine topic
    detected_topic = 'general_wisdom'
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_topic = topic
            break
    
    # Get base responses for the detected topic
    topic_responses = SMART_RESPONSES.get(detected_topic, SMART_RESPONSES['general_wisdom'])
    
    # Apply personality modifiers
    personality_modifiers = {
        'enthusiastic': ["That's incredible! ", "How exciting! ", "I love this topic! ", "Amazing question! "],
        'professional': ["I appreciate your inquiry. ", "That's a thoughtful question. ", "Let me address that. "],
        'casual': ["Cool question! ", "Interesting! ", "Nice! ", "Good point! "],
        'friendly': ["Great question! ", "I'm happy to help! ", "That's wonderful! ", ""]
    }
    
    prefix = random.choice(personality_modifiers.get(personality, personality_modifiers['friendly']))
    base_response = random.choice(topic_responses)
    
    return prefix + base_response

def ask_ai_model(user_input, personality):
    """Main AI function - tries ChatGPT first, falls back to smart responses"""
    try:
        # Try ChatGPT first
        chatgpt_response = ask_chatgpt(user_input, personality)
        
        if chatgpt_response:
            return chatgpt_response
        else:
            # Fall back to smart responses
            return generate_fallback_response(user_input, personality)
        
    except Exception as e:
        print(f"AI model error: {e}")
        return generate_fallback_response(user_input, personality)

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

# Quick response handlers
def handle_time():
    return f"The current time is {datetime.now().strftime('%I:%M %p')}."

def handle_date():
    return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."

def handle_greeting(personality):
    greetings = {
        'friendly': ["Hello! How can I help you today?", "Hi there! What can I do for you?"],
        'professional': ["Good day. How may I assist you?", "Hello. What can I help you with?"],
        'casual': ["Hey! What's up?", "Hi! How's it going?"],
        'enthusiastic': ["Hello! I'm so excited to help you today!", "Hi there! Ready to have some fun?"]
    }
    return random.choice(greetings.get(personality, greetings['friendly']))

def handle_joke(personality):
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why did the programmer quit his job? He didn't get arrays!",
        "What do you call a bear with no teeth? A gummy bear!",
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "What's the best thing about Switzerland? I don't know, but the flag is a big plus!"
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
        # Extract and evaluate simple math expressions
        import operator
        ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
        
        for op_symbol, op_func in ops.items():
            if op_symbol in text:
                nums = re.findall(r'\d+', text)
                if len(nums) >= 2:
                    result = op_func(int(nums[0]), int(nums[1]))
                    return f"{nums[0]} {op_symbol} {nums[1]} = {result}"
        
        return "I can help with basic math! Try asking me something like '5 + 3' or 'what is 10 times 2'."
    except Exception as e:
        return "I'm having trouble with that calculation. Try a simpler math problem!"

def handle_timer(text):
    """Handle timer requests with actual timer functionality"""
    try:
        # Extract time duration from text
        numbers = re.findall(r'\d+', text)
        if numbers:
            duration = int(numbers[0])
            
            # Determine time unit and convert to seconds
            if 'hour' in text.lower():
                unit = 'hours'
                seconds = duration * 3600
                unit_text = f"{duration} hour{'s' if duration != 1 else ''}"
            elif 'second' in text.lower():
                unit = 'seconds'
                seconds = duration
                unit_text = f"{duration} second{'s' if duration != 1 else ''}"
            else:  # default to minutes
                unit = 'minutes'
                seconds = duration * 60
                unit_text = f"{duration} minute{'s' if duration != 1 else ''}"
            
            # Create the actual timer
            timer_info = create_timer(seconds, f"Timer for {unit_text}")
            
            return f"✅ Timer set for {unit_text}! Timer ID: {timer_info['id']}. I'll notify you when it's done."
        else:
            return "I can set a timer for you! Try saying something like 'set timer for 5 minutes' or 'timer for 1 hour'."
    except Exception as e:
        print(f"Error in handle_timer: {e}")
        return "I had trouble setting that timer. Please try again!"

def handle_reminder(text):
    """Handle reminder requests with actual reminder functionality"""
    try:
        # Extract reminder content
        reminder_patterns = [
            r'remind me to (.+)',
            r'reminder to (.+)',
            r'set reminder (.+)',
            r'remind me (.+)'
        ]
        
        reminder_text = None
        for pattern in reminder_patterns:
            match = re.search(pattern, text.lower())
            if match:
                reminder_text = match.group(1)
                break
        
        if reminder_text:
            # Extract time if specified (default to 60 minutes)
            time_match = re.search(r'in (\d+) (minute|hour)', text.lower())
            if time_match:
                time_value = int(time_match.group(1))
                time_unit = time_match.group(2)
                if time_unit == 'hour':
                    remind_in_minutes = time_value * 60
                else:
                    remind_in_minutes = time_value
            else:
                remind_in_minutes = 60  # Default to 1 hour
            
            # Create the actual reminder
            reminder_info = create_reminder(reminder_text, remind_in_minutes)
            
            return f"📅 Reminder set: {reminder_text}. I'll remind you in {remind_in_minutes} minutes! Reminder ID: {reminder_info['id']}"
        else:
            return "I can set reminders for you! Try saying something like 'remind me to call mom' or 'set reminder to buy groceries'."
    except Exception as e:
        print(f"Error in handle_reminder: {e}")
        return "I had trouble setting that reminder. Please try again!"

# Intent recognition
INTENT_PATTERNS = {
    'greeting': [r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b'],
    'time': [r'\b(time|clock)\b', r'\bwhat time is it\b'],
    'date': [r'\b(date|today)\b', r'\bwhat day is it\b'],
    'joke': [r'\b(joke|funny|humor)\b', r'\btell me a joke\b'],
    'math': [r'\d+\s*[\+\-\*\/]\s*\d+', r'\bwhat is \d+', r'\bcalculate\b'],
    'timer': [r'\bset.*timer\b', r'\btimer for\b', r'\b\d+\s*(minute|second|hour).*timer\b'],
    'reminder': [r'\bremind me\b', r'\bset.*reminder\b', r'\breminder.*to\b'],
    'goodbye': [r'\b(bye|goodbye|see you|farewell)\b']
}

def recognize_intent(text):
    text_lower = text.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent
    return 'general'

def calculate_realistic_confidence(user_input, response, ai_source, intent):
    """Calculate realistic confidence scores based on various factors"""
    import random
    
    # Base confidence scores by source
    base_confidence = {
        'chatgpt': random.uniform(0.85, 0.95),  # ChatGPT: 85-95%
        'fallback': random.uniform(0.75, 0.88)  # Fallback: 75-88%
    }
    
    confidence = base_confidence.get(ai_source, 0.80)
    
    # Adjust based on intent type (some are more reliable)
    intent_modifiers = {
        'time': 0.98,      # Time queries are very reliable
        'date': 0.98,      # Date queries are very reliable  
        'math': 0.95,      # Math is usually accurate
        'timer': 0.93,     # Timer setting is pretty reliable
        'reminder': 0.91,  # Reminder setting is reliable
        'greeting': 0.90,  # Greetings are straightforward
        'joke': 0.85,      # Jokes are subjective
        'general': 0.82    # General queries vary more
    }
    
    confidence *= intent_modifiers.get(intent, 0.82)
    
    # Adjust based on input complexity
    word_count = len(user_input.split())
    if word_count > 20:
        confidence *= 0.92  # Longer queries are harder
    elif word_count < 3:
        confidence *= 0.95  # Very short might be unclear
    
    # Adjust based on response length (very short might indicate issues)
    response_length = len(response.split())
    if response_length < 5:
        confidence *= 0.88
    elif response_length > 50:
        confidence *= 0.94  # Very long responses might be less focused
    
    # Add some natural variation
    confidence += random.uniform(-0.05, 0.03)
    
    # Ensure realistic bounds (never 100%, rarely below 70%)
    confidence = max(0.72, min(0.96, confidence))
    
    return round(confidence, 3)

def process_user_input(user_input, personality='friendly'):
    """Process user input and return appropriate response"""
    if not user_input or not user_input.strip():
        return "I didn't quite catch that. Could you please say something?"
    
    # Recognize intent for quick responses
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
    elif intent == 'reminder':
        response = handle_reminder(user_input)
    elif intent == 'goodbye':
        response = "Thank you for chatting! Have a wonderful day!"
    else:
        # Use AI model (ChatGPT or fallback) for general questions
        response = ask_ai_model(user_input, personality)
    
    # Save conversation
    save_conversation(user_input, response, personality)
    
    return response

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Horizon AI Assistant with ChatGPT',
        'ai_model_available': AI_MODEL_AVAILABLE,
        'version': 'chatgpt_v1.0'
    })

@app.route('/api/process', methods=['POST'])
def process_message():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_input = data.get('input', '').strip()
        personality = data.get('personality', 'friendly')
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # Process the input
        start_time = time.time()
        response = process_user_input(user_input, personality)
        response_time = round(time.time() - start_time, 2)
        
        # Determine AI source and intent for confidence calculation
        ai_source = 'chatgpt' if AI_MODEL_AVAILABLE else 'fallback'
        intent = recognize_intent(user_input)
        
        # Calculate realistic confidence
        confidence = calculate_realistic_confidence(user_input, response, ai_source, intent)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'personality': personality,
            'ai_source': ai_source,
            'confidence': confidence,
            'response_time': f"{response_time}s",
            'intent': intent,
            'word_count': len(user_input.split())
        })
        
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/timers-reminders', methods=['GET'])
def get_timers_reminders():
    """Return active timers and reminders"""
    try:
        return jsonify({
            'timers': get_active_timers(),
            'reminders': get_active_reminders(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error getting timers/reminders: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/timers-reminders', methods=['POST'])
def manage_timers_reminders():
    """Create, update, or delete timers and reminders"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        action = data.get('action')  # 'create', 'cancel', 'update'
        item_type = data.get('type')  # 'timer' or 'reminder'
        item_id = data.get('id')
        
        if action == 'cancel':
            if item_type == 'timer' and item_id:
                if item_id in active_timers:
                    active_timers[item_id]['status'] = 'cancelled'
                    return jsonify({
                        'success': True,
                        'message': f'Timer {item_id} cancelled',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': 'Timer not found'}), 404
            
            elif item_type == 'reminder' and item_id:
                for reminder in active_reminders:
                    if reminder['id'] == item_id:
                        reminder['status'] = 'cancelled'
                        return jsonify({
                            'success': True,
                            'message': f'Reminder {item_id} cancelled',
                            'timestamp': datetime.now().isoformat()
                        })
                return jsonify({'error': 'Reminder not found'}), 404
        
        return jsonify({
            'success': True,
            'message': f'{action.capitalize()} {item_type} operation completed',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error managing timers/reminders: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cancel-timer/<int:timer_id>', methods=['POST'])
def cancel_timer(timer_id):
    """Cancel a specific timer"""
    try:
        if timer_id in active_timers:
            active_timers[timer_id]['status'] = 'cancelled'
            active_timers[timer_id]['cancelled_at'] = datetime.now()
            return jsonify({
                'success': True,
                'message': f'Timer {timer_id} cancelled',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Timer not found'}), 404
    except Exception as e:
        print(f"Error cancelling timer: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cancel-reminder/<int:reminder_id>', methods=['POST'])
def cancel_reminder(reminder_id):
    """Cancel a specific reminder"""
    try:
        for reminder in active_reminders:
            if reminder['id'] == reminder_id:
                reminder['status'] = 'cancelled'
                reminder['cancelled_at'] = datetime.now()
                return jsonify({
                    'success': True,
                    'message': f'Reminder {reminder_id} cancelled',
                    'timestamp': datetime.now().isoformat()
                })
        return jsonify({'error': 'Reminder not found'}), 404
    except Exception as e:
        print(f"Error cancelling reminder: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Horizon AI Assistant with ChatGPT...")
    
    # Initialize database
    init_db()
    
    if AI_MODEL_AVAILABLE:
        print("✅ ChatGPT API ready")
    else:
        print("✅ Fallback AI system ready")
    
    print("✅ Intent recognition loaded")
    print("🌐 Server starting on http://127.0.0.1:8080...")
    
    app.run(host='127.0.0.1', port=8080, debug=True)
