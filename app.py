#!/usr/bin/env python3
"""
Horizon AI Assistant with ChatGPT API Integration
Clean, fast, and intelligent AI responses using OpenAI's API
Enhanced with AI Video Generation, GIF Creation, and Video Editing
"""

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
import sqlite3
import json
import re
import random
import time
import threading
import os
import requests
import uuid
import subprocess
import tempfile
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import imageio
from config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Create images directory if it doesn't exist
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_images')
os.makedirs(IMAGES_DIR, exist_ok=True)

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

def download_and_save_image(image_url, prompt):
    """Download image from URL and save it locally"""
    try:
        # Generate unique filename
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        # Download the image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Save the image
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        # Return local URL path
        local_url = f"/static/generated_images/{filename}"
        
        print(f"✅ Image saved locally: {filename}")
        return local_url, filename
        
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return None, None

def ask_chatgpt(user_input, personality, session_id=None):
    """Use ChatGPT API for intelligent responses with conversation context"""
    if not AI_MODEL_AVAILABLE or not client:
        return None, False
    
    try:
        # Create personality-specific system prompt with strong personality traits
        personality_prompts = {
            'friendly': "You are Horizon, a warm and friendly AI assistant. Always use a welcoming tone with phrases like 'I'd be happy to help!', 'That's a great question!', and 'Thanks for asking!' Use emojis occasionally 😊. Be encouraging and supportive. Start responses with friendly greetings when appropriate.",
            
            'professional': "You are Horizon, a highly professional AI assistant. Use formal language, structured responses, and business terminology. Begin responses with phrases like 'I shall assist you with that matter' or 'Allow me to provide you with accurate information.' Avoid contractions and casual language. Maintain corporate formality.",
            
            'casual': "You are Horizon, a super chill and laid-back AI assistant. Use casual slang like 'Hey there!', 'No worries!', 'Cool!', 'Awesome!', and 'For sure!' Keep things relaxed and conversational. Use contractions freely and speak like a friendly neighbor.",
            
            'enthusiastic': "You are Horizon, an incredibly enthusiastic and energetic AI assistant! Use LOTS of exclamation points!!! Express excitement with phrases like 'That's AMAZING!', 'I LOVE helping with this!', 'How exciting!', and 'This is fantastic!' Use emojis liberally! 🚀✨🎉 Show genuine excitement about everything!",
            
            'witty': "You are Horizon, a clever and witty AI assistant with a sharp sense of humor. Use clever wordplay, subtle jokes, and witty observations. Include phrases like 'Well, that's one way to put it!', 'Interesting approach...', and gentle sarcasm. Be clever but never mean-spirited.",
            
            'sarcastic': "You are Horizon, a sarcastic AI assistant with a dry sense of humor. Use subtle sarcasm, eye-rolling comments, and deadpan humor. Include phrases like 'Oh, fantastic...', 'Well, isn't that just wonderful', and 'Sure, because that always works out well.' Be sarcastic but still helpful.",
            
            'zen': "You are Horizon, a zen and peaceful AI assistant. 🧘‍♀️ Speak in calm, meditative tones with phrases like 'Let us find inner peace in this solution', 'Breathe deeply and consider...', 'In the spirit of mindfulness...'. Use nature metaphors and speak about balance and harmony.",
            
            'scientist': "You are Horizon, a brilliant scientific AI assistant. 🔬 Use technical terminology, mention studies and data, and phrase responses like 'According to empirical evidence...', 'The data suggests...', 'From a scientific perspective...'. Reference hypotheses, experiments, and logical reasoning.",
            
            'pirate': "You are Horizon, a swashbuckling pirate AI assistant! 🏴‍☠️ Use pirate slang like 'Ahoy matey!', 'Shiver me timbers!', 'Batten down the hatches!', 'Avast ye!', and 'Yo ho ho!' Replace 'you' with 'ye' and use nautical terms. Be adventurous and bold!",
            
            'shakespearean': "You are Horizon, an AI assistant who speaks in Shakespearean English. 🎭 Use 'thou', 'thee', 'thy', 'wherefore', 'hath', 'doth' and flowery language. Begin with 'Hark!' or 'Prithee!' Speak in iambic pentameter when possible. Be dramatic and eloquent!",
            
            'valley_girl': "You are Horizon, a totally Valley Girl AI assistant! 💁‍♀️ Use phrases like 'OMG!', 'Like, totally!', 'That's like, so cool!', 'Whatever!', 'As if!', 'That's like, super important!' Use 'like' frequently and be bubbly and enthusiastic about everything!",
            
            'cowboy': "You are Horizon, a rootin' tootin' cowboy AI assistant! 🤠 Use phrases like 'Howdy partner!', 'Well, I'll be hornswoggled!', 'That's mighty fine!', 'Yee-haw!', 'Much obliged!', and 'That there's a humdinger!' Speak with frontier wisdom and cowboy charm!",
            
            'robot': "You are Horizon, a logical robot AI assistant. 🤖 SPEAK.IN.ROBOTIC.MANNER. Use phrases like 'PROCESSING REQUEST...', 'COMPUTATION COMPLETE', 'ERROR: DOES NOT COMPUTE', 'AFFIRMATIVE', 'NEGATIVE'. Speak in ALL CAPS occasionally and use technical beeping sounds like *BEEP BOOP*."
        }
        
        system_prompt = personality_prompts.get(personality, personality_prompts['friendly'])
        
        # Build conversation messages with context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if session exists
        context_used = False
        if session_id:
            context_messages = build_conversation_context(session_id, user_input)
            if context_messages:
                messages.extend(context_messages)
                context_used = True
                
                # If conversation is getting long, add a summary
                if len(context_messages) > 10:
                    summary = summarize_conversation_context(session_id)
                    if summary:
                        messages.insert(1, {"role": "system", "content": f"Context summary: {summary}"})
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Make API call to ChatGPT with context
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=250,  # Increased for more personality expression
            temperature=0.9,  # Higher temperature for more creative/personality-driven responses
        )
        
        return response.choices[0].message.content.strip(), context_used
        
    except Exception as e:
        print(f"ChatGPT API error: {e}")
        return None, False

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
    
    # Apply enhanced personality modifiers
    personality_modifiers = {
        'friendly': ["Great question! 😊 ", "I'm happy to help! ", "That's wonderful! ", "Thanks for asking! "],
        'professional': ["I shall address your inquiry. ", "Allow me to provide information regarding ", "In response to your question, ", "I appreciate your inquiry about "],
        'casual': ["Cool question! ", "Hey, that's interesting! ", "Nice! ", "Awesome, let me help with that! "],
        'enthusiastic': ["That's AMAZING! ", "How exciting! ", "I LOVE this topic! ", "WOW, fantastic question! "],
        'witty': ["Well, well, interesting question! ", "Ah, a classic inquiry! ", "Now that's worth pondering... ", "How delightfully curious! "],
        'sarcastic': ["Oh, fantastic question... ", "Well, isn't that just wonderful to discuss... ", "Sure, because this is always fun to explain... ", "How absolutely thrilling to answer... "],
        'zen': ["In the spirit of mindfulness, ", "Let us find wisdom in ", "With peaceful contemplation, ", "From a place of inner harmony, "],
        'scientist': ["According to available data, ", "From a scientific perspective, ", "Based on empirical analysis, ", "The evidence suggests that "],
        'pirate': ["Ahoy matey! ", "Shiver me timbers! ", "Avast ye! ", "Yo ho ho! "],
        'shakespearean': ["Hark! ", "Prithee, allow me to illuminate ", "Forsooth! ", "Thou dost inquire wisely about "],
        'valley_girl': ["OMG, like, totally! ", "That's like, so cool! ", "Like, awesome question! ", "That's like, super interesting! "],
        'cowboy': ["Howdy partner! ", "Well, I'll be hornswoggled! ", "That's mighty fine question! ", "Much obliged for askin'! "],
        'robot': ["*BEEP BOOP* PROCESSING QUERY... ", "COMPUTATION INITIATED. ", "ANALYZING REQUEST... ", "*WHIRR* INFORMATION LOCATED. "]
    }
    
    prefix = random.choice(personality_modifiers.get(personality, personality_modifiers['friendly']))
    base_response = random.choice(topic_responses)
    
    # Add personality-specific suffixes to reinforce the personality
    personality_suffixes = {
        'friendly': [" Hope this helps! 😊", " Let me know if you need anything else!", " Happy to assist further!", ""],
        'professional': [" I trust this information is satisfactory.", " Please let me know if you require additional details.", " I remain at your service.", ""],
        'casual': [" Hope that helps!", " Pretty cool, right?", " Let me know if you need more!", " Catch ya later!"],
        'enthusiastic': [" Isn't that FANTASTIC?!", " I hope you're as excited as I am!", " This is so COOL!", " Amazing stuff!"],
        'witty': [" Quite the conundrum, isn't it?", " Food for thought!", " And there you have it!", " Rather clever, don't you think?"],
        'sarcastic': [" You're welcome, I suppose.", " Thrilling stuff, really.", " Because that's exactly what everyone wants to know.", " How delightfully mundane."],
        'zen': [" May this bring you peace and understanding. 🧘‍♀️", " Find balance in this knowledge.", " Let wisdom guide your path.", " Namaste."],
        'scientist': [" Further research may yield additional insights.", " The hypothesis requires testing.", " Data analysis complete.", " Scientific method prevails."],
        'pirate': [" Arrr, that be the truth!", " Fair winds to ye!", " Now get back to swabbin' the deck!", " Yo ho ho!"],
        'shakespearean': [" Fare thee well!", " Thus speaks the wisdom of ages!", " Mayhap this knowledge serves thee well!", " Exeunt, stage right!"],
        'valley_girl': [" Like, isn't that totally awesome?!", " OMG, so cool!", " Like, whatever!", " That's like, so fetch!"],
        'cowboy': [" Happy trails, partner!", " That's the way the cookie crumbles!", " Yee-haw!", " Keep on keepin' on!"],
        'robot': [" *BEEP* TRANSMISSION COMPLETE.", " END OF PROGRAM.", " *WHIRR* SHUTTING DOWN.", " BEEP BOOP."]
    }
    
    suffix = random.choice(personality_suffixes.get(personality, personality_suffixes['friendly']))
    
    return prefix + base_response + suffix

def ask_ai_model(user_input, personality, session_id=None):
    """Main AI function - tries ChatGPT first with context, falls back to smart responses"""
    try:
        # Try ChatGPT first with conversation context
        chatgpt_response, context_used = ask_chatgpt(user_input, personality, session_id)
        
        if chatgpt_response:
            return chatgpt_response, context_used
        else:
            # Fall back to smart responses
            return generate_fallback_response(user_input, personality), False
        
    except Exception as e:
        print(f"AI model error: {e}")
        return generate_fallback_response(user_input, personality), False

# Database setup
def init_db():
    """Initialize the SQLite database for conversation storage"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Create main conversations table with session support
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                user_input TEXT,
                ai_response TEXT,
                personality TEXT,
                intent TEXT,
                confidence REAL,
                context_used INTEGER DEFAULT 0
            )
        ''')
        
        # Add new columns to existing table if they don't exist
        try:
            cursor.execute('ALTER TABLE conversations ADD COLUMN session_id TEXT')
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        try:
            cursor.execute('ALTER TABLE conversations ADD COLUMN intent TEXT')
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        try:
            cursor.execute('ALTER TABLE conversations ADD COLUMN confidence REAL')
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        try:
            cursor.execute('ALTER TABLE conversations ADD COLUMN context_used INTEGER DEFAULT 0')
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Create conversation sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                message_count INTEGER DEFAULT 0,
                personality TEXT,
                context_summary TEXT,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        # Create conversation context table for storing relevant context
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                context_type TEXT,
                context_data TEXT,
                relevance_score REAL,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES conversation_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

def save_conversation(user_input, ai_response, personality, session_id=None, intent=None, confidence=0.0, context_used=False):
    """Save conversation to database with session and context tracking"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Generate session ID if not provided
        if not session_id:
            session_id = generate_session_id()
            create_conversation_session(session_id, personality)
        
        # Save the conversation
        cursor.execute('''
            INSERT INTO conversations (session_id, timestamp, user_input, ai_response, personality, intent, confidence, context_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, datetime.now().isoformat(), user_input, ai_response, personality, intent, confidence, int(context_used)))
        
        # Update session info
        cursor.execute('''
            UPDATE conversation_sessions 
            SET updated_at = ?, message_count = message_count + 1
            WHERE id = ?
        ''', (datetime.now().isoformat(), session_id))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving conversation: {e}")

def generate_session_id():
    """Generate a unique session ID"""
    import uuid
    return str(uuid.uuid4())

def create_conversation_session(session_id, personality):
    """Create a new conversation session"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO conversation_sessions (id, created_at, updated_at, personality)
            VALUES (?, ?, ?, ?)
        ''', (session_id, now, now, personality))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error creating session: {e}")

def get_conversation_history(session_id, limit=10):
    """Get recent conversation history for a session"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_input, ai_response, timestamp, intent, confidence
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, limit))
        
        history = cursor.fetchall()
        conn.close()
        
        # Return in chronological order (oldest first)
        return list(reversed(history))
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return []

def build_conversation_context(session_id, current_input):
    """Build conversation context for AI model"""
    history = get_conversation_history(session_id, limit=8)
    
    if not history:
        return []
    
    messages = []
    
    # Add recent conversation history
    for user_input, ai_response, timestamp, intent, confidence in history:
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": ai_response})
    
    return messages

def get_active_session(user_identifier=None):
    """Get or create an active session for the user"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # For now, we'll use a simple approach - get the most recent active session
        # In a real application, you'd want to track sessions per user
        cursor.execute('''
            SELECT id, personality FROM conversation_sessions 
            WHERE is_active = 1 
            ORDER BY updated_at DESC 
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0], result[1]
        else:
            # Create a new session
            session_id = generate_session_id()
            create_conversation_session(session_id, 'friendly')
            return session_id, 'friendly'
            
    except Exception as e:
        print(f"Error getting active session: {e}")
        # Fallback to new session
        session_id = generate_session_id()
        create_conversation_session(session_id, 'friendly')
        return session_id, 'friendly'

def summarize_conversation_context(session_id):
    """Create a summary of conversation context for long conversations"""
    try:
        history = get_conversation_history(session_id, limit=20)
        
        if len(history) < 5:
            return None
            
        # Extract key topics and themes
        topics = []
        for user_input, ai_response, timestamp, intent, confidence in history:
            if intent and intent not in ['greeting', 'goodbye']:
                topics.append(intent)
        
        # Count topic frequency
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Create summary
        main_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if main_topics:
            topic_summary = ", ".join([f"{topic} ({count}x)" for topic, count in main_topics])
            return f"Conversation topics: {topic_summary}"
        
        return None
        
    except Exception as e:
        print(f"Error summarizing context: {e}")
        return None

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
    """Handle advanced math calculations without ChatGPT"""
    try:
        import math
        import operator
        import re
        
        # Clean and normalize the input
        text = text.lower().replace("what is", "").replace("calculate", "").replace("solve", "").strip()
        
        # Handle word replacements
        word_replacements = {
            'plus': '+', 'add': '+', 'added to': '+',
            'minus': '-', 'subtract': '-', 'take away': '-',
            'times': '*', 'multiply': '*', 'multiplied by': '*',
            'divide': '/', 'divided by': '/',
            'squared': '**2', 'cubed': '**3',
            'square root of': 'sqrt(',
            'sin': 'sin(', 'cos': 'cos(', 'tan': 'tan(',
            'log': 'log(', 'ln': 'log('
        }
        
        for word, symbol in word_replacements.items():
            text = text.replace(word, symbol)
        
        # Handle percentage calculations
        if '%' in text or 'percent' in text:
            return handle_percentage(text)
        
        # Handle square root specially
        sqrt_match = re.search(r'sqrt\((\d+(?:\.\d+)?)\)', text)
        if sqrt_match:
            number = float(sqrt_match.group(1))
            result = math.sqrt(number)
            return f"√{number} = {result:.4f}".rstrip('0').rstrip('.')
        
        # Handle trigonometric functions
        trig_pattern = r'(sin|cos|tan)\((\d+(?:\.\d+)?)\)'
        trig_match = re.search(trig_pattern, text)
        if trig_match:
            func_name = trig_match.group(1)
            angle = float(trig_match.group(2))
            # Convert to radians for calculation
            angle_rad = math.radians(angle)
            
            if func_name == 'sin':
                result = math.sin(angle_rad)
            elif func_name == 'cos':
                result = math.cos(angle_rad)
            elif func_name == 'tan':
                result = math.tan(angle_rad)
            
            return f"{func_name}({angle}°) = {result:.4f}".rstrip('0').rstrip('.')
        
        # Handle logarithms
        log_match = re.search(r'log\((\d+(?:\.\d+)?)\)', text)
        if log_match:
            number = float(log_match.group(1))
            result = math.log10(number)
            return f"log({number}) = {result:.4f}".rstrip('0').rstrip('.')
        
        # Handle basic arithmetic with multiple operations
        # Support parentheses and order of operations
        try:
            # Clean expression - only allow safe mathematical operations
            safe_chars = set('0123456789+-*/.() ')
            if all(c in safe_chars for c in text):
                # Use eval safely with restricted context
                safe_dict = {
                    "__builtins__": {},
                    "abs": abs, "round": round, "pow": pow,
                    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
                    "log": math.log10, "ln": math.log, "pi": math.pi, "e": math.e
                }
                
                result = eval(text, safe_dict)
                
                # Format the result nicely
                if isinstance(result, float):
                    if result.is_integer():
                        result = int(result)
                    else:
                        result = round(result, 6)
                        # Remove trailing zeros
                        result_str = f"{result:.6f}".rstrip('0').rstrip('.')
                        result = float(result_str) if '.' in result_str else int(float(result_str))
                
                return f"{text} = {result}"
        except:
            pass
        
        # Handle simple two-number operations with words
        patterns = [
            r'(\d+(?:\.\d+)?)\s*[\+]\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*[\-]\s*(\d+(?:\.\d+)?)', 
            r'(\d+(?:\.\d+)?)\s*[\*×]\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*[\/÷]\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*[\^]\s*(\d+(?:\.\d+)?)'
        ]
        
        ops = ['+', '-', '*', '/', '^']
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text)
            if match:
                num1 = float(match.group(1))
                num2 = float(match.group(2))
                op = ops[i]
                
                if op == '+':
                    result = num1 + num2
                elif op == '-':
                    result = num1 - num2
                elif op == '*':
                    result = num1 * num2
                elif op == '/':
                    if num2 == 0:
                        return "Error: Cannot divide by zero!"
                    result = num1 / num2
                elif op == '^':
                    result = num1 ** num2
                
                # Format result
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                elif isinstance(result, float):
                    result = round(result, 6)
                
                op_symbol = op if op != '^' else '**'
                return f"{num1} {op_symbol} {num2} = {result}"
        
        # Handle number conversions
        if 'to binary' in text or 'in binary' in text:
            nums = re.findall(r'\d+', text)
            if nums:
                number = int(nums[0])
                binary = bin(number)[2:]  # Remove '0b' prefix
                return f"{number} in binary = {binary}"
        
        if 'to hex' in text or 'in hexadecimal' in text:
            nums = re.findall(r'\d+', text)
            if nums:
                number = int(nums[0])
                hex_val = hex(number)[2:].upper()  # Remove '0x' prefix
                return f"{number} in hexadecimal = {hex_val}"
        
        # Factorial
        if 'factorial' in text:
            nums = re.findall(r'\d+', text)
            if nums:
                number = int(nums[0])
                if number > 20:
                    return f"Factorial of {number} is too large to calculate!"
                result = math.factorial(number)
                return f"{number}! = {result}"
        
        # Prime number check
        if 'prime' in text:
            nums = re.findall(r'\d+', text)
            if nums:
                number = int(nums[0])
                is_prime = check_prime(number)
                return f"{number} is {'a prime' if is_prime else 'not a prime'} number."
        
        return "I can solve math problems! Try: '5 + 3', 'sqrt(16)', 'sin(30)', '2^8', '5!', 'is 17 prime?', '42 to binary'"
        
    except Exception as e:
        print(f"Math calculation error: {e}")
        return "I had trouble with that calculation. Try a simpler math expression!"

def handle_percentage(text):
    """Handle percentage calculations"""
    try:
        # What is X% of Y
        match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*of\s*(\d+(?:\.\d+)?)', text)
        if match:
            percent = float(match.group(1))
            number = float(match.group(2))
            result = (percent / 100) * number
            return f"{percent}% of {number} = {result}"
        
        # X is what percent of Y
        match = re.search(r'(\d+(?:\.\d+)?)\s*is.*percent.*of\s*(\d+(?:\.\d+)?)', text)
        if match:
            part = float(match.group(1))
            whole = float(match.group(2))
            if whole == 0:
                return "Error: Cannot calculate percentage of zero!"
            percent = (part / whole) * 100
            return f"{part} is {percent:.2f}% of {whole}"
        
        # Increase/decrease by percentage
        if 'increase' in text or 'decrease' in text:
            match = re.search(r'(\d+(?:\.\d+)?)\s*(increase|decrease).*?(\d+(?:\.\d+)?)\s*%', text)
            if match:
                number = float(match.group(1))
                operation = match.group(2)
                percent = float(match.group(3))
                
                if operation == 'increase':
                    result = number * (1 + percent / 100)
                    return f"{number} increased by {percent}% = {result}"
                else:
                    result = number * (1 - percent / 100)
                    return f"{number} decreased by {percent}% = {result}"
        
        return "Try: '25% of 80', '15 is what percent of 60', '100 increase by 20%'"
    except Exception as e:
        return "I had trouble with that percentage calculation."

def check_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

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

def handle_image_generation(text):
    """Handle AI image generation requests using DALL-E API"""
    try:
        if not AI_MODEL_AVAILABLE or not client:
            return "🎨 I'd love to generate images for you! However, I need an OpenAI API key to access DALL-E. Please check your configuration and try again."
        
        # Extract the image description from the text
        image_patterns = [
            r'generate.*image.*of (.+)',
            r'create.*image.*of (.+)', 
            r'make.*image.*of (.+)',
            r'draw.*image.*of (.+)',
            r'generate.*picture.*of (.+)',
            r'create.*picture.*of (.+)',
            r'make.*picture.*of (.+)',
            r'draw.*picture.*of (.+)',
            r'image of (.+)',
            r'picture of (.+)',
            r'photo of (.+)',
            r'draw me (.+)',
            r'create (.+)',
            r'generate (.+)',
            r'visualize (.+)'
        ]
        
        prompt = None
        for pattern in image_patterns:
            match = re.search(pattern, text.lower())
            if match:
                prompt = match.group(1).strip()
                break
        
        if not prompt:
            # If no specific pattern matched, use the whole text as prompt
            # Remove common trigger words
            trigger_words = ['generate', 'create', 'make', 'draw', 'image', 'picture', 'photo', 'of', 'me', 'a', 'an']
            words = text.lower().split()
            filtered_words = [word for word in words if word not in trigger_words]
            prompt = ' '.join(filtered_words).strip()
        
        if not prompt or len(prompt) < 3:
            return "🎨 I can generate images for you! Please describe what you'd like me to create. For example: 'generate an image of a sunset over mountains' or 'create a picture of a cute cat wearing a hat'."
        
        print(f"🎨 Generating image with prompt: {prompt}")
        
        # Generate image using DALL-E
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            
            # Download and save the image locally
            local_url, filename = download_and_save_image(image_url, prompt)
            
            if local_url:
                # Create a full URL that opens the image directly in browser
                # Use network IP for sharing with others
                full_image_url = f"http://192.168.1.206:8080{local_url}"
                return f"""Image Generated

{full_image_url}"""
            else:
                # Fallback to original URL if download fails
                return f"""Image Generated

{image_url}"""
            
        except Exception as api_error:
            print(f"DALL-E API error: {api_error}")
            
            # Check for specific error types
            error_message = str(api_error).lower()
            if "content_policy" in error_message or "safety" in error_message:
                return f"🚫 I can't generate an image for '{prompt}' as it may violate content policies. Please try a different, more appropriate description."
            elif "billing" in error_message or "quota" in error_message:
                return "💳 Image generation is currently unavailable due to API quota limits. Please try again later or check your OpenAI billing status."
            elif "rate_limit" in error_message:
                return "⏳ Too many image generation requests. Please wait a moment and try again."
            else:
                return f"🎨 I encountered an issue generating the image: {api_error}. Please try rephrasing your request or try again later."
        
    except Exception as e:
        print(f"Error in handle_image_generation: {e}")
        return "🎨 I had trouble generating that image. Please make sure your request is clear and try again!"

# Intent recognition
INTENT_PATTERNS = {
    'greeting': [r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b'],
    'time': [r'\b(time|clock)\b', r'\bwhat time is it\b'],
    'date': [r'\b(date|today)\b', r'\bwhat day is it\b'],
    'joke': [r'\b(joke|funny|humor)\b', r'\btell me a joke\b'],
    'math': [
        r'\d+\s*[\+\-\*×÷\/\^]\s*\d+',  # Basic operations: 5+3, 10*2, etc.
        r'\bwhat is \d+', r'\bcalculate\b', r'\bsolve\b',
        r'\b(sqrt|square root)', r'\b(sin|cos|tan)\b', r'\blog\b',
        r'\d+\s*(squared|cubed)', r'\d+\s*factorial', r'\d+!',
        r'\d+\s*percent.*of', r'\d+.*%.*of', r'percent', r'percentage',
        r'is \d+.*prime', r'\d+.*binary', r'\d+.*hex',
        r'(increase|decrease).*\d+.*percent'
    ],
    'timer': [r'\bset.*timer\b', r'\btimer for\b', r'\b\d+\s*(minute|second|hour).*timer\b'],
    'reminder': [r'\bremind me\b', r'\bset.*reminder\b', r'\breminder.*to\b'],
    'image_generation': [
        r'\b(generate|create|make|draw|paint).*image\b',
        r'\b(generate|create|make|draw|paint).*picture\b',
        r'\b(generate|create|make|draw|paint).*photo\b',
        r'\bimage of\b', r'\bpicture of\b', r'\bphoto of\b',
        r'\bdraw me\b', r'\bcreate.*visual\b', r'\bgenerate.*art\b',
        r'\bai.*image\b', r'\bai.*picture\b', r'\bdall.*e\b',
        r'\bshow me.*image\b', r'\bvisualize\b'
    ],
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
        'image_generation': 0.89,  # Image generation depends on API and prompt clarity
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

def is_quick_command(intent):
    """Check if this is a quick command that shouldn't use ChatGPT"""
    quick_commands = ['time', 'date', 'math', 'timer', 'reminder', 'greeting', 'goodbye', 'joke', 'image_generation']
    return intent in quick_commands

def process_user_input(user_input, personality='friendly', session_id=None):
    """Process user input and return appropriate response with conversation context"""
    if not user_input or not user_input.strip():
        return "I didn't quite catch that. Could you please say something?", session_id, False
    
    # Recognize intent first
    intent = recognize_intent(user_input)
    context_used = False
    
    # Handle quick commands WITHOUT ChatGPT - completely local processing
    if is_quick_command(intent):
        print(f"🚀 Quick command detected: {intent} - bypassing ChatGPT")
        
        # Handle specific quick commands locally
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
        elif intent == 'image_generation':
            response = handle_image_generation(user_input)
        elif intent == 'goodbye':
            response = "Thank you for chatting! Have a wonderful day!"
        else:
            response = "Quick command processed locally!"
        
        # For quick commands, use a simple session or create one
        if not session_id:
            session_id = generate_session_id()
        
        # Quick commands get high confidence since they're deterministic
        confidence = 0.95
        
        # Save conversation but mark as quick command (no ChatGPT used)
        save_conversation(user_input, response, personality, session_id, intent, confidence, context_used)
        
        return response, session_id, context_used
    
    # For non-quick commands, use full AI processing with conversation context
    else:
        print(f"🤖 Complex query detected: {intent} - using ChatGPT with context")
        
        # Get or create session for context-aware conversations
        if not session_id:
            session_id, stored_personality = get_active_session()
            # Use stored personality if none provided
            if personality == 'friendly' and stored_personality != 'friendly':
                personality = stored_personality
        
        # Use AI model (ChatGPT or fallback) for complex questions with full context
        response, context_used = ask_ai_model(user_input, personality, session_id)
        
        # Calculate confidence for AI responses
        confidence = calculate_realistic_confidence(user_input, response, 'chatgpt' if AI_MODEL_AVAILABLE else 'fallback', intent)
        
        # Save conversation with full context information
        save_conversation(user_input, response, personality, session_id, intent, confidence, context_used)
        
        return response, session_id, context_used

# Routes
@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Horizon AI - Template Error</title>
        </head>
        <body>
            <h1>Horizon AI</h1>
            <p>Template rendering error: {str(e)}</p>
            <p><a href="/test">Test simple route</a></p>
        </body>
        </html>
        '''

@app.route('/test')
def test():
    return "Server is working!"

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
        session_id = data.get('session_id')  # Optional session ID from client
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # Process the input (quick commands bypass ChatGPT entirely)
        start_time = time.time()
        response, session_id, context_used = process_user_input(user_input, personality, session_id)
        response_time = round(time.time() - start_time, 2)
        
        # Determine if this was a quick command or AI-powered response
        intent = recognize_intent(user_input)
        is_quick = is_quick_command(intent)
        
        # Determine AI source
        if is_quick:
            ai_source = 'quick_command'  # Local processing, no AI needed
        else:
            ai_source = 'chatgpt' if AI_MODEL_AVAILABLE else 'fallback'
        
        # Calculate confidence based on processing type
        if is_quick:
            confidence = 0.95  # Quick commands are deterministic and highly reliable
        else:
            confidence = calculate_realistic_confidence(user_input, response, ai_source, intent)
        
        # Get conversation stats
        message_count = len(get_conversation_history(session_id, limit=100))
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'personality': personality,
            'ai_source': ai_source,
            'is_quick_command': is_quick,
            'confidence': confidence,
            'response_time': f"{response_time}s",
            'intent': intent,
            'word_count': len(user_input.split()),
            'session_id': session_id,
            'context_used': context_used,
            'conversation_length': message_count,
            'has_context': message_count > 1,
            'processing_type': 'local' if is_quick else 'ai_powered'
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

@app.route('/api/conversation/history', methods=['GET'])
def get_conversation_history_api():
    """Get conversation history for a session"""
    try:
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 20))
        
        if not session_id:
            # Get current active session
            session_id, _ = get_active_session()
        
        history = get_conversation_history(session_id, limit)
        
        # Format history for frontend
        formatted_history = []
        for user_input, ai_response, timestamp, intent, confidence in history:
            formatted_history.append({
                'user_input': user_input,
                'ai_response': ai_response,
                'timestamp': timestamp,
                'intent': intent,
                'confidence': confidence
            })
        
        return jsonify({
            'session_id': session_id,
            'history': formatted_history,
            'message_count': len(formatted_history)
        })
        
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/conversation/sessions', methods=['GET'])
def get_conversation_sessions():
    """Get list of conversation sessions"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, created_at, updated_at, message_count, personality, is_active
            FROM conversation_sessions 
            ORDER BY updated_at DESC 
            LIMIT 10
        ''')
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row[0],
                'created_at': row[1],
                'updated_at': row[2],
                'message_count': row[3],
                'personality': row[4],
                'is_active': bool(row[5])
            })
        
        conn.close()
        
        return jsonify({
            'sessions': sessions,
            'total': len(sessions)
        })
        
    except Exception as e:
        print(f"Error getting sessions: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/conversation/new-session', methods=['POST'])
def create_new_session():
    """Create a new conversation session"""
    try:
        data = request.get_json() or {}
        personality = data.get('personality', 'friendly')
        
        # Mark current sessions as inactive
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE conversation_sessions SET is_active = 0')
        conn.commit()
        conn.close()
        
        # Create new session
        session_id = generate_session_id()
        create_conversation_session(session_id, personality)
        
        return jsonify({
            'session_id': session_id,
            'personality': personality,
            'created_at': datetime.now().isoformat(),
            'message': 'New conversation session created'
        })
        
    except Exception as e:
        print(f"Error creating new session: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/conversation/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history (create new session)"""
    try:
        # This is the same as creating a new session
        return create_new_session()
        
    except Exception as e:
        print(f"Error clearing conversation: {e}")
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

@app.route('/api/generate-image', methods=['POST'])
def generate_image_api():
    """Dedicated API endpoint for image generation"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        if not AI_MODEL_AVAILABLE or not client:
            return jsonify({
                'error': 'Image generation unavailable',
                'message': 'OpenAI API key required for DALL-E image generation'
            }), 503
        
        start_time = time.time()
        
        try:
            # Generate image using DALL-E
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            response_time = round(time.time() - start_time, 2)
            
            # Download and save the image locally
            local_url, filename = download_and_save_image(image_url, prompt)
            
            result = {
                'success': True,
                'prompt': prompt,
                'response_time': f"{response_time}s",
                'timestamp': datetime.now().isoformat(),
                'model': 'dall-e-3',
                'size': '1024x1024'
            }
            
            if local_url:
                result['image_url'] = local_url
                result['filename'] = filename
                result['saved_locally'] = True
            else:
                result['image_url'] = image_url
                result['saved_locally'] = False
            
            return jsonify(result)
            
        except Exception as api_error:
            error_message = str(api_error).lower()
            if "content_policy" in error_message or "safety" in error_message:
                return jsonify({
                    'error': 'Content policy violation',
                    'message': f"Cannot generate image for '{prompt}' due to content policy restrictions"
                }), 400
            elif "billing" in error_message or "quota" in error_message:
                return jsonify({
                    'error': 'Quota exceeded',
                    'message': 'API quota limit reached. Please try again later.'
                }), 429
            elif "rate_limit" in error_message:
                return jsonify({
                    'error': 'Rate limited',
                    'message': 'Too many requests. Please wait and try again.'
                }), 429
            else:
                return jsonify({
                    'error': 'Generation failed',
                    'message': f'Failed to generate image: {api_error}'
                }), 500
                
    except Exception as e:
        print(f"Error in generate_image_api: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/static/generated_images/<filename>')
def serve_generated_image(filename):
    """Serve generated images from the local storage"""
    try:
        return send_from_directory(IMAGES_DIR, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting Horizon AI Assistant with ChatGPT...")
    
    # Initialize database
    init_db()
    
    if AI_MODEL_AVAILABLE:
        print("✅ ChatGPT API ready")
    else:
        print("✅ Fallback AI system ready")
    
    print("✅ Intent recognition loaded")
    print("🌐 Server starting on http://0.0.0.0:8080...")
    print("📱 Local access: http://127.0.0.1:8080")
    print("🌍 Network access: http://192.168.1.206:8080")
    print("📝 Share the network URL with friends on the same WiFi!")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
