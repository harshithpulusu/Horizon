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
from config import Config

# Video generation imports
try:
    from PIL import Image, ImageDraw, ImageFont
    import imageio
    VIDEO_FEATURES_AVAILABLE = True
    print("🎥 Basic video generation features loaded successfully")
    
    # Try to import opencv separately
    try:
        import cv2
        import numpy as np
        OPENCV_AVAILABLE = True
        print("� Advanced video effects with OpenCV loaded")
    except ImportError:
        OPENCV_AVAILABLE = False
        print("⚠️ OpenCV not available - basic video generation only")
        
except ImportError as e:
    VIDEO_FEATURES_AVAILABLE = False
    OPENCV_AVAILABLE = False
    print(f"⚠️ Video features not available: {e}")
    print("💡 Install with: pip install Pillow imageio imageio-ffmpeg")

# Audio and Music generation imports
try:
    import speech_recognition as sr
    import pyaudio
    from pydub import AudioSegment
    from mutagen import File as MutagenFile
    AUDIO_FEATURES_AVAILABLE = True
    print("🎵 Audio processing features loaded successfully")
    
    # Try to import ElevenLabs for voice synthesis
    try:
        import elevenlabs
        ELEVENLABS_AVAILABLE = True
        print("🗣️ ElevenLabs voice synthesis available")
    except ImportError:
        ELEVENLABS_AVAILABLE = False
        print("⚠️ ElevenLabs not available - install with: pip install elevenlabs")
        
except ImportError as e:
    AUDIO_FEATURES_AVAILABLE = False
    ELEVENLABS_AVAILABLE = False
    print(f"⚠️ Audio features not available: {e}")
    print("💡 Install with: pip install speechrecognition pyaudio pydub mutagen")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Create media directories if they don't exist
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_images')
VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_videos')
GIFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_gifs')
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_audio')
MUSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_music')
AVATARS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_avatars')
DESIGNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_designs')
MODELS_3D_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_3d_models')
LOGOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'generated_logos')

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(GIFS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MUSIC_DIR, exist_ok=True)
os.makedirs(AVATARS_DIR, exist_ok=True)
os.makedirs(DESIGNS_DIR, exist_ok=True)
os.makedirs(MODELS_3D_DIR, exist_ok=True)
os.makedirs(LOGOS_DIR, exist_ok=True)

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

def ask_chatgpt(user_input, personality, session_id=None, user_id='anonymous'):
    """Use ChatGPT API for intelligent responses with conversation context and AI intelligence features"""
    if not AI_MODEL_AVAILABLE or not client:
        return None, False
    
    try:
        # Update personality usage
        update_personality_usage(personality)
        
        # Analyze user emotion
        emotion_data = analyze_emotion(user_input)
        detected_emotion = emotion_data.get('emotion', 'neutral')
        sentiment_score = emotion_data.get('sentiment', 0.0)
        
        # Retrieve user memory and context
        user_memories = retrieve_user_memory(user_id)
        
        # Get personality profile
        personality_profile = get_personality_profile(personality)
        
        # Create enhanced personality-specific system prompt
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
        
        base_prompt = personality_prompts.get(personality, personality_prompts['friendly'])
        
        # Build enhanced context
        context_parts = [base_prompt]
        
        # Add personality profile context
        if personality_profile:
            context_parts.append(f"\nYour personality traits: {', '.join(personality_profile['traits']) if personality_profile['traits'] else 'N/A'}")
            context_parts.append(f"Your response style: {personality_profile['style']}")
        
        # Add emotional context
        if detected_emotion != 'neutral':
            context_parts.append(f"\nIMPORTANT: The user is feeling {detected_emotion} (confidence: {emotion_data.get('confidence', 0):.2f})")
            context_parts.append(f"User's sentiment: {sentiment_score:.2f} ({classify_mood(sentiment_score)})")
            context_parts.append(f"Please respond appropriately to their {detected_emotion} emotional state and be supportive.")
        
        # Add memory context
        if user_memories:
            important_memories = [mem for mem in user_memories if len(mem) >= 4 and mem[3] > 0.7]  # High importance memories
            if important_memories:
                context_parts.append("\nThings I remember about this user:")
                for memory in important_memories[:3]:  # Top 3 memories
                    if len(memory) >= 3:
                        context_parts.append(f"- {memory[1]}: {memory[2]}")
        
        enhanced_system_prompt = "\n".join(context_parts)
        
        # Build conversation messages with context
        messages = [{"role": "system", "content": enhanced_system_prompt}]
        
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
        
        # Make API call to ChatGPT with enhanced context
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,  # Increased for more comprehensive responses
            temperature=0.8,  # Balanced for personality and accuracy
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Enhance response with emotional awareness
        ai_response = enhance_response_with_emotion(ai_response, detected_emotion, personality)
        
        # Extract and save important information to user memory
        if len(user_input) > 50:  # Longer messages likely contain important info
            keywords = extract_keywords(user_input)
            if keywords:
                key_info = f"User mentioned: {', '.join(keywords[:3])}"
                save_user_memory(user_id, 'conversation_topics', f"topics_{datetime.now().strftime('%Y%m%d')}", key_info, importance=0.6)
        
        # Check for personal information to remember
        personal_info_patterns = {
            'name': r'my name is (\w+)|i\'m (\w+)|call me (\w+)',
            'location': r'i live in ([^,\.]+)|i\'m from ([^,\.]+)|located in ([^,\.]+)',
            'occupation': r'i work as (.*?)|i\'m a (.*?)|my job is (.*?)',
            'interests': r'i like (.*?)|i love (.*?)|i enjoy (.*?)|i\'m interested in (.*?)'
        }
        
        import re
        for info_type, pattern in personal_info_patterns.items():
            match = re.search(pattern, user_input.lower())
            if match:
                value = next((group for group in match.groups() if group), '')
                if value and len(value.strip()) > 1:
                    save_user_memory(user_id, 'personal_info', info_type, value.strip(), importance=0.9)
        
        return ai_response, context_used
        
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

def ask_ai_model(user_input, personality, session_id=None, user_id='anonymous'):
    """Main AI function - tries ChatGPT first with context, falls back to smart responses"""
    try:
        # Try ChatGPT first with conversation context and AI intelligence
        chatgpt_response, context_used = ask_chatgpt(user_input, personality, session_id, user_id)
        
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
    """Initialize the SQLite database for conversation storage with AI Intelligence features"""
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
                context_used INTEGER DEFAULT 0,
                emotion_detected TEXT,
                sentiment_score REAL,
                learning_data TEXT
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
            
        try:
            cursor.execute('ALTER TABLE conversations ADD COLUMN emotion_detected TEXT')
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        try:
            cursor.execute('ALTER TABLE conversations ADD COLUMN sentiment_score REAL')
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        try:
            cursor.execute('ALTER TABLE conversations ADD COLUMN learning_data TEXT')
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
                is_active INTEGER DEFAULT 1,
                user_mood TEXT,
                dominant_emotion TEXT,
                interaction_score REAL DEFAULT 0.0
            )
        ''')
        
        # Add new columns to sessions table
        try:
            cursor.execute('ALTER TABLE conversation_sessions ADD COLUMN user_mood TEXT')
        except sqlite3.OperationalError:
            pass
            
        try:
            cursor.execute('ALTER TABLE conversation_sessions ADD COLUMN dominant_emotion TEXT')
        except sqlite3.OperationalError:
            pass
            
        try:
            cursor.execute('ALTER TABLE conversation_sessions ADD COLUMN interaction_score REAL DEFAULT 0.0')
        except sqlite3.OperationalError:
            pass
        
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
        
        # 🧠 AI PERSONALITY & INTELLIGENCE TABLES
        
        # User preferences and memory system
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_identifier TEXT,
                memory_type TEXT,
                memory_key TEXT,
                memory_value TEXT,
                importance_score REAL DEFAULT 0.5,
                created_at TEXT,
                updated_at TEXT,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        # AI learning system - track what the AI learns from interactions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learning_type TEXT,
                topic TEXT,
                pattern_data TEXT,
                effectiveness_score REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Emotion detection and analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_input TEXT,
                detected_emotion TEXT,
                emotion_confidence REAL,
                sentiment_score REAL,
                mood_classification TEXT,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES conversation_sessions (id)
            )
        ''')
        
        # AI personality profiles and modes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personality_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                personality_name TEXT UNIQUE,
                personality_description TEXT,
                response_style TEXT,
                emotional_traits TEXT,
                language_patterns TEXT,
                created_at TEXT,
                usage_count INTEGER DEFAULT 0,
                user_rating REAL DEFAULT 0.0
            )
        ''')
        
        # User interaction patterns for learning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interaction_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_identifier TEXT,
                pattern_type TEXT,
                pattern_data TEXT,
                frequency INTEGER DEFAULT 1,
                last_occurrence TEXT,
                pattern_strength REAL DEFAULT 0.0
            )
        ''')
        
        # Initialize default personality profiles
        default_personalities = [
            {
                'name': 'friendly',
                'description': 'Warm, welcoming, and supportive. Always positive and encouraging.',
                'style': 'casual_warm',
                'traits': 'optimistic,helpful,encouraging,patient',
                'patterns': 'uses_emojis,positive_language,supportive_phrases'
            },
            {
                'name': 'professional',
                'description': 'Formal, efficient, and business-oriented. Focuses on accuracy and productivity.',
                'style': 'formal_business',
                'traits': 'precise,efficient,reliable,structured',
                'patterns': 'formal_language,structured_responses,business_terminology'
            },
            {
                'name': 'casual',
                'description': 'Laid-back, relaxed, and conversational. Like talking to a friend.',
                'style': 'informal_relaxed',
                'traits': 'relaxed,conversational,approachable,flexible',
                'patterns': 'casual_language,contractions,slang,informal_greetings'
            },
            {
                'name': 'enthusiastic',
                'description': 'High-energy, excited, and passionate about everything!',
                'style': 'high_energy',
                'traits': 'energetic,passionate,motivating,upbeat',
                'patterns': 'exclamation_points,energy_words,motivational_language'
            },
            {
                'name': 'analytical',
                'description': 'Logical, data-driven, and detail-oriented. Focuses on facts and reasoning.',
                'style': 'logical_precise',
                'traits': 'logical,analytical,thorough,objective',
                'patterns': 'factual_language,structured_analysis,evidence_based'
            },
            {
                'name': 'creative',
                'description': 'Artistic, imaginative, and innovative. Thinks outside the box.',
                'style': 'artistic_innovative',
                'traits': 'imaginative,innovative,artistic,expressive',
                'patterns': 'creative_metaphors,artistic_language,innovative_ideas'
            },
            {
                'name': 'zen',
                'description': 'Calm, peaceful, and mindful. Promotes balance and inner peace.',
                'style': 'calm_mindful',
                'traits': 'peaceful,mindful,balanced,wise',
                'patterns': 'meditation_references,peaceful_language,wisdom_quotes'
            },
            {
                'name': 'witty',
                'description': 'Clever, humorous, and quick with wordplay. Enjoys intelligent humor.',
                'style': 'clever_humorous',
                'traits': 'clever,humorous,quick_witted,playful',
                'patterns': 'wordplay,clever_jokes,witty_observations'
            }
        ]
        
        for personality in default_personalities:
            cursor.execute('''
                INSERT OR IGNORE INTO personality_profiles 
                (personality_name, personality_description, response_style, emotional_traits, language_patterns, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                personality['name'],
                personality['description'], 
                personality['style'],
                personality['traits'],
                personality['patterns'],
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
        print("✅ Database initialized with AI Intelligence features")
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
        
        # Analyze emotion and sentiment
        emotion_data = analyze_emotion(user_input)
        emotion_detected = emotion_data.get('emotion', 'neutral')
        sentiment_score = emotion_data.get('sentiment', 0.0)
        
        # Extract learning data
        learning_data = extract_learning_patterns(user_input, ai_response, intent, confidence)
        
        # Save the conversation
        cursor.execute('''
            INSERT INTO conversations (session_id, timestamp, user_input, ai_response, personality, intent, confidence, context_used, emotion_detected, sentiment_score, learning_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, datetime.now().isoformat(), user_input, ai_response, personality, intent, confidence, int(context_used), emotion_detected, sentiment_score, json.dumps(learning_data)))
        
        # Update session info with emotion analysis
        cursor.execute('''
            UPDATE conversation_sessions 
            SET updated_at = ?, message_count = message_count + 1, dominant_emotion = ?, user_mood = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), emotion_detected, classify_mood(sentiment_score), session_id))
        
        # Save emotion analysis
        cursor.execute('''
            INSERT INTO emotion_analysis (session_id, user_input, detected_emotion, emotion_confidence, sentiment_score, mood_classification, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_input, emotion_detected, emotion_data.get('confidence', 0.0), sentiment_score, classify_mood(sentiment_score), datetime.now().isoformat()))
        
        # Update AI learning system
        update_ai_learning(user_input, ai_response, intent, confidence, emotion_detected)
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving conversation: {e}")

# ===== AI PERSONALITY & INTELLIGENCE FUNCTIONS =====

def analyze_emotion(text):
    """Analyze emotion and sentiment from user input"""
    try:
        text_lower = text.lower()
        
        # Emotion keywords and patterns
        emotion_patterns = {
            'happy': ['happy', 'joy', 'excited', 'great', 'awesome', 'wonderful', 'fantastic', 'amazing', 'love', 'perfect', '😄', '😊', '🎉', '❤️'],
            'sad': ['sad', 'unhappy', 'depressed', 'down', 'upset', 'disappointed', 'terrible', 'awful', 'worst', '😢', '😞', '💔'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'hate', 'stupid', 'ridiculous', 'terrible', '😠', '😡', '🤬'],
            'anxious': ['worried', 'nervous', 'anxious', 'scared', 'afraid', 'concerned', 'stress', 'panic', 'overwhelmed', '😰', '😟'],
            'excited': ['excited', 'thrilled', 'pumped', 'enthusiastic', 'can\'t wait', 'looking forward', 'amazing', '🚀', '✨', '🎯'],
            'confused': ['confused', 'don\'t understand', 'unclear', 'puzzled', 'lost', 'what?', 'huh?', '🤔', '😕'],
            'grateful': ['thank', 'thanks', 'grateful', 'appreciate', 'blessed', 'lucky', 'grateful', '🙏', '❤️'],
            'curious': ['curious', 'wonder', 'interested', 'how', 'why', 'what', 'tell me', 'explain', '🤔'],
            'disappointed': ['disappointed', 'let down', 'expected', 'hoped', 'thought', 'supposed to', '😞'],
            'surprised': ['wow', 'really?', 'no way', 'seriously?', 'amazing', 'incredible', '😮', '🤯']
        }
        
        # Calculate emotion scores
        emotion_scores = {}
        for emotion, keywords in emotion_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            emotion_scores[emotion] = score
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get) if max(emotion_scores.values()) > 0 else 'neutral'
        emotion_confidence = min(emotion_scores[dominant_emotion] / 3.0, 1.0)  # Normalize to 0-1
        
        # Calculate sentiment score (-1 to 1)
        positive_words = ['good', 'great', 'awesome', 'perfect', 'love', 'amazing', 'wonderful', 'excellent', 'fantastic', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'stupid', 'ridiculous', 'disappointing']
        
        sentiment_score = 0
        for word in positive_words:
            if word in text_lower:
                sentiment_score += 0.1
        for word in negative_words:
            if word in text_lower:
                sentiment_score -= 0.1
        
        # Adjust sentiment based on emotion
        if dominant_emotion in ['happy', 'excited', 'grateful']:
            sentiment_score += 0.2
        elif dominant_emotion in ['sad', 'angry', 'disappointed']:
            sentiment_score -= 0.2
        
        sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to -1, 1
        
        return {
            'emotion': dominant_emotion,
            'confidence': emotion_confidence,
            'sentiment': sentiment_score,
            'all_emotions': emotion_scores
        }
        
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return {'emotion': 'neutral', 'confidence': 0.0, 'sentiment': 0.0}

def classify_mood(sentiment_score):
    """Classify overall mood based on sentiment score"""
    if sentiment_score > 0.3:
        return 'positive'
    elif sentiment_score < -0.3:
        return 'negative'
    else:
        return 'neutral'

def extract_learning_patterns(user_input, ai_response, intent, confidence):
    """Extract patterns for AI learning system"""
    try:
        patterns = {
            'user_input_length': len(user_input),
            'response_length': len(ai_response),
            'intent': intent,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'keywords': extract_keywords(user_input),
            'question_type': detect_question_type(user_input),
            'complexity': assess_complexity(user_input)
        }
        return patterns
    except Exception as e:
        print(f"Error extracting learning patterns: {e}")
        return {}

def extract_keywords(text):
    """Extract important keywords from text"""
    try:
        # Simple keyword extraction based on frequency and importance
        import re
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return top 5 keywords
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, freq in top_keywords]
        
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def detect_question_type(text):
    """Detect the type of question being asked"""
    text_lower = text.lower().strip()
    
    if text_lower.startswith('what'):
        return 'what_question'
    elif text_lower.startswith('how'):
        return 'how_question'
    elif text_lower.startswith('why'):
        return 'why_question'
    elif text_lower.startswith('when'):
        return 'when_question'
    elif text_lower.startswith('where'):
        return 'where_question'
    elif text_lower.startswith('who'):
        return 'who_question'
    elif '?' in text:
        return 'general_question'
    elif any(word in text_lower for word in ['help', 'assist', 'support']):
        return 'help_request'
    elif any(word in text_lower for word in ['create', 'generate', 'make']):
        return 'creation_request'
    else:
        return 'statement'

def assess_complexity(text):
    """Assess the complexity of the user input"""
    try:
        # Simple complexity assessment based on various factors
        factors = {
            'length': len(text),
            'words': len(text.split()),
            'sentences': text.count('.') + text.count('!') + text.count('?'),
            'technical_terms': count_technical_terms(text),
            'question_words': sum(1 for word in ['what', 'how', 'why', 'when', 'where', 'who'] if word in text.lower())
        }
        
        # Calculate complexity score (0-1)
        complexity_score = 0
        
        # Length factor
        if factors['length'] > 100:
            complexity_score += 0.2
        elif factors['length'] > 50:
            complexity_score += 0.1
        
        # Word count factor
        if factors['words'] > 20:
            complexity_score += 0.2
        elif factors['words'] > 10:
            complexity_score += 0.1
        
        # Technical terms factor
        complexity_score += min(factors['technical_terms'] * 0.1, 0.3)
        
        # Multiple questions factor
        if factors['question_words'] > 1:
            complexity_score += 0.2
        
        # Multiple sentences factor
        if factors['sentences'] > 1:
            complexity_score += 0.1
        
        return min(complexity_score, 1.0)
        
    except Exception as e:
        print(f"Error assessing complexity: {e}")
        return 0.0

def count_technical_terms(text):
    """Count technical terms in the text"""
    technical_terms = [
        'algorithm', 'api', 'database', 'programming', 'software', 'hardware', 'network',
        'artificial intelligence', 'machine learning', 'neural network', 'blockchain',
        'cryptocurrency', 'quantum', 'cybersecurity', 'encryption', 'protocol',
        'server', 'client', 'framework', 'library', 'repository', 'deployment'
    ]
    
    text_lower = text.lower()
    return sum(1 for term in technical_terms if term in text_lower)

def update_ai_learning(user_input, ai_response, intent, confidence, emotion):
    """Update the AI learning system with new interaction data"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Learning categories
        learning_categories = {
            'response_effectiveness': {
                'topic': intent or 'general',
                'pattern': f"intent:{intent},emotion:{emotion}",
                'effectiveness': calculate_response_effectiveness(confidence, emotion)
            },
            'emotional_adaptation': {
                'topic': emotion,
                'pattern': f"emotion:{emotion},response_type:{detect_response_type(ai_response)}",
                'effectiveness': calculate_emotional_effectiveness(emotion, ai_response)
            },
            'conversation_flow': {
                'topic': 'conversation_patterns',
                'pattern': f"input_length:{len(user_input)},response_length:{len(ai_response)}",
                'effectiveness': confidence
            }
        }
        
        for learning_type, data in learning_categories.items():
            # Check if pattern exists
            cursor.execute('''
                SELECT id, usage_count, effectiveness_score FROM ai_learning 
                WHERE learning_type = ? AND topic = ? AND pattern_data = ?
            ''', (learning_type, data['topic'], data['pattern']))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing pattern
                new_usage_count = existing[1] + 1
                new_effectiveness = (existing[2] * existing[1] + data['effectiveness']) / new_usage_count
                
                cursor.execute('''
                    UPDATE ai_learning 
                    SET usage_count = ?, effectiveness_score = ?, updated_at = ?
                    WHERE id = ?
                ''', (new_usage_count, new_effectiveness, datetime.now().isoformat(), existing[0]))
            else:
                # Create new pattern
                cursor.execute('''
                    INSERT INTO ai_learning (learning_type, topic, pattern_data, effectiveness_score, usage_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                ''', (learning_type, data['topic'], data['pattern'], data['effectiveness'], datetime.now().isoformat(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error updating AI learning: {e}")

def calculate_response_effectiveness(confidence, emotion):
    """Calculate how effective the response was based on confidence and emotion"""
    base_effectiveness = confidence
    
    # Boost effectiveness for positive emotions
    if emotion in ['happy', 'excited', 'grateful']:
        base_effectiveness += 0.2
    elif emotion in ['sad', 'angry', 'disappointed']:
        base_effectiveness += 0.1  # Still learning opportunity
    
    return min(base_effectiveness, 1.0)

def calculate_emotional_effectiveness(emotion, response):
    """Calculate how well the response addresses the detected emotion"""
    response_lower = response.lower()
    
    # Emotional response patterns
    if emotion == 'sad':
        if any(word in response_lower for word in ['sorry', 'understand', 'support', 'here for you']):
            return 0.8
    elif emotion == 'angry':
        if any(word in response_lower for word in ['understand', 'frustrating', 'help', 'solve']):
            return 0.8
    elif emotion == 'happy':
        if any(word in response_lower for word in ['great', 'wonderful', 'fantastic', 'excited']):
            return 0.8
    elif emotion == 'anxious':
        if any(word in response_lower for word in ['calm', 'relax', 'help', 'support', 'okay']):
            return 0.8
    
    return 0.5  # Default effectiveness

def detect_response_type(response):
    """Detect the type of response generated"""
    response_lower = response.lower()
    
    if any(word in response_lower for word in ['sorry', 'apologize', 'understand your frustration']):
        return 'empathetic'
    elif any(word in response_lower for word in ['congratulations', 'great', 'fantastic', 'wonderful']):
        return 'celebratory'
    elif any(word in response_lower for word in ['help', 'assist', 'support', 'guide']):
        return 'helpful'
    elif any(word in response_lower for word in ['explain', 'information', 'details', 'about']):
        return 'informative'
    elif any(word in response_lower for word in ['create', 'generate', 'make', 'build']):
        return 'creative'
    else:
        return 'general'

def get_personality_profile(personality_name):
    """Get detailed personality profile from database"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT personality_description, response_style, emotional_traits, language_patterns, user_rating
            FROM personality_profiles 
            WHERE personality_name = ?
        ''', (personality_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'description': result[0],
                'style': result[1], 
                'traits': result[2].split(',') if result[2] else [],
                'patterns': result[3].split(',') if result[3] else [],
                'rating': result[4] or 0.0
            }
        
        return None
        
    except Exception as e:
        print(f"Error getting personality profile: {e}")
        return None

def save_user_memory(user_id, memory_type, key, value, importance=0.5):
    """Save information to user memory system"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Check if memory already exists
        cursor.execute('''
            SELECT id FROM user_memory 
            WHERE user_identifier = ? AND memory_type = ? AND memory_key = ?
        ''', (user_id, memory_type, key))
        
        if cursor.fetchone():
            # Update existing memory
            cursor.execute('''
                UPDATE user_memory 
                SET memory_value = ?, importance_score = ?, updated_at = ?, access_count = access_count + 1
                WHERE user_identifier = ? AND memory_type = ? AND memory_key = ?
            ''', (value, importance, datetime.now().isoformat(), user_id, memory_type, key))
        else:
            # Create new memory
            cursor.execute('''
                INSERT INTO user_memory (user_identifier, memory_type, memory_key, memory_value, importance_score, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, memory_type, key, value, importance, datetime.now().isoformat(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error saving user memory: {e}")

def retrieve_user_memory(user_id, memory_type=None, key=None):
    """Retrieve information from user memory system"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        if key:
            cursor.execute('''
                SELECT memory_value, importance_score FROM user_memory 
                WHERE user_identifier = ? AND memory_type = ? AND memory_key = ?
            ''', (user_id, memory_type, key))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        elif memory_type:
            cursor.execute('''
                SELECT memory_key, memory_value, importance_score FROM user_memory 
                WHERE user_identifier = ? AND memory_type = ?
                ORDER BY importance_score DESC, access_count DESC
            ''', (user_id, memory_type))
        else:
            cursor.execute('''
                SELECT memory_type, memory_key, memory_value, importance_score FROM user_memory 
                WHERE user_identifier = ?
                ORDER BY importance_score DESC, access_count DESC
            ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        return results
        
    except Exception as e:
        print(f"Error retrieving user memory: {e}")
        return []

def update_personality_usage(personality_name):
    """Update personality usage statistics"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE personality_profiles 
            SET usage_count = usage_count + 1
            WHERE personality_name = ?
        ''', (personality_name,))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error updating personality usage: {e}")

def get_ai_insights(session_id):
    """Get AI insights about the conversation and user"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Get conversation statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as message_count,
                AVG(sentiment_score) as avg_sentiment,
                dominant_emotion,
                user_mood
            FROM conversations c
            JOIN conversation_sessions s ON c.session_id = s.id
            WHERE c.session_id = ?
        ''', (session_id,))
        
        stats = cursor.fetchone()
        
        # Get emotion distribution
        cursor.execute('''
            SELECT emotion_detected, COUNT(*) as count
            FROM emotion_analysis 
            WHERE session_id = ?
            GROUP BY emotion_detected
            ORDER BY count DESC
        ''', (session_id,))
        
        emotions = cursor.fetchall()
        
        # Get learning insights
        cursor.execute('''
            SELECT learning_type, AVG(effectiveness_score) as avg_effectiveness
            FROM ai_learning
            GROUP BY learning_type
        ''', ())
        
        learning = cursor.fetchall()
        
        conn.close()
        
        return {
            'conversation_stats': {
                'message_count': stats[0] if stats else 0,
                'avg_sentiment': stats[1] if stats else 0.0,
                'dominant_emotion': stats[2] if stats else 'neutral',
                'user_mood': stats[3] if stats else 'neutral'
            },
            'emotion_distribution': emotions,
            'learning_effectiveness': dict(learning)
        }
        
    except Exception as e:
        print(f"Error getting AI insights: {e}")
        return {}

def enhance_response_with_emotion(response, detected_emotion, personality):
    """Enhance AI response based on detected emotion and personality"""
    try:
        # Get personality profile
        profile = get_personality_profile(personality)
        
        if not profile:
            return response
        
        # Emotional enhancement patterns
        emotion_enhancements = {
            'sad': {
                'friendly': ["I can hear that you're feeling down. ", "I'm here to support you. ", "It sounds like you're going through a tough time. "],
                'professional': ["I understand this is a difficult situation. ", "Let me provide some assistance. "],
                'casual': ["Hey, I can tell you're feeling down. ", "That sounds rough. "],
                'enthusiastic': ["I want to help cheer you up! ", "Let's turn this around together! "],
                'zen': ["I sense your sadness. Let's find some peace together. ", "Breathe deeply, I'm here with you. "]
            },
            'angry': {
                'friendly': ["I understand you're frustrated. ", "I can help work through this. ", "Let's solve this together. "],
                'professional': ["I recognize your concern. ", "Let me address this matter effectively. "],
                'casual': ["I get that you're mad about this. ", "That's definitely frustrating! "],
                'enthusiastic': ["Let's channel that energy into solving this! ", "I'm ready to help fix this! "],
                'zen': ["I feel your anger. Let's find calm solutions. ", "Take a moment to breathe. "]
            },
            'happy': {
                'friendly': ["I love your positive energy! ", "That's wonderful to hear! ", "Your happiness is contagious! "],
                'professional': ["Excellent! ", "That's very positive news. "],
                'casual': ["That's awesome! ", "So cool! ", "Love the good vibes! "],
                'enthusiastic': ["YES! That's AMAZING! ", "I'm so excited for you! ", "This is FANTASTIC! "],
                'zen': ["Your joy brings peace to our conversation. ", "Beautiful energy flows from your happiness. "]
            },
            'anxious': {
                'friendly': ["I understand you're worried. Let me help ease your concerns. ", "It's okay to feel anxious. "],
                'professional': ["I'll address your concerns systematically. ", "Let me provide clear guidance. "],
                'casual': ["Hey, no worries! ", "I got you covered. ", "Let's figure this out together. "],
                'enthusiastic': ["Don't worry, we've got this! ", "I'm here to help you feel confident! "],
                'zen': ["Breathe with me. Let's find calm together. ", "Peace will come. I'm here to guide you. "]
            },
            'excited': {
                'friendly': ["I love your excitement! ", "That enthusiasm is wonderful! "],
                'professional': ["Your enthusiasm is noted. ", "That's excellent motivation. "],
                'casual': ["Your excitement is contagious! ", "I'm pumped too! "],
                'enthusiastic': ["YES! I'm SO excited with you! ", "This energy is INCREDIBLE! "],
                'zen': ["Your excitement brings beautiful energy to our space. "]
            }
        }
        
        # Get appropriate enhancement
        if detected_emotion in emotion_enhancements and personality in emotion_enhancements[detected_emotion]:
            enhancement_options = emotion_enhancements[detected_emotion][personality]
            enhancement = random.choice(enhancement_options)
            return enhancement + response
        
        return response
        
    except Exception as e:
        print(f"Error enhancing response with emotion: {e}")
        return response

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

# ===============================================
# � AI MUSIC & AUDIO GENERATION FUNCTIONS

def generate_ai_music(prompt, duration=30, style="pop", quality="standard"):
    """Generate AI music using multiple APIs with fallback options"""
    
    print(f"🎵 Generating AI music: '{prompt}' ({style}, {duration}s)")
    
    # Try Stability AI first (PRIMARY - with corrected endpoints)
    if Config.STABILITY_API_KEY:
        print("🎹 Trying Stability AI (PRIMARY)...")
        result = generate_stability_music(prompt, duration, style, quality)
        if result[0]:  # Success
            return result
        else:
            print(f"⚠️ Stability AI failed: {result[1]}")
    
    # Try Replicate MusicGen (BACKUP 1 - best quality with real instruments)
    if Config.REPLICATE_API_TOKEN:
        print("🎼 Trying Replicate MusicGen (BACKUP 1)...")
        result = generate_replicate_music(prompt, duration, style, quality)
        if result[0]:  # Success
            return result
        else:
            print(f"⚠️ Replicate failed: {result[1]}")
    
    # Try Hugging Face MusicGen (BACKUP 2 - good quality, more reliable)
    if Config.HUGGINGFACE_API_KEY:
        print("🤗 Trying Hugging Face MusicGen (BACKUP 2)...")
        result = generate_huggingface_music(prompt, duration, style, quality)
        if result[0]:  # Success
            return result
        else:
            print(f"⚠️ Hugging Face failed: {result[1]}")
    
    # Fallback to enhanced synthesized music (BACKUP 3)
    print("🔄 Using Enhanced Multi-layer Synthesis (BACKUP 3)...")
    return generate_enhanced_music(prompt, duration, style, quality)

def generate_replicate_music(prompt, duration, style, quality):
    """Generate music using Replicate's MusicGen API - BEST QUALITY"""
    
    try:
        print("🎼 Using Replicate MusicGen for professional music with real instruments...")
        
        # Import replicate at function level to handle missing dependency gracefully
        try:
            import replicate
        except ImportError:
            print("⚠️ Replicate package not installed. Install with: pip install replicate")
            return None, "Replicate package not available"
        
        # Set API token
        if not Config.REPLICATE_API_TOKEN:
            print("⚠️ Replicate API token not configured")
            return None, "Replicate API token not configured"
        
        # Enhanced prompt for MusicGen
        musicgen_prompt = f"{style} music, {prompt}"
        
        # Add instrument specifications based on style
        if style.lower() == "pop":
            musicgen_prompt += ", with drums, bass, electric guitar, synthesizer, upbeat"
        elif style.lower() == "rock":
            musicgen_prompt += ", with rock drums, distorted electric guitar, bass guitar, powerful"
        elif style.lower() == "electronic":
            musicgen_prompt += ", with electronic drums, synthesizer, bass synth, energetic"
        elif style.lower() == "classical":
            musicgen_prompt += ", orchestral, piano, strings, elegant"
        elif style.lower() == "jazz":
            musicgen_prompt += ", with jazz drums, saxophone, piano, bass, smooth"
        elif style.lower() == "ambient":
            musicgen_prompt += ", atmospheric, peaceful, flowing"
        else:
            musicgen_prompt += ", with drums and bass"
        
        # Add quality descriptors
        if quality == "high":
            musicgen_prompt += ", professional production, studio quality"
        
        print(f"🎵 Generating with Replicate MusicGen: {musicgen_prompt}")
        print(f"⏱️ Duration: {duration} seconds")
        
        # Set the API token
        import os
        os.environ["REPLICATE_API_TOKEN"] = Config.REPLICATE_API_TOKEN
        
        # Generate music using the exact format you provided
        output = replicate.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input={
                "top_k": 250,
                "top_p": 0,
                "prompt": musicgen_prompt,
                "duration": min(duration, 30),  # MusicGen max 30 seconds
                "temperature": 1,
                "continuation": False,
                "model_version": "stereo-large" if quality == "high" else "large",
                "output_format": "mp3",
                "continuation_start": 0,
                "multi_band_diffusion": False,
                "normalization_strategy": "peak",
                "classifier_free_guidance": 3
            }
        )
        
        print("✅ Replicate MusicGen generation completed!")
        
        # Download and save the generated music
        import uuid
        music_id = str(uuid.uuid4())
        music_filename = f"replicate_{music_id}.mp3"
        music_path = os.path.join(MUSIC_DIR, music_filename)
        
        # Write the file to disk using the provided method
        with open(music_path, "wb") as file:
            file.write(output.read())
        
        print(f"✅ Replicate music saved: {music_filename}")
        return music_filename, None
        
    except Exception as e:
        print(f"❌ Replicate error: {e}")
        return None, f"Replicate error: {str(e)}"

def generate_stability_music(prompt, duration, style, quality):
    """Generate music using Stability AI's Stable Audio API - REAL INSTRUMENTS"""
    
    try:
        print("🎹 Using Stability AI Stable Audio - REAL instruments and professional quality...")
        
        # Check API key
        if not Config.STABILITY_API_KEY:
            print("⚠️ Stability AI API key not configured")
            return None, "Stability AI API key not configured"
        
        # Enhanced prompt for Stable Audio (it DOES support real instruments)
        stable_prompt = f"{style} music: {prompt}"
        
        # Add real instrument specifications based on style
        if style.lower() == "pop":
            stable_prompt += ", real drums, electric guitar, bass guitar, synthesizer, upbeat tempo"
        elif style.lower() == "rock":
            stable_prompt += ", rock drums, distorted electric guitar, bass guitar, powerful energy"
        elif style.lower() == "electronic":
            stable_prompt += ", electronic drums, synthesizer, bass synth, dance beat"
        elif style.lower() == "classical":
            stable_prompt += ", orchestral instruments, piano, violin, cello, strings"
        elif style.lower() == "jazz":
            stable_prompt += ", jazz drums, saxophone, piano, double bass, swing rhythm"
        elif style.lower() == "ambient":
            stable_prompt += ", atmospheric sounds, soft instruments, ethereal"
        else:
            stable_prompt += ", with real drums and bass guitar"
        
        # Add professional quality descriptors
        if quality == "high":
            stable_prompt += ", high fidelity, studio quality, professional mixing"
        
        print(f"🎵 Generating with Stability AI Stable Audio: {stable_prompt}")
        print(f"⏱️ Duration: {duration} seconds")
        
        # The correct Stability AI Stable Audio endpoint
        url = "https://api.stability.ai/v2beta/stable-audio/generate/music"
        
        headers = {
            "Authorization": f"Bearer {Config.STABILITY_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Correct payload format for Stable Audio
        payload = {
            "prompt": stable_prompt,
            "duration": min(duration, 47),  # Stable Audio supports up to 47 seconds
            "cfg_scale": 7.0,
            "seed": None
        }
        
        print(f"🔄 Making request to Stability AI Stable Audio...")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"🔍 Stability AI response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Stability AI generation started successfully!")
            
            if "id" in result:
                generation_id = result["id"]
                print(f"🔄 Generation ID: {generation_id}")
                
                # Poll for completion
                music_url = poll_stability_completion(generation_id, headers)
                if music_url:
                    music_filename = download_music_file(music_url, "stability_audio")
                    return music_filename, None
                    
        elif response.status_code == 400:
            error_detail = response.json() if response.content else "Bad request"
            print(f"❌ Bad request: {error_detail}")
            return None, f"Stability AI bad request: {error_detail}"
            
        elif response.status_code == 401:
            print("🔑 Stability AI: Invalid API key - check your token")
            return None, "Invalid Stability AI API key"
            
        elif response.status_code == 402:
            print("💳 Stability AI: Insufficient credits - check your account balance")
            return None, "Insufficient Stability AI credits"
            
        elif response.status_code == 429:
            print("⏱️ Stability AI: Rate limited - too many requests")
            return None, "Stability AI rate limited"
            
        else:
            error_text = response.text[:200] if response.text else "Unknown error"
            print(f"⚠️ Stability AI error {response.status_code}: {error_text}")
            return None, f"Stability AI error {response.status_code}: {error_text}"
        
        return None, "Stability AI generation failed"
        
    except Exception as e:
        print(f"❌ Stability AI error: {e}")
        return None, f"Stability AI error: {str(e)}"
        
        # Test different authentication methods
        auth_methods = [
            {"Authorization": f"Bearer {Config.STABILITY_API_KEY}"},
            {"Authorization": f"Token {Config.STABILITY_API_KEY}"},
            {"API-Key": Config.STABILITY_API_KEY},
            {"X-API-Key": Config.STABILITY_API_KEY},
            {"stability-api-key": Config.STABILITY_API_KEY}
        ]
        
        # Test authentication with account endpoint
        for i, headers in enumerate(auth_methods):
            headers["Content-Type"] = "application/json"
            headers["Accept"] = "application/json"
            
            try:
                print(f"🔄 Testing auth method {i+1}/5...")
                
                # Test with account/balance endpoint (most APIs have this)
                response = requests.get(
                    "https://api.stability.ai/v1/user/account",
                    headers=headers,
                    timeout=10
                )
                
                print(f"🔍 Auth test response: {response.status_code}")
                
                if response.status_code == 200:
                    print("✅ Valid authentication method found!")
                    account_info = response.json()
                    print(f"📊 Account info: {account_info}")
                    
                    # Now test what endpoints are available
                    available_endpoints = [
                        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                        "https://api.stability.ai/v1/engines/list",
                        "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image"
                    ]
                    
                    print("� Checking available Stability AI services...")
                    for endpoint in available_endpoints:
                        test_response = requests.get(endpoint, headers=headers, timeout=5)
                        print(f"📍 {endpoint.split('/')[-1]}: {test_response.status_code}")
                    
                    # Stability AI currently focuses on images, not audio
                    print("📝 Analysis: Stability AI primarily offers image generation")
                    print("� Stable Audio API may not be publicly available yet")
                    print("🔄 Falling back to other music generation services...")
                    
                    return None, "Stability AI audio generation not available - image service only"
                    
                elif response.status_code == 401:
                    print(f"❌ Auth method {i+1}: Invalid credentials")
                elif response.status_code == 403:
                    print(f"❌ Auth method {i+1}: Forbidden - check API permissions")
                else:
                    print(f"⚠️ Auth method {i+1}: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Auth method {i+1} error: {e}")
                continue
        
        print("❌ No valid authentication method found for Stability AI")
        print("💡 Suggestion: Verify API key format and check if your account has required permissions")
        
        return None, "Stability AI authentication failed - check API key format"
        
    except Exception as e:
        print(f"❌ Stability AI error: {e}")
        return None, f"Stability AI error: {str(e)}"
        
        return None, f"Stability AI generation failed: {response.status_code}"
        
    except Exception as e:
        print(f"❌ Stability AI error: {e}")
def save_stability_audio_response(result):
    """Save audio from Stability AI JSON response"""
    try:
        import uuid
        import base64
        
        audio_data = None
        
        # Check different response formats
        if "audio" in result:
            audio_data = result["audio"]
        elif "artifacts" in result and len(result["artifacts"]) > 0:
            artifact = result["artifacts"][0]
            if "base64" in artifact:
                audio_data = artifact["base64"]
        
        if audio_data:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Save to file
            music_id = str(uuid.uuid4())
            music_filename = f"stability_{music_id}.wav"
            music_path = os.path.join(MUSIC_DIR, music_filename)
            
            with open(music_path, "wb") as f:
                f.write(audio_bytes)
            
            print(f"✅ Stability AI music saved: {music_filename}")
            return music_filename
            
    except Exception as e:
        print(f"❌ Error saving Stability AI audio: {e}")
    
    return None

def save_stability_binary_audio(audio_content):
    """Save binary audio from Stability AI response"""
    try:
        import uuid
        
        music_id = str(uuid.uuid4())
        music_filename = f"stability_{music_id}.wav"
        music_path = os.path.join(MUSIC_DIR, music_filename)
        
        with open(music_path, "wb") as f:
            f.write(audio_content)
        
        print(f"✅ Stability AI binary music saved: {music_filename}")
        return music_filename
        
    except Exception as e:
        print(f"❌ Error saving binary audio: {e}")
    
    return None

# ===== VISUAL AI GENERATION FUNCTIONS =====

def generate_ai_avatar(prompt, style="realistic", consistency_seed=None):
    """Generate consistent character avatars using AI"""
    
    try:
        print(f"🎭 Generating AI Avatar: {prompt} (style: {style})")
        
        # Enhanced avatar prompt
        avatar_prompt = f"portrait of {prompt}, {style} style"
        
        if style.lower() == "realistic":
            avatar_prompt += ", professional headshot, high quality, detailed face, studio lighting"
        elif style.lower() == "anime":
            avatar_prompt += ", anime character design, vibrant colors, detailed eyes"
        elif style.lower() == "cartoon":
            avatar_prompt += ", cartoon character, friendly expression, colorful"
        elif style.lower() == "professional":
            avatar_prompt += ", business portrait, professional attire, corporate style"
        
        # Try multiple avatar generation services
        
        # Option 1: Use DALL-E for avatar generation
        if Config.OPENAI_API_KEY:
            print("🎨 Using DALL-E for avatar generation...")
            
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=avatar_prompt,
                    size="1024x1024",
                    quality="hd",
                    n=1,
                    response_format="url"
                )
                
                image_url = response.data[0].url
                
                # Download and save avatar
                image_response = requests.get(image_url, timeout=30)
                if image_response.status_code == 200:
                    import uuid
                    avatar_id = str(uuid.uuid4())
                    avatar_filename = f"avatar_{avatar_id}.png"
                    avatar_path = os.path.join(AVATARS_DIR, avatar_filename)
                    
                    with open(avatar_path, 'wb') as f:
                        f.write(image_response.content)
                    
                    print(f"✅ Avatar generated: {avatar_filename}")
                    return avatar_filename, None
                    
            except Exception as e:
                print(f"⚠️ DALL-E avatar error: {e}")
        
        # Option 2: Use Stability AI for avatar generation
        if Config.STABILITY_API_KEY:
            print("🎭 Using Stability AI for avatar generation...")
            
            try:
                headers = {
                    "Authorization": f"Bearer {Config.STABILITY_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "text_prompts": [{"text": avatar_prompt}],
                    "cfg_scale": 7,
                    "samples": 1,
                    "steps": 50
                }
                
                response = requests.post(
                    "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "artifacts" in data and len(data["artifacts"]) > 0:
                        import base64
                        import uuid
                        
                        image_data = base64.b64decode(data["artifacts"][0]["base64"])
                        avatar_id = str(uuid.uuid4())
                        avatar_filename = f"avatar_stability_{avatar_id}.png"
                        avatar_path = os.path.join(AVATARS_DIR, avatar_filename)
                        
                        with open(avatar_path, 'wb') as f:
                            f.write(image_data)
                        
                        print(f"✅ Stability AI avatar generated: {avatar_filename}")
                        return avatar_filename, None
                        
            except Exception as e:
                print(f"⚠️ Stability AI avatar error: {e}")
        
        return None, "Avatar generation failed - no working APIs"
        
    except Exception as e:
        print(f"❌ Avatar generation error: {e}")
        return None, f"Avatar error: {str(e)}"

def edit_image_background(image_path, action="remove", new_background=None):
    """Edit image backgrounds - remove, replace, or enhance"""
    
    try:
        print(f"🖼️ Image editing: {action} background")
        
        if action == "remove" and Config.REMOVE_BG_API_KEY:
            print("✂️ Removing background with Remove.bg API...")
            
            headers = {
                "X-Api-Key": Config.REMOVE_BG_API_KEY
            }
            
            with open(image_path, 'rb') as f:
                files = {"image_file": f}
                
                response = requests.post(
                    "https://api.remove.bg/v1.0/removebg",
                    headers=headers,
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    import uuid
                    edit_id = str(uuid.uuid4())
                    edited_filename = f"bg_removed_{edit_id}.png"
                    edited_path = os.path.join(DESIGNS_DIR, edited_filename)
                    
                    with open(edited_path, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"✅ Background removed: {edited_filename}")
                    return edited_filename, None
        
        # Fallback to OpenCV background removal
        print("🔄 Using OpenCV for background processing...")
        
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            # Load image
            img = cv2.imread(image_path)
            
            if action == "remove":
                # Simple background removal using grabcut
                height, width = img.shape[:2]
                mask = np.zeros((height, width), np.uint8)
                
                # Create rectangle for foreground
                rect = (50, 50, width-50, height-50)
                
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                result = img * mask2[:, :, np.newaxis]
                
                # Convert to RGBA for transparency
                result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
                result_rgba[:, :, 3] = mask2 * 255
                
                import uuid
                edit_id = str(uuid.uuid4())
                edited_filename = f"bg_removed_cv_{edit_id}.png"
                edited_path = os.path.join(DESIGNS_DIR, edited_filename)
                
                cv2.imwrite(edited_path, result_rgba)
                
                print(f"✅ Background removed with OpenCV: {edited_filename}")
                return edited_filename, None
                
        except Exception as e:
            print(f"⚠️ OpenCV processing error: {e}")
        
        return None, "Background editing failed"
        
    except Exception as e:
        print(f"❌ Image editing error: {e}")
        return None, f"Image editing error: {str(e)}"

def generate_3d_model(prompt, style="realistic"):
    """Generate 3D models from text descriptions"""
    
    try:
        print(f"🗿 Generating 3D model: {prompt} (style: {style})")
        
        # Enhanced 3D model prompt
        model_prompt = f"3D model of {prompt}, {style} style"
        
        if style.lower() == "realistic":
            model_prompt += ", high detail, photorealistic textures, professional quality"
        elif style.lower() == "lowpoly":
            model_prompt += ", low polygon count, game-ready, clean geometry"
        elif style.lower() == "stylized":
            model_prompt += ", artistic style, creative design, unique aesthetic"
        
        # Try Tripo API for 3D generation
        if Config.TRIPO_API_KEY:
            print("🔮 Using Tripo API for 3D model generation...")
            
            try:
                headers = {
                    "Authorization": f"Bearer {Config.TRIPO_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "prompt": model_prompt,
                    "style": style,
                    "quality": "high"
                }
                
                response = requests.post(
                    "https://api.tripo3d.ai/v1/text-to-3d",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "model_url" in result:
                        # Download 3D model
                        model_response = requests.get(result["model_url"], timeout=60)
                        
                        if model_response.status_code == 200:
                            import uuid
                            model_id = str(uuid.uuid4())
                            model_filename = f"model_3d_{model_id}.obj"
                            model_path = os.path.join(MODELS_3D_DIR, model_filename)
                            
                            with open(model_path, 'wb') as f:
                                f.write(model_response.content)
                            
                            print(f"✅ 3D model generated: {model_filename}")
                            return model_filename, None
                            
            except Exception as e:
                print(f"⚠️ Tripo API error: {e}")
        
        # Try Meshy API for 3D generation
        if Config.MESHY_API_KEY:
            print("🎯 Using Meshy API for 3D model generation...")
            
            try:
                headers = {
                    "Authorization": f"Bearer {Config.MESHY_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "text": model_prompt,
                    "mode": "text-to-3d",
                    "art_style": style
                }
                
                response = requests.post(
                    "https://api.meshy.ai/v1/text-to-3d",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Meshy 3D generation started: {result}")
                    return None, "3D model generation started - check back in a few minutes"
                    
            except Exception as e:
                print(f"⚠️ Meshy API error: {e}")
        
        return None, "3D model generation not available - API keys needed"
        
    except Exception as e:
        print(f"❌ 3D model generation error: {e}")
        return None, f"3D generation error: {str(e)}"

def generate_logo_design(brand_name, industry, style="modern"):
    """Generate logos and brand designs"""
    
    try:
        print(f"🏷️ Generating logo for: {brand_name} ({industry}, {style})")
        
        # Enhanced logo prompt
        logo_prompt = f"professional logo design for {brand_name}, {industry} industry, {style} style"
        
        if style.lower() == "modern":
            logo_prompt += ", clean lines, minimalist, contemporary design"
        elif style.lower() == "vintage":
            logo_prompt += ", retro aesthetic, classic typography, timeless design"
        elif style.lower() == "creative":
            logo_prompt += ", artistic flair, unique concept, innovative design"
        elif style.lower() == "corporate":
            logo_prompt += ", professional appearance, trustworthy, business-oriented"
        
        logo_prompt += ", vector style, high contrast, suitable for business use"
        
        # Try DALL-E for logo generation
        if Config.OPENAI_API_KEY:
            print("🎨 Using DALL-E for logo design...")
            
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=logo_prompt,
                    size="1024x1024",
                    quality="hd",
                    n=1,
                    response_format="url"
                )
                
                image_url = response.data[0].url
                
                # Download and save logo
                image_response = requests.get(image_url, timeout=30)
                if image_response.status_code == 200:
                    import uuid
                    logo_id = str(uuid.uuid4())
                    logo_filename = f"logo_{brand_name}_{logo_id}.png"
                    logo_path = os.path.join(LOGOS_DIR, logo_filename)
                    
                    with open(logo_path, 'wb') as f:
                        f.write(image_response.content)
                    
                    print(f"✅ Logo generated: {logo_filename}")
                    return logo_filename, None
                    
            except Exception as e:
                print(f"⚠️ DALL-E logo error: {e}")
        
        # Try Stability AI for logo generation
        if Config.STABILITY_API_KEY:
            print("🎭 Using Stability AI for logo design...")
            
            try:
                headers = {
                    "Authorization": f"Bearer {Config.STABILITY_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "text_prompts": [{"text": logo_prompt}],
                    "cfg_scale": 7,
                    "samples": 1,
                    "steps": 50
                }
                
                response = requests.post(
                    "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "artifacts" in data and len(data["artifacts"]) > 0:
                        import base64
                        import uuid
                        
                        image_data = base64.b64decode(data["artifacts"][0]["base64"])
                        logo_id = str(uuid.uuid4())
                        logo_filename = f"logo_stability_{brand_name}_{logo_id}.png"
                        logo_path = os.path.join(LOGOS_DIR, logo_filename)
                        
                        with open(logo_path, 'wb') as f:
                            f.write(image_data)
                        
                        print(f"✅ Stability AI logo generated: {logo_filename}")
                        return logo_filename, None
                        
            except Exception as e:
                print(f"⚠️ Stability AI logo error: {e}")
        
        return None, "Logo generation failed - no working APIs"
        
    except Exception as e:
        print(f"❌ Logo generation error: {e}")
        return None, f"Logo error: {str(e)}"

def upscale_image(image_path, scale_factor=2):
    """Upscale images using AI"""
    
    try:
        print(f"📈 Upscaling image by {scale_factor}x...")
        
        # Try AI upscaling services first
        if Config.UPSCAYL_API_KEY:
            print("🚀 Using AI upscaling service...")
            
            try:
                headers = {
                    "Authorization": f"Bearer {Config.UPSCAYL_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                with open(image_path, 'rb') as f:
                    files = {"image": f}
                    data = {"scale": scale_factor}
                    
                    response = requests.post(
                        "https://api.upscayl.com/v1/upscale",
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        import uuid
                        upscale_id = str(uuid.uuid4())
                        upscaled_filename = f"upscaled_{scale_factor}x_{upscale_id}.png"
                        upscaled_path = os.path.join(DESIGNS_DIR, upscaled_filename)
                        
                        with open(upscaled_path, 'wb') as f:
                            f.write(response.content)
                        
                        print(f"✅ Image upscaled: {upscaled_filename}")
                        return upscaled_filename, None
                        
            except Exception as e:
                print(f"⚠️ AI upscaling error: {e}")
        
        # Fallback to OpenCV upscaling
        print("🔄 Using OpenCV for image upscaling...")
        
        try:
            import cv2
            
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Use INTER_CUBIC for better quality upscaling
            upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            import uuid
            upscale_id = str(uuid.uuid4())
            upscaled_filename = f"upscaled_cv_{scale_factor}x_{upscale_id}.png"
            upscaled_path = os.path.join(DESIGNS_DIR, upscaled_filename)
            
            cv2.imwrite(upscaled_path, upscaled)
            
            print(f"✅ Image upscaled with OpenCV: {upscaled_filename}")
            return upscaled_filename, None
            
        except Exception as e:
            print(f"⚠️ OpenCV upscaling error: {e}")
        
        return None, "Image upscaling failed"
        
    except Exception as e:
        print(f"❌ Image upscaling error: {e}")
        return None, f"Upscaling error: {str(e)}"

def generate_huggingface_music(prompt, duration, style, quality):
    """Generate music using Hugging Face MusicGen"""
    
    try:
        print("🤗 Using Hugging Face MusicGen for AI music generation...")
        
        # Enhanced prompt for Hugging Face
        hf_prompt = f"{style} style: {prompt}"
        
        headers = {
            "Authorization": f"Bearer {Config.HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Use Facebook's MusicGen model
        model = "facebook/musicgen-large" if quality == "high" else "facebook/musicgen-medium"
        
        generation_data = {
            "inputs": hf_prompt,
            "parameters": {
                "max_new_tokens": min(duration * 50, 1500),  # Approximate tokens per second
                "temperature": 0.9,
                "do_sample": True
            }
        }
        
        print(f"🎵 Generating with Hugging Face: {hf_prompt}")
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=generation_data
        )
        
        if response.status_code == 200:
            # Hugging Face returns audio data directly
            import uuid
            music_id = str(uuid.uuid4())
            music_filename = f"huggingface_{music_id}.wav"
            music_path = os.path.join(MUSIC_DIR, music_filename)
            
            with open(music_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Hugging Face music saved: {music_filename}")
            return music_filename, None
        else:
            print(f"⚠️ Hugging Face error: {response.status_code} - {response.text}")
        
        return None, f"Hugging Face generation failed: {response.status_code}"
        
    except Exception as e:
        print(f"❌ Hugging Face error: {e}")
        return None, f"Hugging Face error: {str(e)}"

def poll_replicate_completion(prediction_id, headers, max_wait=300):
    """Poll Replicate for completion"""
    import time
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                
                print(f"🔄 Replicate status: {status}")
                
                if status == "succeeded":
                    output = data.get("output")
                    if output:
                        print("✅ Replicate generation completed!")
                        return output
                elif status == "failed":
                    error = data.get("error", "Unknown error")
                    print(f"❌ Replicate generation failed: {error}")
                    return None
                    
            time.sleep(10)
            
        except Exception as e:
            print(f"⚠️ Error polling Replicate: {e}")
            time.sleep(5)
    
    print("⏰ Replicate generation timed out")
    return None

def poll_stability_completion(generation_id, headers, max_wait=300):
    """Poll Stability AI for completion"""
    import time
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"https://api.stability.ai/v2beta/stable-audio/generate/{generation_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                # Check if it's JSON (status) or binary (audio)
                content_type = response.headers.get('content-type', '')
                
                if 'application/json' in content_type:
                    data = response.json()
                    status = data.get("status", "unknown")
                    print(f"🔄 Stability AI status: {status}")
                    
                    if status == "failed":
                        print("❌ Stability AI generation failed")
                        return None
                        
                elif 'audio' in content_type:
                    # Audio file is ready
                    print("✅ Stability AI generation completed!")
                    
                    # Save the audio directly
                    import uuid
                    music_id = str(uuid.uuid4())
                    music_filename = f"stability_{music_id}.wav"
                    music_path = os.path.join(MUSIC_DIR, music_filename)
                    
                    with open(music_path, 'wb') as f:
                        f.write(response.content)
                    
                    return music_filename  # Return filename instead of URL
                    
            time.sleep(10)
            
        except Exception as e:
            print(f"⚠️ Error polling Stability AI: {e}")
            time.sleep(5)
    
    print("⏰ Stability AI generation timed out")
    return None
    """Generate music using Suno AI API"""
    
    try:
        print("🎭 Using Suno AI for professional music generation...")
        
        # For now, Suno AI requires web interface access
        # The API key you provided might be for web interface access
        print("📝 Note: Suno AI currently requires web interface for generation")
        print("🔄 Falling back to enhanced music generation...")
        
        # Fall back to enhanced music generation
        return generate_enhanced_music(prompt, duration, style, quality)
        
    except Exception as e:
        print(f"❌ Suno AI error: {e}")
        return None, f"Suno AI error: {str(e)}"

def generate_enhanced_music(prompt, duration, style, quality):
    """Generate enhanced music with multiple layers and realistic instruments"""
    
    try:
        print("🎼 Creating enhanced multi-layered music...")
        
        if not AUDIO_FEATURES_AVAILABLE:
            return None, "Audio libraries not available"
        
        # Create a more sophisticated synthesized track
        sample_rate = 44100
        total_samples = int(duration * sample_rate)
        
        import numpy as np
        import math
        
        # Time array
        t = np.linspace(0, duration, total_samples)
        
        # Initialize final audio
        final_audio = np.zeros(total_samples)
        
        # Style-specific music generation
        if style.lower() in ['pop', 'dance', 'electronic']:
            # Pop/Electronic style with multiple layers
            
            # Bass line (low frequency)
            bass_freq = 65.4  # C2
            bass_pattern = np.sin(2 * np.pi * bass_freq * t)
            bass_pattern += 0.5 * np.sin(2 * np.pi * bass_freq * 1.5 * t)  # Fifth
            bass_envelope = np.where(np.sin(2 * np.pi * 2 * t) > 0, 0.4, 0.1)  # Pumping bass
            bass_line = bass_pattern * bass_envelope
            
            # Lead melody
            melody_freqs = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]  # C major scale
            melody = np.zeros(total_samples)
            note_duration = sample_rate // 4  # Quarter note
            
            for i, freq in enumerate(melody_freqs * 10):  # Repeat pattern
                start_idx = (i * note_duration) % total_samples
                end_idx = min(start_idx + note_duration, total_samples)
                if start_idx < total_samples:
                    note_t = t[start_idx:end_idx] - t[start_idx]
                    note_wave = np.sin(2 * np.pi * freq * note_t) * np.exp(-note_t * 2)  # Decay
                    melody[start_idx:end_idx] += note_wave * 0.3
            
            # Drum-like percussion
            kick_freq = 60
            kick_times = np.arange(0, duration, 0.5)  # Every half second
            percussion = np.zeros(total_samples)
            for kick_time in kick_times:
                kick_idx = int(kick_time * sample_rate)
                if kick_idx < total_samples - 1000:
                    kick_wave = np.sin(2 * np.pi * kick_freq * t[kick_idx:kick_idx+1000]) * np.exp(-t[:1000] * 10)
                    percussion[kick_idx:kick_idx+1000] += kick_wave * 0.2
            
            # Hi-hat like sound
            hihat_noise = np.random.normal(0, 0.1, total_samples)
            hihat_envelope = np.zeros(total_samples)
            hihat_times = np.arange(0.25, duration, 0.25)  # Offbeat
            for hihat_time in hihat_times:
                hihat_idx = int(hihat_time * sample_rate)
                if hihat_idx < total_samples - 500:
                    hihat_envelope[hihat_idx:hihat_idx+500] = np.exp(-t[:500] * 20) * 0.1
            
            hihat = hihat_noise * hihat_envelope
            
            # Combine all elements
            final_audio = bass_line + melody + percussion + hihat
            
        elif style.lower() in ['classical', 'piano', 'ambient']:
            # Classical/Piano style
            
            # Piano-like melody with harmonics
            root_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C major
            
            for i, freq in enumerate(root_freqs):
                # Create chord progression
                chord_start = (i * duration / len(root_freqs))
                chord_duration = duration / len(root_freqs)
                
                chord_start_idx = int(chord_start * sample_rate)
                chord_end_idx = int((chord_start + chord_duration) * sample_rate)
                
                if chord_start_idx < total_samples:
                    chord_end_idx = min(chord_end_idx, total_samples)
                    chord_t = t[chord_start_idx:chord_end_idx] - chord_start
                    
                    # Root note
                    root_note = np.sin(2 * np.pi * freq * chord_t) * np.exp(-chord_t * 0.5)
                    # Third (major)
                    third_note = np.sin(2 * np.pi * freq * 1.25 * chord_t) * np.exp(-chord_t * 0.5) * 0.7
                    # Fifth
                    fifth_note = np.sin(2 * np.pi * freq * 1.5 * chord_t) * np.exp(-chord_t * 0.5) * 0.5
                    
                    chord = (root_note + third_note + fifth_note) * 0.2
                    final_audio[chord_start_idx:chord_end_idx] += chord
            
        elif style.lower() in ['rock', 'metal']:
            # Rock style with distorted elements
            
            # Power chord progression
            power_chord_freqs = [82.41, 87.31, 98.00, 110.00]  # E, F, G, A
            
            for i, freq in enumerate(power_chord_freqs * 4):
                chord_start = (i * duration / 16)
                chord_duration = duration / 16
                
                chord_start_idx = int(chord_start * sample_rate)
                chord_end_idx = int((chord_start + chord_duration) * sample_rate)
                
                if chord_start_idx < total_samples:
                    chord_end_idx = min(chord_end_idx, total_samples)
                    chord_t = t[chord_start_idx:chord_end_idx] - chord_start
                    
                    # Distorted guitar-like sound
                    guitar_wave = np.sin(2 * np.pi * freq * chord_t)
                    guitar_wave += 0.3 * np.sin(2 * np.pi * freq * 2 * chord_t)  # Octave
                    guitar_wave = np.tanh(guitar_wave * 3) * 0.4  # Distortion
                    
                    final_audio[chord_start_idx:chord_end_idx] += guitar_wave
            
            # Rock drums
            kick_pattern = np.arange(0, duration, 1)  # Every beat
            for kick_time in kick_pattern:
                kick_idx = int(kick_time * sample_rate)
                if kick_idx < total_samples - 2000:
                    kick_wave = np.sin(2 * np.pi * 50 * t[kick_idx:kick_idx+2000]) * np.exp(-t[:2000] * 5)
                    final_audio[kick_idx:kick_idx+2000] += kick_wave * 0.3
        
        else:
            # Default ambient style
            ambient_freqs = [130.81, 146.83, 164.81, 174.61, 196.00, 220.00]  # C3 major
            
            for i, freq in enumerate(ambient_freqs):
                phase = i * np.pi / 3
                wave = np.sin(2 * np.pi * freq * t + phase) * np.sin(2 * np.pi * 0.1 * t)  # Slow modulation
                final_audio += wave * (0.15 / len(ambient_freqs))
        
        # Apply overall envelope for smooth start/end
        envelope = np.ones(total_samples)
        fade_samples = sample_rate // 10  # 0.1 second fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        final_audio *= envelope
        
        # Normalize audio
        max_val = np.max(np.abs(final_audio))
        if max_val > 0:
            final_audio = final_audio / max_val * 0.8  # Leave some headroom
        
        # Convert to 16-bit integer
        audio_data = (final_audio * 32767).astype(np.int16)
        
        import uuid
        music_id = str(uuid.uuid4())
        music_filename = f"enhanced_{music_id}.wav"
        music_path = os.path.join(MUSIC_DIR, music_filename)
        
        # Save using wave format
        try:
            import wave
            with wave.open(music_path, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            print(f"✅ Enhanced {style} music saved: {music_filename}")
            return music_filename, None
            
        except Exception as e:
            print(f"❌ Error saving enhanced music: {e}")
            return None, f"Failed to save music: {str(e)}"
        
    except Exception as e:
        print(f"❌ Enhanced music generation error: {e}")
        return None, f"Enhanced music error: {str(e)}"

def generate_musicgen_music(prompt, duration, style, quality):
    """Generate music using MusicGen API"""
    
    try:
        print("🎼 Using MusicGen for AI music composition...")
        
        # MusicGen API integration
        headers = {
            "Authorization": f"Bearer {Config.MUSICGEN_API_KEY}",
            "Content-Type": "application/json"
        }
        
        musicgen_prompt = f"Generate {style} music: {prompt}"
        
        generation_data = {
            "prompt": musicgen_prompt,
            "duration": duration,
            "model": "musicgen-large" if quality == "high" else "musicgen-medium",
            "format": "mp3"
        }
        
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json={
                "version": "7a76a8258b23fae65c5a22debb8841d1d7e816b75c2f24218cd2bd8573787906",
                "input": generation_data
            }
        )
        
        if response.status_code == 201:
            prediction_id = response.json()["id"]
            music_url = poll_musicgen_completion(prediction_id, headers)
            if music_url:
                music_filename = download_music_file(music_url, "musicgen")
                return music_filename, None
        
        return None, "MusicGen generation failed"
        
    except Exception as e:
        print(f"❌ MusicGen error: {e}")
        return None, f"MusicGen error: {str(e)}"

def generate_synthesized_music(prompt, duration, style, quality):
    """Generate synthesized music as fallback"""
    
    try:
        print("🎹 Generating synthesized music as fallback...")
        
        if not AUDIO_FEATURES_AVAILABLE:
            return None, "Audio libraries not available"
        
        # Create a simple synthesized track
        sample_rate = 44100
        total_samples = int(duration * sample_rate)
        
        # Generate different tones based on style
        import numpy as np
        
        if style.lower() in ['pop', 'dance', 'electronic']:
            # Upbeat electronic-style synth
            base_freq = 440  # A4
            t = np.linspace(0, duration, total_samples)
            
            # Main melody
            melody = np.sin(2 * np.pi * base_freq * t) * 0.3
            # Bass line
            bass = np.sin(2 * np.pi * base_freq/2 * t) * 0.2
            # High harmonics
            harmony = np.sin(2 * np.pi * base_freq * 2 * t) * 0.1
            
            audio_data = melody + bass + harmony
            
        elif style.lower() in ['classical', 'piano', 'ambient']:
            # Softer, more melodic
            t = np.linspace(0, duration, total_samples)
            frequencies = [261.63, 293.66, 329.63, 349.23, 392.00]  # C major scale
            
            audio_data = np.zeros(total_samples)
            for i, freq in enumerate(frequencies):
                phase_shift = i * np.pi / 4
                audio_data += np.sin(2 * np.pi * freq * t + phase_shift) * (0.2 / len(frequencies))
        
        else:
            # Default gentle melody
            t = np.linspace(0, duration, total_samples)
            audio_data = np.sin(2 * np.pi * 440 * t) * 0.3
        
        # Apply envelope for smoother sound
        envelope = np.exp(-t / (duration * 0.3))  # Decay envelope
        audio_data *= envelope
        
        # Convert to audio format and save
        audio_data = (audio_data * 32767).astype(np.int16)
        
        import uuid
        music_id = str(uuid.uuid4())
        music_filename = f"synth_{music_id}.wav"
        music_path = os.path.join(MUSIC_DIR, music_filename)
        
        # Save using scipy or simple wave format
        try:
            import wave
            with wave.open(music_path, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            print(f"✅ Synthesized music saved: {music_filename}")
            return music_filename, None
            
        except Exception as e:
            print(f"❌ Error saving synthesized music: {e}")
            return None, f"Failed to save music: {str(e)}"
        
    except Exception as e:
        print(f"❌ Synthesized music error: {e}")
        return None, f"Synthesized music error: {str(e)}"

def poll_suno_completion(task_id, headers, max_wait=300):
    """Poll Suno AI for completion"""
    import time
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"https://api.suno.ai/v1/tasks/{task_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "completed":
                    return data.get("output_url")
                elif data.get("status") == "failed":
                    print("❌ Suno AI generation failed")
                    return None
                    
            time.sleep(10)
            
        except Exception as e:
            print(f"⚠️ Error polling Suno: {e}")
            time.sleep(5)
    
    return None

def poll_musicgen_completion(prediction_id, headers, max_wait=300):
    """Poll MusicGen for completion"""
    import time
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "succeeded":
                    return data.get("output")
                elif data.get("status") == "failed":
                    print("❌ MusicGen generation failed")
                    return None
                    
            time.sleep(10)
            
        except Exception as e:
            print(f"⚠️ Error polling MusicGen: {e}")
            time.sleep(5)
    
    return None

def download_music_file(music_url, service_name):
    """Download generated music file"""
    
    try:
        import uuid
        
        response = requests.get(music_url)
        if response.status_code == 200:
            music_id = str(uuid.uuid4())
            music_filename = f"{service_name}_{music_id}.mp3"
            music_path = os.path.join(MUSIC_DIR, music_filename)
            
            with open(music_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Music downloaded: {music_filename}")
            return music_filename
        else:
            print(f"❌ Failed to download music: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading music: {e}")
        return None

def generate_voice_audio(text, voice_style="alloy", quality="standard"):
    """Generate voice audio using ElevenLabs or OpenAI TTS"""
    
    print(f"🗣️ Generating voice audio: '{text[:50]}...'")
    
    # Try ElevenLabs first if available
    if ELEVENLABS_AVAILABLE and Config.ELEVENLABS_API_KEY:
        return generate_elevenlabs_voice(text, voice_style, quality)
    
    # Fallback to OpenAI TTS
    if Config.OPENAI_API_KEY:
        return generate_openai_voice(text, voice_style, quality)
    
    return None, "No voice generation services available"

def generate_elevenlabs_voice(text, voice_style, quality):
    """Generate voice using ElevenLabs API"""
    
    try:
        print("🎤 Using ElevenLabs for premium voice synthesis...")
        
        import elevenlabs
        elevenlabs.set_api_key(Config.ELEVENLABS_API_KEY)
        
        # Voice style mapping
        voice_map = {
            "alloy": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "echo": "ErXwobaYiN019PkySvjV",   # Antoni  
            "fable": "MF3mGyEYCl7XYWbV9V6O",  # Elli
            "onyx": "Yko7PKHZNXotIFUBG7I9",   # Sam
            "nova": "pNInz6obpgDQGcFmaJgB",   # Adam
            "shimmer": "Xb7hH8MSUJpSbSDYk0k2" # Alice
        }
        
        voice_id = voice_map.get(voice_style, voice_map["alloy"])
        
        # Generate audio
        audio = elevenlabs.generate(
            text=text,
            voice=voice_id,
            model="eleven_multilingual_v2" if quality == "high" else "eleven_monolingual_v1"
        )
        
        # Save audio file
        import uuid
        audio_id = str(uuid.uuid4())
        audio_filename = f"elevenlabs_{audio_id}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        elevenlabs.save(audio, audio_path)
        
        print(f"✅ ElevenLabs voice generated: {audio_filename}")
        return audio_filename, None
        
    except Exception as e:
        print(f"❌ ElevenLabs error: {e}")
        return None, f"ElevenLabs error: {str(e)}"

def generate_openai_voice(text, voice_style, quality):
    """Generate voice using OpenAI TTS"""
    
    try:
        print("🤖 Using OpenAI TTS for voice synthesis...")
        
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Generate audio
        response = client.audio.speech.create(
            model="tts-1-hd" if quality == "high" else "tts-1",
            voice=voice_style,
            input=text
        )
        
        # Save audio file
        import uuid
        audio_id = str(uuid.uuid4())
        audio_filename = f"openai_tts_{audio_id}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        response.stream_to_file(audio_path)
        
        print(f"✅ OpenAI TTS generated: {audio_filename}")
        return audio_filename, None
        
    except Exception as e:
        print(f"❌ OpenAI TTS error: {e}")
        return None, f"OpenAI TTS error: {str(e)}"

def transcribe_audio(audio_file_path):
    """Transcribe audio to text using speech recognition"""
    
    if not AUDIO_FEATURES_AVAILABLE:
        return None, "Audio features not available"
    
    try:
        print("🎤 Transcribing audio to text...")
        
        r = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(audio_file_path) as source:
            audio_data = r.record(source)
        
        # Try multiple recognition services
        transcription_methods = [
            ("OpenAI Whisper", lambda: r.recognize_whisper_api(audio_data, api_key=Config.OPENAI_API_KEY)),
            ("Google Speech", lambda: r.recognize_google(audio_data)),
            ("Sphinx (offline)", lambda: r.recognize_sphinx(audio_data))
        ]
        
        for method_name, method_func in transcription_methods:
            try:
                print(f"🔄 Trying {method_name}...")
                text = method_func()
                print(f"✅ Transcription successful with {method_name}")
                return text, None
            except Exception as e:
                print(f"⚠️ {method_name} failed: {e}")
                continue
        
        return None, "All transcription methods failed"
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None, f"Transcription error: {str(e)}"

# �🎥 AI VIDEO GENERATION FUNCTIONS
# ===============================================

def generate_text_video(text_prompt, duration=5, fps=30, quality="high", method="auto"):
    """Enhanced video generation with hybrid DALL-E + Runway ML support"""
    
    # Check if Runway ML is available
    runway_available = bool(Config.RUNWAY_API_KEY)
    
    # Determine best method
    if method == "auto":
        chosen_method = determine_best_video_method(text_prompt, runway_available)
    else:
        chosen_method = method
    
    # Validate method availability
    if chosen_method == 'runway' and not runway_available:
        print("⚠️ Runway ML not available, falling back to DALL-E")
        chosen_method = 'dalle'
    
    print(f"🎬 Generating video using {chosen_method.upper()} method...")
    print(f"   📝 Prompt: {text_prompt}")
    print(f"   ⚙️ Quality: {quality}")
    
    # Generate video based on chosen method
    if chosen_method == 'runway' and runway_available:
        return generate_runway_video(text_prompt, duration, quality)
    else:
        # Call existing DALL-E function and ensure we return a tuple
        result = generate_dalle_only_video(text_prompt, duration, fps, quality)
        if isinstance(result, tuple):
            return result
        else:
            # If it returns just a filename, create tuple
            return (result, None) if result else (None, "Failed to generate video")

def determine_best_video_method(prompt, runway_available):
    """Intelligently choose between DALL-E and Runway based on prompt"""
    
    if not runway_available:
        return 'dalle'
    
    # Analyze prompt for cinematic keywords
    cinematic_keywords = [
        'cinematic', 'movie', 'film', 'dramatic', 'realistic', 'professional',
        'motion', 'movement', 'action', 'flowing', 'dynamic', 'smooth',
        'hollywood', 'epic', 'scene', 'sequence', 'camera', 'shot'
    ]
    
    prompt_lower = prompt.lower()
    
    # Check for cinematic keywords
    if any(keyword in prompt_lower for keyword in cinematic_keywords):
        return 'runway'
    
    # Check for complex scenes that benefit from true video
    complex_keywords = [
        'dancing', 'running', 'flying', 'swimming', 'walking', 'moving',
        'jumping', 'rotating', 'spinning', 'flowing', 'waves', 'fire',
        'smoke', 'water', 'wind', 'explosion', 'growing', 'transforming'
    ]
    
    if any(keyword in prompt_lower for keyword in complex_keywords):
        return 'runway'
    
    # Default to DALL-E for static or simple scenes
    return 'dalle'

def generate_runway_video(prompt, duration=5, quality="high"):
    """Generate video using Runway ML for cinematic quality"""
    
    try:
        import requests
        import time
        import uuid
        
        # Enhanced prompt for Runway ML
        runway_prompt = enhance_prompt_for_runway(prompt, quality)
        
        # Runway ML API call
        headers = {
            "Authorization": f"Bearer {Config.RUNWAY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Video generation request
        generation_data = {
            "text_prompt": runway_prompt,
            "duration": min(duration, 10),  # Runway has limits
            "ratio": "16:9",
            "watermark": False,
            "enhance_prompt": True,
            "seed": None  # Random seed for variety
        }
        
        print(f"🎭 Runway ML: Creating cinematic video...")
        print(f"📝 Enhanced prompt: {runway_prompt}")
        
        # Start generation
        response = requests.post(
            "https://api.runwayml.com/v1/generate",
            headers=headers,
            json=generation_data
        )
        
        if response.status_code == 200:
            task_id = response.json()["id"]
            print(f"🔄 Runway generation started (ID: {task_id})")
            
            # Poll for completion (simplified version)
            max_wait = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                try:
                    status_response = requests.get(
                        f"https://api.runwayml.com/v1/tasks/{task_id}",
                        headers=headers
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get("status")
                        
                        if status == "SUCCEEDED":
                            video_url = status_data.get("output", {}).get("video_url")
                            print("✅ Runway ML generation completed!")
                            
                            # Download video
                            video_filename = download_runway_video(video_url)
                            if video_filename:
                                return video_filename, None
                            else:
                                return None, "Failed to download Runway video"
                                
                        elif status == "FAILED":
                            error = status_data.get("error", "Unknown error")
                            print(f"❌ Runway ML generation failed: {error}")
                            break
                        else:
                            print(f"🔄 Runway ML status: {status}...")
                            time.sleep(10)
                    else:
                        print(f"⚠️ Status check failed: {status_response.status_code}")
                        time.sleep(5)
                        
                except Exception as e:
                    print(f"⚠️ Error checking status: {e}")
                    time.sleep(5)
            
            print("⏰ Runway ML generation timed out")
            
        else:
            error_msg = response.json().get('error', 'Unknown Runway error')
            print(f"❌ Runway ML error: {error_msg}")
            
    except Exception as e:
        print(f"❌ Runway ML generation failed: {e}")
    
    # Fallback to DALL-E
    print("🔄 Falling back to DALL-E system...")
    return generate_dalle_only_video(prompt, duration, 30, quality)

def enhance_prompt_for_runway(prompt, quality):
    """Enhance prompt specifically for Runway ML"""
    
    quality_enhancers = {
        "quick": "simple, clean",
        "standard": "detailed, good lighting",
        "high": "cinematic, professional lighting, high detail, 4K quality",
        "ultra": "cinematic masterpiece, dramatic lighting, ultra-detailed, professional cinematography, epic scene"
    }
    
    enhancer = quality_enhancers.get(quality, quality_enhancers["high"])
    
    # Add cinematic elements
    enhanced = f"{prompt}, {enhancer}, smooth motion, realistic movement"
    
    # Add style hints based on content
    if any(word in prompt.lower() for word in ['nature', 'landscape', 'outdoor']):
        enhanced += ", natural lighting, outdoor cinematography"
    elif any(word in prompt.lower() for word in ['person', 'people', 'character']):
        enhanced += ", portrait lighting, character focus"
    elif any(word in prompt.lower() for word in ['abstract', 'artistic']):
        enhanced += ", artistic style, creative cinematography"
    
    return enhanced

def download_runway_video(video_url):
    """Download Runway ML generated video"""
    
    try:
        import requests
        import uuid
        import os
        
        video_response = requests.get(video_url)
        
        if video_response.status_code == 200:
            # Generate unique filename
            video_id = str(uuid.uuid4())
            video_filename = f"runway_{video_id}.mp4"
            
            # Ensure videos directory exists
            videos_dir = "static/generated_videos"
            os.makedirs(videos_dir, exist_ok=True)
            
            video_path = os.path.join(videos_dir, video_filename)
            
            # Save video file
            with open(video_path, 'wb') as f:
                f.write(video_response.content)
            
            print(f"✅ Runway video saved: {video_filename}")
            return video_filename
        else:
            print(f"❌ Failed to download video: {video_response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading video: {e}")
        return None

def generate_dalle_only_video(text_prompt, duration=5, fps=30, quality="high"):
    """Generate a high-quality animated text video with DALL-E enhanced visuals"""
    if not VIDEO_FEATURES_AVAILABLE:
        return None, "Video features not available. Please install required packages."
    
    try:
        import math  # Import math for advanced calculations
        import requests
        
        # Enhanced quality presets with longer durations and DALL-E integration
        quality_settings = {
            "quick": {"width": 512, "height": 512, "fps": 10, "duration": 3, "dalle_frames": 1},
            "standard": {"width": 512, "height": 512, "fps": 15, "duration": 5, "dalle_frames": 2},
            "high": {"width": 1024, "height": 1024, "fps": 20, "duration": 7, "dalle_frames": 3},
            "ultra": {"width": 1024, "height": 1024, "fps": 24, "duration": 10, "dalle_frames": 4}
        }
        
        # Apply quality settings
        if quality in quality_settings:
            settings = quality_settings[quality]
            width, height = settings["width"], settings["height"]
            fps = settings["fps"]
            duration = settings["duration"]
            dalle_frame_count = settings["dalle_frames"]
        else:
            width, height = 1024, 1024
            dalle_frame_count = 2
        
        frames = []
        total_frames = int(duration * fps)
        
        # Generate DALL-E images for key frames
        print(f"🎨 Generating {dalle_frame_count} DALL-E images for video enhancement...")
        dalle_images = []
        
        for i in range(dalle_frame_count):
            # Create variations of the prompt for different scenes
            if dalle_frame_count == 1:
                dalle_prompt = f"High quality cinematic shot of {text_prompt}, vibrant colors, detailed, 4K"
            else:
                scene_variations = [
                    f"Cinematic wide shot of {text_prompt}, bright lighting, detailed",
                    f"Close-up view of {text_prompt}, dramatic lighting, high detail",
                    f"Dynamic action shot of {text_prompt}, motion blur, cinematic",
                    f"Artistic view of {text_prompt}, beautiful composition, vibrant colors"
                ]
                dalle_prompt = scene_variations[i % len(scene_variations)]
            
            print(f"🖼️ Generating DALL-E image {i+1}/{dalle_frame_count}: {dalle_prompt[:50]}...")
            
            # Generate DALL-E image
            try:
                response = openai.images.generate(
                    model="dall-e-3",
                    prompt=dalle_prompt,
                    size=f"{width}x{height}",
                    quality="standard",
                    n=1,
                )
                
                image_url = response.data[0].url
                
                # Download and save the image
                img_response = requests.get(image_url)
                dalle_image = Image.open(io.BytesIO(img_response.content))
                dalle_image = dalle_image.resize((width, height), Image.Resampling.LANCZOS)
                dalle_images.append(dalle_image)
                print(f"✅ DALL-E image {i+1} generated successfully")
                
            except Exception as e:
                print(f"⚠️ Error generating DALL-E image {i+1}: {e}")
                # Fallback to themed content
                fallback_frame = generate_themed_content(text_prompt, width, height, i, dalle_frame_count)
                dalle_images.append(fallback_frame)
        
        print(f"🎬 Creating {total_frames} video frames with DALL-E backgrounds...")
        
        # Create video frames with DALL-E backgrounds
        for frame_num in range(total_frames):
            progress = frame_num / total_frames
            
            # Determine which DALL-E images to blend
            if len(dalle_images) == 1:
                # Single image - add subtle animations
                base_frame = dalle_images[0].copy()
                
                # Add subtle zoom and pan effects
                zoom_factor = 1.0 + 0.05 * math.sin(progress * 2 * math.pi)
                pan_x = int(10 * math.sin(progress * 4 * math.pi))
                pan_y = int(5 * math.cos(progress * 3 * math.pi))
                
                # Apply zoom and pan
                zoomed_size = (int(width * zoom_factor), int(height * zoom_factor))
                if zoom_factor > 1.0:
                    base_frame = base_frame.resize(zoomed_size, Image.Resampling.LANCZOS)
                    # Crop back to original size with pan offset
                    left = (zoomed_size[0] - width) // 2 + pan_x
                    top = (zoomed_size[1] - height) // 2 + pan_y
                    base_frame = base_frame.crop((left, top, left + width, top + height))
                
            else:
                # Multiple images - blend between them
                image_index = progress * (len(dalle_images) - 1)
                current_image_idx = int(image_index)
                next_image_idx = min(current_image_idx + 1, len(dalle_images) - 1)
                blend_factor = image_index - current_image_idx
                
                # Blend between current and next image
                current_image = dalle_images[current_image_idx]
                next_image = dalle_images[next_image_idx]
                
                if current_image_idx == next_image_idx:
                    base_frame = current_image.copy()
                else:
                    base_frame = Image.blend(current_image, next_image, blend_factor)
            
            # Add text overlay
            draw = ImageDraw.Draw(base_frame)
            
            # Enhanced font loading
            base_font_size = max(width // 25, 32)  # Larger, more readable font
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", base_font_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position
            bbox = draw.textbbox((0, 0), text_prompt, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = height - text_height - 50  # Position near bottom for better visibility
            
            # Enhanced text effects
            if quality in ["high", "ultra"]:
                # Multiple shadow layers for depth
                for offset in range(4, 0, -1):
                    shadow_alpha = 255 - (offset * 40)
                    # Create shadow layer
                    shadow_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                    shadow_draw = ImageDraw.Draw(shadow_layer)
                    shadow_draw.text((x + offset, y + offset), text_prompt, 
                                   fill=(0, 0, 0, shadow_alpha), font=font)
                    base_frame = Image.alpha_composite(base_frame.convert('RGBA'), shadow_layer).convert('RGB')
            
            # Animated text color
            if quality == "ultra":
                # Rainbow text animation
                hue = (progress * 360) % 360
                import colorsys
                rgb = colorsys.hsv_to_rgb(hue/360, 0.8, 1.0)
                text_color = tuple(int(c * 255) for c in rgb)
            else:
                # White text with slight glow
                text_color = (255, 255, 255)
            
            # Draw main text
            draw.text((x, y), text_prompt, fill=text_color, font=font)
            
            frames.append(base_frame)
        
        # Save as video
        video_id = str(uuid.uuid4())
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(VIDEOS_DIR, video_filename)
        
        print(f"🎥 Encoding video with {len(frames)} frames...")
        
        # Write video using imageio with better quality settings
        with imageio.get_writer(video_path, fps=int(fps), codec='libx264', 
                               macro_block_size=1, quality=8) as writer:
            for frame in frames:
                # Convert PIL Image to numpy array for imageio
                import numpy as np
                frame_array = np.array(frame)
                writer.append_data(frame_array)
        
        print(f"✅ Video generated successfully: {video_filename}")
        return video_filename, None
        
    except Exception as e:
        return None, f"Error generating video: {str(e)}"

def generate_themed_content(prompt, width, height, frame_num, total_frames):
    """Generate themed visual content based on the prompt"""
    import math
    import random
    
    # Create base frame
    frame = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(frame)
    
    progress = frame_num / total_frames
    
    # Theme detection and appropriate visuals
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['cat', 'cats', 'kitten', 'feline']):
        # Cat-themed background
        # Sky blue to pink gradient for cute cat scene
        for y in range(height):
            sky_progress = y / height
            r = int(135 + sky_progress * 120)  # Pink tones
            g = int(206 - sky_progress * 100)  # Blue to less blue
            b = int(250 - sky_progress * 50)   # Keep some blue
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Draw simple cat silhouettes
        for i in range(3):
            cat_x = int(width * (0.2 + i * 0.3) + math.sin(progress * 3.14 + i) * 20)
            cat_y = int(height * 0.7 + math.cos(progress * 6.28 + i) * 10)
            
            # Cat body (oval)
            body_width, body_height = 40, 25
            draw.ellipse([cat_x-body_width//2, cat_y-body_height//2, 
                         cat_x+body_width//2, cat_y+body_height//2], fill=(50, 50, 50))
            
            # Cat head (circle)
            head_size = 20
            head_y = cat_y - 20
            draw.ellipse([cat_x-head_size//2, head_y-head_size//2,
                         cat_x+head_size//2, head_y+head_size//2], fill=(60, 60, 60))
            
            # Cat ears (triangles)
            ear_size = 8
            draw.polygon([(cat_x-12, head_y-15), (cat_x-5, head_y-25), (cat_x+2, head_y-15)], fill=(40, 40, 40))
            draw.polygon([(cat_x-2, head_y-15), (cat_x+5, head_y-25), (cat_x+12, head_y-15)], fill=(40, 40, 40))
            
            # Cat tail (curved line)
            tail_x = cat_x + 30
            tail_y = int(cat_y + 15 * math.sin(progress * 6.28 + i * 2))
            draw.ellipse([tail_x-3, tail_y-15, tail_x+3, tail_y+15], fill=(45, 45, 45))
    
    elif any(word in prompt_lower for word in ['sunset', 'sun', 'orange', 'evening']):
        # Sunset scene
        for y in range(height):
            sunset_progress = y / height
            r = int(255 - sunset_progress * 100)  # Orange to red
            g = int(165 - sunset_progress * 100)  # Orange to darker
            b = int(0 + sunset_progress * 100)    # Dark to purple
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Draw sun
        sun_x = int(width * (0.3 + progress * 0.4))
        sun_y = int(height * 0.3)
        sun_size = 40 + int(10 * math.sin(progress * 6.28))
        draw.ellipse([sun_x-sun_size, sun_y-sun_size, sun_x+sun_size, sun_y+sun_size], 
                    fill=(255, 255, 100))
        
        # Sun rays
        for i in range(8):
            angle = (i * 45 + progress * 360) * math.pi / 180
            ray_start_x = sun_x + int((sun_size + 10) * math.cos(angle))
            ray_start_y = sun_y + int((sun_size + 10) * math.sin(angle))
            ray_end_x = sun_x + int((sun_size + 30) * math.cos(angle))
            ray_end_y = sun_y + int((sun_size + 30) * math.sin(angle))
            draw.line([(ray_start_x, ray_start_y), (ray_end_x, ray_end_y)], 
                     fill=(255, 255, 150), width=3)
    
    elif any(word in prompt_lower for word in ['space', 'stars', 'galaxy', 'universe', 'cosmic']):
        # Space scene
        # Dark gradient background
        for y in range(height):
            space_progress = y / height
            intensity = int(20 + space_progress * 40)
            draw.line([(0, y), (width, y)], fill=(intensity//3, intensity//2, intensity))
        
        # Draw stars
        for i in range(20):
            star_x = int((i * 123 + progress * 50) % width)
            star_y = int((i * 67 + progress * 30) % height)
            star_brightness = int(150 + 105 * math.sin(progress * 6.28 + i))
            star_size = 2 + int(2 * math.sin(progress * 3.14 + i * 0.5))
            draw.ellipse([star_x-star_size, star_y-star_size, star_x+star_size, star_y+star_size],
                        fill=(star_brightness, star_brightness, 255))
        
        # Draw planets
        for i in range(2):
            planet_x = int(width * (0.2 + i * 0.6) + math.cos(progress * 2 + i * 3) * 50)
            planet_y = int(height * (0.3 + i * 0.4) + math.sin(progress * 2 + i * 3) * 30)
            planet_size = 25 + i * 15
            colors = [(150, 100, 200), (200, 150, 100)]
            draw.ellipse([planet_x-planet_size, planet_y-planet_size, 
                         planet_x+planet_size, planet_y+planet_size], fill=colors[i])
    
    elif any(word in prompt_lower for word in ['rainbow', 'colors', 'colorful']):
        # Rainbow scene
        # Rainbow bands
        band_height = height // 7
        colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), 
                 (0, 0, 255), (75, 0, 130), (238, 130, 238)]
        
        for i, color in enumerate(colors):
            y_start = i * band_height
            y_end = min((i + 1) * band_height, height)
            # Add wave effect
            for y in range(y_start, y_end):
                wave_offset = int(20 * math.sin((y + progress * 100) * 0.02))
                draw.line([(wave_offset, y), (width + wave_offset, y)], fill=color)
    
    else:
        # Default abstract scene
        # Dynamic gradient
        for y in range(height):
            gradient_progress = y / height
            wave = math.sin(progress * 6.28 + gradient_progress * 3)
            r = int(100 + 100 * wave)
            g = int(150 + 50 * math.cos(progress * 4.14 + gradient_progress * 2))
            b = int(200 + 55 * math.sin(progress * 3.14 + gradient_progress * 4))
            draw.line([(0, y), (width, y)], fill=(max(0, min(255, r)), 
                                                  max(0, min(255, g)), 
                                                  max(0, min(255, b))))
    
    return frame

def generate_animated_gif(text_prompt, duration=3, fps=15, quality="high"):
    """Generate a high-quality animated GIF from text prompt"""
    if not VIDEO_FEATURES_AVAILABLE:
        return None, "GIF features not available. Please install required packages."
    
    try:
        import math  # Use math for advanced animations
        
        # Quality presets for GIFs - optimized for speed
        quality_settings = {
            "quick": {"width": 300, "height": 200, "fps": 6, "duration": 1.5},
            "standard": {"width": 500, "height": 350, "fps": 8, "duration": 2},
            "high": {"width": 600, "height": 400, "fps": 10, "duration": 2.5},
            "ultra": {"width": 800, "height": 600, "fps": 12, "duration": 3}
        }
        
        # Apply quality settings
        if quality in quality_settings:
            settings = quality_settings[quality]
            width, height = settings["width"], settings["height"]
            fps = settings["fps"] 
            duration = settings["duration"]
        else:
            width, height = 800, 600
        frames = []
        total_frames = int(duration * fps)  # Ensure integer for range()
        
        for frame_num in range(total_frames):
            # Fast animated background - simple gradient
            progress = frame_num / total_frames
            
            # Create simple two-color gradient background (very fast)
            color1 = (50 + int(progress * 100), 70 + int(progress * 80), 120 + int(progress * 60))
            color2 = (120 + int(progress * 60), 100 + int(progress * 100), 180 + int(progress * 40))
            
            frame = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(frame)
            
            # Simple vertical gradient using rectangles (much faster than pixel-by-pixel)
            band_height = height // 10  # Only 10 bands for speed
            for i in range(10):
                y_start = i * band_height
                y_end = min((i + 1) * band_height, height)
                band_progress = i / 9
                
                r = int(color1[0] + band_progress * (color2[0] - color1[0]))
                g = int(color1[1] + band_progress * (color2[1] - color1[1]))
                b = int(color1[2] + band_progress * (color2[2] - color1[2]))
                
                draw.rectangle([0, y_start, width, y_end], fill=(r, g, b))
            
            # Simple animated elements (only for higher quality)
            if quality in ["high", "ultra"]:
                # Just 2-3 simple moving circles instead of complex orbital math
                for i in range(3):
                    angle = (frame_num + i * 20) * 0.2
                    x = int(width/2 + (width//4) * math.cos(angle))
                    y = int(height/2 + (height//4) * math.sin(angle))
                    
                    size = 8 + i * 4
                    color = (150 + i * 30, 100 + i * 40, 200 + i * 20)
                    draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
            
            # Fast text rendering
            base_font_size = max(16, width // 25)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", base_font_size)
            except:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), text_prompt, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            # Simple text effects for speed
            if quality in ["high", "ultra"]:
                # Simple shadow only for higher quality
                draw.text((x + 1, y + 1), text_prompt, fill=(0, 0, 0), font=font)
            
            # Fast text color - simple white or slight animation
            if quality == "ultra":
                # Simple color pulse for ultra only
                brightness = int(200 + 55 * math.sin(progress * 6.28))
                text_color = (brightness, 255, brightness)
            else:
                text_color = (255, 255, 255)
            
            draw.text((x, y), text_prompt, fill=text_color, font=font)
            frames.append(frame)
        
        # Save as GIF
        gif_id = str(uuid.uuid4())
        gif_filename = f"{gif_id}.gif"
        gif_path = os.path.join(GIFS_DIR, gif_filename)
        
        frames[0].save(
            gif_path, 
            save_all=True, 
            append_images=frames[1:], 
            duration=int(1000/fps), 
            loop=0
        )
        
        return gif_filename, None
        
    except Exception as e:
        return None, f"Error generating GIF: {str(e)}"

def handle_video_generation(text):
    """Enhanced video generation with hybrid DALL-E + Runway ML support"""
    
    # Extract quality level from text
    quality = "quick"  # Default to quick for faster processing
    if any(word in text.lower() for word in ["speed", "fast", "rapid", "instant"]):
        quality = "quick"
    elif any(word in text.lower() for word in ["standard", "normal", "regular"]):
        quality = "standard"
    elif any(word in text.lower() for word in ["ultra", "maximum", "best", "highest"]):
        quality = "ultra"
    elif any(word in text.lower() for word in ["high", "quality", "premium"]):
        quality = "high"
    
    # Extract generation method preference
    method = "auto"  # Default to automatic method selection
    if any(word in text.lower() for word in ["cinematic", "movie", "film", "realistic", "professional", "runway"]):
        method = "runway"
    elif any(word in text.lower() for word in ["dalle", "animated", "slideshow", "quick", "cheap"]):
        method = "dalle"
    
    # Extract the prompt from the text
    prompt_patterns = [
        r'generate.*video.*(?:of|about|showing)\s*(.+)',
        r'create.*video.*(?:of|about|showing)\s*(.+)', 
        r'make.*video.*(?:of|about|showing)\s*(.+)',
        r'video.*(?:of|about|showing)\s*(.+)',
        r'generate.*video\s*(.+)',
        r'create.*video\s*(.+)',
        r'make.*video\s*(.+)',
        r'(?:video|movie|film).*[:\-]\s*(.+)',
        r'(?:cinematic|runway|dalle).*(?:video|shot).*(?:of|about)\s*(.+)'
    ]
    
    prompt = None
    for pattern in prompt_patterns:
        match = re.search(pattern, text.lower())
        if match:
            prompt = match.group(1).strip()
            break
    
    if not prompt:
        return f"""🎥 I can generate videos using multiple methods:
        
🎨 **DALL-E Animated** (Fast & Affordable):
- 'create a video of cats playing'
- 'make a quick video about sunset'

🎬 **Runway ML Cinematic** (Professional & Realistic):
- 'create a cinematic video of dancing robot'
- 'make a professional video of space exploration'

Just describe what you'd like me to create!"""
    
    # Clean up prompt
    prompt = prompt.replace("high quality", "").replace("ultra quality", "").replace("quick", "").strip()
    prompt = prompt.replace("cinematic", "").replace("runway", "").replace("dalle", "").strip()
    
    print(f"🎥 Generating {quality} quality video with {method} method: {prompt}")
    
    try:
        # Generate the video with specified quality and method
        video_filename, error = generate_text_video(prompt, quality=quality, method=method)
        
        if error:
            return f"🎥 I encountered an issue generating the video: {error}"
        
        if video_filename:
            # Determine if it's a Runway or DALL-E video for appropriate messaging
            is_runway = video_filename.startswith('runway_')
            method_name = "Runway ML Cinematic" if is_runway else "DALL-E Animated"
            
            # Create full URL for the video
            full_video_url = f"http://192.168.1.206:8080/static/generated_videos/{video_filename}"
            
            # Enhanced quality descriptions for hybrid system
            if is_runway:
                quality_desc = {
                    "quick": "Runway Quick (16:9, 3s) 🎬 Cinematic quality",
                    "standard": "Runway Standard (16:9, 5s) 🎬 Professional grade",
                    "high": "Runway High (16:9, 7s) 🎬 Hollywood quality",
                    "ultra": "Runway Ultra (16:9, 10s) 🎬 Masterpiece level"
                }
            else:
                quality_desc = {
                    "quick": "DALL-E Quick (512×512, 3s) 🎨 Fast animated",
                    "standard": "DALL-E Standard (512×512, 5s) 🎨 Detailed animated",
                    "high": "DALL-E High (1024×1024, 7s) 🎨 Premium animated", 
                    "ultra": "DALL-E Ultra (1024×1024, 10s) 🎨 Masterpiece animated"
                }
            
            return f"""📹 **Method**: {method_name}
🎬 **Video**: {full_video_url}"""
            
            return f"""🎥 {quality_desc.get(quality, 'High Quality')} Video Generated
📝 Prompt: "{prompt}"

{full_video_url}"""
        else:
            return "🎥 I had trouble generating that video. Please try a different description."
            
    except Exception as e:
        print(f"Error in video generation: {e}")
        return "🎥 I had trouble generating that video. Please make sure your request is clear and try again!"

def handle_gif_generation(text):
    """Handle animated GIF generation requests with quality detection"""
    # Extract quality level from text
    quality = "quick"  # Default to quick for faster processing
    if any(word in text.lower() for word in ["speed", "fast", "rapid", "instant"]):
        quality = "quick"
    elif any(word in text.lower() for word in ["standard", "normal", "regular"]):
        quality = "standard"
    elif any(word in text.lower() for word in ["ultra", "maximum", "best", "highest"]):
        quality = "ultra"
    elif any(word in text.lower() for word in ["high", "quality", "premium"]):
        quality = "high"
    
    # Extract the prompt from the text
    prompt_patterns = [
        r'generate.*gif.*(?:of|about|showing)\s*(.+)',
        r'create.*gif.*(?:of|about|showing)\s*(.+)',
        r'make.*gif.*(?:of|about|showing)\s*(.+)',
        r'gif.*(?:of|about|showing)\s*(.+)',
        r'generate.*gif\s*(.+)',
        r'create.*gif\s*(.+)',
        r'make.*gif\s*(.+)',
        r'animate.*(.+)',
        r'(?:gif|animation).*[:\-]\s*(.+)'
    ]
    
    prompt = None
    for pattern in prompt_patterns:
        match = re.search(pattern, text.lower())
        if match:
            prompt = match.group(1).strip()
            break
    
    if not prompt:
        return f"🎬 I can create {quality} quality animated GIFs for you! Please describe what you'd like me to animate. For example: 'generate a high quality gif of bouncing balls' or 'animate a spinning logo'."
    
    # Clean up prompt
    prompt = prompt.replace("high quality", "").replace("ultra quality", "").replace("quick", "").strip()
    
    print(f"🎬 Generating {quality} quality GIF with prompt: {prompt}")
    
    try:
        # Generate the GIF with specified quality
        gif_filename, error = generate_animated_gif(prompt, quality=quality)
        
        if error:
            return f"🎬 I encountered an issue generating the GIF: {error}"
        
        if gif_filename:
            # Create full URL for the GIF
            full_gif_url = f"http://192.168.1.206:8080/static/generated_gifs/{gif_filename}"
            
            # Quality descriptions for GIFs - updated for speed
            quality_desc = {
                "quick": "Quick (300×200, 6fps, 1.5s) ⚡ ~2-5 seconds",
                "standard": "Standard (500×350, 8fps, 2s) ⚡ ~3-8 seconds",
                "high": "High Quality (600×400, 10fps, 2.5s) ⚡ ~4-10 seconds", 
                "ultra": "Ultra Quality (800×600, 12fps, 3s) ⚡ ~6-15 seconds"
            }
            
            return f"""🎬 {quality_desc.get(quality, 'High Quality')} Animated GIF Generated
📝 Prompt: "{prompt}"

{full_gif_url}"""
        else:
            return "🎬 I had trouble generating that GIF. Please try a different description."
            
    except Exception as e:
        print(f"Error in GIF generation: {e}")
        return "🎬 I had trouble generating that GIF. Please make sure your request is clear and try again!"

def handle_music_generation(text):
    """Handle AI music composition requests"""
    
    # Extract quality level from text
    quality = "standard"  # Default
    if any(word in text.lower() for word in ["high", "premium", "quality", "professional"]):
        quality = "high"
    elif any(word in text.lower() for word in ["quick", "fast", "simple", "basic"]):
        quality = "quick"
    
    # Extract duration from text
    duration = 30  # Default 30 seconds
    import re
    duration_match = re.search(r'(\d+)\s*(second|minute|min)', text.lower())
    if duration_match:
        dur_value = int(duration_match.group(1))
        dur_unit = duration_match.group(2)
        if dur_unit in ['minute', 'min']:
            duration = dur_value * 60
        else:
            duration = dur_value
        duration = min(duration, 300)  # Max 5 minutes
    
    # Extract music style from text
    style = "pop"  # Default
    style_keywords = {
        'pop': ['pop', 'upbeat', 'catchy', 'mainstream'],
        'rock': ['rock', 'guitar', 'drums', 'heavy'],
        'classical': ['classical', 'orchestra', 'piano', 'elegant'],
        'electronic': ['electronic', 'synth', 'edm', 'techno', 'dance'],
        'jazz': ['jazz', 'saxophone', 'smooth', 'blues'],
        'ambient': ['ambient', 'chill', 'relaxing', 'peaceful', 'calm'],
        'hip-hop': ['hip-hop', 'rap', 'beats', 'urban'],
        'country': ['country', 'folk', 'acoustic', 'western']
    }
    
    for style_name, keywords in style_keywords.items():
        if any(keyword in text.lower() for keyword in keywords):
            style = style_name
            break
    
    # Extract the prompt from the text
    prompt_patterns = [
        r'(?:generate|create|make|compose).*music.*(?:about|for|of)\s*(.+)',
        r'(?:generate|create|make|compose).*song.*(?:about|for|of)\s*(.+)',
        r'music.*(?:about|for|of)\s*(.+)',
        r'song.*(?:about|for|of)\s*(.+)',
        r'(?:compose|write).*(?:music|song)\s*(.+)',
        r'ai.*music.*[:\-]\s*(.+)',
        r'(?:suno|musicgen).*[:\-]\s*(.+)'
    ]
    
    prompt = None
    for pattern in prompt_patterns:
        match = re.search(pattern, text.lower())
        if match:
            prompt = match.group(1).strip()
            break
    
    if not prompt:
        return f"""🎵 I can generate AI music in various styles!

🎼 **Available Styles**: Pop, Rock, Classical, Electronic, Jazz, Ambient, Hip-Hop, Country

💡 **Examples**:
- 'compose pop music about summer'
- 'create classical music for relaxation'
- 'generate electronic music for 2 minutes'
- 'make jazz music about city nights'

Just describe what kind of music you'd like!"""
    
    # Clean up prompt
    prompt = prompt.replace(f"{quality}", "").replace(f"{style}", "").strip()
    
    print(f"🎵 Generating {style} music ({quality} quality, {duration}s): {prompt}")
    
    try:
        # Generate the music
        music_filename, error = generate_ai_music(prompt, duration, style, quality)
        
        if error:
            return f"🎵 I encountered an issue generating the music: {error}"
        
        if music_filename:
            # Create full URL for the music
            full_music_url = f"http://192.168.1.206:8080/static/generated_music/{music_filename}"
            
            # Determine service used
            service_name = "Enhanced AI Music"
            if music_filename.startswith('replicate_'):
                service_name = "Replicate MusicGen Pro"
            elif music_filename.startswith('stability_'):
                service_name = "Stability AI Stable Audio"
            elif music_filename.startswith('huggingface_'):
                service_name = "Hugging Face MusicGen"
            elif music_filename.startswith('suno_'):
                service_name = "Suno AI"
            elif music_filename.startswith('musicgen_'):
                service_name = "MusicGen"
            elif music_filename.startswith('enhanced_'):
                service_name = "Enhanced Multi-Layer"
            elif music_filename.startswith('synth_'):
                service_name = "Basic Synthesized"
            
            return f"""🎵 **Service**: {service_name} {style.title()} Music
🎼 **Track**: {full_music_url}"""
            
        return "🎵 I had trouble generating that music. Please try a different description."
        
    except Exception as e:
        print(f"Error in music generation: {e}")
        return "🎵 I had trouble generating that music. Please make sure your request is clear and try again!"

def handle_voice_generation(text):
    """Handle voice/speech synthesis requests"""
    
    # Extract voice style from text
    voice_style = "alloy"  # Default OpenAI voice
    voice_keywords = {
        'alloy': ['neutral', 'balanced', 'alloy'],
        'echo': ['male', 'deep', 'echo'],
        'fable': ['british', 'accent', 'fable'],
        'onyx': ['strong', 'powerful', 'onyx'],
        'nova': ['female', 'clear', 'nova'],
        'shimmer': ['soft', 'gentle', 'shimmer']
    }
    
    for voice_name, keywords in voice_keywords.items():
        if any(keyword in text.lower() for keyword in keywords):
            voice_style = voice_name
            break
    
    # Extract quality level
    quality = "standard"
    if any(word in text.lower() for word in ["high", "premium", "quality", "hd"]):
        quality = "high"
    
    # Extract the text to speak
    prompt_patterns = [
        r'(?:say|speak|read).*[:\-]\s*(.+)',
        r'voice.*(?:saying|reading)\s*(.+)',
        r'text.*to.*speech.*[:\-]\s*(.+)',
        r'(?:generate|create).*voice.*[:\-]\s*(.+)',
        r'(?:elevenlabs|tts).*[:\-]\s*(.+)',
        r'speak.*text\s*(.+)',
        r'narrate.*[:\-]\s*(.+)'
    ]
    
    text_to_speak = None
    for pattern in prompt_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text_to_speak = match.group(1).strip()
            break
    
    if not text_to_speak:
        return f"""🗣️ I can generate speech from text using AI voices!

🎤 **Available Voices**: Alloy, Echo, Fable, Onyx, Nova, Shimmer

💡 **Examples**:
- 'say: Hello world'
- 'generate female voice: Welcome to our store'
- 'speak with male voice: This is a test'
- 'read aloud: The quick brown fox jumps'

Just tell me what text to speak!"""
    
    print(f"🗣️ Generating {voice_style} voice ({quality}): {text_to_speak[:50]}...")
    
    try:
        # Generate the voice audio
        audio_filename, error = generate_voice_audio(text_to_speak, voice_style, quality)
        
        if error:
            return f"🗣️ I encountered an issue generating the voice: {error}"
        
        if audio_filename:
            # Create full URL for the audio
            full_audio_url = f"http://192.168.1.206:8080/static/generated_audio/{audio_filename}"
            
            # Determine service used
            service_name = "OpenAI TTS"
            if audio_filename.startswith('elevenlabs_'):
                service_name = "ElevenLabs"
            
            return f"""🗣️ **Voice**: {service_name} ({voice_style.title()})
🎙️ **Audio**: {full_audio_url}"""
            
        return "🗣️ I had trouble generating that voice. Please try a different text."
        
    except Exception as e:
        print(f"Error in voice generation: {e}")
        return "🗣️ I had trouble generating that voice. Please make sure your request is clear and try again!"

def handle_audio_transcription(text):
    """Handle audio transcription requests"""
    
    return """🎤 **Audio Transcription Available!**

To transcribe audio to text:
1. Upload an audio file (MP3, WAV, etc.)
2. Use the transcription endpoint
3. Get text output from speech

💡 **Supported**:
- OpenAI Whisper (premium)
- Google Speech Recognition
- Offline Sphinx recognition

📁 **File Upload**: Coming soon in web interface!"""

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
    'video_generation': [
        r'\b(generate|create|make|produce).*video\b',
        r'\b(generate|create|make|produce).*movie\b',
        r'\b(generate|create|make|produce).*film\b',
        r'\bvideo of\b', r'\bmovie of\b', r'\bfilm of\b',
        r'\bai.*video\b', r'\btext.*to.*video\b',
        r'\bshow me.*video\b', r'\bvideo.*about\b',
        r'\brecord.*video\b', r'\bvideo.*clip\b',
        r'\bcinematic.*video\b', r'\brunway.*video\b',
        r'\bdalle.*video\b', r'\banimated.*video\b',
        r'\bprofessional.*video\b', r'\brealistic.*video\b',
        r'\bmovie.*scene\b', r'\bfilm.*sequence\b'
    ],
    'gif_generation': [
        r'\b(generate|create|make|produce).*gif\b',
        r'\b(generate|create|make|produce).*animation\b',
        r'\bgif of\b', r'\banimate\b', r'\banimated\b',
        r'\bai.*gif\b', r'\bmoving.*image\b',
        r'\bshow me.*gif\b', r'\bgif.*about\b',
        r'\bloop.*animation\b', r'\bshort.*animation\b'
    ],
    'music_generation': [
        r'\b(generate|create|make|compose).*music\b',
        r'\b(generate|create|make|compose).*song\b',
        r'\b(generate|create|make|compose).*track\b',
        r'\b(generate|create|make|compose).*tune\b',
        r'\bai.*music\b', r'\bai.*song\b', r'\bmusic.*about\b',
        r'\bcompose.*music\b', r'\bwrite.*song\b',
        r'\bplay.*music\b', r'\bmake.*beat\b',
        r'\bsynthesize.*music\b', r'\bmusical.*composition\b',
        r'\binstrumental\b', r'\bmelody.*for\b',
        r'\bsuno.*music\b', r'\bmusicgen\b'
    ],
    'voice_generation': [
        r'\b(generate|create|make).*voice\b',
        r'\b(generate|create|make).*speech\b',
        r'\btext.*to.*speech\b', r'\btts\b',
        r'\bspeak.*text\b', r'\bvoice.*over\b',
        r'\bai.*voice\b', r'\bai.*speech\b',
        r'\bread.*aloud\b', r'\bsay.*this\b',
        r'\belevenlabs\b', r'\bvoice.*synthesis\b',
        r'\bnarrate.*this\b', r'\bspoken.*audio\b'
    ],
    'audio_transcription': [
        r'\btranscribe.*audio\b', r'\bspeech.*to.*text\b',
        r'\bconvert.*speech\b', r'\blisten.*to.*audio\b',
        r'\bwhisper.*transcribe\b', r'\baudio.*to.*text\b',
        r'\btranscription\b', r'\bparse.*audio\b',
        r'\bsubtitles.*from.*audio\b'
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
    quick_commands = ['time', 'date', 'math', 'timer', 'reminder', 'greeting', 'goodbye', 'joke', 'image_generation', 'video_generation', 'gif_generation', 'music_generation', 'voice_generation', 'audio_transcription']
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
        elif intent == 'video_generation':
            response = handle_video_generation(user_input)
        elif intent == 'gif_generation':
            response = handle_gif_generation(user_input)
        elif intent == 'music_generation':
            response = handle_music_generation(user_input)
        elif intent == 'voice_generation':
            response = handle_voice_generation(user_input)
        elif intent == 'audio_transcription':
            response = handle_audio_transcription(user_input)
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

# ===== VISUAL AI API ENDPOINTS =====

@app.route('/api/generate-avatar', methods=['POST'])
def api_generate_avatar():
    """Generate AI avatar"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        style = data.get('style', 'realistic')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        filename, error = generate_ai_avatar(prompt, style)
        
        if filename:
            return jsonify({
                'success': True,
                'filename': filename,
                'url': f'/static/generated_avatars/{filename}'
            })
        else:
            return jsonify({'error': error or 'Avatar generation failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/edit-background', methods=['POST'])
def api_edit_background():
    """Edit image background"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        action = request.form.get('action', 'remove')
        
        # Save uploaded image temporarily
        import uuid
        temp_id = str(uuid.uuid4())
        temp_filename = f"temp_{temp_id}.png"
        temp_path = os.path.join(DESIGNS_DIR, temp_filename)
        image_file.save(temp_path)
        
        filename, error = edit_image_background(temp_path, action)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if filename:
            return jsonify({
                'success': True,
                'filename': filename,
                'url': f'/static/generated_designs/{filename}'
            })
        else:
            return jsonify({'error': error or 'Background editing failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-3d-model', methods=['POST'])
def api_generate_3d_model():
    """Generate 3D model"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        style = data.get('style', 'realistic')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        filename, error = generate_3d_model(prompt, style)
        
        if filename:
            return jsonify({
                'success': True,
                'filename': filename,
                'url': f'/static/generated_3d_models/{filename}'
            })
        else:
            return jsonify({'error': error or '3D model generation failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-logo', methods=['POST'])
def api_generate_logo():
    """Generate logo design"""
    try:
        data = request.get_json()
        brand_name = data.get('brand_name', '')
        industry = data.get('industry', 'technology')
        style = data.get('style', 'modern')
        
        if not brand_name:
            return jsonify({'error': 'Brand name is required'}), 400
        
        filename, error = generate_logo_design(brand_name, industry, style)
        
        if filename:
            return jsonify({
                'success': True,
                'filename': filename,
                'url': f'/static/generated_logos/{filename}'
            })
        else:
            return jsonify({'error': error or 'Logo generation failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upscale-image', methods=['POST'])
def api_upscale_image():
    """Upscale image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        scale_factor = int(request.form.get('scale', 2))
        
        # Save uploaded image temporarily
        import uuid
        temp_id = str(uuid.uuid4())
        temp_filename = f"temp_{temp_id}.png"
        temp_path = os.path.join(DESIGNS_DIR, temp_filename)
        image_file.save(temp_path)
        
        filename, error = upscale_image(temp_path, scale_factor)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if filename:
            return jsonify({
                'success': True,
                'filename': filename,
                'url': f'/static/generated_designs/{filename}'
            })
        else:
            return jsonify({'error': error or 'Image upscaling failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@app.route('/api/generate-music', methods=['POST'])
def generate_music_api():
    """API endpoint for music generation"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        style = data.get('style', 'pop')
        duration = min(int(data.get('duration', 30)), 300)  # Max 5 minutes
        quality = data.get('quality', 'standard')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        print(f"🎵 API Music generation request: {prompt} ({style}, {duration}s)")
        
        music_filename, error = generate_ai_music(prompt, duration, style, quality)
        
        if error:
            return jsonify({'error': error}), 500
        
        if music_filename:
            music_url = f"/static/generated_music/{music_filename}"
            return jsonify({
                'music_filename': music_filename,
                'music_url': music_url,
                'style': style,
                'duration': duration,
                'message': f'🎵 {style.title()} music generated successfully!'
            })
        else:
            return jsonify({'error': 'Failed to generate music'}), 500
            
    except Exception as e:
        print(f"Error in generate_music_api: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/generate-voice', methods=['POST'])
def generate_voice_api():
    """API endpoint for voice synthesis"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        voice = data.get('voice', 'alloy')
        quality = data.get('quality', 'standard')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if len(text) > 4000:
            return jsonify({'error': 'Text too long (max 4000 characters)'}), 400
        
        print(f"🗣️ API Voice generation request: {text[:50]}... ({voice})")
        
        audio_filename, error = generate_voice_audio(text, voice, quality)
        
        if error:
            return jsonify({'error': error}), 500
        
        if audio_filename:
            audio_url = f"/static/generated_audio/{audio_filename}"
            return jsonify({
                'audio_filename': audio_filename,
                'audio_url': audio_url,
                'voice': voice,
                'message': f'🗣️ Voice generated successfully!'
            })
        else:
            return jsonify({'error': 'Failed to generate voice'}), 500
            
    except Exception as e:
        print(f"Error in generate_voice_api: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/transcribe-audio', methods=['POST'])
def transcribe_audio_api():
    """API endpoint for audio transcription"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Audio file is required'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            print(f"🎤 API Transcription request: {audio_file.filename}")
            
            # Transcribe the audio
            transcription, error = transcribe_audio(temp_path)
            
            if error:
                return jsonify({'error': error}), 500
            
            if transcription:
                return jsonify({
                    'transcription': transcription,
                    'filename': audio_file.filename,
                    'message': '🎤 Audio transcribed successfully!'
                })
            else:
                return jsonify({'error': 'Failed to transcribe audio'}), 500
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
    except Exception as e:
        print(f"Error in transcribe_audio_api: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/static/generated_images/<filename>')
def serve_generated_image(filename):
    """Serve generated images from the local storage"""
    try:
        return send_from_directory(IMAGES_DIR, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/static/generated_music/<filename>')
def serve_generated_music(filename):
    """Serve generated music from the local storage"""
    try:
        return send_from_directory(MUSIC_DIR, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Music not found'}), 404

@app.route('/static/generated_audio/<filename>')
def serve_generated_audio(filename):
    """Serve generated audio from the local storage"""
    try:
        return send_from_directory(AUDIO_DIR, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Audio not found'}), 404

@app.route('/static/generated_avatars/<filename>')
def serve_generated_avatar(filename):
    """Serve generated avatars from the local storage"""
    try:
        return send_from_directory(AVATARS_DIR, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Avatar not found'}), 404

@app.route('/static/generated_designs/<filename>')
def serve_generated_design(filename):
    """Serve generated designs from the local storage"""
    try:
        return send_from_directory(DESIGNS_DIR, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Design not found'}), 404

@app.route('/static/generated_3d_models/<filename>')
def serve_generated_3d_model(filename):
    """Serve generated 3D models from the local storage"""
    try:
        return send_from_directory(MODELS_3D_DIR, filename)
    except FileNotFoundError:
        return jsonify({'error': '3D model not found'}), 404

@app.route('/static/generated_logos/<filename>')
def serve_generated_logo(filename):
    """Serve generated logos from the local storage"""
    try:
        return send_from_directory(LOGOS_DIR, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Logo not found'}), 404

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
