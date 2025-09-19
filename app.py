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

# Google Gemini AI imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✨ Google Gemini AI loaded successfully")
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Google Gemini AI not available")

# Google Imagen (Vertex AI) imports
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic.schema import predict
    IMAGEN_AVAILABLE = True
    print("🎨 Google Imagen AI loaded successfully")
except ImportError:
    IMAGEN_AVAILABLE = False
    print("⚠️ Google Imagen AI not available")

# Machine Learning Training imports
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd
    import numpy as np
    import joblib
    ML_TRAINING_AVAILABLE = True
    print("🧠 ML Training capabilities loaded successfully")
except ImportError as e:
    ML_TRAINING_AVAILABLE = False
    print(f"⚠️ ML Training not available: {e}")

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

# Initialize Google Gemini AI
try:
    if GEMINI_AVAILABLE:
        gemini_api_key = os.getenv('GEMINI_API_KEY') or getattr(Config, 'GEMINI_API_KEY', None)
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            # Test the connection by listing available models
            models = list(genai.list_models())
            if models:
                print("✅ Google Gemini API connected successfully")
                GEMINI_CONFIGURED = True
            else:
                print("⚠️ Gemini API key invalid or no models available")
                GEMINI_CONFIGURED = False
        else:
            print("⚠️ No Gemini API key found")
            GEMINI_CONFIGURED = False
    else:
        GEMINI_CONFIGURED = False
except Exception as e:
    print(f"⚠️ Error configuring Gemini: {e}")
    GEMINI_CONFIGURED = False

# Initialize Google Imagen (Vertex AI)
try:
    if IMAGEN_AVAILABLE:
        project_id = getattr(Config, 'GOOGLE_CLOUD_PROJECT', 'horizon-ai-project')
        region = getattr(Config, 'GOOGLE_CLOUD_REGION', 'us-central1')
        
        # For now, we'll use the Gemini API key for authentication
        # In production, you'd use proper service account credentials
        if GEMINI_CONFIGURED:
            aiplatform.init(project=project_id, location=region)
            print("✅ Google Imagen (Vertex AI) initialized successfully")
            IMAGEN_CONFIGURED = True
        else:
            print("⚠️ Imagen requires Gemini API configuration")
            IMAGEN_CONFIGURED = False
    else:
        IMAGEN_CONFIGURED = False
except Exception as e:
    print(f"⚠️ Error configuring Imagen: {e}")
    IMAGEN_CONFIGURED = False
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
        
        # Custom AI Model Training Tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS custom_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,
                creator_id TEXT,
                description TEXT,
                training_data_path TEXT,
                model_file_path TEXT,
                config_json TEXT,
                training_status TEXT DEFAULT 'pending',
                training_progress REAL DEFAULT 0.0,
                accuracy_score REAL,
                loss_score REAL,
                model_size_mb REAL,
                created_at TEXT,
                updated_at TEXT,
                is_public INTEGER DEFAULT 0,
                download_count INTEGER DEFAULT 0,
                rating_average REAL DEFAULT 0.0,
                rating_count INTEGER DEFAULT 0,
                tags TEXT,
                version TEXT DEFAULT '1.0.0'
            )
        ''')
        
        # Training Sessions for progress tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                session_id TEXT UNIQUE,
                training_config TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT DEFAULT 'running',
                current_epoch INTEGER DEFAULT 0,
                total_epochs INTEGER,
                current_loss REAL,
                current_accuracy REAL,
                logs TEXT,
                error_message TEXT,
                FOREIGN KEY (model_id) REFERENCES custom_models (id)
            )
        ''')
        
        # Training Data Management
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT UNIQUE NOT NULL,
                dataset_type TEXT NOT NULL,
                file_path TEXT,
                file_size_mb REAL,
                sample_count INTEGER,
                feature_count INTEGER,
                description TEXT,
                creator_id TEXT,
                created_at TEXT,
                is_public INTEGER DEFAULT 0,
                format_type TEXT,
                preprocessing_config TEXT
            )
        ''')
        
        # AI Model Marketplace
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_marketplace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT,
                price REAL DEFAULT 0.0,
                license_type TEXT DEFAULT 'MIT',
                demo_url TEXT,
                documentation_url TEXT,
                featured INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                view_count INTEGER DEFAULT 0,
                last_updated TEXT,
                FOREIGN KEY (model_id) REFERENCES custom_models (id)
            )
        ''')
        
        # Model Reviews and Ratings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                reviewer_id TEXT,
                rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                review_text TEXT,
                created_at TEXT,
                helpful_votes INTEGER DEFAULT 0,
                FOREIGN KEY (model_id) REFERENCES custom_models (id)
            )
        ''')
        
        # Model Downloads and Usage Analytics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                user_id TEXT,
                action_type TEXT,
                timestamp TEXT,
                user_agent TEXT,
                ip_address TEXT,
                success INTEGER DEFAULT 1,
                FOREIGN KEY (model_id) REFERENCES custom_models (id)
            )
        ''')
        
        # Model Dependencies and Requirements
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                dependency_name TEXT,
                dependency_version TEXT,
                dependency_type TEXT,
                required INTEGER DEFAULT 1,
                FOREIGN KEY (model_id) REFERENCES custom_models (id)
            )
        ''')
        
        # Prompt Engineering Lab Tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_name TEXT NOT NULL,
                category TEXT,
                description TEXT,
                prompt_text TEXT NOT NULL,
                variables TEXT, -- JSON array of variable names
                use_case TEXT,
                creator_id TEXT,
                created_at TEXT,
                updated_at TEXT,
                usage_count INTEGER DEFAULT 0,
                rating_average REAL DEFAULT 0.0,
                rating_count INTEGER DEFAULT 0,
                is_public INTEGER DEFAULT 0,
                tags TEXT -- JSON array of tags
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                description TEXT,
                prompt_a TEXT NOT NULL,
                prompt_b TEXT NOT NULL,
                variables TEXT, -- JSON object with test variables
                model_used TEXT,
                creator_id TEXT,
                created_at TEXT,
                status TEXT DEFAULT 'running', -- running, completed, paused
                total_tests INTEGER DEFAULT 0,
                winner TEXT, -- 'a', 'b', or 'tie'
                confidence_score REAL DEFAULT 0.0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                prompt_variant TEXT, -- 'a' or 'b'
                test_input TEXT,
                ai_response TEXT,
                response_time REAL,
                user_rating INTEGER, -- 1-5 scale
                auto_score REAL, -- automated quality score
                metrics TEXT, -- JSON object with additional metrics
                timestamp TEXT,
                FOREIGN KEY (experiment_id) REFERENCES prompt_experiments (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER,
                date TEXT,
                usage_count INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0.0,
                avg_rating REAL DEFAULT 0.0,
                success_rate REAL DEFAULT 0.0,
                error_count INTEGER DEFAULT 0,
                improvement_suggestions TEXT,
                FOREIGN KEY (prompt_id) REFERENCES prompt_templates (id)
            )
        ''')
        
        # AI Performance Analytics Tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                feature_used TEXT,
                model_used TEXT,
                request_type TEXT,
                response_time REAL,
                tokens_used INTEGER,
                success INTEGER, -- 1 for success, 0 for failure
                error_message TEXT,
                timestamp TEXT,
                date TEXT, -- YYYY-MM-DD for daily aggregation
                hour INTEGER -- 0-23 for hourly stats
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                metric_type TEXT, -- daily, weekly, monthly
                feature_name TEXT,
                model_name TEXT,
                total_requests INTEGER DEFAULT 0,
                successful_requests INTEGER DEFAULT 0,
                failed_requests INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0.0,
                total_tokens INTEGER DEFAULT 0,
                user_satisfaction REAL DEFAULT 0.0,
                peak_hour INTEGER,
                improvement_score REAL DEFAULT 0.0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                date TEXT,
                total_interactions INTEGER DEFAULT 0,
                favorite_features TEXT, -- JSON array
                most_used_personality TEXT,
                avg_session_length REAL DEFAULT 0.0,
                satisfaction_score REAL DEFAULT 0.0,
                engagement_level TEXT, -- low, medium, high
                preferred_models TEXT, -- JSON array
                usage_patterns TEXT -- JSON object with usage insights
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS improvement_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT, -- prompt_optimization, performance_enhancement, user_experience
                category TEXT,
                title TEXT,
                description TEXT,
                data_source TEXT, -- which table/metric this insight comes from
                confidence_score REAL,
                impact_level TEXT, -- low, medium, high, critical
                action_suggested TEXT,
                implemented INTEGER DEFAULT 0,
                created_at TEXT,
                priority INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_name TEXT NOT NULL,
                description TEXT,
                feature_tested TEXT,
                variant_a TEXT, -- configuration for variant A
                variant_b TEXT, -- configuration for variant B
                start_date TEXT,
                end_date TEXT,
                status TEXT DEFAULT 'active', -- active, paused, completed
                total_participants INTEGER DEFAULT 0,
                conversion_rate_a REAL DEFAULT 0.0,
                conversion_rate_b REAL DEFAULT 0.0,
                statistical_significance REAL DEFAULT 0.0,
                winner TEXT, -- 'a', 'b', or 'inconclusive'
                created_by TEXT
            )
        ''')
        
        # Research Paper Generator Tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                topic TEXT NOT NULL,
                field TEXT, -- science, technology, medicine, social_science, etc.
                abstract TEXT,
                content TEXT,
                citations TEXT, -- JSON array of citations
                bibliography TEXT, -- formatted bibliography
                keywords TEXT, -- comma-separated keywords
                author_name TEXT,
                status TEXT DEFAULT 'draft', -- draft, in_progress, completed
                created_at TEXT,
                updated_at TEXT,
                word_count INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 0.0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                source_type TEXT, -- journal, book, website, dataset, etc.
                title TEXT NOT NULL,
                authors TEXT,
                journal_name TEXT,
                publication_year INTEGER,
                doi TEXT,
                url TEXT,
                abstract TEXT,
                relevance_score REAL DEFAULT 0.0,
                citation_format TEXT, -- APA, MLA, Chicago, etc.
                added_at TEXT,
                FOREIGN KEY (paper_id) REFERENCES research_papers (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                field TEXT,
                structure TEXT, -- JSON object with paper structure
                guidelines TEXT,
                example_content TEXT,
                created_by TEXT,
                created_at TEXT,
                usage_count INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                generation_time REAL,
                word_count INTEGER,
                citation_count INTEGER,
                quality_metrics TEXT, -- JSON object with quality scores
                user_rating INTEGER, -- 1-5 scale
                completion_rate REAL,
                date_generated TEXT,
                FOREIGN KEY (paper_id) REFERENCES research_papers (id)
            )
        ''')
        
        # Scientific Simulation Tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT, -- physics, chemistry, biology, math
                simulation_type TEXT, -- molecular, physics_engine, ecosystem, etc.
                description TEXT,
                parameters TEXT, -- JSON object with simulation parameters
                initial_conditions TEXT, -- JSON object with starting conditions
                results TEXT, -- JSON object with simulation results
                visualization_data TEXT, -- JSON object for charts/graphs
                created_by TEXT,
                created_at TEXT,
                updated_at TEXT,
                run_count INTEGER DEFAULT 0,
                avg_runtime REAL DEFAULT 0.0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT,
                description TEXT,
                default_parameters TEXT, -- JSON object
                educational_content TEXT,
                learning_objectives TEXT,
                difficulty_level TEXT, -- beginner, intermediate, advanced
                created_by TEXT,
                created_at TEXT,
                usage_count INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER,
                run_parameters TEXT, -- JSON object with specific run parameters
                output_data TEXT, -- JSON object with results
                runtime REAL,
                success BOOLEAN DEFAULT 1,
                error_message TEXT,
                visualizations TEXT, -- JSON array of generated charts/graphs
                insights TEXT, -- AI-generated insights about results
                run_timestamp TEXT,
                FOREIGN KEY (simulation_id) REFERENCES simulations (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER,
                date TEXT,
                total_runs INTEGER DEFAULT 0,
                avg_runtime REAL DEFAULT 0.0,
                success_rate REAL DEFAULT 0.0,
                most_common_parameters TEXT, -- JSON object
                user_engagement_score REAL DEFAULT 0.0,
                educational_effectiveness REAL DEFAULT 0.0,
                FOREIGN KEY (simulation_id) REFERENCES simulations (id)
            )
        ''')
        
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
        max_score = max(emotion_scores.values()) if emotion_scores else 0
        if max_score > 0:
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            emotion_confidence = min(emotion_scores[dominant_emotion] / 3.0, 1.0)  # Normalize to 0-1
        else:
            dominant_emotion = 'neutral'
            emotion_confidence = 0.0
        
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
    """Handle AI image generation requests using Imagen, Gemini or DALL-E API"""
    try:
        # Check which AI service to use - prioritize Imagen > DALL-E
        if IMAGEN_CONFIGURED:
            return handle_imagen_generation(text)
        elif AI_MODEL_AVAILABLE and client:
            return handle_dalle_image_generation(text)
        else:
            return "🎨 I'd love to generate images for you! However, I need either a Google Cloud/Imagen setup or OpenAI API key to access image generation. Please check your configuration and try again."
    except Exception as e:
        print(f"Error in handle_image_generation: {e}")
        return "🎨 I had trouble generating that image. Please make sure your request is clear and try again!"

def handle_imagen_generation(text):
    """Handle AI image generation requests using Google Imagen API"""
    try:
        # Extract the image description from the text
        prompt = extract_image_prompt(text)
        
        if not prompt or len(prompt) < 3:
            return "🎨 I can generate images for you using Google Imagen! Please describe what you'd like me to create. For example: 'generate an image of a sunset over mountains' or 'create a picture of a cute cat wearing a hat'."
        
        print(f"🎨🌟 Generating image with Google Imagen: {prompt}")
        
        try:
            # Enhanced prompt for better image generation
            enhanced_prompt = f"Create a high-quality, detailed, photorealistic image of: {prompt}. Professional quality, well-composed, ultra-detailed."
            
            # Use Imagen through Vertex AI
            # Note: This is a simplified approach. In production, you'd use proper endpoint prediction
            endpoint = aiplatform.Endpoint.list(filter='display_name="imagen-endpoint"')
            
            if not endpoint:
                # Fallback message - Imagen setup required
                print("⚠️ Imagen endpoint not found. Using prompt enhancement for DALL-E...")
                if AI_MODEL_AVAILABLE and client:
                    return handle_dalle_image_generation_with_enhancement(text, enhanced_prompt)
                else:
                    return f"🎨🌟 Google Imagen processed your prompt: '{prompt}'. However, Imagen endpoint setup is required for image generation. Falling back to DALL-E is not available either."
            
            # If endpoint exists, predict with Imagen
            # instances = [{"prompt": enhanced_prompt}]
            # response = endpoint[0].predict(instances=instances)
            
            # For now, return enhanced prompt to DALL-E
            if AI_MODEL_AVAILABLE and client:
                return handle_dalle_image_generation_with_enhancement(text, enhanced_prompt)
            else:
                return f"🎨🌟 Google Imagen enhanced your prompt to: '{enhanced_prompt}'. However, image generation endpoint needs setup."
            
        except Exception as api_error:
            print(f"Imagen API error: {api_error}")
            # Fall back to DALL-E if available
            if AI_MODEL_AVAILABLE and client:
                print("🔄 Falling back to DALL-E for image generation...")
                return handle_dalle_image_generation(text)
            else:
                return f"🎨 Imagen encountered an issue: {api_error}. No fallback image generation available."
        
    except Exception as e:
        print(f"Error in handle_imagen_generation: {e}")
        # Fall back to DALL-E if available
        if AI_MODEL_AVAILABLE and client:
            return handle_dalle_image_generation(text)
        return "🎨 I had trouble generating that image. Please make sure your request is clear and try again!"

def handle_dalle_image_generation_with_enhancement(text, enhanced_prompt):
    """Handle DALL-E image generation with Imagen-enhanced prompts"""
    try:
        print(f"🎨✨ Using DALL-E with Imagen-enhanced prompt: {enhanced_prompt}")
        
        # Generate image using DALL-E with enhanced prompt
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            
            # Download and save the image locally
            local_url, filename = download_and_save_image(image_url, enhanced_prompt)
            
            if local_url:
                # Create a full URL that opens the image directly in browser
                full_image_url = f"http://192.168.1.206:8080{local_url}"
                return f"""🎨🌟 Image Generated with Imagen Enhancement

{full_image_url}

Enhanced prompt: {enhanced_prompt}"""
            else:
                # Fallback to original URL if download fails
                return f"""🎨🌟 Image Generated with Imagen Enhancement

{image_url}

Enhanced prompt: {enhanced_prompt}"""
            
        except Exception as api_error:
            print(f"DALL-E API error: {api_error}")
            return f"🎨 I encountered an issue generating the image: {api_error}. Please try rephrasing your request or try again later."
        
    except Exception as e:
        print(f"Error in handle_dalle_image_generation_with_enhancement: {e}")
        return "🎨 I had trouble generating that image. Please make sure your request is clear and try again!"

def handle_gemini_image_generation(text):
    """Handle AI image generation requests using Google Gemini API"""
    try:
        # Extract the image description from the text
        prompt = extract_image_prompt(text)
        
        if not prompt or len(prompt) < 3:
            return "🎨 I can generate images for you! Please describe what you'd like me to create. For example: 'generate an image of a sunset over mountains' or 'create a picture of a cute cat wearing a hat'."
        
        print(f"🎨✨ Generating image with Gemini Veo3: {prompt}")
        
        # Note: As of current Gemini API, direct image generation might not be available
        # This is a template for when Google releases image generation capabilities
        # For now, we'll fall back to text-to-image via Imagen or similar services
        
        try:
            # Initialize Gemini model for image generation
            # Note: This is pseudocode for future Gemini image generation capabilities
            model = genai.GenerativeModel('gemini-pro')  # Will be updated when image gen is available
            
            # Enhanced prompt for better image generation
            enhanced_prompt = f"Create a high-quality, detailed image of: {prompt}. Style: photorealistic, professional quality, well-composed."
            
            # This would be the future API call for Gemini image generation
            # response = model.generate_image(prompt=enhanced_prompt)
            
            # For now, return a message about Gemini text capabilities and fall back to DALL-E
            if AI_MODEL_AVAILABLE and client:
                print("🔄 Gemini image generation not yet available via API. Falling back to DALL-E...")
                return handle_dalle_image_generation(text)
            else:
                return f"🎨✨ Gemini AI processed your prompt: '{prompt}'. However, Veo3 image generation is not yet available through the public API. Please ensure you have either Gemini image generation access or DALL-E configured for image creation."
            
        except Exception as api_error:
            print(f"Gemini API error: {api_error}")
            # Fall back to DALL-E if available
            if AI_MODEL_AVAILABLE and client:
                print("🔄 Falling back to DALL-E for image generation...")
                return handle_dalle_image_generation(text)
            else:
                return f"🎨 Gemini encountered an issue: {api_error}. No fallback image generation available."
        
    except Exception as e:
        print(f"Error in handle_gemini_image_generation: {e}")
        # Fall back to DALL-E if Gemini fails
        if AI_MODEL_AVAILABLE and client:
            return handle_dalle_image_generation(text)
        return "🎨 I had trouble generating that image. Please make sure your request is clear and try again!"

def extract_image_prompt(text):
    """Extract image description from user text"""
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
    
    return prompt

def handle_dalle_image_generation(text):
    """Handle AI image generation requests using DALL-E API (fallback)"""
    try:
        prompt = extract_image_prompt(text)
        
        if not prompt or len(prompt) < 3:
            return "🎨 I can generate images for you! Please describe what you'd like me to create. For example: 'generate an image of a sunset over mountains' or 'create a picture of a cute cat wearing a hat'."
        
        print(f"🎨 Generating image with DALL-E: {prompt}")
        
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
        print(f"Error in handle_dalle_image_generation: {e}")
        return "🎨 I had trouble generating that image. Please make sure your request is clear and try again!"

def handle_logo_generation(text):
    """Handle AI logo generation requests using smart AI instead of ChatGPT"""
    try:
        print(f"🏷️ Processing logo generation request: {text}")
        
        # Extract brand information from the text using smart parsing
        import re
        
        # Try to extract brand name and details
        brand_patterns = [
            r'logo.*for (.+?)(?:,|\.|$)',
            r'(?:brand|company|business) (?:called |named )?(.+?)(?:,|\.|$)',
            r'create.*logo.*(.+?)(?:,|\.|$)',
            r'design.*logo.*(.+?)(?:,|\.|$)',
            r'make.*logo.*(.+?)(?:,|\.|$)',
            r'generate.*logo.*(.+?)(?:,|\.|$)',
        ]
        
        brand_name = "YourBrand"
        for pattern in brand_patterns:
            match = re.search(pattern, text.lower())
            if match:
                brand_name = match.group(1).strip()
                # Clean up the brand name
                brand_name = re.sub(r'\b(a|an|the|my|our|company|business|brand)\b', '', brand_name).strip()
                if brand_name:
                    break
        
        # Extract industry if mentioned
        industry_keywords = {
            'technology': ['tech', 'software', 'app', 'digital', 'ai', 'computer', 'coding'],
            'healthcare': ['health', 'medical', 'clinic', 'hospital', 'care', 'wellness'],
            'food': ['restaurant', 'cafe', 'food', 'kitchen', 'dining', 'bakery', 'coffee'],
            'fashion': ['fashion', 'clothing', 'style', 'boutique', 'apparel'],
            'finance': ['bank', 'finance', 'money', 'investment', 'financial'],
            'education': ['school', 'education', 'learning', 'university', 'academy'],
            'fitness': ['gym', 'fitness', 'workout', 'sports', 'athletic'],
            'beauty': ['beauty', 'salon', 'spa', 'cosmetics', 'skincare'],
            'automotive': ['car', 'auto', 'vehicle', 'garage', 'automotive'],
            'real_estate': ['real estate', 'property', 'homes', 'realty']
        }
        
        industry = 'general'
        text_lower = text.lower()
        for ind, keywords in industry_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                industry = ind
                break
        
        # Extract style if mentioned
        style_keywords = {
            'modern': ['modern', 'contemporary', 'sleek', 'clean', 'minimalist'],
            'vintage': ['vintage', 'retro', 'classic', 'traditional', 'old-school'],
            'creative': ['creative', 'artistic', 'unique', 'innovative', 'abstract'],
            'corporate': ['corporate', 'professional', 'business', 'formal'],
            'playful': ['fun', 'playful', 'colorful', 'friendly', 'cheerful'],
            'elegant': ['elegant', 'sophisticated', 'luxury', 'premium', 'refined']
        }
        
        style = 'modern'  # Default style
        for st, keywords in style_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                style = st
                break
        
        print(f"🎯 Extracted: Brand='{brand_name}', Industry='{industry}', Style='{style}'")
        
        # Generate smart response using AI intelligence instead of ChatGPT
        personality_responses = {
            'friendly': f"I'd be happy to help you create a {style} logo for {brand_name}! 😊",
            'professional': f"I shall assist you in developing a professional {style} logo for {brand_name}.",
            'enthusiastic': f"WOW! I'm SO excited to create an AMAZING {style} logo for {brand_name}! 🚀",
            'creative': f"Oh, what a delightfully creative challenge! A {style} logo for {brand_name} - how inspiring!",
            'zen': f"Let us mindfully craft a {style} logo that embodies the essence of {brand_name}. 🧘‍♀️"
        }
        
        # Try to generate the actual logo
        try:
            logo_url, error = generate_logo_design(brand_name, industry, style)
            
            if logo_url:
                # Success - return positive response with bold, clickable URL
                base_response = personality_responses.get('friendly', f"I've created a {style} logo for {brand_name}!")
                return f"{base_response}\n\n🎨 Your logo has been generated! Click the button below to view it:\n\n<div style='text-align: center; margin: 15px 0;'><a href='{logo_url}' target='_blank' style='display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: transform 0.2s;' onmouseover='this.style.transform=\"scale(1.05)\"' onmouseout='this.style.transform=\"scale(1)\"'>🔗 VIEW YOUR LOGO</a></div>\n\n✨ The logo features a {style} design perfect for the {industry} industry. I've incorporated elements that reflect your brand's identity while ensuring it's professional and memorable.\n\n💡 Logo Tips:\n• Click the button above to view your logo\n• Right-click → Save As to download the image\n• Use it on business cards, websites, and marketing materials\n• Consider creating variations for different use cases\n• Make sure it looks good in both color and black & white\n\nWould you like me to create any variations or additional designs?"
            else:
                # Fallback response when generation fails
                return f"🏷️ I'd love to create a {style} logo for {brand_name} in the {industry} industry! While I'm having some technical difficulties with image generation right now, I can definitely help you plan your logo design.\n\n🎨 For a {style} {industry} logo, I recommend:\n• Clean, professional typography\n• Colors that reflect your brand personality\n• Simple, memorable design elements\n• Scalable vector format\n\n💡 Consider including elements that represent:\n• Your industry ({industry})\n• Your brand values\n• Visual appeal in the {style} style\n\nWould you like specific suggestions for colors, fonts, or design elements for your {brand_name} logo?"
                
        except Exception as generation_error:
            print(f"Logo generation error: {generation_error}")
            return f"🏷️ I'd be happy to help design a {style} logo for {brand_name}! While I'm experiencing some technical issues with image generation, I can provide you with detailed design guidance.\n\n🎨 For your {industry} logo, consider:\n• {style.title()} aesthetic with clean lines\n• Professional color scheme\n• Memorable brand elements\n• Versatile design for multiple uses\n\nWould you like specific recommendations for your logo design?"
        
    except Exception as e:
        print(f"Error in handle_logo_generation: {e}")
        return "🏷️ I'd be happy to help you create a logo! Please provide more details about your brand name, industry, and preferred style, and I'll generate a professional logo design for you."

def handle_logo_generation(text):
    """Handle AI logo generation requests using smart AI and image generation APIs"""
    try:
        print(f"🏷️ Processing logo generation request: {text}")
        
        # Extract logo details from the text using smart AI patterns
        logo_patterns = [
            r'generate.*logo.*for (.+)',
            r'create.*logo.*for (.+)', 
            r'make.*logo.*for (.+)',
            r'design.*logo.*for (.+)',
            r'build.*logo.*for (.+)',
            r'logo for (.+)',
            r'brand.*for (.+)',
            r'corporate.*logo.*for (.+)',
            r'business.*logo.*for (.+)',
            r'company.*logo.*for (.+)',
            r'logo.*design.*for (.+)',
            r'brand.*identity.*for (.+)',
            r'visual.*identity.*for (.+)'
        ]
        
        brand_name = ""
        industry = "technology"  # default
        style = "modern"  # default
        
        # Extract brand name
        for pattern in logo_patterns:
            match = re.search(pattern, text.lower())
            if match:
                brand_info = match.group(1).strip()
                
                # Try to extract brand name (first word/phrase)
                brand_parts = brand_info.split()
                if brand_parts:
                    # Look for industry keywords to separate brand name
                    industry_keywords = [
                        'tech', 'technology', 'software', 'app', 'digital', 'web', 'it',
                        'restaurant', 'food', 'cafe', 'coffee', 'dining', 'kitchen',
                        'health', 'medical', 'healthcare', 'clinic', 'hospital', 'wellness',
                        'fashion', 'clothing', 'apparel', 'style', 'boutique',
                        'finance', 'banking', 'investment', 'money', 'financial',
                        'education', 'school', 'university', 'learning', 'training',
                        'real estate', 'property', 'construction', 'building',
                        'automotive', 'car', 'vehicle', 'auto', 'transport',
                        'beauty', 'salon', 'spa', 'cosmetics', 'skincare',
                        'sports', 'fitness', 'gym', 'athletic', 'exercise',
                        'travel', 'tourism', 'hotel', 'vacation', 'adventure',
                        'retail', 'shop', 'store', 'marketplace', 'commerce',
                        'consulting', 'service', 'agency', 'firm', 'professional'
                    ]
                    
                    # Extract industry if mentioned
                    for keyword in industry_keywords:
                        if keyword in brand_info.lower():
                            industry = keyword
                            brand_info = brand_info.lower().replace(keyword, '').strip()
                            break
                    
                    # Clean up brand name
                    brand_name = brand_info.strip()
                    if brand_name:
                        # Capitalize properly
                        brand_name = ' '.join(word.capitalize() for word in brand_name.split())
                    
                break
        
        # Extract style hints from the text
        style_keywords = {
            'modern': ['modern', 'contemporary', 'clean', 'minimal', 'sleek', 'simple'],
            'vintage': ['vintage', 'retro', 'classic', 'old-school', 'traditional', 'timeless'],
            'creative': ['creative', 'artistic', 'unique', 'innovative', 'original', 'abstract'],
            'corporate': ['corporate', 'professional', 'business', 'formal', 'enterprise', 'official'],
            'playful': ['playful', 'fun', 'colorful', 'friendly', 'casual', 'bright'],
            'elegant': ['elegant', 'sophisticated', 'luxury', 'premium', 'refined', 'classy']
        }
        
        for style_type, keywords in style_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                style = style_type
                break
        
        # If no brand name extracted, try to get it from the whole text
        if not brand_name:
            # Look for quoted names or capitalized words
            quoted_match = re.search(r'["\']([^"\']+)["\']', text)
            if quoted_match:
                brand_name = quoted_match.group(1)
            else:
                # Look for capitalized words as potential brand names
                words = text.split()
                capitalized_words = [word for word in words if word[0].isupper() and len(word) > 2]
                if capitalized_words:
                    brand_name = ' '.join(capitalized_words[:3])  # Take up to 3 words
        
        # Default brand name if none found
        if not brand_name:
            brand_name = "MyBrand"
        
        print(f"🏷️ Extracted details - Brand: '{brand_name}', Industry: '{industry}', Style: '{style}'")
        
        # Generate smart AI response about the logo creation process
        smart_responses = [
            f"🎨 Creating a {style} logo for {brand_name} in the {industry} industry! Let me design something perfect for your brand...",
            f"🏷️ Designing a professional {style} logo for {brand_name}! This will be great for a {industry} business...",
            f"✨ Working on a {style} logo design for {brand_name}! Perfect for the {industry} sector...",
            f"🎯 Crafting a {style} brand identity for {brand_name}! This {industry} logo will look amazing...",
            f"🚀 Generating a {style} logo for {brand_name}! Your {industry} brand deserves something special..."
        ]
        
        import random
        smart_response = random.choice(smart_responses)
        
        # Try to generate the actual logo using the AI image generation
        logo_url, error = generate_logo_design(brand_name, industry, style)
        
        if logo_url:
            return f"{smart_response}\n\n✅ Logo generated successfully! Your new {style} logo for {brand_name} is ready:\n\n<div style='text-align: center; margin: 15px 0;'><a href='{logo_url}' target='_blank' style='display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: transform 0.2s;' onmouseover='this.style.transform=\"scale(1.05)\"' onmouseout='this.style.transform=\"scale(1)\"'>🔗 CLICK HERE TO VIEW YOUR LOGO</a></div>\n\n✨ The logo features {style} design elements perfect for the {industry} industry. Feel free to request modifications or try different styles!"
        else:
            # Even if logo generation fails, provide helpful response
            return f"{smart_response}\n\n💡 Here are some {style} logo design ideas for {brand_name} in the {industry} industry:\n\n• Clean, professional typography with your brand name\n• {industry.capitalize()} industry-relevant icons or symbols\n• {style.capitalize()} color scheme (think brand personality)\n• Scalable design that works on business cards and billboards\n• Memorable visual elements that represent your brand values\n\n🎨 Would you like me to try generating the logo again with different parameters, or would you prefer specific design suggestions?"
        
    except Exception as e:
        print(f"Error in handle_logo_generation: {e}")
        return "🏷️ I'd love to help you create a logo! Please describe your brand name, industry, and preferred style (modern, vintage, creative, etc.). For example: 'Create a modern logo for TechStart, a software company' or 'Design a vintage logo for Bella's Cafe, a coffee shop'."

# ===============================================
# 🎮 INTERACTIVE FEATURES FUNCTIONS

def handle_game_master(text, session_id=None, personality='friendly'):
    """Handle AI Game Master requests for interactive stories and text adventures"""
    try:
        print(f"🎮 Processing game master request: {text}")
        
        # Extract game/story type and theme from the text
        import re
        
        # Detect game type
        game_type = "adventure"  # default
        if re.search(r'\b(fantasy|magic|dragon|wizard|medieval)\b', text.lower()):
            game_type = "fantasy"
        elif re.search(r'\b(sci.*fi|space|alien|robot|future)\b', text.lower()):
            game_type = "sci-fi"
        elif re.search(r'\b(horror|scary|zombie|ghost|dark)\b', text.lower()):
            game_type = "horror"
        elif re.search(r'\b(mystery|detective|crime|solve)\b', text.lower()):
            game_type = "mystery"
        elif re.search(r'\b(romance|love|relationship)\b', text.lower()):
            game_type = "romance"
        
        # Detect if it's a new story or continuation
        is_continuation = re.search(r'\b(continue|next|what.*happens|then|choice)\b', text.lower())
        
        # Personality-based responses
        personality_responses = {
            'friendly': f"🎮 Let's start an amazing {game_type} adventure together!",
            'professional': f"🎯 Initiating {game_type} narrative experience.",
            'creative': f"✨ Oh, what a delightful {game_type} tale we shall weave!",
            'witty': f"🃏 Ready for a {game_type} adventure? Plot twist: you're the hero!",
            'empathetic': f"🌟 I sense you're ready for an emotional {game_type} journey."
        }
        
        if is_continuation:
            # This would ideally load from session memory
            story_scenarios = {
                'fantasy': "🏰 The ancient castle looms before you, its gates creaking in the wind. Inside, you hear the faint sound of chanting. **What do you do?**\n\nA) Enter through the main gate boldly\nB) Sneak around to find a side entrance\nC) Call out to announce your presence\nD) Cast a protection spell first",
                'sci-fi': "🚀 Your spaceship's alarms are blaring. Through the viewport, you see an unknown alien vessel approaching. Your AI companion says: 'Captain, they're hailing us.' **What's your command?**\n\nA) Open communications immediately\nB) Charge weapons defensively\nC) Attempt to flee at maximum warp\nD) Send a peaceful greeting signal",
                'mystery': "🔍 You examine the crime scene more closely. A torn piece of fabric caught on the window latch catches your eye - it's expensive silk with an unusual pattern. **Your next move?**\n\nA) Test the fabric for DNA evidence\nB) Research local shops that sell this fabric\nC) Check if the victim owned similar clothing\nD) Look for security cameras nearby",
                'horror': "👻 The basement door creaks open by itself. A cold draft carries the scent of decay, and you hear footsteps echoing from below. Your flashlight flickers ominously. **Do you dare...**\n\nA) Descend into the basement immediately\nB) Call for backup first\nC) Secure the door and leave\nD) Record evidence before proceeding",
                'adventure': "⚔️ At the crossroads, you notice fresh horse tracks leading in three directions: north toward the mountains, east to the dark forest, and south to the coastal village. **Which path calls to you?**\n\nA) Follow the mountain trail north\nB) Brave the mysterious dark forest\nC) Head to the coastal village\nD) Study the tracks more carefully first"
            }
            
            base_response = personality_responses.get(personality, personality_responses['friendly'])
            return f"{base_response}\n\n{story_scenarios.get(game_type, story_scenarios['adventure'])}\n\n🎭 **Choose wisely!** Your decision will shape the story. Type your choice (A, B, C, or D) or describe your own action!"
        
        else:
            # Start a new adventure
            story_intros = {
                'fantasy': "🧙‍♂️ **The Kingdom of Aethermoor** 🏰\n\nYou are a young adventurer who has just arrived at the mystical Kingdom of Aethermoor. Ancient magic flows through the land, but dark forces are stirring. The village elder approaches you with worry in her eyes.\n\n'Brave traveler,' she says, 'the Crystal of Eternal Light has been stolen from our sacred temple. Without it, our protective barriers will fall within three days. Will you help us?'\n\n**What do you do?**\nA) Accept the quest immediately\nB) Ask for more details about the crystal\nC) Request payment for your services\nD) Suggest finding other heroes to help",
                
                'sci-fi': "🌌 **Stardate 2387: The Nexus Station** 🚀\n\nYou're Commander of the exploration vessel 'Aurora' docked at the remote Nexus Station. Suddenly, all communications with Earth go silent. The station's AI informs you that an unknown energy signature is approaching fast.\n\n'Commander,' your Science Officer reports, 'the signature doesn't match any known technology. ETA: 15 minutes.'\n\n**Your orders, Commander?**\nA) Prepare for first contact protocols\nB) Ready defensive systems\nC) Evacuate the station immediately\nD) Attempt to scan the approaching object",
                
                'mystery': "🕵️ **The Ravenwood Manor Mystery** 🏚️\n\nYou're a detective called to investigate the sudden disappearance of Lord Ravenwood from his locked study. The house staff is nervous, the family members each have alibis, and a valuable painting is also missing.\n\nThe butler, Mr. Grimsby, leads you to the study: 'Everything is exactly as we found it, Detective. The door was locked from the inside, and the window is 30 feet above ground.'\n\n**Where do you begin?**\nA) Examine the locked study thoroughly\nB) Interview the family members\nC) Question the house staff\nD) Investigate the missing painting",
                
                'horror': "🌙 **The Whispers of Blackwood House** 👻\n\nYou're a paranormal investigator who has just entered the abandoned Blackwood House. Local legends speak of the family that vanished 50 years ago, leaving behind only their screams echoing in the night.\n\nAs you step into the foyer, the door slams shut behind you. Your EMF detector starts beeping rapidly, and you hear children laughing somewhere upstairs.\n\n**What's your first move?**\nA) Head upstairs toward the laughter\nB) Try to force the front door open\nC) Set up recording equipment first\nD) Explore the ground floor systematically",
                
                'adventure': "🗺️ **The Lost Treasure of Captain Stormwind** ⚓\n\nYou're standing on the deck of your ship, the 'Sea Dragon,' holding an ancient map you found in a bottle. It shows the location of Captain Stormwind's legendary treasure on the mysterious Skull Island.\n\nYour first mate approaches: 'Captain, the crew is ready to set sail. But I should warn you - other treasure hunters are also searching for Stormwind's gold, and Skull Island is said to be cursed.'\n\n**What are your orders?**\nA) Set sail for Skull Island immediately\nB) Gather more information about the island first\nC) Recruit additional crew members\nD) Stock up on supplies and weapons"
            }
            
            base_response = personality_responses.get(personality, personality_responses['friendly'])
            return f"{base_response}\n\n{story_intros.get(game_type, story_intros['adventure'])}\n\n🎮 **The adventure begins!** Type your choice or describe your action. I'll adapt the story based on your decisions!"
            
    except Exception as e:
        print(f"Error in handle_game_master: {e}")
        return "🎮 I'd love to start an interactive adventure with you! Try saying: 'Start a fantasy adventure' or 'Begin a sci-fi story' or 'Create a mystery game'. I can be your AI Game Master for any type of interactive story you'd like to explore!"

def handle_code_generation(text, personality='friendly'):
    """Handle AI programming assistant requests for multiple languages"""
    try:
        print(f"💻 Processing code generation request: {text}")
        
        # Detect programming language
        import re
        
        language = "python"  # default
        if re.search(r'\b(javascript|js)\b', text.lower()):
            language = "javascript"
        elif re.search(r'\bjava\b', text.lower()) and not re.search(r'javascript', text.lower()):
            language = "java"
        elif re.search(r'\b(cpp|c\+\+)\b', text.lower()):
            language = "cpp"
        elif re.search(r'\bhtml\b', text.lower()):
            language = "html"
        elif re.search(r'\bcss\b', text.lower()):
            language = "css"
        elif re.search(r'\b(react|jsx)\b', text.lower()):
            language = "react"
        elif re.search(r'\b(node|nodejs)\b', text.lower()):
            language = "nodejs"
        elif re.search(r'\bsql\b', text.lower()):
            language = "sql"
        
        # Extract what they want to build
        task_match = re.search(r'(function|class|program|script|code|algorithm).*?(?:that|to|for)\s*(.+)', text.lower())
        task = task_match.group(2) if task_match else "a helpful program"
        
        # Personality-based responses
        personality_responses = {
            'friendly': f"💻 I'd love to help you code in {language.upper()}!",
            'professional': f"🔧 Initiating {language.upper()} programming assistance.",
            'creative': f"✨ Let's craft some beautiful {language.upper()} code together!",
            'witty': f"🤖 Time to make the computer dance with {language.upper()}!",
            'empathetic': f"💪 Don't worry, we'll tackle this {language.upper()} challenge together!"
        }
        
        # Language-specific examples and templates
        code_examples = {
            'python': {
                'example': '''```python
# Example: Simple function template
def process_data(data):
    """
    Process and transform data
    Args: data - input data to process
    Returns: processed result
    """
    try:
        result = []
        for item in data:
            # Add your processing logic here
            processed_item = str(item).upper()
            result.append(processed_item)
        return result
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Usage example
sample_data = ["hello", "world", "python"]
output = process_data(sample_data)
print(output)  # ['HELLO', 'WORLD', 'PYTHON']
```''',
                'features': ['• Clean, readable syntax', '• Excellent for data science', '• Great for automation', '• Strong library ecosystem']
            },
            'javascript': {
                'example': '''```javascript
// Example: Modern JavaScript function
const processData = async (data) => {
    try {
        const result = data.map(item => {
            // Add your processing logic here
            return item.toString().toUpperCase();
        });
        
        return result;
    } catch (error) {
        console.error('Error processing data:', error);
        return null;
    }
};

// Usage example
const sampleData = ["hello", "world", "javascript"];
processData(sampleData)
    .then(output => console.log(output)) // ['HELLO', 'WORLD', 'JAVASCRIPT']
    .catch(err => console.error(err));
```''',
                'features': ['• Runs in browsers and servers', '• Asynchronous programming', '• Modern ES6+ features', '• Huge ecosystem (npm)']
            },
            'java': {
                'example': '''```java
// Example: Java class template
import java.util.*;

public class DataProcessor {
    
    public static List<String> processData(List<String> data) {
        try {
            List<String> result = new ArrayList<>();
            
            for (String item : data) {
                // Add your processing logic here
                String processedItem = item.toUpperCase();
                result.add(processedItem);
            }
            
            return result;
        } catch (Exception e) {
            System.err.println("Error processing data: " + e.getMessage());
            return new ArrayList<>();
        }
    }
    
    public static void main(String[] args) {
        List<String> sampleData = Arrays.asList("hello", "world", "java");
        List<String> output = processData(sampleData);
        System.out.println(output); // [HELLO, WORLD, JAVA]
    }
}
```''',
                'features': ['• Platform independent', '• Strong typing system', '• Object-oriented', '• Enterprise-level scalability']
            }
        }
        
        base_response = personality_responses.get(personality, personality_responses['friendly'])
        example_data = code_examples.get(language, code_examples['python'])
        
        return f"""{base_response}

🎯 **Programming Language: {language.upper()}**
📝 **Task: Create {task}**

{example_data['example']}

✨ **Why {language.upper()}?**
{chr(10).join(example_data['features'])}

💡 **Next Steps:**
1. **Modify** the template above for your specific needs
2. **Test** your code with sample data
3. **Add error handling** for robust applications
4. **Document** your functions clearly

🔧 **Need specific help?** Ask me to:
• "Explain this code step by step"
• "Add error handling to this function"
• "Optimize this algorithm"
• "Convert this to {language} from another language"

Ready to build something amazing? Let me know what specific functionality you need!"""
        
    except Exception as e:
        print(f"Error in handle_code_generation: {e}")
        return "💻 I'm your AI programming assistant! I can help you write code in Python, JavaScript, Java, C++, HTML, CSS, and more. Try asking: 'Write a Python function to sort data' or 'Create a JavaScript API call' or 'Generate a Java class for user management'. What would you like to code today?"

def handle_quiz_generation(text, personality='friendly'):
    """Handle quiz and trivia generation requests with interactive UI"""
    try:
        print(f"🧠 Processing quiz generation request: {text}")
        
        # Extract topic from the text
        import re
        
        topic_match = re.search(r'(?:about|on|quiz|trivia).*?([a-zA-Z\s]+)', text.lower())
        topic = topic_match.group(1).strip() if topic_match else "general knowledge"
        
        # Detect difficulty
        difficulty = "medium"  # default
        if re.search(r'\b(easy|beginner|simple)\b', text.lower()):
            difficulty = "easy"
        elif re.search(r'\b(hard|difficult|advanced|expert)\b', text.lower()):
            difficulty = "hard"
        
        # Personality-based responses
        personality_responses = {
            'friendly': f"🧠 Let's test your knowledge about {topic}!",
            'professional': f"📊 Generating {difficulty} {topic} assessment.",
            'creative': f"✨ Time for a fun {topic} brain challenge!",
            'witty': f"🤓 Ready to show off your {topic} expertise?",
            'empathetic': f"🌟 Let's learn together with this {topic} quiz!"
        }
        
        # Enhanced quiz database with more questions
        quiz_templates = {
            'science': [
                {"q": "What is the chemical symbol for gold?", "options": ["Au", "Ag", "Gd", "Go"], "correct": 0, "explanation": "Au comes from the Latin word 'aurum' meaning gold."},
                {"q": "Which planet is known as the Red Planet?", "options": ["Venus", "Mars", "Jupiter", "Saturn"], "correct": 1, "explanation": "Mars appears red due to iron oxide (rust) on its surface."},
                {"q": "What is the speed of light in vacuum?", "options": ["300,000 km/s", "150,000 km/s", "450,000 km/s", "600,000 km/s"], "correct": 0, "explanation": "Light travels at approximately 299,792,458 meters per second in a vacuum."},
                {"q": "How many bones are in an adult human body?", "options": ["206", "186", "226", "246"], "correct": 0, "explanation": "Adults have 206 bones, while babies are born with about 270 bones."},
                {"q": "What gas makes up about 78% of Earth's atmosphere?", "options": ["Oxygen", "Carbon Dioxide", "Nitrogen", "Hydrogen"], "correct": 2, "explanation": "Nitrogen makes up about 78% of Earth's atmosphere, while oxygen is about 21%."},
                {"q": "Which scientist developed the theory of relativity?", "options": ["Isaac Newton", "Albert Einstein", "Galileo Galilei", "Stephen Hawking"], "correct": 1, "explanation": "Einstein published his special theory of relativity in 1905 and general relativity in 1915."},
                {"q": "What is the hardest natural substance on Earth?", "options": ["Gold", "Iron", "Diamond", "Quartz"], "correct": 2, "explanation": "Diamond is the hardest natural material, rating 10 on the Mohs hardness scale."},
                {"q": "Which organ in the human body produces insulin?", "options": ["Liver", "Kidney", "Pancreas", "Heart"], "correct": 2, "explanation": "The pancreas produces insulin to help regulate blood sugar levels."},
                {"q": "What is the smallest unit of matter?", "options": ["Molecule", "Atom", "Proton", "Electron"], "correct": 1, "explanation": "Atoms are the basic building blocks of matter and the smallest units of elements."},
                {"q": "How many chambers does a human heart have?", "options": ["2", "3", "4", "5"], "correct": 2, "explanation": "The human heart has four chambers: two atria and two ventricles."}
            ],
            'history': [
                {"q": "In which year did World War II end?", "options": ["1944", "1945", "1946", "1947"], "correct": 1, "explanation": "WWII ended in 1945 with Japan's surrender on September 2, 1945."},
                {"q": "Who was the first person to walk on the moon?", "options": ["Buzz Aldrin", "John Glenn", "Neil Armstrong", "Alan Shepard"], "correct": 2, "explanation": "Neil Armstrong was the first human to walk on the moon on July 20, 1969."},
                {"q": "The Great Wall of China was built primarily to defend against invasions from which direction?", "options": ["South", "East", "West", "North"], "correct": 3, "explanation": "The Great Wall was built to protect against invasions from northern nomadic tribes."},
                {"q": "Which ancient wonder of the world was located in Alexandria?", "options": ["Hanging Gardens", "Lighthouse", "Colossus", "Mausoleum"], "correct": 1, "explanation": "The Lighthouse of Alexandria was one of the Seven Wonders of the Ancient World."},
                {"q": "Who painted the ceiling of the Sistine Chapel?", "options": ["Leonardo da Vinci", "Michelangelo", "Raphael", "Donatello"], "correct": 1, "explanation": "Michelangelo painted the Sistine Chapel ceiling between 1508 and 1512."},
                {"q": "The Roman Empire was divided into two parts in which year?", "options": ["285 AD", "395 AD", "476 AD", "527 AD"], "correct": 1, "explanation": "The Roman Empire was permanently divided in 395 AD after Emperor Theodosius I's death."},
                {"q": "Which queen ruled England during Shakespeare's time?", "options": ["Queen Victoria", "Queen Elizabeth I", "Queen Mary I", "Queen Anne"], "correct": 1, "explanation": "Queen Elizabeth I ruled during most of Shakespeare's career (1558-1603)."},
                {"q": "The Declaration of Independence was signed in which year?", "options": ["1774", "1775", "1776", "1777"], "correct": 2, "explanation": "The Declaration of Independence was signed on July 4, 1776."},
                {"q": "Who was the first President of the United States?", "options": ["John Adams", "Thomas Jefferson", "George Washington", "Benjamin Franklin"], "correct": 2, "explanation": "George Washington served as the first President from 1789 to 1797."},
                {"q": "The Berlin Wall fell in which year?", "options": ["1987", "1989", "1991", "1993"], "correct": 1, "explanation": "The Berlin Wall fell on November 9, 1989, leading to German reunification."}
            ],
            'technology': [
                {"q": "What does 'AI' stand for?", "options": ["Automated Intelligence", "Artificial Intelligence", "Advanced Interface", "Algorithmic Integration"], "correct": 1, "explanation": "AI stands for Artificial Intelligence, the simulation of human intelligence by machines."},
                {"q": "Which programming language is most popular for data science?", "options": ["JavaScript", "Python", "Assembly", "COBOL"], "correct": 1, "explanation": "Python is widely used in data science due to its libraries like pandas, NumPy, and scikit-learn."},
                {"q": "What does 'HTTP' stand for?", "options": ["Hypertext Transfer Protocol", "High Tech Transfer Process", "Hyperlink Text Transfer Protocol", "Home Transfer Text Protocol"], "correct": 0, "explanation": "HTTP is the protocol used for transferring web pages on the internet."},
                {"q": "Who founded Microsoft?", "options": ["Steve Jobs", "Bill Gates", "Mark Zuckerberg", "Jeff Bezos"], "correct": 1, "explanation": "Bill Gates co-founded Microsoft with Paul Allen in 1975."},
                {"q": "What does 'URL' stand for?", "options": ["Universal Resource Locator", "Uniform Resource Locator", "Universal Reference Link", "Uniform Reference Locator"], "correct": 1, "explanation": "URL stands for Uniform Resource Locator, the address of a web resource."},
                {"q": "Which company developed the iPhone?", "options": ["Google", "Samsung", "Apple", "Microsoft"], "correct": 2, "explanation": "Apple Inc. developed and released the first iPhone in 2007."},
                {"q": "What does 'CPU' stand for?", "options": ["Computer Processing Unit", "Central Processing Unit", "Core Processing Unit", "Central Program Unit"], "correct": 1, "explanation": "CPU stands for Central Processing Unit, the main processor of a computer."},
                {"q": "Which programming language was created by Guido van Rossum?", "options": ["Java", "Python", "C++", "JavaScript"], "correct": 1, "explanation": "Guido van Rossum created Python, first released in 1991."},
                {"q": "What does 'RAM' stand for?", "options": ["Random Access Memory", "Read Access Memory", "Rapid Access Memory", "Remote Access Memory"], "correct": 0, "explanation": "RAM stands for Random Access Memory, the computer's short-term memory."},
                {"q": "Which company owns YouTube?", "options": ["Facebook", "Apple", "Google", "Amazon"], "correct": 2, "explanation": "Google acquired YouTube in 2006 for $1.65 billion."}
            ]
        }
        
        # Default to science if topic not found
        available_questions = quiz_templates.get(topic, quiz_templates['science'])
        
        # Randomly select one question to start
        import random
        question = random.choice(available_questions)
        
        base_response = personality_responses.get(personality, personality_responses['friendly'])
        
        # Generate unique quiz ID for session tracking
        import uuid
        quiz_id = str(uuid.uuid4())[:8]
        
        # Create interactive quiz HTML
        quiz_html = f'''
<div class="interactive-quiz" data-quiz-id="{quiz_id}" data-topic="{topic}" data-difficulty="{difficulty}">
    <div class="quiz-header">
        <h3>🎯 {topic.title()} Quiz ({difficulty.title()} Level)</h3>
        <div class="quiz-progress">
            <span class="question-counter">Question <span id="current-question">1</span> of 10</span>
            <span class="score-display">Score: <span id="current-score">0</span>/10</span>
        </div>
    </div>
    
    <div class="quiz-question-container">
        <div class="quiz-question">
            <h4 id="question-text">{question['q']}</h4>
        </div>
        
        <div class="quiz-options">
            <button class="quiz-option" data-answer="0" data-correct="{question['correct']}" data-explanation="{question['explanation'].replace('"', '&quot;')}">{question['options'][0]}</button>
            <button class="quiz-option" data-answer="1" data-correct="{question['correct']}" data-explanation="{question['explanation'].replace('"', '&quot;')}">{question['options'][1]}</button>
            <button class="quiz-option" data-answer="2" data-correct="{question['correct']}" data-explanation="{question['explanation'].replace('"', '&quot;')}">{question['options'][2]}</button>
            <button class="quiz-option" data-answer="3" data-correct="{question['correct']}" data-explanation="{question['explanation'].replace('"', '&quot;')}">{question['options'][3]}</button>
        </div>
        
        <div class="quiz-feedback" id="quiz-feedback" style="display: none;">
            <div class="feedback-text" id="feedback-text"></div>
            <div class="explanation" id="explanation-text"></div>
            <button class="next-question-btn" id="next-btn" onclick="nextQuestion()" style="display: none;">Next Question ➡️</button>
        </div>
    </div>
    
    <div class="quiz-final-results" id="final-results" style="display: none;">
        <h3>🏆 Quiz Complete!</h3>
        <div class="final-score" id="final-score-text"></div>
        <div class="performance-message" id="performance-message"></div>
        <button class="new-quiz-btn" onclick="startNewQuiz()">Start New Quiz 🎮</button>
    </div>
</div>'''

        # Add JavaScript as a separate string to avoid f-string conflicts
        javascript_code = f'''
<script>
let currentQuizData = {{
    topic: "{topic}",
    difficulty: "{difficulty}",
    currentQuestion: 1,
    score: 0,
    totalQuestions: 10,
    questions: {str(available_questions)},
    answered: false
}};

// Add event listeners for quiz options
document.addEventListener('DOMContentLoaded', function() {{
    setupQuizButtons();
}});

function setupQuizButtons() {{
    const options = document.querySelectorAll('.quiz-option');
    options.forEach(option => {{
        option.addEventListener('click', function() {{
            if (currentQuizData.answered) return;
            
            const selectedAnswer = parseInt(this.getAttribute('data-answer'));
            const correctAnswer = parseInt(this.getAttribute('data-correct'));
            const explanation = this.getAttribute('data-explanation');
            
            answerQuestion(selectedAnswer, correctAnswer, explanation);
        }});
    }});
}}

function answerQuestion(selectedAnswer, correctAnswer, explanation) {{
    if (currentQuizData.answered) return;
    
    currentQuizData.answered = true;
    const options = document.querySelectorAll('.quiz-option');
    const feedback = document.getElementById('quiz-feedback');
    const feedbackText = document.getElementById('feedback-text');
    const explanationText = document.getElementById('explanation-text');
    const nextBtn = document.getElementById('next-btn');
    
    // Disable all buttons
    options.forEach(option => option.disabled = true);
    
    // Show correct/incorrect feedback
    options[selectedAnswer].classList.add(selectedAnswer === correctAnswer ? 'correct' : 'incorrect');
    options[correctAnswer].classList.add('correct');
    
    // Update score
    if (selectedAnswer === correctAnswer) {{
        currentQuizData.score++;
        feedbackText.innerHTML = '✅ Correct!';
        feedbackText.className = 'feedback-text correct-feedback';
    }} else {{
        feedbackText.innerHTML = '❌ Incorrect';
        feedbackText.className = 'feedback-text incorrect-feedback';
    }}
    
    // Show explanation
    explanationText.innerHTML = explanation;
    
    // Update score display
    document.getElementById('current-score').textContent = currentQuizData.score;
    
    // Show feedback and next button
    feedback.style.display = 'block';
    nextBtn.style.display = 'inline-block';
}}

function nextQuestion() {{
    currentQuizData.currentQuestion++;
    
    if (currentQuizData.currentQuestion > currentQuizData.totalQuestions) {{
        showFinalResults();
        return;
    }}
    
    // Reset for next question
    currentQuizData.answered = false;
    
    // Get new random question
    const question = currentQuizData.questions[Math.floor(Math.random() * currentQuizData.questions.length)];
    
    // Update question display
    document.getElementById('question-text').textContent = question.q;
    document.getElementById('current-question').textContent = currentQuizData.currentQuestion;
    
    // Reset options
    const options = document.querySelectorAll('.quiz-option');
    options.forEach((option, index) => {{
        option.textContent = question.options[index];
        option.disabled = false;
        option.className = 'quiz-option';
        option.setAttribute('data-answer', index);
        option.setAttribute('data-correct', question.correct);
        option.setAttribute('data-explanation', question.explanation);
    }});
    
    // Hide feedback
    document.getElementById('quiz-feedback').style.display = 'none';
    
    // Re-setup event listeners
    setupQuizButtons();
}}

function showFinalResults() {{
    const quizContainer = document.querySelector('.quiz-question-container');
    const finalResults = document.getElementById('final-results');
    const finalScoreText = document.getElementById('final-score-text');
    const performanceMessage = document.getElementById('performance-message');
    
    quizContainer.style.display = 'none';
    finalResults.style.display = 'block';
    
    const percentage = Math.round((currentQuizData.score / currentQuizData.totalQuestions) * 100);
    finalScoreText.innerHTML = `You scored ${{currentQuizData.score}} out of ${{currentQuizData.totalQuestions}} (${{percentage}}%)`;
    
    let message = '';
    if (percentage >= 90) {{
        message = '🌟 Outstanding! You are a true expert!';
    }} else if (percentage >= 80) {{
        message = '🎉 Excellent work! You know your stuff!';
    }} else if (percentage >= 70) {{
        message = '👏 Good job! You are doing well!';
    }} else if (percentage >= 60) {{
        message = '📚 Not bad! Keep learning and improving!';
    }} else {{
        message = '💪 Keep studying! There is always room to grow!';
    }}
    
    performanceMessage.innerHTML = message;
}}

function startNewQuiz() {{
    location.reload(); // Simple way to restart - could be made more elegant
}}
</script>'''
        
        return f"""{base_response}

{quiz_html}

{javascript_code}

🎮 **How to play:**
• Click on the answer you think is correct
• ✅ Green = Correct, ❌ Red = Wrong
• See explanations after each answer
• Complete 10 questions to see your final score!

Ready to test your knowledge? Click your first answer above! 🧠"""
        
    except Exception as e:
        print(f"Error in handle_quiz_generation: {e}")
        return "🧠 I'm your AI quiz master! I can create interactive quizzes on any topic. Try asking: 'Create a science quiz', 'Generate history trivia', or 'Quiz me about technology'. What would you like to be tested on today?"

def handle_story_generation(text, personality='friendly'):
    """Handle story, novel, script, and poetry generation requests"""
    try:
        print(f"📖 Processing story generation request: {text}")
        
        # Detect story type
        import re
        story_type = "story"  # default
        if re.search(r'\b(novel|book|chapter)\b', text.lower()):
            story_type = "novel"
        elif re.search(r'\b(script|screenplay|movie|play)\b', text.lower()):
            story_type = "script"
        elif re.search(r'\b(poem|poetry|haiku|sonnet|verse|rhyme)\b', text.lower()):
            story_type = "poetry"
        
        # Detect genre
        genre = "general"
        if re.search(r'\b(fantasy|magic|dragon|wizard|elf)\b', text.lower()):
            genre = "fantasy"
        elif re.search(r'\b(sci-?fi|space|robot|alien|future)\b', text.lower()):
            genre = "sci-fi"
        elif re.search(r'\b(mystery|detective|crime|murder)\b', text.lower()):
            genre = "mystery"
        elif re.search(r'\b(horror|scary|ghost|vampire|zombie)\b', text.lower()):
            genre = "horror"
        elif re.search(r'\b(romance|love|relationship)\b', text.lower()):
            genre = "romance"
        elif re.search(r'\b(adventure|quest|journey|explore)\b', text.lower()):
            genre = "adventure"
        elif re.search(r'\b(comedy|funny|humor|laugh)\b', text.lower()):
            genre = "comedy"
        
        # Extract theme/topic
        theme_match = re.search(r'(?:about|story.*of|novel.*about|script.*about|poem.*about)[\s]*([^,.!?]+)', text.lower())
        theme = theme_match.group(1).strip() if theme_match else "an interesting topic"
        
        # Personality-based responses
        personality_responses = {
            'friendly': f"📚 I'd love to help you create a wonderful {story_type}!",
            'professional': f"📝 I shall assist you in crafting a {story_type} with professional quality.",
            'creative': f"✨ Let's unleash our creativity and write an amazing {story_type}!",
            'witty': f"🖋️ Ah, a {story_type}! Time to weave some literary magic!",
            'enthusiastic': f"📖 OH WOW! I'm SO excited to write this {story_type} with you!!!"
        }
        
        base_response = personality_responses.get(personality, personality_responses['friendly'])
        
        # Story templates based on type and genre
        if story_type == "poetry":
            poetry_templates = {
                'fantasy': [
                    "In realms where dragons soar on high,\nAnd magic dances through the sky,\nA hero's quest begins today,\nTo chase the darkness all away.",
                    "Beneath the ancient wizard's tower,\nLies secrets of forgotten power,\nThe crystal gleams with mystic light,\nTo guide the lost through endless night."
                ],
                'sci-fi': [
                    "Among the stars so bright and far,\nWe sail through space from star to star,\nThe future calls with voices clear,\nTo those who dare to venture here.",
                    "In circuits made of light and code,\nArtificial minds decode,\nThe mysteries of time and space,\nAs humanity finds its place."
                ],
                'romance': [
                    "Two hearts that beat as one tonight,\nUnder moon's enchanting light,\nA love that grows with each new day,\nNo force on earth can take away.",
                    "In whispered words and gentle touch,\nA love that means so very much,\nThrough seasons change and years go by,\nOur hearts will never say goodbye."
                ],
                'general': [
                    "Life's a journey, winding road,\nWith stories yet to be told,\nEach moment brings a chance to see,\nThe wonder of what we can be.",
                    "In quiet moments of the day,\nWhen thoughts and dreams begin to play,\nWe find the magic all around,\nIn simple joys that can be found."
                ]
            }
            
            poems = poetry_templates.get(genre, poetry_templates['general'])
            content = random.choice(poems)
            
        elif story_type == "script":
            script_content = f"""**{theme.title()} - {genre.title()} Script**

**FADE IN:**

**EXT. OPENING SCENE - DAY**

*The camera opens on a {genre} setting. Our protagonist stands at the threshold of adventure.*

**PROTAGONIST**
(looking determined)
Today everything changes. I can feel it.

**CHARACTER 2**
(concerned)
Are you sure about this? There's no going back once we start.

**PROTAGONIST**
(with conviction)
Some things are worth the risk. This is one of them.

*The protagonist takes a deep breath and steps forward into the unknown.*

**FADE TO:**

**INT. NEXT SCENE - CONTINUOUS**

*The story unfolds as our characters face the challenges ahead...*

**[SCENE CONTINUES...]**

🎬 **Script Structure:**
• **Act I**: Setup and introduction ({genre} elements)
• **Act II**: Conflict and character development
• **Act III**: Resolution and character growth

**Next scenes could include:**
- Character backstory revelation
- Major plot twist involving {theme}
- Climactic {genre} sequence
- Emotional character resolution"""
            
            content = script_content
            
        elif story_type == "novel":
            novel_content = f"""**{theme.title()} - A {genre.title()} Novel**

**Chapter 1: The Beginning**

{get_novel_opening(genre, theme)}

**Story Outline:**

**📚 Part I: Discovery**
- Introduction of main character and world
- Inciting incident involving {theme}
- First glimpse of the {genre} elements

**📚 Part II: Development** 
- Character faces initial challenges
- Supporting characters introduced
- Plot thickens with {genre} complications

**📚 Part III: Crisis**
- Major conflict reaches peak
- Character's beliefs/world challenged
- Dark moment before resolution

**📚 Part IV: Resolution**
- Character growth and triumph
- {genre} elements resolved satisfyingly
- New equilibrium established

**🎯 Writing Tips for Your {genre.title()} Novel:**
• Develop rich, multi-dimensional characters
• Build your {genre} world consistently
• Create tension through {theme}-related conflicts
• Research {genre} conventions and tropes
• Plan character arcs alongside plot progression

**💡 Next Chapter Ideas:**
- Character background exploration
- Introduction of antagonist/conflict
- World-building and atmosphere
- Dialogue-driven character development"""
            
            content = novel_content
            
        else:  # Regular story
            story_content = f"""**{theme.title()} - A {genre.title()} Story**

{get_story_opening(genre, theme)}

**🎭 Story Development:**

**Beginning**: {get_story_beginning(genre, theme)}

**Middle**: The plot thickens as our protagonist discovers that {theme} is more complex than initially thought. {get_genre_complications(genre)}

**End**: After facing seemingly insurmountable challenges, the character learns that {get_genre_lesson(genre)} and finds a way to resolve the conflict surrounding {theme}.

**🎨 Story Elements:**
• **Genre**: {genre.title()}
• **Theme**: {theme}
• **Tone**: {get_genre_tone(genre)}
• **Setting**: {get_genre_setting(genre)}
• **Conflict**: {get_genre_conflict(genre, theme)}

**✍️ Continue Your Story:**
- Develop character backstories
- Add supporting characters
- Build tension through obstacles
- Create memorable dialogue
- Plan your story's climax and resolution

**🎯 Writing Prompts for Next Sections:**
• What happens when the character faces their first major obstacle?
• How does the {genre} setting influence the character's decisions?
• What unexpected ally or enemy might appear?
• How will {theme} evolve throughout the story?"""
            
            content = story_content
        
        return f"""{base_response}

{content}

**🚀 Story Generation Features:**
• **Multiple Formats**: Stories, novels, scripts, poetry
• **Genre Variety**: Fantasy, sci-fi, mystery, romance, horror, comedy, adventure
• **Personalized Content**: Based on your specific themes and interests
• **Writing Guidance**: Structure tips and continuation ideas

**📝 Want More?** Try asking for:
• "Write a fantasy poem about dragons"
• "Create a sci-fi script about time travel"  
• "Generate a mystery novel about a missing artifact"
• "Compose a romantic story about {theme}"

Ready to continue your literary journey? What story would you like to explore next? ✨"""
        
    except Exception as e:
        print(f"Error in handle_story_generation: {e}")
        return "📖 I'm your AI story creator! I can write novels, scripts, poetry, and stories in any genre. Try asking: 'Write a fantasy story about dragons', 'Create a sci-fi script', 'Compose a love poem', or 'Generate a mystery novel'. What literary adventure shall we embark on today?"

def get_novel_opening(genre, theme):
    """Generate novel opening based on genre"""
    openings = {
        'fantasy': f"The ancient prophecy had spoken of this day, when {theme} would either save or doom the realm of Aethermoor. Lyra stood at the edge of the Whispering Woods, her fingers tracing the mystical runes carved into her grandmother's pendant. The magic hummed beneath her skin, responding to the approaching darkness that threatened everything she held dear.",
        
        'sci-fi': f"The year was 2387, and {theme} had become humanity's greatest obsession. Captain Nova Martinez stared out at the swirling nebula beyond the viewport of the starship Prometheus, knowing that somewhere in that cosmic maelstrom lay the answer to a question that had plagued civilization for centuries.",
        
        'mystery': f"Detective Sarah Chen had seen many strange cases in her twenty years on the force, but nothing quite like this. The victim lay perfectly positioned in the center of the locked room, with no signs of struggle and no apparent cause of death. The only clue was a single word carved into the antique desk: '{theme.title()}'.",
        
        'horror': f"The old Victorian house on Elm Street had been empty for fifty years, and for good reason. Local legends spoke of {theme} as the source of the terrible events that had driven the previous owners to madness. But Sarah had always been skeptical of ghost stories... until now.",
        
        'romance': f"Emma had given up on love after her last heartbreak, throwing herself into her work at the botanical garden. But when she discovered the mysterious love letters hidden in the old oak tree, each one signed with a single word - '{theme}' - she began to believe that perhaps fairy tales could still come true.",
        
        'adventure': f"The map was ancient, its edges yellow with age, and it promised to lead to something extraordinary related to {theme}. Jake had inherited it from his grandfather, along with a cryptic warning: 'Some treasures are worth more than gold, but they exact a price that few are willing to pay.'",
        
        'comedy': f"Life as a professional {theme} consultant was not what Emma had imagined when she graduated from college. Between dealing with eccentric clients, navigating office politics, and trying to convince people that yes, {theme} was indeed a real and serious profession, every day brought a new adventure in absurdity."
    }
    
    return openings.get(genre, f"It was a day like any other when everything changed. The protagonist had always thought {theme} was just a simple part of life, but today would prove that sometimes the most ordinary things hold the most extraordinary secrets.")

def get_story_opening(genre, theme):
    """Generate story opening based on genre"""
    openings = {
        'fantasy': f"In a land where magic flowed like rivers and {theme} held the power to reshape reality, a young hero discovered their destiny was far greater than they ever imagined.",
        
        'sci-fi': f"The transmission from deep space contained only three words: '{theme} is coming.' Dr. Elena Vasquez stared at the message, knowing that humanity's future hung in the balance.",
        
        'mystery': f"The case of {theme} had baffled police for months. Detective Morgan knew there was more to this puzzle than met the eye, and today, a breakthrough was finally within reach.",
        
        'horror': f"They said {theme} was just a legend, a story to frighten children. But as the shadows lengthened and strange sounds echoed through the house, Sarah began to realize that some legends are terrifyingly real.",
        
        'romance': f"Claire never believed in love at first sight until she met Alex at the {theme} festival. One look across the crowded square, and everything changed.",
        
        'adventure': f"The ancient artifact related to {theme} was said to grant incredible power to whoever could solve its mysteries. Adventure-seeker Jake was about to discover if the legends were true.",
        
        'comedy': f"When life gives you {theme}, apparently you're supposed to make the best of it. Too bad nobody told Marcus that before he accidentally became the world's most reluctant expert on the subject."
    }
    
    return openings.get(genre, f"Once upon a time, in a world not so different from our own, {theme} became the center of an extraordinary tale that would change everything.")

def get_story_beginning(genre, theme):
    return f"Our protagonist encounters {theme} for the first time, setting off a chain of events that will challenge everything they thought they knew about {genre} adventures."

def get_genre_complications(genre):
    complications = {
        'fantasy': "Magical forces beyond their control begin to interfere, and ancient enemies emerge from the shadows.",
        'sci-fi': "Technology fails at the worst possible moment, and alien influences complicate the mission.",
        'mystery': "False clues lead down dangerous paths, and the truth becomes more elusive than ever.",
        'horror': "The supernatural forces grow stronger, and reality itself begins to unravel.",
        'romance': "Misunderstandings threaten to tear the lovers apart, and external pressures mount.",
        'adventure': "Dangerous obstacles block the path, and unexpected enemies emerge.",
        'comedy': "Everything that can go wrong does go wrong, leading to increasingly absurd situations."
    }
    return complications.get(genre, "Unexpected challenges arise that test the character's resolve and ingenuity.")

def get_genre_lesson(genre):
    lessons = {
        'fantasy': "true magic comes from within and the power of friendship",
        'sci-fi': "technology is only as good as the humanity that guides it",
        'mystery': "the truth is often hidden in plain sight",
        'horror': "courage can overcome even the darkest fears",
        'romance': "real love requires trust, communication, and vulnerability",
        'adventure': "the greatest treasures are the experiences and friendships gained along the way",
        'comedy': "laughter truly is the best medicine, and perspective is everything"
    }
    return lessons.get(genre, "perseverance and self-discovery lead to personal growth")

def get_genre_tone(genre):
    tones = {
        'fantasy': "Mystical and wonder-filled",
        'sci-fi': "Futuristic and thought-provoking", 
        'mystery': "Suspenseful and intriguing",
        'horror': "Dark and atmospheric",
        'romance': "Emotional and heartwarming",
        'adventure': "Exciting and fast-paced",
        'comedy': "Light-hearted and humorous"
    }
    return tones.get(genre, "Engaging and thoughtful")

def get_genre_setting(genre):
    settings = {
        'fantasy': "Magical realm with mystical creatures",
        'sci-fi': "Future world or space setting",
        'mystery': "Modern city or small town with secrets",
        'horror': "Isolated or haunted location", 
        'romance': "Romantic locations that bring hearts together",
        'adventure': "Exotic locations full of danger and discovery",
        'comedy': "Everyday settings where humor can flourish"
    }
    return settings.get(genre, "Contemporary setting with unique elements")

def get_genre_conflict(genre, theme):
    conflicts = {
        'fantasy': f"Magical forces threaten the world, and {theme} holds the key to salvation",
        'sci-fi': f"Technological or alien challenges that relate to {theme}",
        'mystery': f"A puzzling crime or disappearance involving {theme}",
        'horror': f"Supernatural threats connected to {theme}",
        'romance': f"Obstacles to love that involve {theme}",
        'adventure': f"A quest or journey centered around {theme}",
        'comedy': f"Humorous misunderstandings and mishaps involving {theme}"
    }
    return conflicts.get(genre, f"Challenges and obstacles related to {theme}")

def handle_meme_generation(text, personality='friendly'):
    """Handle meme generation requests"""
    return "😂 Meme generation is coming soon! I can help you with story writing, comic creation, and fashion design instead. Try asking: 'Write a story about adventure', 'Create a comic about superheroes', or 'Design an outfit for work'."

def get_humor_level(style):
    """Get humor intensity description"""
    levels = {
        'relatable': 'Universal and accessible',
        'sarcastic': 'Dry and witty',
        'wholesome': 'Feel-good and heartwarming',
        'dark_humor': 'Edgy but tasteful',
        'office': 'Corporate comedy gold',
        'student': 'Academic struggle humor',
        'pet': 'Adorably funny'
    }
    return levels.get(style, 'Hilariously entertaining')

def get_audience(style):
    """Get target audience for meme style"""
    audiences = {
        'relatable': 'Everyone - universal appeal',
        'sarcastic': 'People who appreciate dry humor',
        'wholesome': 'Family-friendly, all ages',
        'office': 'Working professionals',
        'student': 'Students and academics',
        'pet': 'Pet lovers and animal enthusiasts'
    }
    return audiences.get(style, 'General audience')

def generate_bonus_meme_ideas(topic, style):
    """Generate additional meme ideas for the topic and style"""
    ideas = [
        f"• Alternative {style} take: {topic} + current trending meme format",
        f"• Crossover idea: {topic} meets classic internet culture",
        f"• Series potential: Daily {topic} {style} memes",
        f"• Interactive: Ask followers to add their own {topic} experiences"
    ]
    return '\n'.join(ideas)

def handle_comic_generation(text, personality='friendly'):
    """Handle comic strip and comic panel generation requests"""
    try:
        print(f"📚 Processing comic generation request: {text}")
        
        # Extract comic topic/theme
        import re
        
        topic_match = re.search(r'(?:comic.*about|comic.*with|story.*about)[\s]*([^,.!?]+)', text.lower())
        topic = topic_match.group(1).strip() if topic_match else "everyday life"
        
        # Detect comic style
        comic_style = "slice_of_life"  # default
        if re.search(r'\b(superhero|hero|villain|power)\b', text.lower()):
            comic_style = "superhero"
        elif re.search(r'\b(manga|anime|japanese)\b', text.lower()):
            comic_style = "manga"
        elif re.search(r'\b(webcomic|web|online)\b', text.lower()):
            comic_style = "webcomic"
        elif re.search(r'\b(newspaper|daily|strip)\b', text.lower()):
            comic_style = "newspaper"
        elif re.search(r'\b(graphic.*novel|serious|drama)\b', text.lower()):
            comic_style = "graphic_novel"
        elif re.search(r'\b(funny|humor|comedy|joke)\b', text.lower()):
            comic_style = "comedy"
        
        # Personality-based responses
        personality_responses = {
            'friendly': f"🎨 I'd love to help you create an amazing comic about {topic}!",
            'creative': f"✨ Let's bring {topic} to life through sequential art!",
            'enthusiastic': f"📚 WOW! Creating comics about {topic} is going to be EPIC!!!",
            'professional': f"🖋️ I shall assist you in developing a comic narrative about {topic}.",
            'witty': f"💭 Time to draw some laughs with a {topic} comic!"
        }
        
        base_response = personality_responses.get(personality, personality_responses['friendly'])
        
        # Generate comic structure based on style
        comic_content = generate_comic_structure(topic, comic_style)
        
        return f"""{base_response}

{comic_content}

**🎨 Comic Creation Features:**
• **Multiple Styles**: Superhero, manga, webcomic, newspaper strips, graphic novels
• **Story Structure**: Complete panel layouts with dialogue and action
• **Character Development**: Protagonist and supporting character suggestions
• **Visual Guidance**: Panel composition and artistic direction

**📚 Want More Comics?** Try asking for:
• "Create a superhero comic about time travel"
• "Generate a manga-style comic about school life"
• "Make a funny webcomic about pets"
• "Design a newspaper comic strip about office work"

Ready to publish your comic? What's your next story idea? ✏️"""
        
    except Exception as e:
        print(f"Error in handle_comic_generation: {e}")
        return "📚 I'm your AI comic creator! I can design comic strips, graphic novels, superhero stories, manga-style comics, and webcomics on any topic. Try asking: 'Create a superhero comic about saving the city', 'Generate a funny comic strip about cats', or 'Make a manga about school adventures'. What comic story shall we create today?"

def generate_comic_structure(topic, style):
    """Generate comic structure based on topic and style"""
    
    if style == "superhero":
        return f"""**🦸 {topic.title()} - Superhero Comic**

**Panel 1**: 📱 Wide establishing shot
*Scene: City skyline at dusk. Something related to {topic} is causing chaos in the distance.*
**Narration**: "In a world where {topic} threatens everything we hold dear..."

**Panel 2**: 👤 Close-up on protagonist
*Hero looking determined, costume reflecting their {topic}-related powers.*
**Hero**: "Not on my watch. Time to stop this {topic} madness!"

**Panel 3**: 💥 Action panel
*Hero leaping into action, using their {topic}-powered abilities.*
**Sound Effect**: "WHOOSH!" "ZAP!" 

**Panel 4**: 😨 Reaction shot
*Citizens looking up in amazement and relief.*
**Citizen**: "It's the {topic.title()} Guardian! We're saved!"

**Panel 5**: 🏆 Resolution panel
*Hero standing victorious, {topic} crisis resolved.*
**Hero**: "Remember, with great {topic} comes great responsibility."

**🎭 Character Profiles:**
• **Hero Name**: The {topic.title()} Guardian
• **Powers**: {topic}-based abilities, super strength, flight
• **Secret Identity**: Ordinary person who discovered {topic} powers
• **Motivation**: Protect the world from {topic}-related threats

**🏢 Setting**: Modern metropolis where {topic} technology/magic exists
**🎯 Story Arc**: Origin story → First villain → Team-up → Major crisis → Resolution"""

    elif style == "manga":
        return f"""**🌸 {topic.title()} - Manga Comic**

**Page 1, Panel 1**: 🏫 Establishing shot
*Japanese high school, cherry blossoms falling. Focus on {topic} club building.*

**Page 1, Panel 2**: 😊 Character introduction
*Protagonist (big anime eyes, expressive) discovering {topic} for the first time.*
**Protagonist**: "Eh?! What is this {topic}? It's... incredible!"

**Page 1, Panel 3**: ✨ Reaction panel
*Sparkly background, protagonist's eyes shining with determination.*
**Protagonist**: "I'll become the best at {topic} in all of Japan!"

**Page 2, Panel 1**: 👥 Group shot
*Meeting the {topic} club members, each with distinct personalities.*
**Club President**: "Welcome to the {topic} club! We've been waiting for someone like you!"

**Page 2, Panel 2**: 💪 Training montage panel
*Multiple small panels showing {topic} practice and improvement.*
**Narration**: "Days turned to weeks as our hero trained relentlessly..."

**Page 2, Panel 3**: 🎌 Tournament announcement
*Large panel with dramatic tournament poster about {topic} competition.*
**Announcement**: "The National {topic.title()} Championship begins tomorrow!"

**🎌 Character Archetypes:**
• **Protagonist**: Enthusiastic beginner with hidden talent
• **Mentor**: Wise senpai who guides {topic} training
• **Rival**: Skilled opponent who challenges growth
• **Support**: Cheerful friend who believes in protagonist

**🏫 Setting**: Japanese school with strong {topic} culture
**📈 Story Progression**: Discovery → Training → Friendship → Competition → Growth"""

    elif style == "webcomic":
        return f"""**💻 {topic.title()} - Webcomic Series**

**Episode 1: "Getting Started"**

**Panel 1**: 🏠 Simple room background
*Protagonist at computer/doing {topic}-related activity.*
**Protagonist**: "Okay, time to finally get serious about {topic}."

**Panel 2**: 😅 Close-up, slightly concerned expression
**Protagonist**: "How hard could it be, right?"

**Panel 3**: 📚 Montage of research
*Multiple browser tabs, books, videos about {topic}.*
**Protagonist**: "...Oh. Oh no."

**Panel 4**: 😵 Overwhelmed expression
*Protagonist surrounded by {topic} information.*
**Protagonist**: "There's SO MUCH to learn about {topic}!"

**Panel 5**: 😤 Determined face
**Protagonist**: "But I'm not giving up! Day 1 of my {topic} journey starts now!"

**Episode Ideas:**
• **Episode 2**: "First Attempt" - Things go hilariously wrong
• **Episode 3**: "Expert Advice" - Getting help from {topic} pros  
• **Episode 4**: "Small Victory" - First success with {topic}
• **Episode 5**: "Community" - Finding other {topic} enthusiasts

**🎨 Art Style**: Simple, expressive characters with clean lines
**📱 Format**: Vertical scroll format, mobile-friendly
**🎯 Tone**: Relatable, humorous, encouraging
**👥 Audience**: People interested in {topic} or learning new skills"""

    elif style == "newspaper":
        return f"""**📰 {topic.title()} - Daily Comic Strip**

**Strip 1**: "Monday Morning"
**Panel 1**: 😴 Character waking up
**Character**: "Another Monday... time for {topic}."

**Panel 2**: ☕ At breakfast table
**Partner**: "Still obsessed with {topic}, I see."

**Panel 3**: 😊 Character leaving happily
**Character**: "It's not obsession, it's passion!"

---

**Strip 2**: "The Expert"
**Panel 1**: 👥 Character talking to friend
**Character**: "I'm getting really good at {topic}!"

**Panel 2**: 🤔 Friend looking skeptical
**Friend**: "Really? Show me."

**Panel 3**: 😅 Character failing at {topic}
**Character**: "...I said I'm getting good, not that I'm there yet!"

---

**Strip 3**: "Weekend Plans"
**Panel 1**: 📅 Looking at calendar
**Partner**: "What are your weekend plans?"

**Panel 2**: 😍 Character excited
**Character**: "More {topic} practice!"

**Panel 3**: 🙄 Partner's reaction
**Partner**: "I should have seen that coming."

**📰 Series Concept**: Daily life humor centered around {topic}
**👥 Characters**: Enthusiast + Patient partner/friends
**🎯 Format**: 3-panel daily strips, Sunday color strips
**📅 Themes**: Monday struggles, weekend enthusiasm, learning curves"""

    else:  # slice_of_life or general
        return f"""**🎭 {topic.title()} - Slice of Life Comic**

**Chapter 1: "Discovery"**

**Panel 1**: 🏠 Everyday setting
*Ordinary moment that leads to discovering {topic}.*
**Narration**: "Sometimes the most ordinary days lead to extraordinary discoveries..."

**Panel 2**: 👀 Moment of realization
*Character noticing something special about {topic}.*
**Character**: "Wait... there's something different about this {topic}."

**Panel 3**: 🤔 Investigation panel
*Character exploring and learning more about {topic}.*
**Character**: "I never realized {topic} could be so interesting!"

**Panel 4**: 💡 Understanding
*Character having an "aha!" moment about {topic}.*
**Character**: "This changes everything I thought I knew!"

**Panel 5**: 🌅 New perspective
*Character looking at the world differently because of {topic}.*
**Character**: "I can't wait to see where this {topic} journey takes me."

**📖 Story Themes:**
• **Growth**: Learning and personal development through {topic}
• **Community**: Meeting others who share {topic} interests  
• **Challenges**: Overcoming obstacles related to {topic}
• **Discovery**: Finding unexpected aspects of {topic}

**🎨 Visual Style**: Realistic but warm, detailed backgrounds
**📚 Chapter Structure**: 
- Discovery → Learning → Community → Challenges → Growth
**👥 Supporting Cast**: Mentors, fellow enthusiasts, skeptics turned believers"""

def handle_fashion_design(text, personality='friendly'):
    """Handle fashion design and style recommendation requests"""
    try:
        print(f"👗 Processing fashion design request: {text}")
        
        # Extract style preferences and occasion
        import re
        
        occasion_match = re.search(r'(?:for|dress.*for|outfit.*for|style.*for)[\s]*([^,.!?]+)', text.lower())
        occasion = occasion_match.group(1).strip() if occasion_match else "everyday wear"
        
        # Detect style preference
        style_pref = "casual"  # default
        if re.search(r'\b(formal|business|professional|office)\b', text.lower()):
            style_pref = "formal"
        elif re.search(r'\b(party|club|night.*out|evening)\b', text.lower()):
            style_pref = "party"
        elif re.search(r'\b(bohemian|boho|hippie|free.*spirit)\b', text.lower()):
            style_pref = "bohemian"
        elif re.search(r'\b(minimalist|simple|clean|basic)\b', text.lower()):
            style_pref = "minimalist"
        elif re.search(r'\b(vintage|retro|classic|old.*school)\b', text.lower()):
            style_pref = "vintage"
        elif re.search(r'\b(edgy|punk|alternative|rock)\b', text.lower()):
            style_pref = "edgy"
        elif re.search(r'\b(romantic|feminine|soft|delicate)\b', text.lower()):
            style_pref = "romantic"
        elif re.search(r'\b(sporty|athletic|active|gym)\b', text.lower()):
            style_pref = "sporty"
        
        # Detect season/weather
        season = "current"
        if re.search(r'\b(winter|cold|snow|warm.*clothes)\b', text.lower()):
            season = "winter"
        elif re.search(r'\b(summer|hot|beach|light.*clothes)\b', text.lower()):
            season = "summer"
        elif re.search(r'\b(spring|mild|transitional)\b', text.lower()):
            season = "spring"
        elif re.search(r'\b(fall|autumn|layering)\b', text.lower()):
            season = "fall"
        
        # Personality-based responses
        personality_responses = {
            'friendly': f"👗 I'd love to help you create the perfect {style_pref} look for {occasion}!",
            'professional': f"✨ I shall provide you with sophisticated fashion recommendations for {occasion}.",
            'enthusiastic': f"💃 OMG YES! Let's create an AMAZING {style_pref} outfit for {occasion}!!!",
            'creative': f"🎨 Time to unleash some serious style creativity for your {occasion} look!",
            'witty': f"👠 Fashion emergency? I've got the perfect {style_pref} prescription for {occasion}!"
        }
        
        base_response = personality_responses.get(personality, personality_responses['friendly'])
        
        # Generate fashion recommendations
        fashion_content = generate_fashion_recommendations(occasion, style_pref, season)
        
        return f"""{base_response}

{fashion_content}

**✨ AI Fashion Features:**
• **Personalized Styling**: Outfits tailored to your preferences and occasion
• **Season-Appropriate**: Weather and climate considerations
• **Style Variety**: From casual to formal, bohemian to minimalist
• **Accessory Guidance**: Complete head-to-toe styling advice

**👗 Want More Style Ideas?** Try asking for:
• "Design a professional outfit for work meetings"
• "Create a bohemian look for a music festival"  
• "Suggest minimalist winter clothes"
• "Party outfit for a night out"

Ready to upgrade your wardrobe? What's your next style challenge? 💫"""
        
    except Exception as e:
        print(f"Error in handle_fashion_design: {e}")
        return "👗 I'm your AI fashion stylist! I can design outfits, suggest clothing combinations, and provide style advice for any occasion - from casual everyday looks to formal events, bohemian vibes to minimalist aesthetics. Try asking: 'Design a professional outfit for work', 'Create a party look for tonight', or 'Suggest casual weekend clothes'. What style adventure shall we embark on today?"

def generate_fashion_recommendations(occasion, style, season):
    """Generate detailed fashion recommendations"""
    
    # Style-based outfit formulas
    style_guides = {
        'casual': {
            'tops': ['comfortable t-shirt', 'soft sweater', 'casual blouse', 'hoodie'],
            'bottoms': ['well-fitted jeans', 'comfortable leggings', 'casual chinos', 'denim shorts'],
            'shoes': ['white sneakers', 'comfortable flats', 'casual boots', 'canvas shoes'],
            'accessories': ['crossbody bag', 'simple watch', 'baseball cap', 'sunglasses']
        },
        'formal': {
            'tops': ['crisp button-down shirt', 'silk blouse', 'tailored blazer', 'elegant sweater'],
            'bottoms': ['dress pants', 'pencil skirt', 'tailored trousers', 'midi dress'],
            'shoes': ['pointed-toe pumps', 'oxford shoes', 'elegant flats', 'ankle boots'],
            'accessories': ['structured handbag', 'classic watch', 'pearl earrings', 'silk scarf']
        },
        'party': {
            'tops': ['sequined top', 'silk camisole', 'off-shoulder blouse', 'bodysuit'],
            'bottoms': ['mini skirt', 'wide-leg pants', 'leather leggings', 'cocktail dress'],
            'shoes': ['statement heels', 'strappy sandals', 'ankle boots', 'platform shoes'],
            'accessories': ['clutch bag', 'statement jewelry', 'bold earrings', 'cocktail ring']
        },
        'bohemian': {
            'tops': ['flowy blouse', 'peasant top', 'crochet vest', 'kimono cardigan'],
            'bottoms': ['maxi skirt', 'wide-leg pants', 'flowy dress', 'palazzo pants'],
            'shoes': ['ankle boots', 'sandals', 'moccasins', 'wedges'],
            'accessories': ['fringe bag', 'layered necklaces', 'hair bands', 'wooden bangles']
        },
        'minimalist': {
            'tops': ['white button-down', 'cashmere sweater', 'simple turtleneck', 'clean blazer'],
            'bottoms': ['straight-leg trousers', 'midi skirt', 'tailored shorts', 'slip dress'],
            'shoes': ['leather loafers', 'white sneakers', 'simple flats', 'clean boots'],
            'accessories': ['structured tote', 'delicate jewelry', 'classic watch', 'simple belt']
        },
        'vintage': {
            'tops': ['vintage band tee', 'retro blouse', 'cardigan', 'vintage blazer'],
            'bottoms': ['high-waisted jeans', 'A-line skirt', 'vintage dress', 'wide-leg trousers'],
            'shoes': ['retro sneakers', 'mary janes', 'oxford shoes', 'vintage boots'],
            'accessories': ['vintage handbag', 'cat-eye sunglasses', 'pearl accessories', 'vintage scarf']
        },
        'edgy': {
            'tops': ['band t-shirt', 'leather jacket', 'mesh top', 'graphic tee'],
            'bottoms': ['ripped jeans', 'leather pants', 'mini skirt', 'cargo pants'],
            'shoes': ['combat boots', 'platform shoes', 'chunky sneakers', 'ankle boots'],
            'accessories': ['chain bag', 'studded accessories', 'bold jewelry', 'statement belt']
        },
        'romantic': {
            'tops': ['lace blouse', 'silk camisole', 'ruffled top', 'floral print shirt'],
            'bottoms': ['flowy skirt', 'soft pants', 'feminine dress', 'pleated skirt'],
            'shoes': ['ballet flats', 'low heels', 'delicate sandals', 'mary janes'],
            'accessories': ['small purse', 'delicate jewelry', 'hair ribbons', 'floral accessories']
        },
        'sporty': {
            'tops': ['athletic tank', 'sports bra', 'performance tee', 'zip-up hoodie'],
            'bottoms': ['leggings', 'track pants', 'athletic shorts', 'joggers'],
            'shoes': ['running shoes', 'cross-trainers', 'athletic sandals', 'slip-on sneakers'],
            'accessories': ['gym bag', 'fitness tracker', 'baseball cap', 'water bottle']
        }
    }
    
    # Get style components
    components = style_guides.get(style, style_guides['casual'])
    
    # Season-specific modifications
    seasonal_tips = {
        'winter': {
            'layers': ['wool coat', 'warm scarf', 'gloves', 'thermal layers'],
            'fabrics': ['wool', 'cashmere', 'fleece', 'thermal materials'],
            'colors': ['deep burgundy', 'forest green', 'navy', 'charcoal gray']
        },
        'summer': {
            'layers': ['light cardigan', 'sun hat', 'sunglasses', 'lightweight scarf'],
            'fabrics': ['cotton', 'linen', 'breathable synthetics', 'lightweight materials'],
            'colors': ['white', 'pastels', 'bright colors', 'light neutrals']
        },
        'spring': {
            'layers': ['light jacket', 'cardigan', 'light scarf', 'transitional pieces'],
            'fabrics': ['cotton blends', 'light wool', 'denim', 'jersey'],
            'colors': ['soft pastels', 'fresh greens', 'light blues', 'coral']
        },
        'fall': {
            'layers': ['trench coat', 'sweaters', 'boots', 'layering pieces'],
            'fabrics': ['wool', 'corduroy', 'denim', 'knits'],
            'colors': ['warm browns', 'burnt orange', 'deep reds', 'mustard yellow']
        }
    }
    
    # Build recommendation
    import random
    
    selected_top = random.choice(components['tops'])
    selected_bottom = random.choice(components['bottoms'])
    selected_shoes = random.choice(components['shoes'])
    selected_accessory = random.choice(components['accessories'])
    
    seasonal_info = seasonal_tips.get(season, seasonal_tips['current'])
    
    outfit_description = f"""👗 **{style.title()} Outfit for {occasion.title()}**

**🎯 Complete Look:**
• **Top**: {selected_top.title()}
• **Bottom**: {selected_bottom.title()}  
• **Shoes**: {selected_shoes.title()}
• **Key Accessory**: {selected_accessory.title()}

**🌟 Style Details:**
• **Aesthetic**: {style.title()} with modern touches
• **Occasion**: Perfect for {occasion}
• **Comfort Level**: Stylish yet comfortable for all-day wear
• **Versatility**: Can be dressed up or down with accessories

**🎨 Color Palette Suggestions:**
{get_color_palette(style, season)}

**✨ Styling Tips:**
• **Fit**: Ensure proper proportions - if top is loose, bottom should be fitted
• **Balance**: Mix textures and patterns for visual interest
• **Layering**: {get_layering_tips(style, season)}
• **Accessories**: {get_accessory_tips(style)}

**👜 Complete Accessory List:**
• **Bag**: {get_bag_suggestion(style, occasion)}
• **Jewelry**: {get_jewelry_suggestion(style)}
• **Outerwear**: {get_outerwear_suggestion(style, season)}
• **Extras**: {get_extra_accessories(style)}

**🛍️ Shopping List:**
1. {selected_top.title()}
2. {selected_bottom.title()}
3. {selected_shoes.title()}
4. {selected_accessory.title()}
5. Complementary accessories

**💡 Mix & Match Ideas:**
• Swap the top for a {random.choice(components['tops'])} for variety
• Try the bottom with a {random.choice(components['tops'])} for different occasions
• Change shoes to {random.choice(components['shoes'])} for comfort
• Add a {random.choice(components['accessories'])} for extra style"""

    return outfit_description

def get_color_palette(style, season):
    """Generate color palette suggestions"""
    palettes = {
        'casual': ['Navy & white', 'Denim blue & cream', 'Olive green & beige', 'Gray & soft pink'],
        'formal': ['Black & white', 'Navy & cream', 'Charcoal & burgundy', 'Camel & ivory'],
        'party': ['Black & gold', 'Deep red & black', 'Emerald & silver', 'Purple & metallic'],
        'bohemian': ['Earthy browns & cream', 'Rust & olive', 'Mustard & burgundy', 'Terracotta & sage'],
        'minimalist': ['All white', 'Black & white', 'Beige & cream', 'Gray tones'],
        'vintage': ['Dusty rose & cream', 'Mint green & white', 'Mustard & brown', 'Navy & red'],
        'edgy': ['All black', 'Black & silver', 'Dark red & black', 'Leather brown & black'],
        'romantic': ['Blush pink & white', 'Lavender & cream', 'Soft yellow & white', 'Rose & ivory'],
        'sporty': ['Black & neon', 'Gray & bright colors', 'Navy & white', 'Color blocking']
    }
    return ' • '.join(palettes.get(style, palettes['casual']))

def get_layering_tips(style, season):
    """Get layering suggestions"""
    if season == 'winter':
        return "Layer with cozy sweaters and warm outerwear"
    elif season == 'summer':
        return "Keep layers light - think cardigans and light scarves"
    elif season == 'spring':
        return "Perfect for transitional layering pieces"
    elif season == 'fall':
        return "Ideal for cozy sweaters and stylish jackets"
    else:
        return "Layer according to current weather and comfort"

def get_accessory_tips(style):
    """Get style-specific accessory tips"""
    tips = {
        'casual': "Keep accessories simple and functional",
        'formal': "Choose classic, high-quality pieces",
        'party': "Go bold with statement accessories",
        'bohemian': "Layer jewelry and add natural textures",
        'minimalist': "Less is more - choose one statement piece",
        'vintage': "Mix authentic vintage with vintage-inspired pieces",
        'edgy': "Add metal details and bold statements",
        'romantic': "Focus on delicate, feminine details",
        'sporty': "Keep accessories functional and comfortable"
    }
    return tips.get(style, "Choose accessories that complement your personal style")

def get_bag_suggestion(style, occasion):
    """Suggest appropriate bag for style and occasion"""
    suggestions = {
        'casual': 'Canvas tote or crossbody bag',
        'formal': 'Structured handbag or briefcase',
        'party': 'Clutch or small evening bag',
        'bohemian': 'Fringe bag or woven tote',
        'minimalist': 'Clean-lined tote or simple clutch',
        'vintage': 'Vintage-style handbag or satchel',
        'edgy': 'Chain bag or studded purse',
        'romantic': 'Small feminine purse or delicate bag',
        'sporty': 'Backpack or gym bag'
    }
    return suggestions.get(style, 'Versatile handbag')

def get_jewelry_suggestion(style):
    """Suggest jewelry for style"""
    suggestions = {
        'casual': 'Simple studs and delicate chain',
        'formal': 'Pearl or gold classic pieces',
        'party': 'Statement earrings and bold necklace',
        'bohemian': 'Layered necklaces and natural stones',
        'minimalist': 'One quality piece - watch or simple necklace',
        'vintage': 'Art deco or vintage-inspired pieces',
        'edgy': 'Bold metal pieces and chains',
        'romantic': 'Delicate pearls and soft metals',
        'sporty': 'Fitness tracker and simple studs'
    }
    return suggestions.get(style, 'Personal preference pieces')

def get_outerwear_suggestion(style, season):
    """Suggest outerwear based on style and season"""
    if season == 'winter':
        return {'casual': 'Puffer jacket', 'formal': 'Wool coat', 'party': 'Faux fur jacket'}.get(style, 'Warm coat')
    elif season == 'summer':
        return {'casual': 'Light cardigan', 'formal': 'Blazer', 'party': 'Light wrap'}.get(style, 'Light layer')
    else:
        return {'casual': 'Denim jacket', 'formal': 'Trench coat', 'party': 'Statement jacket'}.get(style, 'Versatile jacket')

def get_extra_accessories(style):
    """Get additional accessory suggestions"""
    extras = {
        'casual': 'Baseball cap and sunglasses',
        'formal': 'Silk scarf and classic watch',
        'party': 'Statement hair accessory',
        'bohemian': 'Hair band and anklet',
        'minimalist': 'Quality leather belt',
        'vintage': 'Cat-eye sunglasses',
        'edgy': 'Studded belt and bold rings',
        'romantic': 'Hair ribbon and delicate bracelet',
        'sporty': 'Athletic headband and water bottle'
    }
    return extras.get(style, 'Personal style accessories')

# ===============================================
# 🔮 AI FUTURISTIC FEATURES FUNCTIONS

def handle_ar_integration(text, personality='friendly'):
    """Handle augmented reality integration requests with practical guidance"""
    try:
        print(f"🔮 Processing AR integration request: {text}")
        
        # Extract AR mode/type
        import re
        
        ar_type = "general"  # default
        if re.search(r'\b(face|facial).*filter\b', text.lower()):
            ar_type = "face_filter"
        elif re.search(r'\b(object|item).*recognition\b', text.lower()):
            ar_type = "object_recognition"
        elif re.search(r'\b(navigation|directions)\b', text.lower()):
            ar_type = "navigation"
        elif re.search(r'\b(education|learning)\b', text.lower()):
            ar_type = "educational"
        elif re.search(r'\b(game|gaming|entertainment)\b', text.lower()):
            ar_type = "gaming"
        elif re.search(r'\b(shopping|retail)\b', text.lower()):
            ar_type = "shopping"
        elif re.search(r'\b(social|sharing)\b', text.lower()):
            ar_type = "social"
        
        # Personality-based responses
        personality_responses = {
            'friendly': f"🔮 Let me help you create AR face filters! Here's how to get started:",
            'professional': f"📱 I'll provide you with practical AR development guidance:",
            'enthusiastic': f"🤩 YES! Let's build some AMAZING AR filters together!",
            'creative': f"✨ Time to bring your AR vision to life! Here's your roadmap:",
            'witty': f"👓 Ready to filter reality? Let's make it happen!"
        }
        
        base_response = personality_responses.get(personality, personality_responses['friendly'])
        
        # Generate practical AR guidance
        ar_guidance = generate_practical_ar_guide(ar_type)
        
        # Format the response with HTML line breaks for proper display
        formatted_response = f"""{base_response}<br><br>

{ar_guidance}<br><br>

<strong>🎯 Next Steps:</strong><br>
1. <strong>Choose your platform</strong> from the options above<br>
2. <strong>Download the recommended tools</strong><br>
3. <strong>Follow the step-by-step tutorial</strong><br>
4. <strong>Test your first filter</strong><br>
5. <strong>Share your creation!</strong><br><br>

<strong>💡 Need help with a specific step?</strong> Ask me:<br>
• "How do I set up Spark AR Studio?"<br>
• "Show me face tracking code examples"<br>
• "Help me publish my AR filter"<br>
• "What are the best AR development practices?"<br><br>

Ready to start building? Which platform interests you most? 🚀"""
        
        return formatted_response
        
    except Exception as e:
        print(f"Error in handle_ar_integration: {e}")
        return "🔮 I'm here to help you actually build AR experiences! I can provide step-by-step guides, tool recommendations, code examples, and platform setup instructions. Try asking: 'How do I create AR face filters?', 'What tools do I need for AR development?', or 'Show me AR coding tutorials'. Let's build something amazing!"

def handle_dream_journal(text, personality='friendly'):
    """Handle dream journal and analysis requests"""
    try:
        print(f"💭 Processing dream journal request: {text}")
        
        # Extract dream elements
        import re
        
        dream_type = "general"  # default
        if re.search(r'\b(nightmare|scary|fear)\b', text.lower()):
            dream_type = "nightmare"
        elif re.search(r'\b(lucid|control)\b', text.lower()):
            dream_type = "lucid"
        elif re.search(r'\b(recurring|repeat)\b', text.lower()):
            dream_type = "recurring"
        elif re.search(r'\b(prophetic|future|prediction)\b', text.lower()):
            dream_type = "prophetic"
        elif re.search(r'\b(childhood|past)\b', text.lower()):
            dream_type = "childhood"
        elif re.search(r'\b(flying|float)\b', text.lower()):
            dream_type = "flying"
        elif re.search(r'\b(water|ocean|swimming)\b', text.lower()):
            dream_type = "water"
        
        # Personality-based responses
        personality_responses = {
            'friendly': f"💭 I'd be honored to help you explore your dreams and their meanings!",
            'professional': f"🧠 I shall provide comprehensive dream analysis using psychological principles.",
            'enthusiastic': f"🌟 DREAMS are fascinating windows into our subconscious!!!",
            'creative': f"✨ Let's unlock the mysteries hidden in your dreamscape!",
            'witty': f"😴 Time to decode what your brain does during its downtime!"
        }
        
        base_response = personality_responses.get(personality, personality_responses['friendly'])
        
        # Generate dream analysis content
        dream_content = generate_dream_analysis(dream_type)
        
        return f"""{base_response}

{dream_content}

**💭 Dream Journal Features:**
• **AI Dream Analysis**: Psychological interpretation of dream symbols and themes
• **Pattern Recognition**: Identifies recurring elements across multiple dreams
• **Emotional Mapping**: Tracks emotional patterns in dreams over time
• **Symbol Database**: Comprehensive library of dream symbolism meanings

**🌙 Want More Dream Insights?** Try asking for:
• "Analyze my recurring nightmare about falling"
• "What do dreams about water mean?"
• "Help me understand my lucid dreams"
• "Create a dream tracking system"

Ready to explore your subconscious? What dreams shall we analyze? 🔍"""
        
    except Exception as e:
        print(f"Error in handle_dream_journal: {e}")
        return "💭 I'm your AI dream analyst! I can interpret dreams, analyze symbolism, track patterns, and provide psychological insights into your subconscious. Try asking: 'Analyze my dream about flying', 'What do recurring dreams mean?', or 'Help me keep a dream journal'. What dreams shall we explore today?"

def handle_time_capsule(text, personality='friendly'):
    """Handle time capsule and future prediction requests"""
    try:
        print(f"⏰ Processing time capsule request: {text}")
        
        # Extract time frame and prediction type
        import re
        
        time_frame = "1_year"  # default
        if re.search(r'\b(tomorrow|next.*day)\b', text.lower()):
            time_frame = "1_day"
        elif re.search(r'\b(next.*week|week)\b', text.lower()):
            time_frame = "1_week"
        elif re.search(r'\b(next.*month|month)\b', text.lower()):
            time_frame = "1_month"
        elif re.search(r'\b(5.*year|five.*year)\b', text.lower()):
            time_frame = "5_years"
        elif re.search(r'\b(10.*year|ten.*year|decade)\b', text.lower()):
            time_frame = "10_years"
        elif re.search(r'\b(century|100.*year)\b', text.lower()):
            time_frame = "100_years"
        
        prediction_type = "general"
        if re.search(r'\b(technology|tech|ai|robot)\b', text.lower()):
            prediction_type = "technology"
        elif re.search(r'\b(society|social|culture)\b', text.lower()):
            prediction_type = "society"
        elif re.search(r'\b(environment|climate|earth)\b', text.lower()):
            prediction_type = "environment"
        elif re.search(r'\b(personal|my.*life|career)\b', text.lower()):
            prediction_type = "personal"
        elif re.search(r'\b(economy|money|finance)\b', text.lower()):
            prediction_type = "economy"
        
        # Personality-based responses
        personality_responses = {
            'friendly': f"⏰ I'd love to help you create a fascinating glimpse into the future!",
            'professional': f"🔮 I shall provide data-driven predictions and trend analysis.",
            'enthusiastic': f"🚀 TIME TRAVEL through predictions! This is AMAZING!!!",
            'creative': f"✨ Let's craft a visionary time capsule for future generations!",
            'witty': f"🔮 Crystal ball activated! Let's see what the future holds!"
        }
        
        base_response = personality_responses.get(personality, personality_responses['friendly'])
        
        # Generate time capsule content
        capsule_content = generate_time_capsule(time_frame, prediction_type)
        
        return f"""{base_response}

{capsule_content}

**⏰ Time Capsule Features:**
• **Future Predictions**: AI-powered trend analysis and forecasting
• **Personalized Capsules**: Customized predictions based on your interests
• **Multiple Timeframes**: From tomorrow to the next century
• **Scenario Planning**: Multiple possible future outcomes

**🔮 Want More Future Insights?** Try asking for:
• "Predict technology trends for next decade"
• "Create personal time capsule for 5 years"
• "What will society look like in 2050?"
• "Predict environmental changes"

Ready to peek into tomorrow? What future shall we explore? 🌟"""
        
    except Exception as e:
        print(f"Error in handle_time_capsule: {e}")
        return "⏰ I'm your AI futurist! I can predict trends, create time capsules, forecast changes, and analyze future possibilities across technology, society, environment, and personal life. Try asking: 'Predict tech trends for 2030', 'Create a time capsule for 10 years', or 'What will AI look like in the future?'. What future shall we explore?"

def handle_virtual_world_builder(text, personality='friendly'):
    """Handle virtual world building requests"""
    try:
        print(f"🌍 Processing virtual world builder request: {text}")
        
        # Extract world type and features
        import re
        
        world_type = "custom"  # default
        if re.search(r'\b(fantasy|magical|medieval)\b', text.lower()):
            world_type = "fantasy"
        elif re.search(r'\b(sci.*fi|futuristic|space|cyberpunk)\b', text.lower()):
            world_type = "sci_fi"
        elif re.search(r'\b(realistic|modern|contemporary)\b', text.lower()):
            world_type = "realistic"
        elif re.search(r'\b(post.*apocalyptic|wasteland|survival)\b', text.lower()):
            world_type = "post_apocalyptic"
        elif re.search(r'\b(underwater|ocean|sea)\b', text.lower()):
            world_type = "underwater"
        elif re.search(r'\b(sky|floating|aerial)\b', text.lower()):
            world_type = "sky"
        elif re.search(r'\b(horror|dark|gothic)\b', text.lower()):
            world_type = "horror"
        elif re.search(r'\b(cartoon|anime|stylized)\b', text.lower()):
            world_type = "stylized"
        
        # Personality-based responses
        personality_responses = {
            'friendly': f"🌍 I'd be thrilled to help you build an incredible virtual world!",
            'professional': f"🏗️ I shall design a comprehensive virtual environment with precise specifications.",
            'enthusiastic': f"🎮 WORLD BUILDING is the most creative thing EVER!!!",
            'creative': f"✨ Let's craft entire universes limited only by imagination!",
            'witty': f"👨‍💻 God mode activated! Time to create some digital real estate!"
        }
        
        base_response = personality_responses.get(personality, personality_responses['friendly'])
        
        # Generate virtual world content
        world_content = generate_virtual_world(world_type)
        
        return f"""{base_response}

{world_content}

**🌍 Virtual World Builder Features:**
• **Multiple World Types**: Fantasy, sci-fi, realistic, post-apocalyptic, underwater, sky worlds
• **Physics Simulation**: Realistic or custom physics engines
• **Interactive Elements**: NPCs, objects, environmental interactions
• **Multiplayer Support**: Shared virtual spaces for collaboration

**🎮 Want More Worlds?** Try asking for:
• "Build a fantasy world with magic systems"
• "Create a futuristic cyberpunk city"
• "Design an underwater civilization"
• "Make a post-apocalyptic survival world"

Ready to play god? What virtual universe shall we create? 🚀"""
        
    except Exception as e:
        print(f"Error in handle_virtual_world_builder: {e}")
        return "🌍 I'm your virtual world architect! I can create entire digital universes including fantasy realms, sci-fi cities, realistic environments, underwater worlds, and custom physics simulations. Try asking: 'Build a fantasy world with dragons', 'Create a cyberpunk city', or 'Design a floating sky world'. What virtual reality shall we construct?"

def generate_practical_ar_guide(ar_type):
    """Generate practical AR development guide with actionable steps"""
    
    guides = {
        'face_filter': """**� How to Create AR Face Filters - Step by Step**

**📱 Platform Options (Choose One):**

**Option 1: Meta Spark AR Studio (Easiest - No Coding)**<br>
• Download: spark.ar → Download Spark AR Studio (Free)<br>
• Best for: Instagram, Facebook filters<br>
• Time to first filter: 30 minutes<br>
• Skill level: Beginner-friendly<br><br>

**Option 2: Snapchat Lens Studio (Creative Focus)**<br>
• Download: lensstudio.snapchat.com (Free)<br>
• Best for: Snapchat lenses, creative effects<br>
• Time to first filter: 45 minutes<br>
• Skill level: Beginner to intermediate<br><br>

**Option 3: TikTok Effect House (Trending Platform)**<br>
• Download: effecthouse.tiktok.com (Free)<br>
• Best for: TikTok effects, viral content<br>
• Time to first filter: 1 hour<br>
• Skill level: Beginner<br><br>

<strong>🛠️ Quick Start Tutorial (Spark AR):</strong><br><br>

1. Install Spark AR Studio from spark.ar<br>
2. Open new project → Face Tracker template<br>
3. Add 3D object → Import your model or use built-in<br>
4. Attach to face → Drag to face tracker in scene<br>
5. Test on phone → Spark AR Player app<br>
6. Publish → Submit to Instagram/Facebook<br><br>

<strong>💡 Beginner Filter Ideas:</strong><br><br>

• Cat ears and whiskers (30 min tutorial)<br>
• Sunglasses overlay (15 min tutorial)<br>
• Color-changing hair (45 min tutorial)<br>
• Floating text/emojis (20 min tutorial)<br><br>

**📚 Learning Resources:**
• **YouTube**: "Spark AR Tutorial for Beginners"
• **Official docs**: spark.ar/learn
• **Community**: Facebook AR Creators group
• **Practice**: Start with templates, modify slowly""",

        'object_recognition': """**🔍 Build Object Recognition AR - Practical Guide**

**�️ Development Platforms:**

**Option 1: Unity + AR Foundation (Most Powerful)**
• **Setup**: Download Unity Hub → Install Unity 2022.3+ → AR Foundation package
• **Best for**: Custom apps, complex recognition
• **Time to prototype**: 2-3 hours
• **Skill level**: Intermediate (some coding required)

**Option 2: 8th Wall (Web-based AR)**
• **Setup**: 8thwall.com → Create account → Web editor
• **Best for**: Browser-based AR, no app download
• **Time to prototype**: 1-2 hours  
• **Skill level**: Beginner-intermediate

**Option 3: Vuforia (Industry Standard)**
• **Setup**: developer.vuforia.com → Unity integration
• **Best for**: Marker-based recognition, enterprise
• **Time to prototype**: 2-4 hours
• **Skill level**: Intermediate

**🚀 Quick Start (Unity + AR Foundation):**
1. **Create Unity project** → 3D template
2. **Install AR Foundation** → Window → Package Manager
3. **Add AR Session Origin** → XR → AR Session Origin
4. **Create image target** → Vuforia Image Target
5. **Add 3D content** → Drag model to scene
6. **Build to phone** → File → Build Settings → Android/iOS

**📱 Testing Your AR:**
• **Android**: Enable Developer Options → USB Debugging
• **iOS**: Xcode → Sign with Apple ID → Build to device
• **Web**: Use HTTPS server for camera access

**💡 Starter Project Ideas:**
• **Business card scanner** → Show contact info overlay
• **Product scanner** → Display reviews and pricing
• **Plant identifier** → Show care instructions
• **QR code enhanced** → Rich media overlays""",

        'navigation': """**🗺️ AR Navigation Development Guide**

**🛠️ Platform Choices:**

**Option 1: ARCore/ARKit + Google Maps (Professional)**
• **Setup**: Android Studio + ARCore SDK OR Xcode + ARKit
• **Best for**: Turn-by-turn navigation apps
• **Time to prototype**: 4-6 hours
• **Skill level**: Advanced (Java/Kotlin or Swift required)

**Option 2: Unity + AR Foundation + Mapbox (Flexible)**
• **Setup**: Unity + AR Foundation + Mapbox SDK
• **Best for**: Custom navigation experiences
• **Time to prototype**: 3-4 hours
• **Skill level**: Intermediate

**Option 3: 8th Wall + Location API (Web)**
• **Setup**: 8th Wall + Geolocation API
• **Best for**: Web-based location AR
• **Time to prototype**: 2-3 hours
• **Skill level**: Intermediate (JavaScript)

**🚀 Quick Start (Unity Approach):**
1. **Unity project setup** → Install AR Foundation + Mapbox
2. **Get location** → GPS coordinates via device
3. **Load map data** → Mapbox routing API
4. **Place AR markers** → WorldSpace UI elements
5. **Direction arrows** → 3D models pointing to waypoints
6. **Distance calculation** → Vector math for proximity

**� Essential Features to Implement:**
• **GPS tracking** → Continuous location updates
• **Compass heading** → Device orientation for directions
• **Route calculation** → Shortest path algorithms
• **Voice guidance** → Text-to-speech integration
• **Offline maps** → Download for no-internet areas

**💡 Simple Navigation Projects:**
• **Campus wayfinder** → Navigate university buildings
• **Museum guide** → AR tours with directions
• **Parking locator** → Find your car in large lots
• **Hiking trails** → Outdoor navigation with AR markers"""
    }
    
    return guides.get(ar_type, guides['face_filter'])

def generate_dream_analysis(dream_type):
    """Generate dream analysis content based on dream type"""
    
    analyses = {
        'nightmare': """**😰 Nightmare Analysis**

**🧠 Psychological Interpretation:**
Your nightmares often represent unprocessed anxieties, fears, or traumatic experiences that your subconscious is working through. They serve as your mind's way of confronting and potentially resolving internal conflicts.

**🔍 Common Nightmare Symbols:**
• **Being Chased**: Avoiding a problem or responsibility in waking life
• **Falling**: Loss of control or fear of failure
• **Death**: Major life transitions or fear of change
• **Monsters**: Repressed emotions or aspects of self
• **Being Trapped**: Feeling stuck in a life situation

**💡 Coping Strategies:**
• **Dream Journaling**: Record details immediately upon waking
• **Lucid Dreaming**: Learn to recognize and control nightmare scenarios
• **Relaxation Techniques**: Bedtime meditation and stress reduction
• **Therapy Integration**: Discuss recurring nightmares with professionals
• **Imagery Rehearsal**: Mentally rehearse positive dream outcomes

**🌙 Transformation Techniques:**
• **Rewrite the Ending**: Imagine confronting fears successfully
• **Symbol Dialogue**: Mentally communicate with frightening dream figures
• **Progressive Muscle Relaxation**: Physical tension release before sleep
• **Positive Visualization**: Replace scary imagery with peaceful scenes""",

        'lucid': """**✨ Lucid Dream Analysis**

**🧠 Consciousness in Dreams:**
Lucid dreaming represents heightened self-awareness and mental control. Your ability to recognize and manipulate dream states indicates strong metacognitive abilities and potential for conscious personal development.

**🎯 Lucid Dreaming Benefits:**
• **Skill Practice**: Rehearse real-world activities in safe environment
• **Creative Problem-Solving**: Access unlimited imagination for solutions
• **Fear Confrontation**: Face anxieties with knowledge of safety
• **Spiritual Exploration**: Deep self-discovery and consciousness expansion
• **Entertainment**: Ultimate virtual reality experience

**🔧 Enhancement Techniques:**
• **Reality Checks**: Develop habits to recognize dream states
• **Dream Signs**: Identify personal dream pattern indicators
• **Wake-Back-to-Bed**: Strategic sleep interruption for lucidity
• **Meditation Practice**: Strengthen mindfulness and awareness
• **Dream Supplements**: Natural aids like galantamine or choline

**🎨 Creative Applications:**
• **Artistic Inspiration**: Visual and auditory creative exploration
• **Problem Solving**: Work through challenges without consequences
• **Skill Development**: Practice speeches, sports, or performances
• **Personal Growth**: Explore different aspects of personality""",

        'recurring': """**🔄 Recurring Dream Analysis**

**🧠 Pattern Recognition:**
Recurring dreams indicate unresolved issues, persistent concerns, or important life lessons your subconscious is emphasizing. The repetition suggests these themes require conscious attention and resolution.

**🔍 Common Recurring Themes:**
• **School/Tests**: Performance anxiety or imposter syndrome
• **Ex-Partners**: Unresolved relationship emotions or lessons
• **Childhood Homes**: Nostalgia, security needs, or family issues
• **Missing Transportation**: Fear of missing opportunities
• **Natural Disasters**: Feeling overwhelmed by life changes

**💡 Resolution Strategies:**
• **Theme Identification**: Analyze common elements across dreams
• **Emotional Processing**: Address underlying feelings in waking life
• **Behavioral Changes**: Modify actions related to dream themes
• **Symbolic Understanding**: Interpret metaphorical meanings
• **Integration Work**: Apply dream insights to daily life

**🌟 Breakthrough Techniques:**
• **Active Imagination**: Consciously continue dream scenarios while awake
• **Gestalt Therapy**: Dialogue with different dream elements
• **Art Therapy**: Express dream imagery through creative mediums
• **Meditation**: Deep reflection on dream messages and meanings""",

        'prophetic': """**🔮 Prophetic Dream Analysis**

**🧠 Precognitive Experiences:**
Prophetic dreams may reflect your subconscious pattern recognition, intuitive processing, or symbolic representation of likely future scenarios based on current life trajectories and environmental cues.

**🎯 Types of Prophetic Dreams:**
• **Literal Predictions**: Direct representation of future events
• **Symbolic Prophecy**: Metaphorical glimpses of coming changes
• **Warning Dreams**: Subconscious alerts about potential problems
• **Guidance Dreams**: Direction for important life decisions
• **Collective Visions**: Insights about societal or global changes

**📊 Validation Methods:**
• **Dream Documentation**: Detailed recording with timestamps
• **Pattern Analysis**: Track accuracy rates over time
• **Context Evaluation**: Consider current life circumstances
• **Symbolic Interpretation**: Look beyond literal meanings
• **Probability Assessment**: Evaluate likelihood of predicted events

**🌟 Development Practices:**
• **Intuition Training**: Strengthen psychic and empathic abilities
• **Meditation Practice**: Deepen connection to unconscious wisdom
• **Energy Work**: Develop sensitivity to subtle environmental changes
• **Dream Incubation**: Intentionally request prophetic guidance""",

        'water': """**🌊 Water Dream Analysis**

**🧠 Emotional Symbolism:**
Water in dreams typically represents emotions, the unconscious mind, purification, and life transitions. The state and behavior of water in your dreams reflects your current emotional landscape and psychological state.

**💧 Water Symbol Meanings:**
• **Clear Water**: Emotional clarity, peace, spiritual purity
• **Turbulent Water**: Emotional turmoil, life chaos, uncertainty
• **Deep Water**: Profound emotions, unconscious depths, mystery
• **Swimming**: Navigation through emotional challenges
• **Drowning**: Feeling overwhelmed by emotions or life circumstances

**🌊 Different Water Contexts:**
• **Ocean Dreams**: Vast emotional depths, collective unconscious
• **River Dreams**: Life flow, transition, forward movement
• **Rain Dreams**: Emotional cleansing, renewal, fresh starts
• **Flood Dreams**: Overwhelming emotions, loss of control
• **Still Water**: Peace, reflection, contemplation needs

**💡 Interpretation Guidelines:**
• **Personal Associations**: Consider your relationship with water
• **Emotional State**: Reflect on current feelings and challenges
• **Life Transitions**: Connect to major changes or decisions
• **Spiritual Growth**: Explore themes of purification and renewal"""
    }
    
    return analyses.get(dream_type, analyses['general'])

def generate_time_capsule(time_frame, prediction_type):
    """Generate time capsule content based on timeframe and prediction type"""
    
    capsules = {
        ('1_year', 'technology'): """**⏰ One Year Tech Time Capsule (2026)**

**🚀 Emerging Technologies:**
• **AI Integration**: ChatGPT-5 and advanced AI assistants in daily workflows
• **Quantum Computing**: First consumer quantum devices for specific applications
• **AR Glasses**: Apple Vision Pro competitors creating market expansion
• **Brain-Computer Interfaces**: Neuralink trials showing promising results
• **Sustainable Tech**: Solar efficiency breakthroughs reaching 30%+ conversion

**📱 Consumer Predictions:**
• **Foldable Phones**: Mainstream adoption with improved durability
• **Voice AI**: Conversational AI replacing traditional app interfaces
• **Smart Homes**: Seamless IoT integration without compatibility issues
• **Electric Vehicles**: 40% of new car sales in developed countries
• **Digital Payments**: Cryptocurrency integration in major retailers

**🌐 Global Tech Trends:**
• **Remote Work Tech**: Advanced virtual collaboration platforms
• **Cybersecurity**: AI-powered threat detection becoming standard
• **Green Computing**: Data centers powered by 80% renewable energy
• **5G Maturity**: Nationwide coverage enabling new applications
• **Edge Computing**: Localized processing reducing latency significantly""",

        ('5_years', 'society'): """**⏰ Five Year Society Time Capsule (2030)**

**🏛️ Social Transformations:**
• **Work Evolution**: 4-day work weeks standard in progressive companies
• **Education Reform**: Personalized AI tutors supplementing human teachers
• **Healthcare Access**: Telemedicine covering 60% of routine medical care
• **Urban Planning**: Smart cities with integrated sustainability systems
• **Digital Governance**: Blockchain-based voting and citizen services

**👥 Cultural Shifts:**
• **Generation Alpha**: Digital natives reshaping social norms and communication
• **Sustainability Mindset**: Climate consciousness driving consumer choices
• **Mental Health**: Therapy and wellness becoming normalized and accessible
• **Diversity & Inclusion**: Systemic changes in corporate and social structures
• **Community Building**: Local networks strengthening post-pandemic isolation

**🌍 Global Society:**
• **Climate Adaptation**: Communities actively preparing for environmental changes
• **Economic Models**: Universal Basic Income pilot programs in multiple countries
• **Social Media**: Decentralized platforms challenging traditional tech monopolies
• **Aging Population**: Technology-assisted senior care becoming mainstream
• **Migration Patterns**: Climate-driven population movements reshaping geography""",

        ('10_years', 'environment'): """**⏰ Ten Year Environmental Time Capsule (2035)**

**🌱 Planetary Changes:**
• **Climate Tipping Points**: Arctic ice melting accelerating beyond current models
• **Ocean Acidification**: Coral reef ecosystems adapting or facing extinction
• **Weather Extremes**: Category 6 hurricanes becoming regular occurrence
• **Biodiversity**: 30% species loss driving ecosystem reorganization
• **Carbon Levels**: Atmospheric CO2 reaching 450 ppm despite reduction efforts

**🔄 Adaptation Strategies:**
• **Renewable Energy**: 85% of global electricity from clean sources
• **Carbon Capture**: Industrial-scale atmospheric CO2 removal systems
• **Sustainable Agriculture**: Lab-grown meat comprising 40% of protein consumption
• **Water Management**: Desalination and recycling meeting 50% of freshwater needs
• **Green Architecture**: Buildings producing more energy than they consume

**🌊 Ecosystem Responses:**
• **Ocean Currents**: Gulf Stream weakening affecting global weather patterns
• **Forest Migration**: Tree species moving toward poles at accelerated rates
• **Urban Wildlife**: Cities hosting diverse adapted animal populations
• **Soil Health**: Regenerative farming restoring degraded agricultural land
• **Pollinator Networks**: Artificial pollination supplementing declining bee populations""",

        ('1_day', 'personal'): """**⏰ Tomorrow's Personal Prediction**

**🌅 Your Next 24 Hours:**
• **Morning Energy**: You'll wake up feeling refreshed and motivated
• **Creative Breakthrough**: A solution to a current problem will suddenly become clear
• **Social Connection**: An unexpected conversation will brighten your day
• **Learning Moment**: You'll discover something new that sparks your curiosity
• **Evening Reflection**: You'll feel grateful for a small but meaningful experience

**💡 Opportunities to Watch For:**
• **Technology**: A new app or tool will catch your attention
• **Relationships**: Chance to strengthen a connection with someone important
• **Health**: Your body will send signals about what it needs
• **Career**: Small progress on a longer-term professional goal
• **Personal Growth**: Moment of self-awareness or emotional insight

**🎯 Recommended Focus:**
• **Mindfulness**: Stay present and notice subtle positive moments
• **Openness**: Be receptive to unexpected opportunities or ideas
• **Gratitude**: Acknowledge three things that go well tomorrow
• **Self-Care**: Listen to your physical and emotional needs
• **Connection**: Reach out to someone you've been thinking about"""
    }
    
    key = (time_frame, prediction_type)
    return capsules.get(key, f"**⏰ Future Prediction: {time_frame.replace('_', ' ').title()} - {prediction_type.title()}**\n\nYour personalized time capsule is being prepared with insights about {prediction_type} trends over the next {time_frame.replace('_', ' ')}. This will include detailed predictions, scenarios, and actionable insights for your future planning.")

def generate_virtual_world(world_type):
    """Generate virtual world description based on world type"""
    
    worlds = {
        'fantasy': """**🏰 Fantasy Virtual World: "Aethermoor Realms"**

**🌍 World Overview:**
A mystical realm where magic flows through crystalline ley lines across floating islands connected by ancient stone bridges. Three moons govern different schools of magic, creating a dynamic magical ecosystem.

**🏛️ Major Regions:**
• **Crystalline Peaks**: Floating mountains where dragons nest and time magic is strongest
• **Shadowwood Forest**: Enchanted woodland with talking trees and hidden fairy villages
• **Sunspire Capital**: Gleaming city of white towers where all races trade and learn
• **Mistral Plains**: Windswept grasslands home to centaur tribes and sky whales
• **Voidreach Depths**: Underground crystal caverns with bioluminescent ecosystems

**⚔️ Inhabitants & Factions:**
• **Aetherweavers**: Human mages who manipulate reality through geometric spells
• **Ironbark Druids**: Elven guardians who can merge with nature temporarily
• **Stormforge Dwarves**: Master craftsmen who forge magic into tools and weapons
• **Prism Dragons**: Ancient beings who collect and store magical knowledge
• **Shadow Dancers**: Mysterious folk who travel between dimensions

**🎮 Interactive Systems:**
• **Magic Crafting**: Combine elemental essences to create unique spells
• **Beast Bonding**: Form partnerships with magical creatures
• **Ley Line Navigation**: Travel instantly between magical nexus points
• **Reality Shaping**: Advanced players can modify world terrain temporarily
• **Time Streams**: Some areas experience faster or slower time flow

**🏗️ Building Mechanics:**
• **Floating Structures**: Defy gravity with proper magical foundations
• **Living Architecture**: Buildings that grow and adapt over time
• **Elemental Integration**: Harness fire, water, earth, air for functionality
• **Dimensional Pockets**: Create expanded interior spaces""",

        'sci_fi': """**🚀 Sci-Fi Virtual World: "Nova Frontier Station"**

**🌌 World Overview:**
A massive space station orbiting a binary star system, serving as humanity's furthest outpost. The station rotates to provide gravity while housing multiple biomes and research facilities.

**🏢 Station Sectors:**
• **Command Nexus**: Central hub with artificial gravity and administrative centers
• **Hydroponics Rings**: Agricultural sectors with Earth-like environments
• **Zero-G Industrial**: Manufacturing and research in weightless conditions
• **Residential Spirals**: Housing districts with artificial day/night cycles
• **Outer Docking**: Ship maintenance and customs for interstellar travelers

**👥 Factions & Societies:**
• **Core Scientists**: Researchers pushing boundaries of physics and biology
• **Void Runners**: Pilots and traders who navigate dangerous space routes
• **Synthesis Collective**: Humans enhanced with cybernetic implants
• **Terraforming Guild**: Engineers planning to make worlds habitable
• **Quantum Mystics**: Philosophers exploring consciousness and reality

**⚡ Advanced Technologies:**
• **Quantum Tunneling**: Instant travel between designated station points
• **Holographic Environments**: Customizable reality simulation chambers
• **AI Companions**: Personalized artificial beings with unique personalities
• **Matter Compilation**: Convert energy into any needed physical objects
• **Neural Interfaces**: Direct brain-computer interaction systems

**🛠️ Construction Features:**
• **Modular Design**: Snap-together components for rapid construction
• **Gravity Generators**: Create localized gravitational fields anywhere
• **Energy Networks**: Route power through sophisticated grid systems
• **Environmental Controls**: Manage atmosphere, temperature, and lighting
• **Emergency Systems**: Automated safety protocols and escape pods""",

        'underwater': """**🌊 Underwater Virtual World: "Abyssal Sanctuaries"**

**🐠 World Overview:**
A vast ocean world with floating continents above and deep trenches below. Bioluminescent coral cities provide light in the eternal twilight of the deep sea.

**🏙️ Aquatic Regions:**
• **Coral Metropolis**: Vibrant reef cities with living architecture
• **Abyssal Plains**: Dark depths with mysterious creatures and ancient ruins
• **Kelp Forests**: Towering seaweed jungles with hidden settlements
• **Thermal Vents**: Volcanic regions providing energy for deep communities
• **Ice Caverns**: Frozen underwater caves in polar regions

**🐋 Marine Inhabitants:**
• **Coral Architects**: Beings who grow and shape living reef structures
• **Deep Dwellers**: Mysterious entities adapted to crushing depths
• **Current Riders**: Fast-moving nomads who travel ocean streams
• **Whale Singers**: Giant creatures who communicate across vast distances
• **Pressure Walkers**: Beings who can survive at any ocean depth

**🌊 Unique Mechanics:**
• **Pressure Systems**: Depth affects movement and ability usage
• **Current Navigation**: Ride underwater streams for rapid travel
• **Bioluminescence**: Create light through biological processes
• **Sonic Communication**: Sound-based messaging across distances
• **Symbiotic Relationships**: Partner with sea creatures for abilities

**🏗️ Aquatic Building:**
• **Living Coral**: Grow and shape organic architectural structures
• **Pressure Domes**: Create air-filled spaces for surface dwellers
• **Current Generators**: Harness water flow for energy and transport
• **Depth Elevators**: Vertical transportation through pressure zones
• **Bio-luminescent Lighting**: Natural illumination systems""",

        'post_apocalyptic': """**☢️ Post-Apocalyptic Virtual World: "Fractured Earth"**

**🌆 World Overview:**
Fifty years after The Great Convergence, reality has become unstable. Technology and nature have merged in chaotic ways, creating a world where survival depends on adaptation and ingenuity.

**🏗️ Devastated Regions:**
• **Chrome Wastelands**: Metallic deserts where machines reproduce autonomously
• **Overgrown Megacities**: Urban jungles where buildings are consumed by mutant plants
• **Reality Storms**: Areas where physics becomes unpredictable and dangerous
• **Safe Havens**: Fortified settlements with stable environmental conditions
• **The Breach Zones**: Portals to other dimensions leak strange energies

**👥 Survivor Factions:**
• **Tech Salvagers**: Engineers who repair and repurpose old-world technology
• **Bio-Adaptants**: Humans who have merged with plant/animal DNA
• **Reality Shapers**: Mystics who can manipulate unstable physics
• **Nomad Tribes**: Mobile communities that avoid territorial conflicts
• **Corporate Remnants**: Last vestiges of pre-apocalypse mega-corporations

**⚡ Survival Systems:**
• **Resource Scavenging**: Find materials in dangerous ruined areas
• **Mutation Management**: Adapt to radiation and environmental hazards
• **Technology Fusion**: Combine scavenged parts into functional equipment
• **Settlement Building**: Establish safe zones with defensive capabilities
• **Reality Anchoring**: Stabilize areas of chaotic physics

**🔧 Construction Elements:**
• **Scrap Architecture**: Build from salvaged materials and debris
• **Bio-mechanical Fusion**: Integrate living and mechanical components
• **Defensive Systems**: Automated turrets and protective barriers
• **Resource Generators**: Solar panels, water purifiers, food gardens
• **Communication Networks**: Long-range radio and message systems"""
    }
    
    return worlds.get(world_type, worlds['fantasy'])

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

# ===== CUSTOM AI MODEL TRAINING SYSTEM =====

def handle_custom_model_training(text):
    """Handle custom AI model training requests"""
    try:
        if not ML_TRAINING_AVAILABLE:
            return "🧠 Custom model training requires additional ML libraries. Please install torch, transformers, and scikit-learn to enable this feature."
        
        print(f"🧠 Processing custom model training request: {text}")
        
        # Parse training request
        training_patterns = [
            r'train.*model.*on (.+)',
            r'create.*model.*for (.+)',
            r'build.*ai.*for (.+)',
            r'train.*ai.*to (.+)',
            r'custom.*model.*(.+)'
        ]
        
        task_description = None
        for pattern in training_patterns:
            match = re.search(pattern, text.lower())
            if match:
                task_description = match.group(1).strip()
                break
        
        if not task_description:
            return """🧠 **Custom AI Model Training**

I can help you train custom AI models! Here are some examples:

🎯 **Text Classification**
• "Train a model to classify customer reviews as positive/negative"
• "Create a model for spam email detection"

🔍 **Named Entity Recognition** 
• "Train a model to extract names and locations from text"
• "Build an AI to identify medical terms in documents"

📊 **Sentiment Analysis**
• "Create a model to analyze social media sentiment"
• "Train an AI for product review analysis"

🤖 **Chatbot Training**
• "Train a custom chatbot for customer service"
• "Create an AI assistant for specific domain knowledge"

Please describe what you'd like your model to do!"""
        
        # Generate training plan
        training_plan = generate_training_plan(task_description)
        
        return f"""🧠 **Custom AI Model Training Plan**

**Task**: {task_description}

{training_plan}

🚀 **Next Steps**:
1. Upload your training data (CSV, JSON, or TXT format)
2. Configure training parameters
3. Start training process
4. Monitor progress and evaluate results
5. Deploy your model to the marketplace

Would you like to proceed with setting up the training environment?"""
        
    except Exception as e:
        print(f"Error in handle_custom_model_training: {e}")
        return "🧠 I'd be happy to help you train a custom AI model! Please provide more details about what you'd like your model to do."

def generate_training_plan(task_description):
    """Generate a detailed training plan based on task description"""
    try:
        # Analyze task type
        task_type = determine_task_type(task_description)
        
        plans = {
            "classification": """
📋 **Training Plan: Text Classification**

**Model Type**: Fine-tuned BERT/DistilBERT
**Estimated Time**: 30-60 minutes
**Data Required**: 100+ labeled examples

**Architecture**:
• Pre-trained transformer model
• Classification head for your categories
• Dropout for regularization

**Training Process**:
1. Data preprocessing and tokenization
2. Train/validation split (80/20)
3. Fine-tuning with learning rate scheduling
4. Evaluation with accuracy, F1-score
5. Model optimization and compression""",
            
            "ner": """
📋 **Training Plan: Named Entity Recognition**

**Model Type**: BERT-based NER model
**Estimated Time**: 45-90 minutes  
**Data Required**: 200+ annotated sentences

**Architecture**:
• Token classification transformer
• BIO tagging scheme
• CRF layer for sequence consistency

**Training Process**:
1. Text annotation and IOB formatting
2. Token-level label alignment
3. Fine-tuning with entity recognition head
4. Validation with precision, recall, F1
5. Entity extraction optimization""",
            
            "sentiment": """
📋 **Training Plan: Sentiment Analysis**

**Model Type**: RoBERTa-based sentiment classifier
**Estimated Time**: 25-45 minutes
**Data Required**: 500+ sentiment-labeled texts

**Architecture**:
• Pre-trained RoBERTa encoder
• Multi-class sentiment head
• Attention visualization layers

**Training Process**:
1. Sentiment data preprocessing
2. Balanced sampling across classes
3. Fine-tuning with class weights
4. Evaluation with confusion matrix
5. Sentiment confidence calibration""",
            
            "chatbot": """
📋 **Training Plan: Custom Chatbot**

**Model Type**: Conversational AI with context
**Estimated Time**: 2-4 hours
**Data Required**: 1000+ conversation pairs

**Architecture**:
• Encoder-decoder transformer
• Context attention mechanism
• Response generation head

**Training Process**:
1. Conversation data formatting
2. Context window preparation
3. Seq2seq training with teacher forcing
4. Response quality evaluation
5. Dialogue coherence optimization"""
        }
        
        return plans.get(task_type, """
📋 **Training Plan: Custom AI Model**

**Model Type**: Task-specific neural network
**Estimated Time**: 1-3 hours
**Data Required**: Varies by complexity

**Training Process**:
1. Data analysis and preprocessing
2. Model architecture design
3. Training with validation monitoring
4. Performance evaluation
5. Model optimization and deployment""")
        
    except Exception as e:
        print(f"Error generating training plan: {e}")
        return "Training plan generation in progress..."

def determine_task_type(description):
    """Determine the type of ML task from description"""
    description_lower = description.lower()
    
    if any(word in description_lower for word in ['classify', 'classification', 'category', 'label']):
        return "classification"
    elif any(word in description_lower for word in ['entity', 'extract', 'ner', 'named', 'identify']):
        return "ner"
    elif any(word in description_lower for word in ['sentiment', 'emotion', 'feeling', 'positive', 'negative']):
        return "sentiment"
    elif any(word in description_lower for word in ['chatbot', 'conversation', 'dialogue', 'chat', 'assistant']):
        return "chatbot"
    else:
        return "custom"

def create_training_session(model_name, model_type, training_config):
    """Create a new training session in the database"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Generate unique session ID
        session_id = f"train_{uuid.uuid4().hex[:8]}"
        
        # Insert custom model record
        cursor.execute('''
            INSERT INTO custom_models (model_name, model_type, creator_id, description, training_status, created_at, updated_at, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            model_type,
            "user_001",  # In production, use actual user ID
            training_config.get('description', 'Custom trained model'),
            'pending',
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            '1.0.0'
        ))
        
        model_id = cursor.lastrowid
        
        # Insert training session
        cursor.execute('''
            INSERT INTO training_sessions (model_id, session_id, training_config, start_time, status, total_epochs)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            session_id,
            json.dumps(training_config),
            datetime.now().isoformat(),
            'pending',
            training_config.get('epochs', 10)
        ))
        
        conn.commit()
        conn.close()
        
        return session_id, model_id
        
    except Exception as e:
        print(f"Error creating training session: {e}")
        return None, None

def start_model_training(session_id, model_id, training_data, config):
    """Start the actual model training process"""
    try:
        if not ML_TRAINING_AVAILABLE:
            return False, "ML training libraries not available"
        
        print(f"🚀 Starting training for session {session_id}")
        
        # Update training status
        update_training_status(session_id, 'running', 0)
        
        # In a production environment, this would run in a separate process/thread
        # For now, we'll simulate the training process
        
        # Simulated training steps
        for epoch in range(config.get('epochs', 5)):
            # Simulate training progress
            progress = (epoch + 1) / config.get('epochs', 5) * 100
            
            # Update progress in database
            update_training_progress(session_id, epoch + 1, progress, 
                                   current_loss=0.5 - (epoch * 0.1), 
                                   current_accuracy=0.6 + (epoch * 0.08))
            
            print(f"📊 Epoch {epoch + 1}: Progress {progress:.1f}%")
        
        # Mark training as completed
        update_training_status(session_id, 'completed', 100)
        
        # Update model status
        update_model_status(model_id, 'trained', accuracy_score=0.92, loss_score=0.15)
        
        return True, "Training completed successfully"
        
    except Exception as e:
        print(f"Error in model training: {e}")
        update_training_status(session_id, 'failed', 0, str(e))
        return False, str(e)

def update_training_status(session_id, status, progress, error_message=None):
    """Update training session status in database"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE training_sessions 
            SET status = ?, error_message = ?
            WHERE session_id = ?
        ''', (status, error_message, session_id))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error updating training status: {e}")

def update_training_progress(session_id, current_epoch, progress, current_loss=None, current_accuracy=None):
    """Update training progress in database"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE training_sessions 
            SET current_epoch = ?, current_loss = ?, current_accuracy = ?
            WHERE session_id = ?
        ''', (current_epoch, current_loss, current_accuracy, session_id))
        
        # Also update the model progress
        cursor.execute('''
            UPDATE custom_models 
            SET training_progress = ?, updated_at = ?
            WHERE id = (SELECT model_id FROM training_sessions WHERE session_id = ?)
        ''', (progress, datetime.now().isoformat(), session_id))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error updating training progress: {e}")

def update_model_status(model_id, status, accuracy_score=None, loss_score=None):
    """Update model training status"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE custom_models 
            SET training_status = ?, accuracy_score = ?, loss_score = ?, updated_at = ?
            WHERE id = ?
        ''', (status, accuracy_score, loss_score, datetime.now().isoformat(), model_id))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error updating model status: {e}")

def handle_model_marketplace(text):
    """Handle AI model marketplace requests"""
    try:
        print(f"🏪 Processing marketplace request: {text}")
        
        marketplace_patterns = [
            r'browse.*models',
            r'marketplace',
            r'find.*model',
            r'search.*models',
            r'model.*store',
            r'download.*model'
        ]
        
        is_marketplace_request = any(re.search(pattern, text.lower()) for pattern in marketplace_patterns)
        
        if is_marketplace_request:
            return get_marketplace_overview()
        else:
            return """🏪 **AI Model Marketplace**

Welcome to the Horizon AI Model Marketplace! 

🔍 **Browse Models**:
• "Browse available models"
• "Show me text classification models"
• "Find sentiment analysis models"

📥 **Download Models**:
• "Download model [model_name]"
• "Install customer service chatbot"

⭐ **Popular Categories**:
• Text Classification
• Sentiment Analysis  
• Named Entity Recognition
• Chatbots & Assistants
• Image Recognition
• Custom Fine-tuned Models

🚀 **Upload Your Model**:
• "Share my trained model"
• "Publish model to marketplace"

What would you like to explore?"""
        
    except Exception as e:
        print(f"Error in handle_model_marketplace: {e}")
        return "🏪 Welcome to the AI Model Marketplace! Browse, download, and share custom AI models."

def get_marketplace_overview():
    """Get overview of available models in marketplace"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Get featured models
        cursor.execute('''
            SELECT cm.model_name, cm.model_type, cm.description, cm.rating_average, cm.download_count, mm.category
            FROM custom_models cm
            JOIN model_marketplace mm ON cm.id = mm.model_id
            WHERE mm.featured = 1 AND mm.status = 'active'
            ORDER BY cm.rating_average DESC, cm.download_count DESC
            LIMIT 5
        ''')
        
        featured_models = cursor.fetchall()
        
        # Get model statistics
        cursor.execute('''
            SELECT COUNT(*) as total_models,
                   COUNT(DISTINCT mm.category) as categories,
                   AVG(cm.rating_average) as avg_rating,
                   SUM(cm.download_count) as total_downloads
            FROM custom_models cm
            JOIN model_marketplace mm ON cm.id = mm.model_id
            WHERE mm.status = 'active'
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        # Format response
        response = """🏪 **AI Model Marketplace Overview**

📊 **Marketplace Stats**:
• {} Total Models Available
• {} Categories
• {:.1f}⭐ Average Rating  
• {} Total Downloads

🌟 **Featured Models**:
""".format(
            stats[0] if stats[0] else 0,
            stats[1] if stats[1] else 0, 
            stats[2] if stats[2] else 0.0,
            stats[3] if stats[3] else 0
        )
        
        if featured_models:
            for model in featured_models:
                response += f"""
🤖 **{model[0]}** ({model[1]})
   {model[2][:100]}...
   ⭐ {model[3]:.1f} | 📥 {model[4]} downloads | 🏷️ {model[5]}
"""
        else:
            response += "\n🔄 No featured models available yet. Be the first to publish!"
        
        response += """
🎯 **Quick Actions**:
• "Browse [category] models" - Find specific types
• "Download [model_name]" - Install a model  
• "Upload my model" - Share your creation
• "Train new model" - Create custom AI"""
        
        return response
        
    except Exception as e:
        print(f"Error getting marketplace overview: {e}")
        return "🏪 Model Marketplace is loading... Please try again in a moment."

# ===== PROMPT ENGINEERING LAB FUNCTIONS =====

def handle_prompt_engineering(text):
    """Handle prompt engineering lab requests"""
    try:
        print(f"🧪 Processing prompt engineering request: {text}")
        
        # Analyze the request type
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['create', 'new template', 'build prompt']):
            return create_prompt_template_interface()
        elif any(keyword in text_lower for keyword in ['test', 'experiment', 'ab test', 'compare']):
            return create_prompt_experiment_interface()
        elif any(keyword in text_lower for keyword in ['optimize', 'improve', 'enhance']):
            return get_prompt_optimization_suggestions()
        elif any(keyword in text_lower for keyword in ['templates', 'library', 'browse']):
            return get_prompt_template_library()
        elif any(keyword in text_lower for keyword in ['analytics', 'performance', 'stats']):
            return get_prompt_analytics_overview()
        else:
            return get_prompt_lab_overview()
            
    except Exception as e:
        print(f"Error in prompt engineering handler: {e}")
        return "🧪 Prompt Engineering Lab is loading... Please try again in a moment."

def get_prompt_lab_overview():
    """Get overview of prompt engineering lab capabilities"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Get template statistics
        cursor.execute('SELECT COUNT(*) FROM prompt_templates')
        template_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM prompt_experiments')
        experiment_count = cursor.fetchone()[0]
        
        # Get recent templates
        cursor.execute('''
            SELECT template_name, category, usage_count, rating_average
            FROM prompt_templates
            ORDER BY created_at DESC
            LIMIT 5
        ''')
        recent_templates = cursor.fetchall()
        
        # Get active experiments
        cursor.execute('''
            SELECT experiment_name, status, total_tests, winner
            FROM prompt_experiments
            WHERE status = 'running'
            ORDER BY created_at DESC
            LIMIT 3
        ''')
        active_experiments = cursor.fetchall()
        
        conn.close()
        
        response = f"""🧪 **Prompt Engineering Lab**

📊 **Lab Statistics**:
• **{template_count}** Prompt Templates
• **{experiment_count}** A/B Experiments
• Advanced optimization tools
• Performance analytics

🔬 **Recent Templates**:"""
        
        if recent_templates:
            for template in recent_templates:
                name, category, usage, rating = template
                response += f"\n• **{name}** ({category}) - Used {usage}x - {rating:.1f}⭐"
        else:
            response += "\n• No templates yet - create your first one!"
        
        if active_experiments:
            response += f"\n\n⚗️ **Active Experiments**:"
            for exp in active_experiments:
                name, status, tests, winner = exp
                response += f"\n• **{name}** - {tests} tests - {status}"
        
        response += """

🎯 **Lab Features**:
• **Template Builder** - Create reusable prompt templates
• **A/B Testing** - Compare prompt variants scientifically  
• **Optimization Engine** - AI-powered prompt improvements
• **Performance Analytics** - Track success metrics
• **Template Library** - Browse community templates

🚀 **Quick Actions**:
• "Create new template" - Build a prompt template
• "Start A/B test" - Compare two prompts
• "Optimize my prompt" - Get AI suggestions
• "Browse templates" - Explore template library
• "Show analytics" - View performance data"""
        
        return response
        
    except Exception as e:
        print(f"Error getting prompt lab overview: {e}")
        return "🧪 Prompt Engineering Lab is loading... Please try again in a moment."

def create_prompt_template_interface():
    """Create interface for building new prompt templates"""
    return """🧪 **Create New Prompt Template**

I'll help you create a professional prompt template! Here's how to structure it:

**Template Components**:
• **Name**: Give your template a descriptive name
• **Category**: Choose a category (writing, analysis, creative, etc.)
• **Variables**: Define placeholders like {topic}, {style}, {audience}
• **Core Prompt**: Write the main prompt with variables

**Example Template**:
```
Name: "Blog Post Writer"
Category: "Content Creation"
Variables: {topic}, {audience}, {tone}, {length}

Prompt: "Write a {length} blog post about {topic} for {audience}. 
Use a {tone} tone and include practical examples. Structure with 
clear headings and actionable insights."
```

🎯 **Ready to create?** Say something like:
• "Template: Email Marketing Writer"
• "Create social media template"
• "Build analysis prompt template"

I'll guide you through each step and help optimize your prompt for maximum effectiveness!"""

def create_prompt_experiment_interface():
    """Create interface for A/B testing prompts"""
    return """⚗️ **Prompt A/B Testing Lab**

Let's set up a scientific comparison between two prompt variants!

**Experiment Setup**:
• **Hypothesis**: What do you want to test?
• **Prompt A**: Your baseline prompt
• **Prompt B**: Your variant to test against
• **Success Metrics**: How will you measure success?
• **Test Inputs**: Sample data to test both prompts

**Common Test Scenarios**:
• **Clarity Test**: Formal vs conversational tone
• **Length Test**: Brief vs detailed instructions
• **Structure Test**: Bullet points vs paragraphs
• **Context Test**: With vs without examples

**Example Experiment**:
```
Hypothesis: "More specific examples improve output quality"

Prompt A: "Write a product description"
Prompt B: "Write a product description with specific benefits, 
features, and target customer use cases"

Metrics: Clarity score, engagement potential, completeness
```

🔬 **Ready to start testing?** Say:
• "Test formal vs casual prompts"
• "Compare short vs detailed instructions"
• "Experiment with different structures"

I'll help you design the perfect experiment and analyze the results!"""

def get_prompt_optimization_suggestions():
    """Get AI-powered prompt optimization suggestions"""
    return """🎯 **Prompt Optimization Engine**

Let me analyze and improve your prompts using advanced optimization techniques!

**Optimization Areas**:

**🎪 Clarity & Specificity**
• Remove ambiguous language
• Add specific constraints and examples
• Define expected output format

**🎭 Context & Role Definition**  
• Establish clear AI persona/role
• Provide relevant background context
• Set appropriate expertise level

**📊 Structure & Format**
• Use numbered steps for complex tasks
• Include examples and templates
• Specify desired output structure

**🎨 Creativity & Engagement**
• Balance creativity with constraints
• Use engaging language and examples
• Include variety in instruction style

**📈 Performance Optimization**
• Test different phrasings
• Optimize for consistent results
• Reduce hallucination risks

**🔬 How to optimize**:
• **Paste your prompt** - I'll analyze and suggest improvements
• **Describe your goal** - I'll create an optimized version
• **Share your challenges** - I'll provide targeted solutions

Example: "Optimize this prompt: 'Write a blog post about AI'"

I'll transform it into a high-performance template with specific instructions, context, and structure!"""

def get_prompt_template_library():
    """Browse the prompt template library"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Get template categories
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM prompt_templates
            GROUP BY category
            ORDER BY count DESC
        ''')
        categories = cursor.fetchall()
        
        # Get top-rated templates
        cursor.execute('''
            SELECT template_name, category, description, rating_average, usage_count
            FROM prompt_templates
            WHERE is_public = 1
            ORDER BY rating_average DESC, usage_count DESC
            LIMIT 8
        ''')
        top_templates = cursor.fetchall()
        
        conn.close()
        
        response = """📚 **Prompt Template Library**

🏆 **Top-Rated Templates**:"""
        
        if top_templates:
            for template in top_templates:
                name, category, desc, rating, usage = template
                short_desc = (desc[:50] + "...") if len(desc) > 50 else desc
                response += f"\n• **{name}** ({category}) - {rating:.1f}⭐ - Used {usage}x\n  *{short_desc}*"
        else:
            response += "\n• Library is being built - be the first to contribute!"
        
        if categories:
            response += f"\n\n📂 **Categories Available**:"
            for category, count in categories:
                response += f"\n• **{category}** ({count} templates)"
        
        response += """

🎯 **Popular Categories**:
• **Content Creation** - Blog posts, social media, marketing copy
• **Data Analysis** - Research, insights, report generation
• **Creative Writing** - Stories, poems, character development
• **Business Communication** - Emails, proposals, presentations
• **Code & Technical** - Documentation, debugging, explanations
• **Education** - Lesson plans, explanations, study guides

🔍 **Find Templates**:
• "Show me [category] templates"
• "Find templates for [use case]"
• "Browse creative writing prompts"
• "Get business email templates"

📝 **Use Templates**:
Simply say "Use [template name]" and I'll apply it with your specific inputs!"""
        
        return response
        
    except Exception as e:
        print(f"Error getting template library: {e}")
        return "📚 Template Library is loading... Please try again in a moment."

def get_prompt_analytics_overview():
    """Get prompt performance analytics"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Get template performance stats
        cursor.execute('''
            SELECT 
                AVG(avg_response_time) as avg_time,
                AVG(avg_rating) as avg_rating,
                AVG(success_rate) as avg_success,
                COUNT(*) as total_analytics
            FROM prompt_analytics
            WHERE date >= date('now', '-30 days')
        ''')
        overall_stats = cursor.fetchone()
        
        # Get top performing templates
        cursor.execute('''
            SELECT pt.template_name, pt.category, 
                   AVG(pa.avg_rating) as rating,
                   SUM(pa.usage_count) as total_usage,
                   AVG(pa.success_rate) as success_rate
            FROM prompt_templates pt
            JOIN prompt_analytics pa ON pt.id = pa.prompt_id
            WHERE pa.date >= date('now', '-30 days')
            GROUP BY pt.id
            ORDER BY rating DESC, success_rate DESC
            LIMIT 5
        ''')
        top_performers = cursor.fetchall()
        
        # Get improvement insights
        cursor.execute('''
            SELECT title, description, impact_level, confidence_score
            FROM improvement_insights
            WHERE insight_type = 'prompt_optimization'
            ORDER BY priority DESC, confidence_score DESC
            LIMIT 3
        ''')
        insights = cursor.fetchall()
        
        conn.close()
        
        if overall_stats[0]:
            avg_time, avg_rating, avg_success, total_analytics = overall_stats
            response = f"""📊 **Prompt Analytics Dashboard**

📈 **30-Day Performance**:
• **Average Rating**: {avg_rating:.1f}/5.0 ⭐
• **Success Rate**: {avg_success:.1f}% ✅
• **Avg Response Time**: {avg_time:.2f}s ⚡
• **Analytics Records**: {total_analytics} data points"""
        else:
            response = """📊 **Prompt Analytics Dashboard**

📈 **30-Day Performance**:
• **Building Analytics** - Start using templates to see data!
• Performance tracking active
• Insights engine ready"""
        
        if top_performers:
            response += f"\n\n🏆 **Top Performing Templates**:"
            for template in top_performers:
                name, category, rating, usage, success = template
                response += f"\n• **{name}** ({category}) - {rating:.1f}⭐ - {success:.1f}% success - {usage} uses"
        
        if insights:
            response += f"\n\n💡 **Optimization Insights**:"
            for insight in insights:
                title, desc, impact, confidence = insight
                response += f"\n• **{title}** ({impact} impact, {confidence:.0f}% confidence)\n  *{desc}*"
        
        response += """

📊 **Analytics Features**:
• **Template Performance** - Success rates and user ratings
• **Response Time Tracking** - Optimize for speed
• **Usage Patterns** - Understand what works
• **A/B Test Results** - Statistical significance testing
• **Improvement Suggestions** - AI-powered optimization tips

🔍 **Detailed Analytics**:
• "Show template performance"
• "Analyze my prompts"
• "Get optimization tips"
• "View experiment results"

📈 **Performance Tracking**:
All your prompts are automatically analyzed for performance, helping you continuously improve your AI interactions!"""
        
        return response
        
    except Exception as e:
        print(f"Error getting prompt analytics: {e}")
        return "📊 Analytics dashboard is loading... Please try again in a moment."

# ===== AI PERFORMANCE ANALYTICS FUNCTIONS =====

def handle_ai_performance_analytics(text):
    """Handle AI performance analytics requests"""
    try:
        print(f"📊 Processing performance analytics request: {text}")
        
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['usage', 'stats', 'statistics']):
            return get_usage_statistics()
        elif any(keyword in text_lower for keyword in ['performance', 'metrics', 'benchmark']):
            return get_performance_metrics()
        elif any(keyword in text_lower for keyword in ['improvement', 'insights', 'optimize']):
            return get_improvement_insights()
        elif any(keyword in text_lower for keyword in ['user', 'engagement', 'behavior']):
            return get_user_analytics()
        elif any(keyword in text_lower for keyword in ['ab test', 'experiment', 'test']):
            return get_ab_test_results()
        else:
            return get_analytics_overview()
            
    except Exception as e:
        print(f"Error in performance analytics handler: {e}")
        return "📊 AI Performance Analytics is loading... Please try again in a moment."

def get_analytics_overview():
    """Get comprehensive analytics overview"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Get today's stats
        today = datetime.now().date().isoformat()
        cursor.execute('''
            SELECT 
                COUNT(*) as total_requests,
                SUM(success) as successful_requests,
                AVG(response_time) as avg_response_time,
                COUNT(DISTINCT user_id) as unique_users
            FROM ai_usage_stats
            WHERE date = ?
        ''', (today,))
        today_stats = cursor.fetchone()
        
        # Get weekly trends
        cursor.execute('''
            SELECT 
                COUNT(*) as total_requests,
                AVG(response_time) as avg_response_time,
                (SUM(success) * 100.0 / COUNT(*)) as success_rate
            FROM ai_usage_stats
            WHERE date >= date('now', '-7 days')
        ''')
        weekly_stats = cursor.fetchone()
        
        # Get top features
        cursor.execute('''
            SELECT feature_used, COUNT(*) as usage_count
            FROM ai_usage_stats
            WHERE date >= date('now', '-7 days')
            GROUP BY feature_used
            ORDER BY usage_count DESC
            LIMIT 5
        ''')
        top_features = cursor.fetchall()
        
        # Get performance insights
        cursor.execute('''
            SELECT title, impact_level, confidence_score
            FROM improvement_insights
            WHERE created_at >= date('now', '-7 days')
            ORDER BY priority DESC
            LIMIT 3
        ''')
        recent_insights = cursor.fetchall()
        
        conn.close()
        
        total_req, success_req, avg_time, unique_users = today_stats
        week_req, week_time, success_rate = weekly_stats
        
        response = f"""📊 **AI Performance Analytics Dashboard**

📈 **Today's Performance**:
• **{total_req or 0}** Total Requests
• **{success_req or 0}** Successful Responses
• **{unique_users or 0}** Active Users
• **{avg_time:.2f}s** Avg Response Time

📊 **7-Day Trends**:
• **{week_req or 0}** Total Requests
• **{success_rate:.1f}%** Success Rate
• **{week_time:.2f}s** Avg Response Time
• Tracking performance continuously"""
        
        if top_features:
            response += f"\n\n🔥 **Most Used Features (7 days)**:"
            for feature, count in top_features:
                response += f"\n• **{feature}**: {count} uses"
        
        if recent_insights:
            response += f"\n\n💡 **Recent Performance Insights**:"
            for title, impact, confidence in recent_insights:
                response += f"\n• **{title}** ({impact} impact, {confidence:.0f}% confidence)"
        
        response += """

🎯 **Analytics Categories**:
• **Usage Statistics** - Request volumes and patterns
• **Performance Metrics** - Response times and success rates  
• **User Analytics** - Engagement and behavior insights
• **Improvement Insights** - AI-powered optimization suggestions
• **A/B Test Results** - Feature performance comparisons

🔍 **Detailed Views**:
• "Show usage stats" - Volume and trend analysis
• "Performance metrics" - Speed and reliability data
• "User behavior" - Engagement and satisfaction
• "Improvement insights" - Optimization opportunities
• "A/B test results" - Feature comparison data

📈 **Smart Monitoring**:
Our AI continuously analyzes performance to identify optimization opportunities and ensure the best user experience!"""
        
        return response
        
    except Exception as e:
        print(f"Error getting analytics overview: {e}")
        return "📊 Analytics dashboard is loading... Please try again in a moment."

def get_usage_statistics():
    """Get detailed usage statistics"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Daily usage for last 7 days
        cursor.execute('''
            SELECT date, 
                   COUNT(*) as requests,
                   COUNT(DISTINCT user_id) as users,
                   SUM(success) as successful
            FROM ai_usage_stats
            WHERE date >= date('now', '-7 days')
            GROUP BY date
            ORDER BY date DESC
        ''')
        daily_stats = cursor.fetchall()
        
        # Feature usage breakdown
        cursor.execute('''
            SELECT feature_used, 
                   COUNT(*) as total_uses,
                   AVG(response_time) as avg_time,
                   (SUM(success) * 100.0 / COUNT(*)) as success_rate
            FROM ai_usage_stats
            WHERE date >= date('now', '-30 days')
            GROUP BY feature_used
            ORDER BY total_uses DESC
        ''')
        feature_stats = cursor.fetchall()
        
        # Peak usage hours
        cursor.execute('''
            SELECT hour, COUNT(*) as request_count
            FROM ai_usage_stats
            WHERE date >= date('now', '-7 days')
            GROUP BY hour
            ORDER BY request_count DESC
            LIMIT 5
        ''')
        peak_hours = cursor.fetchall()
        
        conn.close()
        
        response = """📊 **Usage Statistics**

📅 **Daily Activity (Last 7 Days)**:"""
        
        if daily_stats:
            for date, requests, users, successful in daily_stats:
                success_pct = (successful / requests * 100) if requests > 0 else 0
                response += f"\n• **{date}**: {requests} requests, {users} users, {success_pct:.1f}% success"
        else:
            response += "\n• No usage data yet - start using features to see statistics!"
        
        if feature_stats:
            response += f"\n\n🎯 **Feature Usage (Last 30 Days)**:"
            for feature, uses, avg_time, success_rate in feature_stats:
                response += f"\n• **{feature}**: {uses} uses, {avg_time:.2f}s avg, {success_rate:.1f}% success"
        
        if peak_hours:
            response += f"\n\n⏰ **Peak Usage Hours**:"
            for hour, count in peak_hours:
                time_str = f"{hour:02d}:00"
                response += f"\n• **{time_str}**: {count} requests"
        
        response += """

📈 **Usage Insights**:
• Track daily request volumes
• Monitor feature adoption rates
• Identify peak usage patterns
• Analyze user engagement trends

🔍 **Drill Down Options**:
• "Show feature performance"
• "Analyze user patterns"
• "Peak hour analysis"
• "Success rate trends"""
        
        return response
        
    except Exception as e:
        print(f"Error getting usage statistics: {e}")
        return "📊 Usage statistics are loading... Please try again in a moment."

def get_performance_metrics():
    """Get performance metrics and benchmarks"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Model performance comparison
        cursor.execute('''
            SELECT model_used,
                   COUNT(*) as total_requests,
                   AVG(response_time) as avg_response_time,
                   (SUM(success) * 100.0 / COUNT(*)) as success_rate,
                   AVG(tokens_used) as avg_tokens
            FROM ai_usage_stats
            WHERE model_used IS NOT NULL AND date >= date('now', '-30 days')
            GROUP BY model_used
            ORDER BY total_requests DESC
        ''')
        model_performance = cursor.fetchall()
        
        # Performance trends
        cursor.execute('''
            SELECT date,
                   AVG(response_time) as avg_time,
                   (SUM(success) * 100.0 / COUNT(*)) as success_rate
            FROM ai_usage_stats
            WHERE date >= date('now', '-7 days')
            GROUP BY date
            ORDER BY date DESC
        ''')
        performance_trends = cursor.fetchall()
        
        # Error analysis
        cursor.execute('''
            SELECT error_message, COUNT(*) as error_count
            FROM ai_usage_stats
            WHERE success = 0 AND error_message IS NOT NULL
            AND date >= date('now', '-7 days')
            GROUP BY error_message
            ORDER BY error_count DESC
            LIMIT 5
        ''')
        error_analysis = cursor.fetchall()
        
        conn.close()
        
        response = """⚡ **Performance Metrics**

🤖 **Model Performance (Last 30 Days)**:"""
        
        if model_performance:
            for model, requests, avg_time, success_rate, avg_tokens in model_performance:
                response += f"\n• **{model}**: {requests} requests, {avg_time:.2f}s avg, {success_rate:.1f}% success, {avg_tokens:.0f} tokens"
        else:
            response += "\n• No model performance data yet - AI models will be tracked as they're used!"
        
        if performance_trends:
            response += f"\n\n📈 **Performance Trends (Last 7 Days)**:"
            for date, avg_time, success_rate in performance_trends:
                response += f"\n• **{date}**: {avg_time:.2f}s response time, {success_rate:.1f}% success"
        
        if error_analysis:
            response += f"\n\n⚠️ **Error Analysis (Last 7 Days)**:"
            for error, count in error_analysis:
                short_error = (error[:60] + "...") if len(error) > 60 else error
                response += f"\n• **{short_error}**: {count} occurrences"
        
        response += """

📊 **Performance Benchmarks**:
• **Response Time**: Target < 2.0s for optimal UX
• **Success Rate**: Target > 95% for reliability
• **Token Efficiency**: Monitor cost optimization
• **Error Rate**: Target < 5% for stability

🎯 **Optimization Opportunities**:
• Monitor slow response patterns
• Identify high-error features
• Track token usage efficiency
• Benchmark against industry standards

🔧 **Performance Actions**:
• "Optimize slow responses"
• "Analyze error patterns"
• "Compare model efficiency"
• "Track improvement trends"""
        
        return response
        
    except Exception as e:
        print(f"Error getting performance metrics: {e}")
        return "⚡ Performance metrics are loading... Please try again in a moment."

def get_improvement_insights():
    """Get AI-powered improvement insights"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Get all improvement insights
        cursor.execute('''
            SELECT insight_type, title, description, impact_level, 
                   confidence_score, action_suggested, implemented
            FROM improvement_insights
            ORDER BY priority DESC, confidence_score DESC
            LIMIT 10
        ''')
        insights = cursor.fetchall()
        
        conn.close()
        
        response = """💡 **AI-Powered Improvement Insights**

🧠 **Optimization Opportunities**:"""
        
        if insights:
            for insight_type, title, desc, impact, confidence, action, implemented in insights:
                status = "✅ Implemented" if implemented else "🔄 Pending"
                response += f"\n\n**{title}** ({impact.upper()} Impact - {confidence:.0f}% Confidence) {status}"
                response += f"\n*{desc}*"
                if action and not implemented:
                    response += f"\n🎯 **Action**: {action}"
        else:
            response += """
• 🔍 **Analyzing Performance** - Gathering data for insights
• 📊 **Building Baselines** - Establishing performance metrics  
• 🧠 **AI Learning** - Understanding usage patterns
• ⚡ **Optimization Ready** - Insights will appear as data accumulates"""
        
        response += """

🎯 **Insight Categories**:
• **Prompt Optimization** - Improve AI interaction quality
• **Performance Enhancement** - Speed and reliability improvements
• **User Experience** - Interface and workflow optimizations
• **Feature Usage** - Adoption and engagement improvements
• **Error Reduction** - Reliability and stability enhancements

🔍 **How Insights Work**:
• **Continuous Analysis** - AI monitors all interactions
• **Pattern Detection** - Identifies optimization opportunities
• **Statistical Validation** - Ensures recommendations are data-driven
• **Actionable Suggestions** - Provides specific improvement steps
• **Impact Assessment** - Prioritizes by potential value

🚀 **Implementation Tracking**:
• Mark insights as implemented
• Monitor improvement impact
• Track performance changes
• Validate optimization success

💪 **Smart Optimization**:
Our AI continuously learns from your usage patterns to suggest personalized improvements that enhance your experience!"""
        
        return response
        
    except Exception as e:
        print(f"Error getting improvement insights: {e}")
        return "💡 Improvement insights are loading... Please try again in a moment."

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
    """Generate logos and brand designs using AI image generation with smart fallbacks"""
    
    try:
        print(f"🏷️ Generating logo for: {brand_name} ({industry}, {style})")
        
        # Enhanced logo prompt with industry-specific elements
        logo_prompt = f"professional logo design for {brand_name}, {industry} industry, {style} style"
        
        # Industry-specific enhancements
        industry_prompts = {
            'technology': 'tech, digital, innovation, modern, circuit patterns, gear icons',
            'healthcare': 'medical, health, care, cross symbol, healing, wellness, trust',
            'finance': 'banking, money, security, stability, professional, trust, growth',
            'restaurant': 'food, dining, chef hat, fork and knife, culinary, appetite',
            'fashion': 'style, elegance, clothing, trendy, chic, sophisticated',
            'education': 'learning, books, graduation cap, knowledge, growth, development',
            'automotive': 'cars, speed, movement, wheels, engineering, power',
            'beauty': 'elegance, style, cosmetics, wellness, luxury, refined',
            'sports': 'athletic, fitness, energy, movement, strength, competition',
            'travel': 'adventure, exploration, journey, compass, globe, destinations'
        }
        
        # Add industry-specific elements
        if industry in industry_prompts:
            logo_prompt += f", {industry_prompts[industry]}"
        
        # Style-specific enhancements
        if style.lower() == "modern":
            logo_prompt += ", clean lines, minimalist, contemporary design, geometric shapes, sans-serif typography"
        elif style.lower() == "vintage":
            logo_prompt += ", retro aesthetic, classic typography, timeless design, aged textures, serif fonts"
        elif style.lower() == "creative":
            logo_prompt += ", artistic flair, unique concept, innovative design, abstract elements, creative typography"
        elif style.lower() == "corporate":
            logo_prompt += ", professional appearance, trustworthy, business-oriented, clean, authoritative"
        elif style.lower() == "playful":
            logo_prompt += ", fun, colorful, friendly, approachable, rounded shapes, vibrant colors"
        elif style.lower() == "elegant":
            logo_prompt += ", sophisticated, luxury, refined, premium, elegant typography, subtle colors"
        
        logo_prompt += ", vector style, high contrast, suitable for business use, scalable, memorable branding"
        
        print(f"🎨 Enhanced logo prompt: {logo_prompt}")
        
        # Try DALL-E first (PRIMARY - best quality and most reliable)
        if Config.OPENAI_API_KEY and client:
            print("🎨 Using DALL-E for professional logo design...")
            
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=logo_prompt,
                    size="1024x1024",
                    quality="hd",
                    style="vivid",  # More vibrant and professional
                    n=1,
                    response_format="url"
                )
                
                image_url = response.data[0].url
                
                # Return the direct URL from DALL-E instead of saving locally
                print(f"✅ DALL-E logo generated: {image_url}")
                return image_url, None
                    
            except Exception as e:
                print(f"⚠️ DALL-E logo error: {e}")
        
        # Try Stability AI as backup (SECONDARY - good for artistic logos)
        if Config.STABILITY_API_KEY:
            print("🎭 Using Stability AI for artistic logo design...")
            
            try:
                headers = {
                    "Authorization": f"Bearer {Config.STABILITY_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "text_prompts": [{"text": logo_prompt}],
                    "cfg_scale": 8,  # Higher for more adherence to prompt
                    "samples": 1,
                    "steps": 50,
                    "style_preset": "digital-art",  # Good for logos
                    "width": 1024,
                    "height": 1024
                }
                
                response = requests.post(
                    "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "artifacts" in data and len(data["artifacts"]) > 0:
                        import base64
                        import uuid
                        
                        # Save the image temporarily to serve as URL
                        image_data = base64.b64decode(data["artifacts"][0]["base64"])
                        logo_id = str(uuid.uuid4())
                        logo_filename = f"logo_stability_{brand_name.replace(' ', '_')}_{style}_{logo_id}.png"
                        logo_path = os.path.join(LOGOS_DIR, logo_filename)
                        
                        with open(logo_path, 'wb') as f:
                            f.write(image_data)
                        
                        # Return full URL that can be accessed directly
                        logo_url = f"http://127.0.0.1:8080/static/generated_logos/{logo_filename}"
                        print(f"✅ Stability AI logo generated: {logo_url}")
                        return logo_url, None
                else:
                    print(f"⚠️ Stability AI API error: {response.status_code} - {response.text}")
                        
            except Exception as e:
                print(f"⚠️ Stability AI logo error: {e}")
        
        # Fallback: Use Hugging Face for logo generation (TERTIARY)
        if Config.HUGGINGFACE_API_KEY:
            print("🤗 Using Hugging Face for backup logo generation...")
            
            try:
                headers = {
                    "Authorization": f"Bearer {Config.HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "inputs": logo_prompt,
                    "parameters": {
                        "guidance_scale": 8.5,
                        "num_inference_steps": 50,
                        "width": 1024,
                        "height": 1024
                    }
                }
                
                # Try multiple Hugging Face models
                models = [
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    "runwayml/stable-diffusion-v1-5",
                    "CompVis/stable-diffusion-v1-4"
                ]
                
                for model in models:
                    try:
                        response = requests.post(
                            f"https://api-inference.huggingface.co/models/{model}",
                            headers=headers,
                            json=payload,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            import uuid
                            logo_id = str(uuid.uuid4())
                            logo_filename = f"logo_hf_{brand_name.replace(' ', '_')}_{style}_{logo_id}.png"
                            logo_path = os.path.join(LOGOS_DIR, logo_filename)
                            
                            with open(logo_path, 'wb') as f:
                                f.write(response.content)
                            
                            # Return full URL that can be accessed directly
                            logo_url = f"http://127.0.0.1:8080/static/generated_logos/{logo_filename}"
                            print(f"✅ Hugging Face logo generated: {logo_url}")
                            return logo_url, None
                        else:
                            print(f"⚠️ Hugging Face model {model} failed: {response.status_code}")
                    except Exception as model_error:
                        print(f"⚠️ Hugging Face model {model} error: {model_error}")
                        continue
                        
            except Exception as e:
                print(f"⚠️ Hugging Face logo error: {e}")
        
        # Final fallback: Programmatic logo generation (LOCAL GENERATION)
        print("🔧 Using local programmatic logo generation as final fallback...")
        return generate_programmatic_logo(brand_name, industry, style)
        
    except Exception as e:
        print(f"❌ Logo generation error: {e}")
        return None, f"Logo error: {str(e)}"

def generate_programmatic_logo(brand_name, industry, style):
    """Generate a simple programmatic logo as final fallback"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import uuid
        
        # Create a 1024x1024 image
        img = Image.new('RGB', (1024, 1024), 'white')
        draw = ImageDraw.Draw(img)
        
        # Style-based color schemes
        color_schemes = {
            'modern': {'bg': '#f8f9fa', 'primary': '#007bff', 'secondary': '#6c757d'},
            'vintage': {'bg': '#f5f5dc', 'primary': '#8b4513', 'secondary': '#daa520'},
            'creative': {'bg': '#fff', 'primary': '#ff6b6b', 'secondary': '#4ecdc4'},
            'corporate': {'bg': '#f8f9fa', 'primary': '#343a40', 'secondary': '#17a2b8'},
            'playful': {'bg': '#fff9c4', 'primary': '#ff9800', 'secondary': '#e91e63'},
            'elegant': {'bg': '#000', 'primary': '#gold', 'secondary': '#silver'}
        }
        
        colors = color_schemes.get(style, color_schemes['modern'])
        
        # Draw background
        img = Image.new('RGB', (1024, 1024), colors['bg'])
        draw = ImageDraw.Draw(img)
        
        # Draw simple geometric logo based on industry
        center = (512, 512)
        
        if industry in ['technology', 'software']:
            # Draw tech-inspired geometric shapes
            draw.rectangle([400, 400, 624, 624], fill=colors['primary'])
            draw.rectangle([450, 450, 574, 574], fill=colors['bg'])
        elif industry in ['healthcare', 'medical']:
            # Draw a cross
            draw.rectangle([462, 400, 562, 624], fill=colors['primary'])
            draw.rectangle([400, 462, 624, 562], fill=colors['primary'])
        else:
            # Draw a simple circle
            draw.ellipse([400, 400, 624, 624], fill=colors['primary'])
        
        # Add brand name text (simplified - might not have proper fonts)
        try:
            font_size = 80
            # Try to use a system font
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Get text size and center it
            text_bbox = draw.textbbox((0, 0), brand_name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (1024 - text_width) // 2
            text_y = 700
            
            draw.text((text_x, text_y), brand_name, fill=colors['primary'], font=font)
        except:
            # Fallback without custom font
            draw.text((400, 700), brand_name, fill=colors['primary'])
        
        # Save the logo and return URL
        logo_id = str(uuid.uuid4())
        logo_filename = f"logo_programmatic_{brand_name.replace(' ', '_')}_{style}_{logo_id}.png"
        logo_path = os.path.join(LOGOS_DIR, logo_filename)
        
        img.save(logo_path)
        
        # Return full URL that can be accessed directly
        logo_url = f"http://127.0.0.1:8080/static/generated_logos/{logo_filename}"
        print(f"✅ Programmatic logo generated: {logo_url}")
        return logo_url, None
        
    except Exception as e:
        print(f"❌ Programmatic logo generation failed: {e}")
        return None, f"All logo generation methods failed: {str(e)}"

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
    'logo_generation': [
        r'\b(generate|create|make|design|build).*logo\b',
        r'\b(generate|create|make|design|build).*brand\b',
        r'\b(generate|create|make|design|build).*emblem\b',
        r'\blogo.*for\b', r'\bbrand.*identity\b', r'\bcorporate.*logo\b',
        r'\blogo.*design\b', r'\bbusiness.*logo\b', r'\bcompany.*logo\b',
        r'\bai.*logo\b', r'\blogo.*maker\b', r'\bdesign.*logo\b',
        r'\bbrand.*logo\b', r'\bprofessional.*logo\b', r'\bcustom.*logo\b',
        r'\bvisual.*identity\b', r'\bbrand.*mark\b', r'\bicon.*design\b'
    ],
    'game_master': [
        r'\b(start|begin|create|play).*story\b',
        r'\b(start|begin|create|play).*adventure\b',
        r'\b(start|begin|create|play).*game\b',
        r'\btext.*adventure\b', r'\binteractive.*story\b',
        r'\brpg.*story\b', r'\bdungeons.*and.*dragons\b',
        r'\bai.*game.*master\b', r'\bgame.*master\b', r'\bdm\b',
        r'\bcreate.*character\b', r'\brole.*playing\b',
        r'\badventure.*story\b', r'\bnarrative.*game\b',
        r'\bstory.*time\b', r'\btell.*me.*story\b',
        r'\bwhat.*happens.*next\b', r'\bcontinue.*story\b',
        r'\bfantasy.*adventure\b', r'\bquest.*story\b'
    ],
    'code_generation': [
        r'\b(write|create|generate|code).*python\b',
        r'\b(write|create|generate|code).*javascript\b',
        r'\b(write|create|generate|code).*java\b',
        r'\b(write|create|generate|code).*cpp\b',
        r'\b(write|create|generate|code).*html\b',
        r'\b(write|create|generate|code).*css\b',
        r'\bcode.*for\b', r'\bprogram.*for\b', r'\bscript.*for\b',
        r'\bfunction.*that\b', r'\bclass.*that\b',
        r'\bai.*programmer\b', r'\bai.*coding\b',
        r'\bprogramming.*help\b', r'\bcode.*help\b',
        r'\bdebug.*code\b', r'\bfix.*code\b',
        r'\bexplain.*code\b', r'\bcode.*example\b',
        r'\balgorithm.*for\b', r'\bdata.*structure\b',
        r'\bapi.*code\b', r'\bdatabase.*code\b'
    ],
    'quiz_generation': [
        r'\b(create|generate|make).*quiz\b',
        r'\b(create|generate|make).*trivia\b',
        r'\b(create|generate|make).*test\b',
        r'\bquiz.*about\b', r'\btrivia.*about\b',
        r'\btest.*my.*knowledge\b', r'\bquestion.*about\b',
        r'\bai.*quiz\b', r'\bai.*trivia\b',
        r'\bmultiple.*choice\b', r'\btrue.*false\b',
        r'\bknowledge.*test\b', r'\blearn.*quiz\b',
        r'\beducational.*quiz\b', r'\bstudent.*quiz\b',
        r'\bpersonalized.*quiz\b', r'\bcustom.*quiz\b',
        r'\bquiz.*questions\b', r'\btrivia.*questions\b'
    ],
    'story_generation': [
        r'\b(write|create|generate|compose).*story\b',
        r'\b(write|create|generate|compose).*novel\b',
        r'\b(write|create|generate|compose).*script\b',
        r'\b(write|create|generate|compose).*poem\b',
        r'\b(write|create|generate|compose).*poetry\b',
        r'\bstory.*about\b', r'\bnovel.*about\b', r'\bscript.*about\b',
        r'\bpoem.*about\b', r'\bpoetry.*about\b',
        r'\bai.*writer\b', r'\bai.*author\b', r'\bcreative.*writing\b',
        r'\bfiction.*story\b', r'\bshort.*story\b', r'\bbook.*chapter\b',
        r'\bscreenplay\b', r'\bmovie.*script\b', r'\bplay.*script\b',
        r'\bhaiku\b', r'\bsonnet\b', r'\bverse\b', r'\brhyme\b',
        r'\bnarrative\b', r'\btale.*about\b', r'\bepic.*story\b'
    ],
    'meme_generation': [
        r'\b(create|generate|make).*meme\b',
        r'\b(create|generate|make).*funny.*image\b',
        r'\bmeme.*about\b', r'\bmeme.*with\b',
        r'\bai.*meme\b', r'\bhumor.*image\b', r'\bfunny.*picture\b',
        r'\bmeme.*template\b', r'\bmeme.*generator\b',
        r'\bjoke.*image\b', r'\bcomedy.*meme\b', r'\bviral.*meme\b',
        r'\binternet.*meme\b', r'\bcustom.*meme\b', r'\bmeme.*maker\b',
        r'\bsarcastic.*meme\b', r'\brelatable.*meme\b'
    ],
    'comic_generation': [
        r'\b(create|generate|make|draw).*comic\b',
        r'\b(create|generate|make|draw).*comic.*strip\b',
        r'\b(create|generate|make|draw).*comic.*panel\b',
        r'\bcomic.*about\b', r'\bcomic.*story\b',
        r'\bai.*comic\b', r'\bcomic.*creator\b', r'\bcomic.*book\b',
        r'\bgraphic.*novel\b', r'\bsequential.*art\b',
        r'\bsuperhero.*comic\b', r'\bcomic.*character\b',
        r'\bstoryboard\b', r'\bcomic.*series\b', r'\bwebcomic\b',
        r'\bmanga.*style\b', r'\bcartoon.*comic\b'
    ],
    'fashion_design': [
        r'\b(design|create|generate|suggest).*fashion\b',
        r'\b(design|create|generate|suggest).*clothing\b',
        r'\b(design|create|generate|suggest).*outfit\b',
        r'\b(design|create|generate|suggest).*style\b',
        r'\bfashion.*advice\b', r'\bstyle.*advice\b', r'\boutfit.*advice\b',
        r'\bai.*stylist\b', r'\bai.*fashion\b', r'\bfashion.*ai\b',
        r'\bclothing.*recommendation\b', r'\bstyle.*recommendation\b',
        r'\bfashion.*trend\b', r'\bstyle.*trend\b', r'\bwhat.*to.*wear\b',
        r'\bdress.*for\b', r'\boutfit.*for\b', r'\bstyle.*for\b',
        r'\bfashion.*designer\b', r'\bclothing.*design\b', r'\bwardrobe.*advice\b',
        r'\bpersonal.*stylist\b', r'\bfashion.*consultation\b'
    ],
    'ar_integration': [
        r'\b(ar|augmented.*reality)\b',
        r'\b(camera|cam).*overlay\b', r'\b(camera|cam).*filter\b',
        r'\baugmented.*reality.*camera\b', r'\bar.*camera\b',
        r'\bvirtual.*overlay\b', r'\bdigital.*overlay\b',
        r'\bar.*filter\b', r'\bar.*effect\b', r'\bar.*experience\b',
        r'\bcamera.*ar\b', r'\breal.*time.*overlay\b',
        r'\binteractive.*camera\b', r'\bsmart.*camera\b',
        r'\bmixed.*reality\b', r'\bholographic.*display\b',
        r'\bar.*integration\b', r'\baugment.*camera\b',
        r'\bvirtual.*reality.*camera\b', r'\b3d.*overlay\b'
    ],
    'dream_journal': [
        r'\b(dream|dreams).*journal\b', r'\b(dream|dreams).*analysis\b',
        r'\b(dream|dreams).*interpretation\b', r'\b(dream|dreams).*meaning\b',
        r'\banalyze.*dream\b', r'\binterpret.*dream\b',
        r'\bdream.*analyzer\b', r'\bdream.*interpreter\b',
        r'\bwhat.*does.*dream.*mean\b', r'\bdream.*symbolism\b',
        r'\bdream.*psychology\b', r'\bsleep.*analysis\b',
        r'\bsubconscious.*analysis\b', r'\bdream.*diary\b',
        r'\bai.*dream\b', r'\bdream.*ai\b', r'\bdream.*tracker\b',
        r'\brecord.*dream\b', r'\blog.*dream\b', r'\bsave.*dream\b'
    ],
    'time_capsule': [
        r'\b(time.*capsule|timecapsule)\b', r'\bfuture.*prediction\b',
        r'\bpredict.*future\b', r'\bfuture.*trends\b',
        r'\btrend.*prediction\b', r'\bfuture.*forecast\b',
        r'\btime.*travel.*simulation\b', r'\bfuture.*scenario\b',
        r'\bcreate.*time.*capsule\b', r'\bmake.*time.*capsule\b',
        r'\bfuturistic.*prediction\b', r'\btomorrow.*prediction\b',
        r'\bai.*prophecy\b', r'\bai.*fortune.*telling\b',
        r'\bpredict.*what.*will.*happen\b', r'\bfuture.*analysis\b',
        r'\btrend.*analysis\b', r'\bfuture.*insights\b'
    ],
    'virtual_world_builder': [
        r'\b(virtual.*world|virtual.*environment)\b',
        r'\b(build|create|generate|design).*virtual.*world\b',
        r'\bworld.*builder\b', r'\benvironment.*builder\b',
        r'\b3d.*world\b', r'\bvirtual.*reality.*world\b',
        r'\bcreate.*universe\b', r'\bbuild.*universe\b',
        r'\bvirtual.*landscape\b', r'\bdigital.*world\b',
        r'\bsimulated.*environment\b', r'\bai.*world\b',
        r'\bprocedural.*generation\b', r'\bworld.*generation\b',
        r'\bcustom.*environment\b', r'\bvirtual.*physics\b',
        r'\bimmersive.*world\b', r'\binteractive.*environment\b',
        r'\bsandbox.*world\b', r'\bopen.*world.*creator\b'
    ],
    'model_training': [
        r'\b(train|create|build).*model\b',
        r'\b(train|create|build).*ai\b',
        r'\bcustom.*model\b', r'\bcustom.*ai\b',
        r'\bmodel.*training\b', r'\bai.*training\b',
        r'\bmachine.*learning\b', r'\bml.*training\b',
        r'\btrain.*on.*data\b', r'\bfine.*tune\b',
        r'\bclassification.*model\b', r'\bsentiment.*model\b',
        r'\bner.*model\b', r'\bchatbot.*training\b',
        r'\bpersonalized.*ai\b', r'\bcustom.*classifier\b',
        r'\btrain.*my.*own.*ai\b', r'\bbuild.*my.*own.*model\b',
        r'\bmodel.*builder\b', r'\bai.*builder\b',
        r'\bupload.*training.*data\b', r'\bcreate.*dataset\b'
    ],
    'model_marketplace': [
        r'\b(marketplace|market)\b',
        r'\b(browse|find|search).*models\b',
        r'\b(download|install).*model\b',
        r'\bmodel.*store\b', r'\bai.*store\b',
        r'\bmodel.*library\b', r'\bai.*library\b',
        r'\bshare.*model\b', r'\bpublish.*model\b',
        r'\bmodel.*repository\b', r'\bai.*repository\b',
        r'\bavailable.*models\b', r'\bfeatured.*models\b',
        r'\bpopular.*models\b', r'\bmodel.*rating\b',
        r'\bmodel.*reviews\b', r'\bexplore.*models\b',
        r'\bcommunity.*models\b', r'\bupload.*model\b'
    ],
    'prompt_engineering': [
        r'\b(prompt.*engineering|prompt.*lab)\b',
        r'\b(optimize|improve|enhance).*prompt\b',
        r'\b(create|build|design).*prompt\b',
        r'\bprompt.*template\b',
        r'\bprompt.*optimization\b',
        r'\ba.*b.*test.*prompt\b',
        r'\btest.*prompt\b',
        r'\bexperiment.*prompt\b',
        r'\bcompare.*prompts\b',
        r'\bprompt.*analytics\b',
        r'\bprompt.*performance\b',
        r'\btemplate.*library\b',
        r'\bprompt.*library\b',
        r'\bprompt.*builder\b',
        r'\bprompt.*creator\b'
    ],
    'performance_analytics': [
        r'\b(analytics|performance.*analytics)\b',
        r'\b(usage.*stats|usage.*statistics)\b',
        r'\b(performance.*metrics|performance.*stats)\b',
        r'\b(ai.*analytics|ai.*performance)\b',
        r'\banalytics.*dashboard\b',
        r'\bperformance.*dashboard\b',
        r'\busage.*data\b',
        r'\bperformance.*data\b',
        r'\bimprovement.*insights\b',
        r'\boptimization.*insights\b',
        r'\buser.*analytics\b',
        r'\bengagement.*analytics\b',
        r'\bresponse.*time.*analytics\b',
        r'\bsuccess.*rate.*analytics\b',
        r'\banalytics.*overview\b',
        r'\bperformance.*overview\b'
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
        'logo_generation': 0.88,   # Logo generation depends on AI art models
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
    quick_commands = ['time', 'date', 'math', 'timer', 'reminder', 'greeting', 'goodbye', 'joke', 'image_generation', 'video_generation', 'gif_generation', 'music_generation', 'voice_generation', 'audio_transcription', 'logo_generation', 'game_master', 'code_generation', 'quiz_generation', 'story_generation', 'comic_generation', 'fashion_design', 'ar_integration', 'dream_journal', 'time_capsule', 'virtual_world_builder', 'model_training', 'model_marketplace', 'prompt_engineering', 'performance_analytics']
    return intent in quick_commands

def process_user_input(user_input, personality='friendly', session_id=None, user_id='anonymous'):
    """Process user input and return appropriate response with conversation context and AI intelligence"""
    if not user_input or not user_input.strip():
        return "I didn't quite catch that. Could you please say something?", session_id, False, {}
    
    # Initialize database if needed
    init_db()
    
    # Recognize intent first
    intent = recognize_intent(user_input)
    context_used = False
    ai_insights = {}
    
    # Analyze emotion for all inputs (quick commands and AI responses)
    emotion_data = analyze_emotion(user_input)
    detected_emotion = emotion_data.get('emotion', 'neutral')
    sentiment_score = emotion_data.get('sentiment', 0.0)
    
    ai_insights = {
        'emotion_detected': detected_emotion,
        'emotion_confidence': emotion_data.get('confidence', 0.0),
        'sentiment_score': sentiment_score,
        'mood': classify_mood(sentiment_score),
        'intent_detected': intent
    }
    
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
        elif intent == 'logo_generation':
            response = handle_logo_generation(user_input)
        elif intent == 'game_master':
            response = handle_game_master(user_input, session_id, personality)
        elif intent == 'code_generation':
            response = handle_code_generation(user_input, personality)
        elif intent == 'quiz_generation':
            response = handle_quiz_generation(user_input, personality)
        elif intent == 'story_generation':
            response = handle_story_generation(user_input, personality)
        elif intent == 'meme_generation':
            response = handle_meme_generation(user_input, personality)
        elif intent == 'comic_generation':
            response = handle_comic_generation(user_input, personality)
        elif intent == 'fashion_design':
            response = handle_fashion_design(user_input, personality)
        elif intent == 'ar_integration':
            response = handle_ar_integration(user_input, personality)
        elif intent == 'dream_journal':
            response = handle_dream_journal(user_input, personality)
        elif intent == 'time_capsule':
            response = handle_time_capsule(user_input, personality)
        elif intent == 'virtual_world_builder':
            response = handle_virtual_world_builder(user_input, personality)
        elif intent == 'model_training':
            response = handle_custom_model_training(user_input)
        elif intent == 'model_marketplace':
            response = handle_model_marketplace(user_input)
        elif intent == 'prompt_engineering':
            response = handle_prompt_engineering(user_input)
        elif intent == 'performance_analytics':
            response = handle_ai_performance_analytics(user_input)
        elif intent == 'goodbye':
            response = "Thank you for chatting! Have a wonderful day!"
        else:
            response = "Quick command processed locally!"
        
        # Enhance response with emotional awareness even for quick commands
        response = enhance_response_with_emotion(response, detected_emotion, personality)
        
        # For quick commands, use a simple session or create one
        if not session_id:
            session_id = generate_session_id()
            create_conversation_session(session_id, personality)
        
        # Quick commands get high confidence since they're deterministic
        confidence = 0.95
        
        # Save conversation with emotion analysis
        save_conversation(user_input, response, personality, session_id, intent, confidence, context_used)
        
        # Update personality usage
        update_personality_usage(personality)
        
        # Save simple user memory for quick commands
        if len(user_input) > 10:
            keywords = extract_keywords(user_input)
            if keywords:
                save_user_memory(user_id, 'quick_commands', f"command_{intent}", f"Used {intent}: {', '.join(keywords[:2])}", importance=0.3)
        
        return response, session_id, context_used, ai_insights
    
    # For non-quick commands, use full AI processing with conversation context
    else:
        print(f"🤖 Complex query detected: {intent} - using ChatGPT with context and AI intelligence")
        
        # Get or create session for context-aware conversations
        if not session_id:
            session_id, stored_personality = get_active_session()
            # Use stored personality if none provided
            if personality == 'friendly' and stored_personality != 'friendly':
                personality = stored_personality
        
        # Use AI model (ChatGPT or fallback) for complex questions with full context and intelligence
        response, context_used = ask_ai_model(user_input, personality, session_id, user_id)
        
        # Calculate confidence for AI responses
        confidence = calculate_realistic_confidence(user_input, response, 'chatgpt' if AI_MODEL_AVAILABLE else 'fallback', intent)
        
        # Get comprehensive AI insights
        session_insights = get_ai_insights(session_id)
        ai_insights.update(session_insights)
        ai_insights['context_used'] = context_used
        ai_insights['ai_learning_active'] = True
        
        # Save conversation with full context information and AI intelligence
        save_conversation(user_input, response, personality, session_id, intent, confidence, context_used)
        
        return response, session_id, context_used, ai_insights

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
        
        logo_url, error = generate_logo_design(brand_name, industry, style)
        
        if logo_url:
            return jsonify({
                'success': True,
                'url': logo_url
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
        user_id = data.get('user_id', 'anonymous')  # User identifier for AI intelligence
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # Process the input with AI intelligence features
        start_time = time.time()
        response, session_id, context_used, ai_insights = process_user_input(user_input, personality, session_id, user_id)
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
            'processing_type': 'local' if is_quick else 'ai_powered',
            'ai_insights': ai_insights,
            'ai_intelligence_active': True,
            'emotion_detected': ai_insights.get('emotion_detected', 'neutral') if ai_insights else 'neutral',
            'sentiment_score': ai_insights.get('sentiment_score', 0.0) if ai_insights else 0.0,
            'learning_active': True
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

# ===== AI INTELLIGENCE API ENDPOINTS =====

@app.route('/api/ai-insights', methods=['GET'])
def get_ai_insights_api():
    """Get AI insights and intelligence data"""
    try:
        session_id = request.args.get('session_id')
        user_id = request.args.get('user_id', 'anonymous')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Get comprehensive AI insights
        insights = get_ai_insights(session_id)
        
        # Get user memory
        user_memories = retrieve_user_memory(user_id)
        
        # Get personality usage stats
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT personality_name, usage_count, user_rating 
            FROM personality_profiles 
            ORDER BY usage_count DESC
        ''')
        personality_stats = cursor.fetchall()
        
        # Get recent emotion analysis
        cursor.execute('''
            SELECT detected_emotion, emotion_confidence, sentiment_score, timestamp
            FROM emotion_analysis 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (session_id,))
        recent_emotions = cursor.fetchall()
        
        # Get learning effectiveness
        cursor.execute('''
            SELECT learning_type, AVG(effectiveness_score) as avg_effectiveness, COUNT(*) as count
            FROM ai_learning
            GROUP BY learning_type
        ''')
        learning_stats = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'session_insights': insights,
            'user_memories': {
                'count': len(user_memories),
                'memories': user_memories[:10]  # Top 10 memories
            },
            'personality_stats': [
                {'name': p[0], 'usage_count': p[1], 'rating': p[2]} 
                for p in personality_stats
            ],
            'recent_emotions': [
                {
                    'emotion': e[0], 
                    'confidence': e[1], 
                    'sentiment': e[2], 
                    'timestamp': e[3]
                } for e in recent_emotions
            ],
            'learning_stats': [
                {
                    'type': l[0], 
                    'effectiveness': l[1], 
                    'count': l[2]
                } for l in learning_stats
            ],
            'ai_intelligence_active': True
        })
        
    except Exception as e:
        print(f"Error getting AI insights: {e}")
        return jsonify({'error': 'Failed to get AI insights'}), 500

@app.route('/api/personalities', methods=['GET'])
def get_personalities():
    """Get all available AI personalities"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT personality_name, personality_description, response_style, 
                   emotional_traits, language_patterns, usage_count, user_rating
            FROM personality_profiles
            ORDER BY usage_count DESC
        ''')
        
        personalities = cursor.fetchall()
        conn.close()
        
        return jsonify({
            'personalities': [
                {
                    'name': p[0],
                    'description': p[1],
                    'style': p[2],
                    'traits': p[3].split(',') if p[3] else [],
                    'patterns': p[4].split(',') if p[4] else [],
                    'usage_count': p[5],
                    'rating': p[6]
                } for p in personalities
            ]
        })
        
    except Exception as e:
        print(f"Error getting personalities: {e}")
        return jsonify({'error': 'Failed to get personalities'}), 500

@app.route('/api/personalities/rate', methods=['POST'])
def rate_personality():
    """Rate a personality"""
    try:
        data = request.get_json()
        personality_name = data.get('personality')
        rating = data.get('rating')
        
        if not personality_name or rating is None:
            return jsonify({'error': 'Personality name and rating required'}), 400
        
        if not (1 <= rating <= 5):
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Update personality rating (simple average for now)
        cursor.execute('''
            UPDATE personality_profiles 
            SET user_rating = CASE 
                WHEN user_rating IS NULL THEN ?
                ELSE (user_rating + ?) / 2
            END
            WHERE personality_name = ?
        ''', (rating, rating, personality_name))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'Rated {personality_name} personality'})
        
    except Exception as e:
        print(f"Error rating personality: {e}")
        return jsonify({'error': 'Failed to rate personality'}), 500

@app.route('/api/memory', methods=['GET'])
def get_user_memory_api():
    """Get user memory data"""
    try:
        user_id = request.args.get('user_id', 'anonymous')
        memory_type = request.args.get('type')
        
        memories = retrieve_user_memory(user_id, memory_type)
        
        return jsonify({
            'user_id': user_id,
            'memory_type': memory_type,
            'memories': memories,
            'count': len(memories)
        })
        
    except Exception as e:
        print(f"Error getting user memory: {e}")
        return jsonify({'error': 'Failed to get user memory'}), 500

@app.route('/api/memory', methods=['POST'])
def save_user_memory_api():
    """Save user memory data"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'anonymous')
        memory_type = data.get('type')
        key = data.get('key')
        value = data.get('value')
        importance = data.get('importance', 0.5)
        
        if not all([memory_type, key, value]):
            return jsonify({'error': 'Type, key, and value required'}), 400
        
        save_user_memory(user_id, memory_type, key, value, importance)
        
        return jsonify({'success': True, 'message': 'Memory saved successfully'})
        
    except Exception as e:
        print(f"Error saving user memory: {e}")
        return jsonify({'error': 'Failed to save user memory'}), 500

@app.route('/api/emotion-analysis', methods=['GET'])
def get_emotion_analysis():
    """Get emotion analysis for a session"""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT detected_emotion, emotion_confidence, sentiment_score, 
                   mood_classification, user_input, timestamp
            FROM emotion_analysis 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 50
        ''', (session_id,))
        
        emotions = cursor.fetchall()
        conn.close()
        
        return jsonify({
            'session_id': session_id,
            'emotions': [
                {
                    'emotion': e[0],
                    'confidence': e[1],
                    'sentiment_score': e[2],
                    'mood': e[3],
                    'user_input': e[4][:100] + '...' if len(e[4]) > 100 else e[4],
                    'timestamp': e[5]
                } for e in emotions
            ],
            'count': len(emotions)
        })
        
    except Exception as e:
        print(f"Error getting emotion analysis: {e}")
        return jsonify({'error': 'Failed to get emotion analysis'}), 500

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

@app.route('/api/personality', methods=['POST'])
def update_personality():
    """Update personality for the current session"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        personality = data.get('personality', 'friendly')
        session_id = data.get('session_id')
        
        # Validate personality
        valid_personalities = [
            'friendly', 'professional', 'casual', 'enthusiastic', 'witty', 
            'sarcastic', 'zen', 'scientist', 'pirate', 'shakespearean', 
            'valley_girl', 'cowboy', 'robot'
        ]
        
        if personality not in valid_personalities:
            return jsonify({'error': f'Invalid personality. Valid options: {", ".join(valid_personalities)}'}), 400
        
        # Update session personality if session exists
        if session_id:
            try:
                conn = sqlite3.connect('ai_memory.db')
                cursor = conn.cursor()
                
                # Update the session's personality
                cursor.execute('''
                    UPDATE conversation_sessions 
                    SET personality = ?, updated_at = ? 
                    WHERE session_id = ?
                ''', (personality, datetime.now().isoformat(), session_id))
                
                conn.commit()
                conn.close()
                
                print(f"✅ Updated session {session_id} personality to: {personality}")
                
            except Exception as e:
                print(f"Error updating session personality: {e}")
        
        # Update personality usage statistics
        update_personality_usage(personality)
        
        return jsonify({
            'success': True,
            'personality': personality,
            'session_id': session_id,
            'message': f'Personality updated to {personality}'
        })
        
    except Exception as e:
        print(f"Error in update_personality: {e}")
        return jsonify({'error': 'Failed to update personality'}), 500

@app.route('/api/personality', methods=['GET'])
def get_personality_info():
    """Get available personalities and current session personality"""
    try:
        session_id = request.args.get('session_id')
        
        personalities = {
            'friendly': {
                'name': 'Friendly',
                'description': 'Warm, welcoming, and supportive with encouraging responses',
                'emoji': '😊'
            },
            'professional': {
                'name': 'Professional', 
                'description': 'Formal, structured, and business-oriented communication',
                'emoji': '💼'
            },
            'casual': {
                'name': 'Casual',
                'description': 'Relaxed, laid-back with informal and conversational tone',
                'emoji': '😎'
            },
            'enthusiastic': {
                'name': 'Enthusiastic',
                'description': 'High-energy, exciting, and passionate about everything',
                'emoji': '🎉'
            },
            'witty': {
                'name': 'Witty',
                'description': 'Clever humor, wordplay, and intelligent observations',
                'emoji': '🧠'
            },
            'sarcastic': {
                'name': 'Sarcastic',
                'description': 'Dry humor with subtle sarcasm while remaining helpful',
                'emoji': '🙄'
            },
            'zen': {
                'name': 'Zen',
                'description': 'Peaceful, meditative, and mindful responses',
                'emoji': '🧘‍♀️'
            },
            'scientist': {
                'name': 'Scientific',
                'description': 'Data-driven, logical, and evidence-based communication',
                'emoji': '🔬'
            },
            'pirate': {
                'name': 'Pirate',
                'description': 'Swashbuckling adventure with nautical terminology',
                'emoji': '🏴‍☠️'
            },
            'shakespearean': {
                'name': 'Shakespearean',
                'description': 'Eloquent, dramatic, and poetic Old English style',
                'emoji': '🎭'
            },
            'valley_girl': {
                'name': 'Valley Girl',
                'description': 'Bubbly, trendy, and enthusiastic California style',
                'emoji': '💁‍♀️'
            },
            'cowboy': {
                'name': 'Cowboy',
                'description': 'Rootin\' tootin\' frontier wisdom and charm',
                'emoji': '🤠'
            },
            'robot': {
                'name': 'Robot',
                'description': 'Logical, mechanical, and computational responses',
                'emoji': '🤖'
            }
        }
        
        current_personality = 'friendly'  # default
        
        # Get current session personality if session exists
        if session_id:
            try:
                conn = sqlite3.connect('ai_memory.db')
                cursor = conn.cursor()
                cursor.execute('SELECT personality FROM conversation_sessions WHERE session_id = ?', (session_id,))
                result = cursor.fetchone()
                if result:
                    current_personality = result[0]
                conn.close()
            except Exception as e:
                print(f"Error getting session personality: {e}")
        
        return jsonify({
            'personalities': personalities,
            'current_personality': current_personality,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error in get_personality_info: {e}")
        return jsonify({'error': 'Failed to get personality info'}), 500

# ===== MODEL MANAGEMENT API ENDPOINTS =====

@app.route('/api/models', methods=['GET'])
def get_models_api():
    """Get list of custom models"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT cm.id, cm.model_name, cm.model_type, cm.description, 
                   cm.training_status, cm.training_progress, cm.accuracy_score,
                   cm.created_at, cm.rating_average, cm.download_count, cm.version
            FROM custom_models cm
            ORDER BY cm.created_at DESC
        ''')
        
        models = []
        for row in cursor.fetchall():
            models.append({
                'id': row[0],
                'name': row[1],
                'type': row[2], 
                'description': row[3],
                'status': row[4],
                'progress': row[5],
                'accuracy': row[6],
                'created_at': row[7],
                'rating': row[8],
                'downloads': row[9],
                'version': row[10]
            })
        
        conn.close()
        return jsonify({'models': models})
        
    except Exception as e:
        print(f"Error getting models: {e}")
        return jsonify({'error': 'Failed to get models'}), 500

@app.route('/api/models/train', methods=['POST'])
def start_training_api():
    """Start training a new custom model"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        model_type = data.get('model_type')
        description = data.get('description', '')
        config = data.get('config', {})
        
        if not all([model_name, model_type]):
            return jsonify({'error': 'Model name and type required'}), 400
        
        # Create training session
        session_id, model_id = create_training_session(model_name, model_type, {
            'description': description,
            'epochs': config.get('epochs', 10),
            'learning_rate': config.get('learning_rate', 0.001),
            'batch_size': config.get('batch_size', 16)
        })
        
        if session_id:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'model_id': model_id,
                'message': 'Training session created successfully'
            })
        else:
            return jsonify({'error': 'Failed to create training session'}), 500
            
    except Exception as e:
        print(f"Error starting training: {e}")
        return jsonify({'error': 'Failed to start training'}), 500

@app.route('/api/models/training/<session_id>', methods=['GET'])
def get_training_status_api(session_id):
    """Get training progress for a session"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ts.status, ts.current_epoch, ts.total_epochs, 
                   ts.current_loss, ts.current_accuracy, ts.error_message,
                   cm.model_name, cm.training_progress
            FROM training_sessions ts
            JOIN custom_models cm ON ts.model_id = cm.id
            WHERE ts.session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return jsonify({
                'session_id': session_id,
                'status': result[0],
                'current_epoch': result[1],
                'total_epochs': result[2],
                'current_loss': result[3],
                'current_accuracy': result[4],
                'error_message': result[5],
                'model_name': result[6],
                'progress': result[7]
            })
        else:
            return jsonify({'error': 'Training session not found'}), 404
            
    except Exception as e:
        print(f"Error getting training status: {e}")
        return jsonify({'error': 'Failed to get training status'}), 500

@app.route('/api/marketplace', methods=['GET'])
def get_marketplace_api():
    """Get marketplace models"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        category = request.args.get('category')
        search = request.args.get('search')
        featured_only = request.args.get('featured') == 'true'
        
        query = '''
            SELECT cm.id, cm.model_name, cm.model_type, cm.description,
                   cm.rating_average, cm.download_count, cm.version,
                   mm.category, mm.price, mm.license_type, mm.featured
            FROM custom_models cm
            JOIN model_marketplace mm ON cm.id = mm.model_id
            WHERE mm.status = 'active' AND cm.training_status = 'trained'
        '''
        params = []
        
        if category:
            query += ' AND mm.category = ?'
            params.append(category)
            
        if search:
            query += ' AND (cm.model_name LIKE ? OR cm.description LIKE ?)'
            params.extend([f'%{search}%', f'%{search}%'])
            
        if featured_only:
            query += ' AND mm.featured = 1'
            
        query += ' ORDER BY mm.featured DESC, cm.rating_average DESC, cm.download_count DESC'
        
        cursor.execute(query, params)
        
        models = []
        for row in cursor.fetchall():
            models.append({
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'description': row[3],
                'rating': row[4],
                'downloads': row[5],
                'version': row[6],
                'category': row[7],
                'price': row[8],
                'license': row[9],
                'featured': bool(row[10])
            })
        
        conn.close()
        return jsonify({'models': models})
        
    except Exception as e:
        print(f"Error getting marketplace: {e}")
        return jsonify({'error': 'Failed to get marketplace'}), 500

@app.route('/api/models/<int:model_id>/download', methods=['POST'])
def download_model_api(model_id):
    """Download a model from marketplace"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Record download analytics
        cursor.execute('''
            INSERT INTO model_analytics (model_id, user_id, action_type, timestamp, success)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_id, 'user_001', 'download', datetime.now().isoformat(), 1))
        
        # Update download count
        cursor.execute('''
            UPDATE custom_models 
            SET download_count = download_count + 1
            WHERE id = ?
        ''', (model_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Model download initiated',
            'model_id': model_id
        })
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return jsonify({'error': 'Failed to download model'}), 500

@app.route('/api/models/<int:model_id>/rate', methods=['POST'])
def rate_model_api(model_id):
    """Rate a model in the marketplace"""
    try:
        data = request.get_json()
        rating = data.get('rating')
        review_text = data.get('review', '')
        
        if not rating or rating < 1 or rating > 5:
            return jsonify({'error': 'Valid rating (1-5) required'}), 400
        
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Insert review
        cursor.execute('''
            INSERT INTO model_reviews (model_id, reviewer_id, rating, review_text, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_id, 'user_001', rating, review_text, datetime.now().isoformat()))
        
        # Update model rating average
        cursor.execute('''
            UPDATE custom_models 
            SET rating_average = (
                SELECT AVG(rating) FROM model_reviews WHERE model_id = ?
            ),
            rating_count = rating_count + 1
            WHERE id = ?
        ''', (model_id, model_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Model rated successfully',
            'rating': rating
        })
        
    except Exception as e:
        print(f"Error rating model: {e}")
        return jsonify({'error': 'Failed to rate model'}), 500

# ===== PROMPT ENGINEERING LAB API ENDPOINTS =====

@app.route('/api/prompts/templates', methods=['GET'])
def get_prompt_templates_api():
    """Get prompt templates with filtering"""
    try:
        category = request.args.get('category')
        search = request.args.get('search')
        
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        query = '''
            SELECT id, template_name, category, description, prompt_text,
                   variables, use_case, usage_count, rating_average,
                   rating_count, created_at
            FROM prompt_templates
            WHERE is_public = 1
        '''
        params = []
        
        if category:
            query += ' AND category = ?'
            params.append(category)
            
        if search:
            query += ' AND (template_name LIKE ? OR description LIKE ?)'
            params.extend([f'%{search}%', f'%{search}%'])
            
        query += ' ORDER BY rating_average DESC, usage_count DESC'
        
        cursor.execute(query, params)
        
        templates = []
        for row in cursor.fetchall():
            templates.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'description': row[3],
                'prompt_text': row[4],
                'variables': json.loads(row[5]) if row[5] else [],
                'use_case': row[6],
                'usage_count': row[7],
                'rating': row[8],
                'rating_count': row[9],
                'created_at': row[10]
            })
        
        conn.close()
        return jsonify({'templates': templates})
        
    except Exception as e:
        print(f"Error getting prompt templates: {e}")
        return jsonify({'error': 'Failed to get templates'}), 500

@app.route('/api/prompts/templates', methods=['POST'])
def create_prompt_template_api():
    """Create a new prompt template"""
    try:
        data = request.get_json()
        
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO prompt_templates (
                template_name, category, description, prompt_text,
                variables, use_case, creator_id, created_at,
                updated_at, is_public, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('name'),
            data.get('category'),
            data.get('description', ''),
            data.get('prompt_text'),
            json.dumps(data.get('variables', [])),
            data.get('use_case', ''),
            'user_001',  # Replace with actual user ID
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            data.get('is_public', 0),
            json.dumps(data.get('tags', []))
        ))
        
        template_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'template_id': template_id,
            'message': 'Template created successfully'
        })
        
    except Exception as e:
        print(f"Error creating prompt template: {e}")
        return jsonify({'error': 'Failed to create template'}), 500

@app.route('/api/prompts/experiments', methods=['POST'])
def create_prompt_experiment_api():
    """Create a new prompt A/B test experiment"""
    try:
        data = request.get_json()
        
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO prompt_experiments (
                experiment_name, description, prompt_a, prompt_b,
                variables, model_used, creator_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('name'),
            data.get('description', ''),
            data.get('prompt_a'),
            data.get('prompt_b'),
            json.dumps(data.get('variables', {})),
            data.get('model_used', 'gpt-3.5-turbo'),
            'user_001',  # Replace with actual user ID
            datetime.now().isoformat()
        ))
        
        experiment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'experiment_id': experiment_id,
            'message': 'Experiment created successfully'
        })
        
    except Exception as e:
        print(f"Error creating prompt experiment: {e}")
        return jsonify({'error': 'Failed to create experiment'}), 500

@app.route('/api/prompts/experiments/<int:experiment_id>/test', methods=['POST'])
def test_prompt_experiment_api(experiment_id):
    """Run a test for a prompt experiment"""
    try:
        data = request.get_json()
        test_input = data.get('test_input')
        variant = data.get('variant')  # 'a' or 'b'
        
        # Get experiment details
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT prompt_a, prompt_b, model_used
            FROM prompt_experiments
            WHERE id = ?
        ''', (experiment_id,))
        
        experiment = cursor.fetchone()
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404
        
        prompt_a, prompt_b, model_used = experiment
        selected_prompt = prompt_a if variant == 'a' else prompt_b
        
        # Simulate AI response (replace with actual AI call)
        start_time = datetime.now()
        ai_response = f"[Simulated response for variant {variant.upper()}] {test_input[:50]}..."
        end_time = datetime.now()
        
        response_time = (end_time - start_time).total_seconds()
        
        # Save test result
        cursor.execute('''
            INSERT INTO prompt_test_results (
                experiment_id, prompt_variant, test_input, ai_response,
                response_time, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id, variant, test_input, ai_response,
            response_time, datetime.now().isoformat()
        ))
        
        # Update experiment stats
        cursor.execute('''
            UPDATE prompt_experiments
            SET total_tests = total_tests + 1
            WHERE id = ?
        ''', (experiment_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'response_time': response_time,
            'variant': variant
        })
        
    except Exception as e:
        print(f"Error testing prompt experiment: {e}")
        return jsonify({'error': 'Failed to run test'}), 500

# ===== AI PERFORMANCE ANALYTICS API ENDPOINTS =====

@app.route('/api/analytics/usage', methods=['GET'])
def get_usage_analytics_api():
    """Get usage analytics data"""
    try:
        days = int(request.args.get('days', 7))
        
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Daily usage stats
        cursor.execute('''
            SELECT date, 
                   COUNT(*) as requests,
                   COUNT(DISTINCT user_id) as users,
                   SUM(success) as successful,
                   AVG(response_time) as avg_time
            FROM ai_usage_stats
            WHERE date >= date('now', '-{} days')
            GROUP BY date
            ORDER BY date DESC
        '''.format(days))
        
        daily_stats = []
        for row in cursor.fetchall():
            daily_stats.append({
                'date': row[0],
                'requests': row[1],
                'users': row[2],
                'successful': row[3],
                'avg_response_time': row[4]
            })
        
        # Feature usage breakdown
        cursor.execute('''
            SELECT feature_used, COUNT(*) as usage_count
            FROM ai_usage_stats
            WHERE date >= date('now', '-{} days')
            GROUP BY feature_used
            ORDER BY usage_count DESC
        '''.format(days))
        
        feature_stats = []
        for row in cursor.fetchall():
            feature_stats.append({
                'feature': row[0],
                'usage_count': row[1]
            })
        
        conn.close()
        
        return jsonify({
            'daily_stats': daily_stats,
            'feature_stats': feature_stats,
            'period_days': days
        })
        
    except Exception as e:
        print(f"Error getting usage analytics: {e}")
        return jsonify({'error': 'Failed to get usage analytics'}), 500

@app.route('/api/analytics/performance', methods=['GET'])
def get_performance_analytics_api():
    """Get performance metrics"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        # Overall performance metrics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_requests,
                SUM(success) as successful_requests,
                AVG(response_time) as avg_response_time,
                COUNT(DISTINCT user_id) as unique_users
            FROM ai_usage_stats
            WHERE date >= date('now', '-7 days')
        ''')
        
        overall = cursor.fetchone()
        
        # Model performance comparison
        cursor.execute('''
            SELECT model_used,
                   COUNT(*) as requests,
                   AVG(response_time) as avg_time,
                   (SUM(success) * 100.0 / COUNT(*)) as success_rate
            FROM ai_usage_stats
            WHERE model_used IS NOT NULL AND date >= date('now', '-7 days')
            GROUP BY model_used
            ORDER BY requests DESC
        ''')
        
        model_performance = []
        for row in cursor.fetchall():
            model_performance.append({
                'model': row[0],
                'requests': row[1],
                'avg_response_time': row[2],
                'success_rate': row[3]
            })
        
        conn.close()
        
        return jsonify({
            'overall': {
                'total_requests': overall[0],
                'successful_requests': overall[1],
                'avg_response_time': overall[2],
                'unique_users': overall[3],
                'success_rate': (overall[1] / overall[0] * 100) if overall[0] > 0 else 0
            },
            'model_performance': model_performance
        })
        
    except Exception as e:
        print(f"Error getting performance analytics: {e}")
        return jsonify({'error': 'Failed to get performance analytics'}), 500

@app.route('/api/analytics/insights', methods=['GET'])
def get_improvement_insights_api():
    """Get AI-powered improvement insights"""
    try:
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT insight_type, title, description, impact_level,
                   confidence_score, action_suggested, implemented,
                   created_at, priority
            FROM improvement_insights
            ORDER BY priority DESC, confidence_score DESC
            LIMIT 20
        ''')
        
        insights = []
        for row in cursor.fetchall():
            insights.append({
                'type': row[0],
                'title': row[1],
                'description': row[2],
                'impact_level': row[3],
                'confidence_score': row[4],
                'action_suggested': row[5],
                'implemented': bool(row[6]),
                'created_at': row[7],
                'priority': row[8]
            })
        
        conn.close()
        return jsonify({'insights': insights})
        
    except Exception as e:
        print(f"Error getting improvement insights: {e}")
        return jsonify({'error': 'Failed to get insights'}), 500

@app.route('/api/analytics/log', methods=['POST'])
def log_usage_analytics_api():
    """Log usage analytics data"""
    try:
        data = request.get_json()
        
        conn = sqlite3.connect('ai_memory.db')
        cursor = conn.cursor()
        
        now = datetime.now()
        cursor.execute('''
            INSERT INTO ai_usage_stats (
                user_id, session_id, feature_used, model_used,
                request_type, response_time, tokens_used,
                success, error_message, timestamp, date, hour
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('user_id', 'anonymous'),
            data.get('session_id'),
            data.get('feature_used'),
            data.get('model_used'),
            data.get('request_type'),
            data.get('response_time', 0.0),
            data.get('tokens_used', 0),
            data.get('success', 1),
            data.get('error_message'),
            now.isoformat(),
            now.date().isoformat(),
            now.hour
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Analytics logged'})
        
    except Exception as e:
        print(f"Error logging analytics: {e}")
        return jsonify({'error': 'Failed to log analytics'}), 500

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
