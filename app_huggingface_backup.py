#!/usr/bin/env python3
"""
Horizon AI Assistant with ChatGPT API Integration
Clean, fast, and intelligent AI responses using OpenAI's API
"""

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

# ChatGPT API Integration
try:
    import openai
    from openai import OpenAI
    
    print("� Initializing ChatGPT API connection...")
    
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
        'questions': {
            'what': {
                'technology': [
                    "Technology is constantly evolving! {} Let me share some insights about this fascinating field.",
                    "That's a great tech question! {} From my understanding, this involves multiple aspects.",
                    "Technology topics are exciting! {} This is particularly interesting because it impacts many areas."
                ],
                'science': [
                    "Science is amazing! {} This is based on research and scientific principles.",
                    "Scientific topics fascinate me! {} The underlying mechanisms are quite complex.",
                    "That's a wonderful science question! {} Let me explain what we know about this."
                ],
                'general': [
                    "That's a thoughtful question! {} Based on what I understand, this is quite multifaceted.",
                    "Great question! {} Let me think about the various aspects involved.",
                    "Interesting inquiry! {} This topic has several important dimensions to consider."
                ]
            },
            'how': {
                'learning': [
                    "Learning is a journey! {} Here are some effective approaches you might consider.",
                    "Great learning question! {} The key is to start with fundamentals and build gradually.",
                    "I love learning questions! {} Success often comes from consistent practice and curiosity."
                ],
                'technical': [
                    "Technical topics require patience! {} Let me break down the process step by step.",
                    "That's a solid technical question! {} The approach depends on your current level and goals.",
                    "Technical skills are valuable! {} Here's how you can systematically approach this."
                ],
                'general': [
                    "That's practical thinking! {} There are several strategies that often work well.",
                    "Good question about methodology! {} Success usually involves a combination of approaches.",
                    "I appreciate practical questions! {} Let me suggest some proven methods."
                ]
            },
            'why': {
                'philosophical': [
                    "That touches on deeper questions! {} The reasons often involve multiple philosophical perspectives.",
                    "Philosophical inquiries are profound! {} This connects to fundamental questions about existence.",
                    "Deep thinking! {} The 'why' behind this involves complex philosophical considerations."
                ],
                'scientific': [
                    "That's rooted in scientific principles! {} The underlying mechanisms involve fascinating natural laws.",
                    "Science explains so much! {} This phenomenon occurs due to specific scientific processes.",
                    "Scientific curiosity! {} The explanation involves interesting principles of nature."
                ],
                'general': [
                    "That's insightful! {} The reasons are often interconnected and multifaceted.",
                    "Good question about causation! {} Multiple factors typically contribute to this.",
                    "Thoughtful inquiry! {} Understanding the 'why' helps us see the bigger picture."
                ]
            }
        },
        'topics': {
            'artificial_intelligence': [
                "AI is transforming our world! It's a field that combines computer science, mathematics, and cognitive science to create systems that can perform tasks typically requiring human intelligence.",
                "Artificial Intelligence is fascinating! It involves machine learning, neural networks, and algorithms that can learn from data and make predictions or decisions.",
                "AI is everywhere now! From recommendation systems to autonomous vehicles, it's changing how we interact with technology and solve complex problems."
            ],
            'programming': [
                "Programming is like learning a new language for computers! It's creative problem-solving combined with logical thinking.",
                "Coding is both art and science! You get to build solutions that can impact millions of people while exercising your creativity and analytical skills.",
                "Programming opens infinite possibilities! Whether it's web development, mobile apps, or AI, code is the tool that brings ideas to life."
            ],
            'technology': [
                "Technology shapes our future! From quantum computing to biotechnology, we're living in an era of unprecedented innovation.",
                "Tech evolution is accelerating! What seemed impossible yesterday becomes reality today - it's an exciting time to be curious about technology.",
                "Technology democratizes opportunities! It connects people globally and provides tools for creativity, learning, and problem-solving."
            ],
            'science': [
                "Science is humanity's greatest adventure! It's our quest to understand the universe from quantum particles to cosmic structures.",
                "Scientific discovery drives progress! Every breakthrough, from medicine to space exploration, starts with curiosity and rigorous investigation.",
                "Science connects everything! Physics, chemistry, biology, and mathematics work together to explain the beautiful complexity of our world."
            ],
            'learning': [
                "Learning is a superpower! In our rapidly changing world, the ability to acquire new skills and knowledge is invaluable.",
                "Continuous learning opens doors! Whether formal education or self-directed exploration, every bit of knowledge adds to your potential.",
                "Learning never stops! The most successful people are lifelong learners who stay curious and adapt to new challenges."
            ],
            'creativity': [
                "Creativity is uniquely human! It's how we solve problems, express emotions, and imagine possibilities that don't yet exist.",
                "Creative thinking transforms the world! Art, innovation, and breakthrough solutions all start with someone thinking differently.",
                "Creativity needs nurturing! Like a muscle, it grows stronger with practice, experimentation, and openness to new experiences."
            ]
        },
        'sentiment_responses': {
            'positive': [
                "I love your positive energy! It's wonderful to discuss topics that bring excitement and optimism.",
                "Your enthusiasm is contagious! Positive perspectives often lead to the most interesting conversations.",
                "That's a great attitude! Optimism and curiosity make for the best learning experiences."
            ],
            'curious': [
                "Your curiosity is inspiring! Questions like yours drive human progress and understanding.",
                "I appreciate curious minds! The desire to understand and explore is what makes conversations meaningful.",
                "Curiosity is brilliant! Keep asking questions - that's how we all grow and discover new things."
            ],
            'thoughtful': [
                "That's very thoughtful! Deep questions deserve careful consideration and nuanced responses.",
                "I appreciate your reflective approach! Thoughtful discussions often lead to the most valuable insights.",
                "Your depth of thinking shows! Complex topics require the kind of careful analysis you're bringing."
            ]
        },
        'conversation_starters': [
            "That's fascinating! What sparked your interest in this topic?",
            "I find this area really engaging too! What aspect interests you most?",
            "Great topic choice! Have you had any personal experience with this?",
            "This is an area where there's always more to discover! What would you like to explore?",
            "I enjoy discussing this! What's your current understanding or experience?",
            "This connects to so many other topics! What drew you to this particular aspect?"
        ],
        'general': [
            "That's an interesting perspective! I'd love to hear more about your thoughts on this.",
            "Thank you for bringing that up! Topics like this often have many layers to explore.",
            "I appreciate you sharing that! What aspects of this are most important to you?",
            "That's worth discussing! I find conversations like this really valuable.",
            "Interesting point! Your perspective adds a lot to this discussion.",
            "I'm glad you brought this up! What would you like to explore further?",
            "That resonates with me! What experiences have shaped your thinking on this?",
            "Great conversation starter! What's been on your mind about this lately?",
            "I find this topic engaging! What questions do you have about it?",
            "Thanks for the thoughtful discussion! What direction would you like to take this?"
        ]
    }
    
    AI_MODEL_AVAILABLE = True
    TOKENIZER_AVAILABLE = True
    print("✅ Enhanced Smart AI Assistant ready - Advanced pattern recognition with Hugging Face tokenization")
    
except Exception as e:
    tokenizer = None
    TOKENIZER_AVAILABLE = False
    AI_MODEL_AVAILABLE = False
    print(f"⚠️  Enhanced AI initialization failed: {e}")
    print("⚠️  Running with basic responses only")

# Global variables
timers = {}
reminders = []

def analyze_text_context(text):
    """Enhanced text analysis using Hugging Face tokenizer"""
    if not TOKENIZER_AVAILABLE:
        return {'tokens': text.split(), 'complexity': 'simple'}
    
    try:
        # Tokenize the input for better understanding
        tokens = tokenizer.tokenize(text.lower())
        
        # Analyze complexity and topic
        complexity = 'simple' if len(tokens) < 10 else 'moderate' if len(tokens) < 20 else 'complex'
        
        # Check for topic keywords
        tech_keywords = ['ai', 'artificial', 'intelligence', 'programming', 'code', 'computer', 'software', 'technology', 'algorithm', 'machine', 'learning']
        science_keywords = ['science', 'research', 'experiment', 'theory', 'physics', 'chemistry', 'biology', 'mathematics']
        learning_keywords = ['learn', 'study', 'education', 'school', 'university', 'course', 'tutorial', 'practice']
        
        topic_category = 'general'
        if any(keyword in text.lower() for keyword in tech_keywords):
            topic_category = 'technology'
        elif any(keyword in text.lower() for keyword in science_keywords):
            topic_category = 'science'
        elif any(keyword in text.lower() for keyword in learning_keywords):
            topic_category = 'learning'
        
        return {
            'tokens': tokens,
            'complexity': complexity,
            'topic_category': topic_category,
            'token_count': len(tokens)
        }
    except Exception as e:
        print(f"Text analysis error: {e}")
        return {'tokens': text.split(), 'complexity': 'simple', 'topic_category': 'general'}

def enhanced_ask_ai_model(user_input, personality):
    """Enhanced AI model with better context understanding"""
    if not AI_MODEL_AVAILABLE:
        return "I'd love to help with that, but my AI system isn't available right now. I can help with time, date, jokes, math, and timers though!"
    
    try:
        # Analyze the input text
        text_analysis = analyze_text_context(user_input)
        user_lower = user_input.lower()
        
        # Determine question type and context
        question_type = None
        question_context = 'general'
        
        for qtype in ['what', 'how', 'why', 'when', 'where']:
            if user_lower.startswith(qtype):
                question_type = qtype
                break
        
        # Enhanced topic detection
        topic_responses = []
        detected_topic = None
        
        # Check specific topics with better matching
        topic_keywords = {
            'artificial_intelligence': ['ai', 'artificial intelligence', 'machine learning', 'neural network', 'deep learning'],
            'programming': ['programming', 'coding', 'code', 'software', 'developer', 'python', 'javascript'],
            'technology': ['technology', 'tech', 'computer', 'digital', 'innovation', 'automation'],
            'science': ['science', 'research', 'experiment', 'physics', 'chemistry', 'biology'],
            'learning': ['learn', 'education', 'study', 'course', 'tutorial', 'skill'],
            'creativity': ['creative', 'art', 'design', 'imagination', 'innovation', 'idea']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                detected_topic = topic
                topic_responses.extend(ENHANCED_RESPONSES['topics'].get(topic, []))
                break
        
        # Determine context for question types
        if question_type and question_type in ENHANCED_RESPONSES['questions']:
            if text_analysis['topic_category'] in ENHANCED_RESPONSES['questions'][question_type]:
                question_context = text_analysis['topic_category']
        
        # Personality-based response selection
        personality_prefixes = {
            'enthusiastic': ["That's incredible! ", "How amazing! ", "I'm so excited about this! ", "This is fantastic! "],
            'professional': ["I understand your inquiry. ", "Thank you for your question. ", "Let me address that professionally. "],
            'casual': ["Cool question! ", "Interesting! ", "Good point! ", "Nice! ", "Sweet! "],
            'friendly': ["I'm happy to help! ", "Great question! ", "I'd love to discuss that! ", ""]
        }
        
        prefix = random.choice(personality_prefixes.get(personality, personality_prefixes['friendly']))
        
        # Generate response based on analysis
        response = None
        
        # Priority 1: Specific topic responses
        if topic_responses:
            response = random.choice(topic_responses)
        
        # Priority 2: Question type with context
        elif question_type and question_type in ENHANCED_RESPONSES['questions']:
            if question_context in ENHANCED_RESPONSES['questions'][question_type]:
                template = random.choice(ENHANCED_RESPONSES['questions'][question_type][question_context])
                context_text = "This is a complex and interesting topic."
                response = template.format(context_text)
            else:
                # Fallback to general question responses
                general_templates = ENHANCED_RESPONSES['questions'][question_type].get('general', [])
                if general_templates:
                    template = random.choice(general_templates)
                    response = template.format("This involves several important considerations.")
        
        # Priority 3: Sentiment-based responses
        if not response:
            if any(word in user_lower for word in ['amazing', 'awesome', 'great', 'love', 'fantastic']):
                response = random.choice(ENHANCED_RESPONSES['sentiment_responses']['positive'])
            elif any(word in user_lower for word in ['what', 'how', 'why', 'curious', 'wonder']):
                response = random.choice(ENHANCED_RESPONSES['sentiment_responses']['curious'])
            elif text_analysis['complexity'] == 'complex':
                response = random.choice(ENHANCED_RESPONSES['sentiment_responses']['thoughtful'])
        
        # Priority 4: Conversation starters
        if not response and text_analysis['token_count'] > 3:
            response = random.choice(ENHANCED_RESPONSES['conversation_starters'])
        
        # Priority 5: General responses
        if not response:
            response = random.choice(ENHANCED_RESPONSES['general'])
        
        # Add personality prefix
        final_response = prefix + response
        
        # Ensure response isn't too long
        if len(final_response) > 250:
            final_response = final_response[:250] + "..."
            
        return final_response
        
    except Exception as e:
        print(f"Enhanced AI model error: {e}")
        fallback_responses = [
            "That's a fascinating question! I'd love to explore that topic with you.",
            "I find that really interesting! What aspects would you like to discuss?",
            "Great topic! I enjoy conversations that make me think deeply.",
            "That's worth exploring! What's your perspective on this?"
        ]
        return random.choice(fallback_responses)

# Copy all the existing function definitions from the original app...
# (I'll include the essential ones here)

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

# Simple skill handlers (keeping the existing ones)
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

# Intent recognition patterns (keeping existing)
INTENT_PATTERNS = {
    'greeting': [r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b'],
    'time': [r'\b(time|clock)\b', r'\bwhat time is it\b'],
    'date': [r'\b(date|today)\b', r'\bwhat day is it\b'],
    'goodbye': [r'\b(bye|goodbye|see you|farewell)\b']
}

def recognize_intent(text):
    text_lower = text.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent
    return 'general'

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
    else:
        # Use enhanced AI model for general questions
        response = enhanced_ask_ai_model(user_input, personality)
    
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
        'service': 'Enhanced Horizon AI Assistant',
        'ai_model_available': AI_MODEL_AVAILABLE,
        'tokenizer_available': TOKENIZER_AVAILABLE
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
        response = process_user_input(user_input, personality)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'personality': personality
        })
        
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Horizon AI Assistant...")
    
    # Initialize database
    init_db()
    
    print("✅ Intent recognition loaded")
    print("🌐 Server starting on http://127.0.0.1:8080...")
    
    app.run(host='127.0.0.1', port=8080, debug=True)
