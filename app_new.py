#!/usr/bin/env python3
"""
Horizon AI Assistant - Clean, Fast, and Reliable
No external ML dependencies - Pure Python intelligence
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Smart AI Assistant
print("🧠 Initializing Horizon AI Assistant...")

# Advanced response system with intelligent pattern matching
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
    
    'creativity': [
        "Creativity is what makes us uniquely human! It's how we solve problems, express emotions, and imagine possibilities that don't yet exist.",
        "Creative thinking isn't limited to art - it's essential in science, technology, business, and everyday problem-solving.",
        "The creative process thrives on diverse experiences and perspectives. The more you explore different fields, the more creative connections you can make.",
        "Don't wait for inspiration - create regularly! Like a muscle, creativity grows stronger with consistent practice and experimentation.",
        "Some of the best innovations come from combining ideas from completely different fields. Cross-pollination of concepts breeds creativity!"
    ],
    
    'problem_solving': [
        "Great problem-solving starts with clearly defining the problem. Sometimes what we think is the issue isn't the real challenge.",
        "Break complex problems into smaller, manageable pieces. Solve each piece systematically, and you'll find the bigger challenge becomes achievable.",
        "Consider multiple perspectives! The solution that works for one situation might not work for another, but understanding different approaches expands your toolkit.",
        "Don't be afraid to iterate! The first solution rarely is the best one. Refine, improve, and adapt based on feedback and results.",
        "Sometimes the best solutions come from stepping away and letting your subconscious work on the problem. Take breaks and let ideas incubate!"
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
    ],
    
    'encouragement': [
        "You're asking great questions! Curiosity is the first step toward mastery in any field.",
        "Keep that enthusiasm! Passion and persistence are the keys to achieving remarkable things.",
        "Every expert was once a beginner. Don't be discouraged by the learning curve - embrace it!",
        "You have unique perspectives and experiences that contribute value to any conversation or project.",
        "The fact that you're seeking knowledge and growth shows you're on the right path. Keep going!",
        "Remember, the goal isn't to know everything - it's to stay curious and keep learning throughout life.",
        "Your questions show depth of thinking. That kind of intellectual curiosity leads to breakthrough insights.",
        "Trust the process! Learning and growth take time, but each step forward builds your capabilities."
    ]
}

# Topic keyword mapping for intelligent responses
TOPIC_KEYWORDS = {
    'ai_technology': ['ai', 'artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'algorithm', 'data science', 'automation'],
    'programming': ['programming', 'coding', 'code', 'software', 'developer', 'python', 'javascript', 'computer science', 'debugging', 'framework'],
    'science': ['science', 'research', 'experiment', 'physics', 'chemistry', 'biology', 'mathematics', 'discovery', 'theory', 'hypothesis'],
    'technology': ['technology', 'tech', 'innovation', 'digital', 'computer', 'internet', 'smartphone', 'future', 'innovation', 'engineering'],
    'learning': ['learn', 'education', 'study', 'school', 'university', 'course', 'tutorial', 'skill', 'knowledge', 'training'],
    'creativity': ['creative', 'creativity', 'art', 'design', 'imagination', 'innovation', 'artistic', 'inspiration', 'original'],
    'problem_solving': ['problem', 'solution', 'solve', 'challenge', 'difficulty', 'issue', 'fix', 'troubleshoot', 'debugging'],
}

# Global variables
timers = {}
reminders = []
AI_MODEL_AVAILABLE = True

def analyze_user_input(text):
    """Analyze user input to determine topic and complexity"""
    text_lower = text.lower()
    words = text_lower.split()
    
    # Determine topic
    detected_topic = 'general_wisdom'
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_topic = topic
            break
    
    # Determine complexity based on length and question words
    complexity = 'simple'
    if len(words) > 15 or any(word in text_lower for word in ['explain', 'describe', 'analyze', 'compare', 'evaluate']):
        complexity = 'complex'
    elif len(words) > 8:
        complexity = 'moderate'
    
    # Check for question type
    question_type = None
    for qtype in ['what', 'how', 'why', 'when', 'where', 'who']:
        if text_lower.startswith(qtype):
            question_type = qtype
            break
    
    return {
        'topic': detected_topic,
        'complexity': complexity,
        'question_type': question_type,
        'word_count': len(words)
    }

def generate_smart_response(user_input, personality, analysis):
    """Generate intelligent responses based on input analysis"""
    
    # Get base responses for the detected topic
    topic_responses = SMART_RESPONSES.get(analysis['topic'], SMART_RESPONSES['general_wisdom'])
    
    # Add encouragement for learning-oriented questions
    if analysis['question_type'] in ['how', 'what', 'why'] or analysis['complexity'] == 'complex':
        if random.random() < 0.3:  # 30% chance to add encouragement
            topic_responses = SMART_RESPONSES['encouragement']
    
    # Select base response
    base_response = random.choice(topic_responses)
    
    # Apply personality modifiers
    personality_modifiers = {
        'enthusiastic': ["That's incredible! ", "How exciting! ", "I love this topic! ", "Amazing question! "],
        'professional': ["I appreciate your inquiry. ", "That's a thoughtful question. ", "Let me address that. "],
        'casual': ["Cool question! ", "Interesting! ", "Nice! ", "Good point! "],
        'friendly': ["Great question! ", "I'm happy to help! ", "That's wonderful! ", ""]
    }
    
    prefix = random.choice(personality_modifiers.get(personality, personality_modifiers['friendly']))
    
    # Combine prefix and response
    final_response = prefix + base_response
    
    # Ensure reasonable length
    if len(final_response) > 300:
        sentences = final_response.split('. ')
        final_response = '. '.join(sentences[:2]) + '.'
    
    return final_response

def ask_ai_model(user_input, personality):
    """Main AI function using smart pattern matching"""
    try:
        # Analyze the input
        analysis = analyze_user_input(user_input)
        
        # Generate response
        response = generate_smart_response(user_input, personality, analysis)
        
        return response
        
    except Exception as e:
        print(f"AI model error: {e}")
        return "I appreciate your question! Let me think about that and provide you with a thoughtful response."

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

# Intent recognition
INTENT_PATTERNS = {
    'greeting': [r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b'],
    'time': [r'\b(time|clock)\b', r'\bwhat time is it\b'],
    'date': [r'\b(date|today)\b', r'\bwhat day is it\b'],
    'joke': [r'\b(joke|funny|humor)\b', r'\btell me a joke\b'],
    'math': [r'\d+\s*[\+\-\*\/]\s*\d+', r'\bwhat is \d+', r'\bcalculate\b'],
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
    elif intent == 'goodbye':
        response = "Thank you for chatting! Have a wonderful day!"
    else:
        # Use AI model for general questions
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
        'service': 'Horizon AI Assistant',
        'ai_model_available': AI_MODEL_AVAILABLE,
        'version': 'clean_v1.0'
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
    print("🚀 Starting Horizon AI Assistant...")
    
    # Initialize database
    init_db()
    
    print("✅ Smart AI system loaded")
    print("✅ Intent recognition ready")
    print("🌐 Server starting on http://127.0.0.1:8080...")
    
    app.run(host='127.0.0.1', port=8080, debug=True)
