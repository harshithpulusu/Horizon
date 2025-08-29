from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from datetime import datetime, timedelta
import random
import re
import os
from dataclasses import dataclass
from typing import Dict, List, Any
import sqlite3
import threading
import time
import requests
import spacy
from spacy.matcher import Matcher
import numpy as np
import openai

app = Flask(__name__)
CORS(app)

@dataclass
class Intent:
    name: str
    confidence: float
    entities: Dict[str, Any]

class AdvancedAIProcessor:
    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {}
        self.context_memory = {}
        self.skills = self.init_skills()
        self.intent_patterns = self.init_intent_patterns()
        self.init_database()
        
        # Initialize spaCy for advanced NLP
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.matcher = Matcher(self.nlp.vocab)
            self.init_spacy_patterns()
            print("✅ spaCy model loaded successfully!")
        except OSError:
            print("❌ spaCy model not found. Using fallback basic NLP.")
            self.nlp = None
            self.matcher = None
        self.active_timers = {}
        self.reminders = []
        
    def init_database(self):
        """Initialize SQLite database for persistent memory"""
        self.conn = sqlite3.connect('ai_memory.db', check_same_thread=False)
        self.lock = threading.Lock()
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    user_input TEXT,
                    ai_response TEXT,
                    intent TEXT,
                    sentiment TEXT,
                    confidence REAL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            self.conn.commit()
    
    def init_skills(self):
        """Initialize AI skills similar to Alexa Skills"""
        return {
            'time': self.get_time,
            'date': self.get_date,
            'math': self.calculate_math,
            'reminder': self.set_reminder,
            'joke': self.tell_joke,
            'news': self.get_news,
            'music': self.play_music,
            'smart_home': self.control_smart_home,
            'translation': self.translate_text,
            'definition': self.define_word,
            'trivia': self.ask_trivia,
            'timer': self.set_timer,
            'alarm': self.set_alarm,
            'calendar': self.check_calendar,
            'email': self.send_email,
            'search': self.web_search
        }
    
    def init_intent_patterns(self):
        """Initialize intent recognition patterns"""
        return {
            'time': [
                r'what time is it',
                r'current time',
                r'time right now',
                r'tell me the time'
            ],
            'date': [
                r'what\'?s today\'?s date',
                r'what day is it',
                r'today\'?s date'
            ],
            'math': [
                r'calculate (.+)',
                r'what\'?s (\d+) (\+|\-|\*|/) (\d+)',
                r'solve (.+)',
                r'math (.+)'
            ],
            'reminder': [
                r'remind me to (.+)',
                r'set a reminder (.+)',
                r'don\'?t forget (.+)'
            ],
            'timer': [
                r'set timer for (.+)',
                r'timer for (.+)',
                r'start timer (.+)',
                r'set a timer for (.+)',
                r'countdown (.+)',
                r'timer (.+) minutes',
                r'timer (.+) seconds'
            ],
            'joke': [
                r'tell me a joke',
                r'make me laugh',
                r'something funny',
                r'another joke'
            ],
            'news': [
                r'what\'?s in the news',
                r'latest news',
                r'news about (.+)'
            ],
            'music': [
                r'play music',
                r'play some music',
                r'music by (.+)',
                r'play song (.+)',
                r'put on (.+) music'
            ],
            'smart_home': [
                r'turn (on|off) (.+)',
                r'dim the lights',
                r'set temperature to (\d+)'
            ],
            'translation': [
                r'translate (.+) to (.+)',
                r'how do you say (.+) in (.+)'
            ],
            'definition': [
                r'what does (.+) mean',
                r'define (.+)',
                r'definition of (.+)'
            ],
            'search': [
                r'search for (.+)',
                r'look up (.+)',
                r'find information about (.+)'
            ]
        }
    
    def init_spacy_patterns(self):
        """Initialize spaCy patterns for better intent recognition"""
        if not self.matcher:
            return
        
        # Time patterns
        time_patterns = [
            [{"LOWER": {"IN": ["what", "whats"]}}, {"LOWER": "time"}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": "current"}, {"LOWER": "time"}],
            [{"LOWER": "tell"}, {"LOWER": "me"}, {"LOWER": "the"}, {"LOWER": "time"}]
        ]
        self.matcher.add("TIME_INTENT", time_patterns)
        
        # Timer patterns - enhanced for better recognition
        timer_patterns = [
            [{"LOWER": "set"}, {"LOWER": {"IN": ["timer", "alarm"]}}, {"LOWER": "for", "OP": "?"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["minutes", "minute", "mins", "min", "seconds", "second", "secs", "sec", "hours", "hour", "hrs", "hr"]}}],
            [{"LOWER": "timer"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["minutes", "minute", "mins", "min", "seconds", "second", "secs", "sec", "hours", "hour", "hrs", "hr"]}}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["minutes", "minute", "mins", "min", "seconds", "second", "secs", "sec", "hours", "hour", "hrs", "hr"]}}, {"LOWER": "timer"}],
            [{"LOWER": "countdown"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["minutes", "minute", "mins", "min"]}}]
        ]
        self.matcher.add("TIMER_INTENT", timer_patterns)
        
        # Math patterns - enhanced
        math_patterns = [
            [{"LOWER": {"IN": ["calculate", "compute", "solve"]}}, {"IS_ALPHA": False, "OP": "+"}],
            [{"LOWER": "what"}, {"LOWER": "is"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["plus", "minus", "times", "divided", "multiplied"]}}, {"LIKE_NUM": True}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["+", "-", "*", "/", "plus", "minus", "times", "divided", "by", "multiplied"]}}, {"LIKE_NUM": True}],
            [{"LOWER": "math"}, {"IS_ALPHA": False, "OP": "+"}]
        ]
        self.matcher.add("MATH_INTENT", math_patterns)
        
        # Reminder patterns - enhanced
        reminder_patterns = [
            [{"LOWER": "remind"}, {"LOWER": "me"}, {"LOWER": "to", "OP": "?"}, {"IS_ALPHA": True, "OP": "+"}],
            [{"LOWER": "set"}, {"LOWER": {"IN": ["reminder", "alert"]}}, {"IS_ALPHA": True, "OP": "+"}],
            [{"LOWER": "remember"}, {"LOWER": "to"}, {"IS_ALPHA": True, "OP": "+"}],
            [{"LOWER": "don't"}, {"LOWER": "forget"}, {"IS_ALPHA": True, "OP": "+"}]
        ]
        self.matcher.add("REMINDER_INTENT", reminder_patterns)
        
        # Joke patterns
        joke_patterns = [
            [{"LOWER": "tell"}, {"LOWER": "me"}, {"LOWER": "a"}, {"LOWER": "joke"}],
            [{"LOWER": {"IN": ["joke", "funny", "laugh"]}}, {"IS_ALPHA": True, "OP": "*"}],
            [{"LOWER": "make"}, {"LOWER": "me"}, {"LOWER": "laugh"}],
            [{"LOWER": "something"}, {"LOWER": "funny"}]
        ]
        self.matcher.add("JOKE_INTENT", joke_patterns)
        
        # Date patterns
        date_patterns = [
            [{"LOWER": {"IN": ["what", "whats"]}}, {"LOWER": {"IN": ["date", "day"]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": "today's"}, {"LOWER": "date"}],
            [{"LOWER": "what"}, {"LOWER": "day"}, {"LOWER": "is"}, {"LOWER": "it"}]
        ]
        self.matcher.add("DATE_INTENT", date_patterns)
        
        # Count total patterns
        total_patterns = sum(len(patterns) for patterns in [
            time_patterns, timer_patterns, math_patterns, 
            reminder_patterns, joke_patterns, date_patterns
        ])
        print(f"✅ Initialized {total_patterns} spaCy patterns for enhanced NLP")

    def clean_wake_words(self, user_input: str) -> str:
        """Remove wake words from user input to avoid confusion with commands"""
        # Define wake words that should be removed from commands
        wake_words = [
            'hey horizon', 'horizon', 'hey assistant', 'assistant',
            'hey siri', 'siri', 'hey alexa', 'alexa'  # Common wake words
        ]
        
        cleaned_input = user_input.lower().strip()
        
        # Remove wake words from the beginning of the input
        for wake_word in wake_words:
            if cleaned_input.startswith(wake_word):
                cleaned_input = cleaned_input[len(wake_word):].strip()
                # Remove common connecting words
                if cleaned_input.startswith((',', 'please', 'can you')):
                    cleaned_input = re.sub(r'^(,|please|can you)\s*', '', cleaned_input)
                break
        
        return cleaned_input if cleaned_input else user_input
    
    def recognize_intent(self, user_input: str) -> Intent:
        """Enhanced intent recognition using spaCy and pattern matching"""
        # Clean wake words first
        cleaned_input = self.clean_wake_words(user_input)
        user_input_lower = cleaned_input.lower().strip()
        best_intent = Intent("general", 0.0, {})
        
        print(f"🧠 NLP Analysis for: '{user_input_lower}'")
        
        # First try spaCy-based recognition if available
        if self.nlp and self.matcher:
            spacy_intent = self._recognize_with_spacy(cleaned_input)
            if spacy_intent.confidence > 0.8:
                print(f"✅ spaCy high-confidence match: {spacy_intent.name} (confidence: {spacy_intent.confidence:.3f})")
                return spacy_intent
            elif spacy_intent.confidence > best_intent.confidence:
                best_intent = spacy_intent
                print(f"📊 spaCy match: {spacy_intent.name} (confidence: {spacy_intent.confidence:.3f})")
        
        # Enhanced pattern matching with better scoring
        print(f"🔍 Pattern matching for: '{user_input_lower}'")
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, user_input_lower)
                if match:
                    confidence = len(match.group(0)) / len(user_input_lower)
                    confidence += 0.3  # Boost for exact pattern match
                    
                    print(f"Pattern match: '{pattern}' -> {intent_name} (confidence: {confidence:.3f})")
                    
                    if confidence > best_intent.confidence:
                        entities = {}
                        if match.groups():
                            entities = {f"entity_{i}": group for i, group in enumerate(match.groups())}
                        
                        best_intent = Intent(intent_name, confidence, entities)
                        print(f"New best intent: {intent_name} with entities: {entities}")
        
        # Fallback intent detection using keywords
        if best_intent.confidence < 0.3:
            print("Using keyword fallback detection...")
            keyword_intents = {
                'time': ['time', 'clock', 'hour', 'minute'],
                'math': ['calculate', 'plus', 'minus', 'times', 'divided', 'equation', 'add', 'subtract', 'multiply'],
                'music': ['play', 'song', 'music', 'artist', 'album'],
                'joke': ['joke', 'funny', 'laugh', 'humor'],
                'timer': ['timer', 'countdown', 'minutes', 'seconds', 'alarm'],
                'reminder': ['remind', 'reminder', 'remember', 'forget']
            }
            
            for intent_name, keywords in keyword_intents.items():
                keyword_matches = sum(1 for keyword in keywords if keyword in user_input_lower)
                if keyword_matches > 0:
                    confidence = keyword_matches / len(keywords) * 0.5
                    print(f"Keyword match: {intent_name} ({keyword_matches} keywords, confidence: {confidence:.3f})")
                    if confidence > best_intent.confidence:
                        best_intent = Intent(intent_name, confidence, {})
        
        return best_intent
    
    def _recognize_with_spacy(self, text: str) -> Intent:
        """Use spaCy for advanced intent recognition with entity extraction"""
        if not self.nlp or not self.matcher:
            return Intent("general", 0.0, {})
        
        # Process text with spaCy
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        if not matches:
            return Intent("general", 0.0, {})
        
        # Find the best match with highest confidence
        best_match = None
        best_confidence = 0.0
        best_entities = {}
        
        for match_id, start, end in matches:
            intent_name = self.nlp.vocab.strings[match_id].replace("_INTENT", "").lower()
            
            # Calculate confidence based on match span and text length
            match_span = end - start
            text_length = len(doc)
            confidence = (match_span / text_length) * 1.2  # Boost spaCy matches
            
            # Extract entities from the matched span and surrounding context
            entities = self._extract_entities_spacy(doc, start, end)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = intent_name
                best_entities = entities
        
        # Cap confidence at 1.0
        best_confidence = min(best_confidence, 1.0)
        
        return Intent(best_match or "general", best_confidence, best_entities)
    
    def _extract_entities_spacy(self, doc, start: int, end: int) -> Dict[str, Any]:
        """Extract entities using spaCy's NER and linguistic features"""
        entities = {}
        entity_count = 0
        
        # Look for numbers (for timers, math, etc.)
        for token in doc:
            if token.like_num and entity_count < 5:
                entities[f'number_{entity_count}'] = token.text
                entity_count += 1
        
        # Look for time expressions
        for ent in doc.ents:
            if ent.label_ in ["TIME", "DATE", "CARDINAL", "ORDINAL"] and entity_count < 5:
                entities[f'entity_{entity_count}'] = ent.text
                entity_count += 1
        
        # Extract duration units (minutes, hours, etc.)
        time_units = ["minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs", "second", "seconds", "sec", "secs"]
        for token in doc:
            if token.lemma_.lower() in time_units and entity_count < 5:
                entities[f'time_unit_{entity_count}'] = token.text
                entity_count += 1
        
        # For reminders, extract the task
        if any(token.lemma_.lower() in ["remind", "remember"] for token in doc):
            # Find text after "to" or "about"
            for i, token in enumerate(doc):
                if token.lemma_.lower() in ["to", "about"] and i < len(doc) - 1:
                    task_text = " ".join([t.text for t in doc[i+1:]])
                    entities['task'] = task_text
                    break
        
        return entities

    def generate_response(self, user_input: str, personality: str = 'friendly') -> Dict[str, Any]:
        """Main response generation with context awareness"""
        # Clean wake words first for better intent recognition
        cleaned_input = self.clean_wake_words(user_input)
        print(f"Original input: '{user_input}' -> Cleaned: '{cleaned_input}'")  # Debug log
        
        intent = self.recognize_intent(cleaned_input)
        print(f"Recognized intent: {intent.name} (confidence: {intent.confidence:.3f})")  # Debug log
        
        sentiment = self.analyze_sentiment(cleaned_input)
        
        # Store conversation in database (use original input for history)
        self.store_conversation(user_input, intent, sentiment)
        
        # Update context memory
        self.update_context(cleaned_input, intent)
        
        # Generate response based on intent
        if intent.name in self.skills and intent.confidence > 0.3:
            print(f"Using skill: {intent.name}")  # Debug log
            response = self.skills[intent.name](cleaned_input, intent.entities, personality)
        else:
            print(f"Using OpenAI fallback (intent: {intent.name}, confidence: {intent.confidence:.3f})")  # Debug log
            response = self.ask_openai(cleaned_input, personality)
        
        # Store AI response
        self.store_ai_response(response, intent)
        
        return {
            'response': response,
            'intent': intent.name,
            'confidence': intent.confidence,
            'sentiment_analysis': sentiment,
            'conversation_count': len(self.conversation_history),
            'personality': personality,
            'context_aware': self.get_context_info()
        }
    
    # OpenAI Integration
    def ask_openai(self, user_input: str, personality: str) -> str:
        """Use OpenAI to answer questions that built-in skills can't handle"""
        try:
            # Set up OpenAI API key
            openai.api_key = os.getenv('OPENAI_API_KEY') or "your-openai-api-key-here"
            
            if openai.api_key == "your-openai-api-key-here":
                return "I'd love to help you with that! To unlock my full knowledge capabilities, please add your OpenAI API key to the environment variables. For now, I can help with time, timers, math, and jokes! 🤖"
            
            # Create personality-aware system prompt
            personality_prompts = {
                'friendly': "You are a friendly and helpful AI assistant named Horizon. Be warm, encouraging, and conversational in your responses.",
                'professional': "You are a professional AI assistant named Horizon. Provide clear, concise, and helpful information in a business-appropriate tone.",
                'enthusiastic': "You are an enthusiastic and energetic AI assistant named Horizon. Be exciting, positive, and use lots of exclamation points!",
                'witty': "You are a witty and clever AI assistant named Horizon. Include humor and wordplay in your responses while being helpful.",
                'sarcastic': "You are a sarcastic but helpful AI assistant named Horizon. Use dry humor and sarcasm while still providing good information.",
                'zen': "You are a zen and peaceful AI assistant named Horizon. Speak with wisdom, calmness, and mindfulness.",
                'scientist': "You are a scientific and analytical AI assistant named Horizon. Provide detailed, evidence-based responses with technical accuracy.",
                'pirate': "You are a pirate-themed AI assistant named Horizon. Use pirate language and expressions while helping users.",
                'shakespearean': "You are an AI assistant named Horizon who speaks like Shakespeare. Use eloquent, poetic language from the Elizabethan era.",
                'valley_girl': "You are a valley girl AI assistant named Horizon. Use valley girl speech patterns and expressions.",
                'cowboy': "You are a cowboy AI assistant named Horizon. Use western expressions and a frontier spirit.",
                'robot': "You are a robot AI assistant named Horizon. Speak in a mechanical, computational manner."
            }
            
            system_prompt = personality_prompts.get(personality, personality_prompts['friendly'])
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"I'm having trouble accessing my advanced knowledge base right now. But I can still help with time, timers, math, jokes, and reminders! Try asking me something like 'What time is it?' or 'Tell me a joke!' 🤖"

    # Rest of the methods remain the same...
    def store_conversation(self, user_input: str, intent: Intent, sentiment: Dict):
        """Store conversation in database for learning"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (timestamp, user_input, intent, sentiment, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                user_input,
                intent.name,
                json.dumps(sentiment),
                intent.confidence
            ))
            self.conn.commit()
    
    def store_ai_response(self, response: str, intent: Intent):
        """Update the last conversation with AI response"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE conversations 
                SET ai_response = ? 
                WHERE id = (SELECT MAX(id) FROM conversations)
            ''', (response,))
            self.conn.commit()
    
    def update_context(self, user_input: str, intent: Intent):
        """Update conversation context for better responses"""
        self.context_memory['last_intent'] = intent.name
        self.context_memory['last_input'] = user_input
        self.context_memory['timestamp'] = datetime.now()
        
        # Keep only recent context (last 5 interactions)
        if len(self.conversation_history) >= 5:
            self.conversation_history = self.conversation_history[-4:]
        
        self.conversation_history.append({
            'input': user_input,
            'intent': intent.name,
            'timestamp': datetime.now()
        })
    
    def get_context_info(self):
        """Get context information for frontend"""
        return {
            'last_intent': self.context_memory.get('last_intent'),
            'conversation_length': len(self.conversation_history),
            'recent_topics': list(set([conv['intent'] for conv in self.conversation_history[-3:]]))
        }
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis with emotion detection"""
        words = text.lower().split()
        sentiment_model = {
            'positive': ['happy', 'good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'fantastic', 
                        'awesome', 'brilliant', 'perfect', 'excited', 'thrilled', 'delighted', 'pleased'],
            'negative': ['sad', 'bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointed', 
                        'frustrated', 'angry', 'upset', 'annoyed', 'worried', 'stressed']
        }
        
        pos_score = sum(1 for word in words if word in sentiment_model['positive'])
        neg_score = sum(1 for word in words if word in sentiment_model['negative'])
        
        total_words = len(words)
        if total_words == 0:
            return {'score': 0.0, 'magnitude': 0.0, 'label': 'neutral', 'confidence': 0.0}
            
        sentiment_score = (pos_score - neg_score) / total_words
        magnitude = abs(sentiment_score)
        
        if sentiment_score > 0.1:
            label = 'positive'
        elif sentiment_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
            
        return {
            'score': sentiment_score,
            'magnitude': magnitude,
            'label': label,
            'confidence': min(magnitude * 2, 1.0)
        }

    # Skill implementations
    def get_time(self, user_input: str, entities: Dict, personality: str) -> str:
        """Get current time"""
        current_time = datetime.now().strftime('%I:%M %p')
        current_day = datetime.now().strftime('%A')
        
        responses = {
            'friendly': f"It's {current_time} on this lovely {current_day}! 🕐",
            'professional': f"The current time is {current_time}.",
            'enthusiastic': f"RIGHT NOW it's {current_time}! Time flies when you're having fun! ⏰",
            'witty': f"According to my atomic clock (just kidding, it's my internal timer), it's {current_time}! ⌚",
        }
        
        return responses.get(personality, responses['friendly'])
    
    def get_date(self, user_input: str, entities: Dict, personality: str) -> str:
        """Get current date"""
        current_date = datetime.now().strftime('%B %d, %Y')
        day_of_week = datetime.now().strftime('%A')
        
        return f"Today is {day_of_week}, {current_date}! 📅"
    
    def calculate_math(self, user_input: str, entities: Dict, personality: str) -> str:
        """Perform mathematical calculations"""
        try:
            # Extract mathematical expression
            math_pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/^])\s*(\d+(?:\.\d+)?)'
            match = re.search(math_pattern, user_input)
            
            if match:
                num1, operator, num2 = match.groups()
                num1, num2 = float(num1), float(num2)
                
                if operator == '+':
                    result = num1 + num2
                elif operator == '-':
                    result = num1 - num2
                elif operator == '*':
                    result = num1 * num2
                elif operator == '/':
                    if num2 != 0:
                        result = num1 / num2
                    else:
                        return "I can't divide by zero! That would break the universe! 🌌"
                else:
                    return "I'm not sure about that operation. Try +, -, *, or /! 🧮"
                
                # Format result nicely
                if result == int(result):
                    result = int(result)
                
                return f"The answer is {result}! 🎯"
            
        except Exception as e:
            pass
        
        return "I can help with math! Try something like '25 * 4' or '100 / 5'. I'm quite good with numbers! 🔢"
    
    def tell_joke(self, user_input: str, entities: Dict, personality: str) -> str:
        """Tell jokes based on personality"""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything! 😄",
            "What do you call a fake noodle? An impasta! 🍝",
            "Why did the scarecrow win an award? He was outstanding in his field! 🌾",
            "Why don't programmers like nature? It has too many bugs! 🐛"
        ]
        
        return random.choice(jokes)
    
    def set_timer(self, user_input: str, entities: Dict, personality: str) -> str:
        """Set timer with simple tracking"""
        # Extract time duration
        time_pattern = r'(\d+)\s*(minute|minutes|min|second|seconds|sec|hour|hours|hr)'
        match = re.search(time_pattern, user_input.lower())
        
        if match:
            duration, unit = match.groups()
            duration = int(duration)
            
            # Convert to seconds
            if unit.startswith('min'):
                seconds = duration * 60
                time_str = f"{duration} minute{'s' if duration != 1 else ''}"
            elif unit.startswith('sec'):
                seconds = duration
                time_str = f"{duration} second{'s' if duration != 1 else ''}"
            elif unit.startswith('hour') or unit.startswith('hr'):
                seconds = duration * 3600
                time_str = f"{duration} hour{'s' if duration != 1 else ''}"
            else:
                seconds = duration * 60
                time_str = f"{duration} minute{'s' if duration != 1 else ''}"
            
            # Simple timer tracking
            timer_id = f"timer_{len(self.active_timers) + 1}"
            end_time = datetime.now() + timedelta(seconds=seconds)
            
            self.active_timers[timer_id] = {
                'duration': time_str,
                'start_time': datetime.now(),
                'end_time': end_time,
                'seconds': seconds
            }
            
            return f"Timer set for {time_str}! I'll track it for you. ⏱️"
        
        return "I can set a timer for you! Try saying 'set timer for 5 minutes' or 'timer for 30 seconds'. ⏰"
    
    def set_reminder(self, user_input: str, entities: Dict, personality: str) -> str:
        """Set reminders (mock implementation)"""
        reminder_text = entities.get('entity_0', 'something important')
        return f"I'll remind you to {reminder_text}! Though I should mention, I'm still learning how to actually send reminders. Consider this a friendly heads up! 📝"
    
    # Mock implementations for other skills
    def get_news(self, user_input: str, entities: Dict, personality: str) -> str:
        return "I'd love to get you the latest news! I'm still working on connecting to news APIs, but I hear AI is trending everywhere! 📰"
    
    def play_music(self, user_input: str, entities: Dict, personality: str) -> str:
        song = entities.get('entity_0', 'some great music')
        return f"I'd love to play {song} for you! I'm still learning how to control music players, but I'm humming along in binary! 🎵"
    
    def control_smart_home(self, user_input: str, entities: Dict, personality: str) -> str:
        return "I'd love to help with smart home control! I'm still learning how to connect to smart devices, but consider it done in the virtual world! 🏠"
    
    def translate_text(self, user_input: str, entities: Dict, personality: str) -> str:
        return "I'd love to help with translation! I'm still learning multiple languages, but I'm fluent in binary and Python! 🌍"
    
    def define_word(self, user_input: str, entities: Dict, personality: str) -> str:
        word = entities.get('entity_0', 'that word')
        return f"Great question about '{word}'! I'm still building my dictionary, but I know it's something wonderful! 📚"
    
    def ask_trivia(self, user_input: str, entities: Dict, personality: str) -> str:
        trivia_facts = [
            "Did you know? The first computer bug was an actual bug - a moth found in a computer in 1947! 🦋",
            "Fun fact: The word 'robot' comes from the Czech word 'robota' meaning 'forced labor'! 🤖",
            "Amazing: Honey never spoils! Archaeologists have found edible honey in ancient Egyptian tombs! 🍯"
        ]
        return random.choice(trivia_facts)
    
    def set_alarm(self, user_input: str, entities: Dict, personality: str) -> str:
        return "Alarm set! I'll do my best to remember, though you might want to use your phone's alarm as backup! ⏰"
    
    def check_calendar(self, user_input: str, entities: Dict, personality: str) -> str:
        return "Your calendar looks great! I'm still learning how to access calendars, but I bet you have exciting things planned! 📅"
    
    def send_email(self, user_input: str, entities: Dict, personality: str) -> str:
        return "I'd love to help with email! I'm still learning how to connect to email services, but your message would be amazing! 📧"
    
    def web_search(self, user_input: str, entities: Dict, personality: str) -> str:
        query = entities.get('entity_0', 'that topic')
        return f"I'd search for '{query}' for you! I'm still learning how to browse the web, but I bet there's lots of great info out there! 🔍"

    def get_active_timers_and_reminders(self):
        """Get currently active timers and reminders"""
        active = {
            'timers': [],
            'reminders': []
        }
        
        current_time = datetime.now()
        
        # Format active timers
        for timer_id, timer_info in list(self.active_timers.items()):
            remaining = timer_info['end_time'] - current_time
            if remaining.total_seconds() > 0:
                active['timers'].append({
                    'id': timer_id,
                    'duration': timer_info['duration'],
                    'remaining': str(remaining).split('.')[0]  # Remove microseconds
                })
            else:
                # Timer finished, remove it
                del self.active_timers[timer_id]
        
        return active

# Initialize AI processor
ai_processor = AdvancedAIProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_input():
    data = request.json
    user_input = data.get('input', '')
    personality = data.get('personality', 'friendly')
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    result = ai_processor.generate_response(user_input, personality)
    return jsonify(result)

@app.route('/api/timers-reminders', methods=['GET'])
def get_timers_reminders():
    """Get active timers and reminders"""
    active = ai_processor.get_active_timers_and_reminders()
    return jsonify(active)

@app.route('/api/cancel-timer', methods=['POST'])
def cancel_timer():
    """Cancel a specific timer"""
    data = request.json
    timer_id = data.get('timer_id', '')
    
    if timer_id in ai_processor.active_timers:
        del ai_processor.active_timers[timer_id]
        return jsonify({'message': 'Timer cancelled successfully!'})
    
    return jsonify({'error': 'Timer not found'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_timers': len(ai_processor.active_timers),
        'total_conversations': len(ai_processor.conversation_history)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
