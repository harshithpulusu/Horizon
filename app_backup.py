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
        """Advanced intent recognition using pattern matching and ML-like scoring"""
        # Clean wake words first
        cleaned_input = self.clean_wake_words(user_input)
        user_input_lower = cleaned_input.lower().strip()
        best_intent = Intent("general", 0.0, {})
        
        # Debug: Show which patterns we're checking
        print(f"Checking patterns for: '{user_input_lower}'")
        
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
            print(f"Using conversational response (intent: {intent.name}, confidence: {intent.confidence:.3f})")  # Debug log
            response = self.generate_conversational_response(cleaned_input, personality, sentiment)
        
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
    
    # Skill implementations
    def get_weather(self, user_input: str, entities: Dict, personality: str) -> str:
        """Get real weather information using OpenWeatherMap API or free fallback"""
        try:
            # Extract location from user input or use default
            location = entities.get('entity_0', DEFAULT_WEATHER_LOCATION)
            if not location or location == 'your location':
                location = DEFAULT_WEATHER_LOCATION
            
            # Check if we have a valid API key for OpenWeatherMap
            if WEATHER_API_KEY and WEATHER_API_KEY not in ["your-openweathermap-api-key-here", "PASTE_YOUR_REAL_API_KEY_HERE"]:
                # Try OpenWeatherMap first (free tier: 1000 calls/day)
                return self._get_openweathermap_weather(location, personality)
            else:
                # Use completely free 7Timer API (no key required)
                return self._get_7timer_weather(location, personality)
                
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_mock_weather(location, personality)
    
    def _get_openweathermap_weather(self, location: str, personality: str) -> str:
        """Get weather from OpenWeatherMap API (free tier)"""
        try:
            # Make API request to OpenWeatherMap
            url = f"{WEATHER_API_BASE_URL}/weather"
            params = {
                'q': location,
                'appid': WEATHER_API_KEY,
                'units': DEFAULT_WEATHER_UNITS
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._format_weather_response(data, location, personality)
            elif response.status_code == 404:
                return f"Sorry, I couldn't find weather information for '{location}'. Could you try a different city? 🌍"
            elif response.status_code == 401:
                print("Invalid API key, falling back to free service")
                return self._get_7timer_weather(location, personality)
            else:
                return "I'm having trouble getting weather data right now. Please try again later! 🌤️"
                
        except requests.exceptions.Timeout:
            return "The weather service is taking too long to respond. Please try again! ⏰"
        except requests.exceptions.ConnectionError:
            return "I can't connect to the weather service right now. Please check your internet connection! 🌐"
        except Exception as e:
            print(f"OpenWeatherMap API error: {e}")
            return self._get_7timer_weather(location, personality)
    
    def _get_7timer_weather(self, location: str, personality: str) -> str:
        """Get weather from completely free 7Timer API (no API key required)"""
        try:
            # Get coordinates for the location
            coords = self._get_city_coordinates(location)
            if not coords:
                return f"Sorry, I couldn't find coordinates for '{location}'. Try a major city like New York, London, or Tokyo! 🌍"
            
            lat, lon = coords
            url = "http://www.7timer.info/bin/api.pl"
            params = {
                'lon': lon,
                'lat': lat,
                'product': 'civil',
                'output': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._format_7timer_response(data, location, personality)
            else:
                return self._get_mock_weather(location, personality)
                
        except Exception as e:
            print(f"7Timer API error: {e}")
            return self._get_mock_weather(location, personality)
    
    def _get_city_coordinates(self, location: str) -> tuple:
        """Simple city to coordinates mapping for major cities"""
        city_coords = {
            'new york': (40.7128, -74.0060),
            'los angeles': (34.0522, -118.2437),
            'chicago': (41.8781, -87.6298),
            'houston': (29.7604, -95.3698),
            'phoenix': (33.4484, -112.0740),
            'philadelphia': (39.9526, -75.1652),
            'san antonio': (29.4241, -98.4936),
            'san diego': (32.7157, -117.1611),
            'dallas': (32.7767, -96.7970),
            'san jose': (37.3382, -121.8863),
            'austin': (30.2672, -97.7431),
            'miami': (25.7617, -80.1918),
            'seattle': (47.6062, -122.3321),
            'denver': (39.7392, -104.9903),
            'boston': (42.3601, -71.0589),
            'london': (51.5074, -0.1278),
            'paris': (48.8566, 2.3522),
            'tokyo': (35.6762, 139.6503),
            'sydney': (-33.8688, 151.2093),
            'toronto': (43.6532, -79.3832),
            'montreal': (45.5017, -73.5673),
            'vancouver': (49.2827, -123.1207),
            'berlin': (52.5200, 13.4050),
            'rome': (41.9028, 12.4964),
            'madrid': (40.4168, -3.7038),
            'amsterdam': (52.3676, 4.9041),
            'stockholm': (59.3293, 18.0686),
            'moscow': (55.7558, 37.6176),
            'beijing': (39.9042, 116.4074),
            'shanghai': (31.2304, 121.4737),
            'mumbai': (19.0760, 72.8777),
            'delhi': (28.7041, 77.1025),
            'bangkok': (13.7563, 100.5018),
            'singapore': (1.3521, 103.8198),
            'hong kong': (22.3193, 114.1694),
            'seoul': (37.5665, 126.9780),
            'melbourne': (-37.8136, 144.9631),
            'perth': (-31.9505, 115.8605),
            'auckland': (-36.8485, 174.7633),
            'cairo': (30.0444, 31.2357),
            'lagos': (6.5244, 3.3792),
            'johannesburg': (-26.2041, 28.0473),
            'sao paulo': (-23.5558, -46.6396),
            'rio de janeiro': (-22.9068, -43.1729),
            'mexico city': (19.4326, -99.1332),
            'buenos aires': (-34.6118, -58.3960)
        }
        
        # Clean location input and try to match
        location_lower = location.lower().split(',')[0].strip()
        # Try exact match first
        if location_lower in city_coords:
            return city_coords[location_lower]
        
        # Try partial matches for common abbreviations
        for city, coords in city_coords.items():
            if location_lower in city or city in location_lower:
                return coords
        
        return None
    
    def _format_7timer_response(self, data: Dict, location: str, personality: str) -> str:
        """Format 7Timer API response based on personality"""
        try:
            current = data['dataseries'][0]
            weather_desc = self._decode_7timer_weather(current['weather'])
            temp_c = current['temp2m']  # Temperature in Celsius
            temp_f = round((temp_c * 9/5) + 32)  # Convert to Fahrenheit
            
            # Format based on personality with "Free Weather Service" note
            weather_responses = {
                'friendly': f"The weather in {location} is {weather_desc} with a temperature of {temp_f}°F ({temp_c}°C)! Have a great day! 😊 (via free weather service)",
                'professional': f"Current conditions in {location}: {weather_desc}, {temp_f}°F ({temp_c}°C). Data provided by 7Timer.",
                'enthusiastic': f"WOW! {location} has {weather_desc} weather and it's {temp_f}°F! What an AMAZING day! 🌟",
                'witty': f"Well, well! {location} is serving up {weather_desc} at a crisp {temp_f}°F. Mother Nature's feeling fancy! 🎭",
                'sarcastic': f"Oh joy, {location} has {weather_desc} weather at {temp_f}°F. How absolutely thrilling. 🙄",
                'zen': f"In {location}, the universe presents {weather_desc} at {temp_f}°F. Feel the harmony of nature's rhythm. 🧘",
                'scientist': f"Meteorological analysis for {location}: {weather_desc} conditions, temperature reading {temp_f}°F. Data source: 7Timer. 🔬",
                'pirate': f"Arrr! The weather be {weather_desc} in {location}, with temperatures at {temp_f}°F! Good sailing weather, matey! 🏴‍☠️",
                'shakespearean': f"Hark! In fair {location}, the heavens doth present {weather_desc} at {temp_f} degrees! What wondrous conditions! 🎭",
                'valley_girl': f"OMG! {location} is like, totally {weather_desc} and it's {temp_f}°F! That's like, so perfect! 💁‍♀️",
                'cowboy': f"Well partner, {location}'s got {weather_desc} weather at {temp_f}°F. Mighty fine conditions! 🤠",
                'robot': f"WEATHER ANALYSIS COMPLETE. LOCATION: {location}. CONDITIONS: {weather_desc}. TEMPERATURE: {temp_f}°F. SOURCE: 7TIMER. 🤖"
            }
            
            return weather_responses.get(personality, weather_responses['friendly'])
            
        except Exception as e:
            print(f"Error formatting 7Timer response: {e}")
            return self._get_mock_weather(location, personality)
    
    def _decode_7timer_weather(self, weather_code: str) -> str:
        """Decode 7Timer weather codes to descriptions"""
        weather_codes = {
            'clear': 'clear skies',
            'pcloudy': 'partly cloudy',
            'mcloudy': 'mostly cloudy',
            'cloudy': 'cloudy',
            'humid': 'humid',
            'lightrain': 'light rain',
            'oshower': 'occasional showers',
            'ishower': 'isolated showers',
            'lightsnow': 'light snow',
            'rain': 'rainy',
            'snow': 'snowy',
            'rainsnow': 'rain and snow',
            'ts': 'thunderstorms',
            'tsrain': 'thunderstorms with rain'
        }
        return weather_codes.get(weather_code, weather_code)
    
    def _format_weather_response(self, data: Dict, location: str, personality: str) -> str:
        """Format weather API response based on personality"""
        temp = round(data['main']['temp'])
        feels_like = round(data['main']['feels_like'])
        description = data['weather'][0]['description'].title()
        humidity = data['main']['humidity']
        wind_speed = round(data['wind']['speed']) if 'wind' in data else 0
        
        # Get appropriate emoji for weather condition
        weather_id = data['weather'][0]['id']
        emoji = self._get_weather_emoji(weather_id)
        
        responses = {
            'friendly': f"The weather in {location} is {temp}°F and {description.lower()} {emoji}! It feels like {feels_like}°F. Perfect day to enjoy the outdoors! 🌟",
            'professional': f"Current conditions in {location}: {temp}°F, {description.lower()}, humidity {humidity}%, wind {wind_speed} mph.",
            'enthusiastic': f"WOW! It's {temp}°F and {description.lower()} in {location}! {emoji} Feels like {feels_like}°F! What an AMAZING day! 🚀",
            'witty': f"Mother Nature reports {temp}°F and {description.lower()} in {location} {emoji}. She says it feels like {feels_like}°F, but who are we to argue with her? 😎",
            'sarcastic': f"Oh, the weather? How original. It's {temp}°F and {description.lower()} in {location}. Feels like {feels_like}°F. There, happy now? 🙄",
            'zen': f"In {location}, the universe provides {temp}°F with {description.lower()} {emoji}. Accept what is, for it feels like {feels_like}°F. 🧘",
            'scientist': f"Meteorological data for {location}: Temperature {temp}°F, atmospheric conditions {description.lower()}, perceived temperature {feels_like}°F, humidity {humidity}% 🔬",
            'pirate': f"Arrr! The skies over {location} be showin' {temp}°F with {description.lower()}! Feels like {feels_like}°F, perfect for sailin' the seven seas! ⚓",
            'shakespearean': f"In fair {location}, where we lay our scene, the heavens decree {temp}°F with {description.lower()}! Though it feels as {feels_like}°F! 🎭",
            'valley_girl': f"So like, {location} is totally {temp}°F right now and it's like, {description.lower()}? Feels like {feels_like}°F which is like, whatever! 💁‍♀️",
            'cowboy': f"Well partner, out there in {location} it's {temp}°F with {description.lower()}. Feels mighty like {feels_like}°F - good ridin' weather! 🤠",
            'robot': f"WEATHER.EXE: {location} TEMPERATURE={temp}°F STATUS={description.upper()} FEELS_LIKE={feels_like}°F HUMIDITY={humidity}% 🤖"
        }
        
        return responses.get(personality, responses['friendly'])
    
    def _get_weather_emoji(self, weather_id: int) -> str:
        """Get appropriate emoji based on weather condition ID"""
        if 200 <= weather_id < 300:  # Thunderstorm
            return "⛈️"
        elif 300 <= weather_id < 400:  # Drizzle
            return "🌦️"
        elif 500 <= weather_id < 600:  # Rain
            return "🌧️"
        elif 600 <= weather_id < 700:  # Snow
            return "❄️"
        elif 700 <= weather_id < 800:  # Atmosphere (fog, etc.)
            return "🌫️"
        elif weather_id == 800:  # Clear sky
            return "☀️"
        elif 801 <= weather_id < 900:  # Clouds
            return "☁️"
        else:
            return "🌤️"
    
    def _get_mock_weather(self, location: str, personality: str) -> str:
        """Fallback mock weather when API is not available"""
        weather_conditions = ['sunny', 'cloudy', 'rainy', 'partly cloudy', 'clear']
        temperature = random.randint(60, 85)
        condition = random.choice(weather_conditions)
        
        responses = {
            'friendly': f"It's currently {temperature}°F and {condition} in {location}! Perfect weather to go outside! ☀️ (Note: This is simulated weather - add your API key for real data!)",
            'professional': f"The current weather in {location} is {temperature}°F with {condition} conditions. (Simulated data)",
            'enthusiastic': f"WOW! It's a gorgeous {temperature}°F and {condition} in {location}! AMAZING weather! 🌟 (Demo data)",
            'witty': f"Well, Mother Nature says it's {temperature}°F and {condition} in {location}. She's in a good mood today! 😎 (Fake news until you add your API key!)"
        }
        
        return responses.get(personality, responses['friendly'])
    
    def get_time(self, user_input: str, entities: Dict, personality: str) -> str:
        """Get current time"""
        current_time = datetime.now().strftime('%I:%M %p')
        current_day = datetime.now().strftime('%A')
        
        responses = {
            'friendly': f"It's {current_time} on this lovely {current_day}! 🕐",
            'professional': f"The current time is {current_time}.",
            'enthusiastic': f"RIGHT NOW it's {current_time}! Time flies when you're having fun! ⏰",
            'witty': f"According to my atomic clock (just kidding, it's my internal timer), it's {current_time}! ⌚",
            'sarcastic': f"Oh, you can't read a clock? It's {current_time}. You're welcome. 🙄",
            'zen': f"Time is but an illusion... but if you must know, it is {current_time}. Be present in this moment. 🧘",
            'scientist': f"Based on atomic oscillations and Earth's rotation, the temporal coordinates indicate {current_time}. 🔬",
            'pirate': f"Ahoy matey! By me calculations, it be {current_time} on this fine {current_day}! ⚓",
            'shakespearean': f"Hark! The hour doth strike {current_time} upon this {current_day}, fair thee well! 🎭",
            'valley_girl': f"OMG, like, it's totally {current_time} right now! So crazy! 💁‍♀️",
            'cowboy': f"Well howdy partner! It's {current_time} here in these parts on {current_day}! 🤠",
            'robot': f"SYSTEM TIME: {current_time}. DAY CYCLE: {current_day}. PROCESSING COMPLETE. BEEP BOOP. 🤖"
        }
        
        return responses.get(personality, responses['friendly'])
    
    def get_date(self, user_input: str, entities: Dict, personality: str) -> str:
        """Get current date"""
        current_date = datetime.now().strftime('%B %d, %Y')
        day_of_week = datetime.now().strftime('%A')
        
        return f"Today is {day_of_week}, {current_date}! 📅"
    
    def calculate_math(self, user_input: str, entities: Dict, personality: str) -> str:
        """Perform mathematical calculations"""
        # Enhanced math parser
        try:
            # Extract mathematical expression
            math_pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/^])\s*(\d+(?:\.\d+)?)'
            match = re.search(math_pattern, user_input)
            
            if match:
                num1, operator, num2 = match.groups()
                num1, num2 = float(num1), float(num2)
                
                operations = {
                    '+': num1 + num2,
                    '-': num1 - num2,
                    '*': num1 * num2,
                    '/': num1 / num2 if num2 != 0 else float('inf'),
                    '^': num1 ** num2
                }
                
                result = operations.get(operator)
                
                if result == float('inf'):
                    return "Oops! Can't divide by zero. Even I have my limits! 🤖"
                
                return f"{num1} {operator} {num2} = {result}"
            
            # Try to evaluate more complex expressions safely
            safe_input = re.sub(r'[^0-9+\-*/.() ]', '', user_input)
            if safe_input:
                result = eval(safe_input)
                return f"The answer is {result}! 🧮"
                
        except Exception as e:
            pass
        
        return "I can help with math! Try something like '25 * 4' or '100 / 5'. I'm quite good with numbers! 🔢"
    
    def set_reminder(self, user_input: str, entities: Dict, personality: str) -> str:
        """Set reminders (mock implementation)"""
        reminder_text = entities.get('entity_0', 'something important')
        return f"I'll remind you to {reminder_text}! Though I should mention, I'm still learning how to actually send reminders. Consider this a friendly heads up! 📝"
    
    def tell_joke(self, user_input: str, entities: Dict, personality: str) -> str:
        """Tell jokes based on personality"""
        
        # Different joke styles for different personalities
        joke_sets = {
            'friendly': [
                "Why don't scientists trust atoms? Because they make up everything! 😄",
                "What do you call a fake noodle? An impasta! 🍝",
                "Why did the scarecrow win an award? He was outstanding in his field! 🌾"
            ],
            'professional': [
                "Here's a business joke: Why don't companies ever get cold? They have plenty of Windows! 💼",
                "A professional joke: What's the best thing about Switzerland? I don't know, but the flag is a big plus! 🏢"
            ],
            'enthusiastic': [
                "OH MY GOSH! Why did the math book look so sad? Because it was FULL of problems! But don't worry, we can solve them ALL! 📚🚀",
                "Get ready for this AMAZING joke! What's orange and sounds like a parrot? A CARROT! Isn't that FANTASTIC?! 🥕"
            ],
            'witty': [
                "I told my computer a joke about UDP, but it didn't get it. Unlike you, I hope. 💻",
                "Why do programmers prefer dark mode? Because light attracts bugs! Much like my personality attracts sarcasm. 🐛"
            ],
            'sarcastic': [
                "Oh, you want a joke? How original. Fine: Why did the hipster burn his mouth? He drank coffee before it was cool. 🙄☕",
                "Here's a joke as funny as your request: What's the difference between a poorly dressed person and a tired cat? One wears a suit badly, the other just wants to nap. Hilarious, right? 😒"
            ],
            'zen': [
                "A monk asked his master: 'What is the sound of one hand clapping?' The master replied: 'The same as no hands applauding your jokes.' �",
                "Why did the meditation student break up with his girlfriend? She said he was too detached. He replied: 'Attachment is the root of suffering.' 🕯️"
            ],
            'scientist': [
                "A neutron walks into a bar and asks 'How much for a drink?' The bartender says 'For you? No charge!' 🔬⚛️",
                "Why can't you trust atoms? Because they make up everything! And I mean everything - literally 99.9999% empty space! 🧪"
            ],
            'pirate': [
                "Why couldn't the pirate play cards? Because he was sitting on the deck! Har har har! 🏴‍☠️",
                "What's a pirate's favorite letter? Ye might think it's 'R', but his first love be the 'C'! Arrr! ⚓"
            ],
            'shakespearean': [
                "What light through yonder window breaks? 'Tis the sun, and WiFi is the internet! Hark, technology doth make fools of us all! 🎭",
                "To pun or not to pun, that is the question! Whether 'tis nobler to suffer the groans of outrageous wordplay... 📜"
            ],
            'valley_girl': [
                "OMG, like, why did the blonde stare at the orange juice container? Because it said 'concentrate'! That's like, so random! 💁‍♀️",
                "So there's this guy, and he's like, 'I'm reading a book about anti-gravity.' And I'm like, 'Cool!' And he's like, 'I can't put it down!' LOL! 📖"
            ],
            'cowboy': [
                "Why don't cowboys ever complain? Because they never want to stirrup trouble! Yeehaw! 🤠",
                "What do you call a sleeping bull in the desert? A bulldozer! That there's some fine humor, partner! 🐂"
            ],
            'robot': [
                "JOKE.EXE INITIATED: WHY DO ROBOTS NEVER PANIC? BECAUSE THEY HAVE NERVES OF STEEL. HUMOR.PROTOCOL COMPLETE. BEEP BOOP. 🤖",
                "PROCESSING HUMOR... ERROR 404: FUNNY NOT FOUND. JUST KIDDING. THAT WAS THE JOKE. INITIATING LAUGH.WAV �"
            ]
        }
        
        # Get jokes for the current personality, fallback to friendly
        jokes = joke_sets.get(personality, joke_sets['friendly'])
        selected_joke = random.choice(jokes)
        
        # Add personality-specific delivery
        if personality == 'enthusiastic':
            return f"HERE'S AN AMAZING JOKE FOR YOU! {selected_joke} WASN'T THAT INCREDIBLE?! 🎉"
        elif personality == 'sarcastic':
            return f"Oh, you want to hear a joke? {selected_joke} There, I've fulfilled my comedy quota for the day. 😏"
        elif personality == 'zen':
            return f"Let me share wisdom through humor: {selected_joke} May this bring lightness to your soul. 🙏"
        elif personality == 'professional':
            return f"Here is a light-hearted anecdote for your consideration: {selected_joke}"
        else:
            return selected_joke
    
    def get_news(self, user_input: str, entities: Dict, personality: str) -> str:
        """Get news (mock implementation)"""
        return "I'd love to get you the latest news! I'm still working on connecting to news APIs, but I hear AI is trending everywhere! 📰"
    
    def play_music(self, user_input: str, entities: Dict, personality: str) -> str:
        """Play music (mock implementation)"""
        song = entities.get('entity_0', 'some great music')
        return f"I'd love to play {song} for you! I'm still learning how to control music players, but I'm humming along in binary! 🎵"
    
    def control_smart_home(self, user_input: str, entities: Dict, personality: str) -> str:
        """Control smart home devices (mock implementation)"""
        action = entities.get('entity_0', 'on')
        device = entities.get('entity_1', 'the lights')
        return f"I'm turning {action} {device}! Well, I would if I were connected to your smart home system. Consider it done in the virtual world! 🏠"
    
    def translate_text(self, user_input: str, entities: Dict, personality: str) -> str:
        """Translate text (mock implementation)"""
        return "I'd love to help with translation! I'm still learning multiple languages, but I'm fluent in binary and Python! 🌍"
    
    def define_word(self, user_input: str, entities: Dict, personality: str) -> str:
        """Define words (mock implementation)"""
        word = entities.get('entity_0', 'that word')
        return f"Great question about '{word}'! I'm still building my dictionary, but I know it's something wonderful! 📚"
    
    def ask_trivia(self, user_input: str, entities: Dict, personality: str) -> str:
        """Trivia questions"""
        trivia_facts = [
            "Did you know? The first computer bug was an actual bug - a moth found in a computer in 1947! 🦋",
            "Fun fact: The word 'robot' comes from the Czech word 'robota' meaning 'forced labor'! 🤖",
            "Amazing: Honey never spoils! Archaeologists have found edible honey in ancient Egyptian tombs! 🍯",
            "Incredible: A group of flamingos is called a 'flamboyance'! How fabulous! 🦩"
        ]
        return random.choice(trivia_facts)
    
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
                time_str = f"{duration} minute{'s' if duration > 1 else ''}"
            elif unit.startswith('sec'):
                seconds = duration
                time_str = f"{duration} second{'s' if duration > 1 else ''}"
            elif unit.startswith('hour') or unit.startswith('hr'):
                seconds = duration * 3600
                time_str = f"{duration} hour{'s' if duration > 1 else ''}"
            else:
                seconds = duration * 60  # Default to minutes
                time_str = f"{duration} minute{'s' if duration > 1 else ''}"
            
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
        """Set reminders with simple tracking"""
        reminder_text = entities.get('entity_0', 'something important')
        
        # Try to extract time information
        time_patterns = [
            r'in (\d+) (minute|minutes|min|hour|hours|hr)',
            r'at (\d{1,2}):(\d{2})',
            r'tomorrow',
            r'in (\d+) days?'
        ]
        
        scheduled = False
        for pattern in time_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                scheduled = True
                reminder_id = f"reminder_{len(self.reminders) + 1}"
                
                if 'tomorrow' in user_input.lower():
                    remind_time = datetime.now().replace(hour=9, minute=0, second=0) + timedelta(days=1)
                elif ':' in pattern:  # Time format like "at 3:30"
                    hour, minute = match.groups()
                    remind_time = datetime.now().replace(hour=int(hour), minute=int(minute), second=0)
                    if remind_time <= datetime.now():
                        remind_time += timedelta(days=1)
                else:  # Duration format like "in 5 minutes"
                    duration, unit = match.groups()
                    duration = int(duration)
                    if unit.startswith('min'):
                        remind_time = datetime.now() + timedelta(minutes=duration)
                    elif unit.startswith('hour') or unit.startswith('hr'):
                        remind_time = datetime.now() + timedelta(hours=duration)
                    elif unit.startswith('day'):
                        remind_time = datetime.now() + timedelta(days=duration)
                
                self.reminders.append({
                    'id': reminder_id,
                    'text': reminder_text,
                    'time': remind_time,
                    'created': datetime.now()
                })
                
                return f"Reminder set! I'll remind you to {reminder_text} at {remind_time.strftime('%I:%M %p on %B %d')}. 📝"
        
        if not scheduled:
            return f"I'll remember that you want to {reminder_text}! Try being more specific about when, like 'remind me in 30 minutes' or 'remind me tomorrow at 9 AM'. 📝"
    
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
                    'remaining_seconds': int(remaining.total_seconds())
                })
            else:
                # Timer expired, remove it
                del self.active_timers[timer_id]
        
        # Format active reminders
        for reminder in self.reminders[:]:
            if reminder['time'] > current_time:
                time_until = reminder['time'] - current_time
                active['reminders'].append({
                    'id': reminder['id'],
                    'text': reminder['text'],
                    'time': reminder['time'].strftime('%I:%M %p on %B %d'),
                    'time_until_seconds': int(time_until.total_seconds())
                })
            else:
                # Reminder expired, remove it
                self.reminders.remove(reminder)
        
        return active
    
    def set_alarm(self, user_input: str, entities: Dict, personality: str) -> str:
        """Set alarm (mock implementation)"""
        return "Alarm set! I'll do my best to remember, though you might want to use your phone's alarm as backup! ⏰"
    
    def check_calendar(self, user_input: str, entities: Dict, personality: str) -> str:
        """Check calendar (mock implementation)"""
        return "Your calendar looks great! I'm still learning how to access calendars, but I bet you have exciting things planned! 📅"
    
    def send_email(self, user_input: str, entities: Dict, personality: str) -> str:
        """Send email (mock implementation)"""
        return "I'd love to help with email! I'm still learning how to connect to email services, but your message would be amazing! 📧"
    
    def web_search(self, user_input: str, entities: Dict, personality: str) -> str:
        """Web search (mock implementation)"""
        query = entities.get('entity_0', 'that topic')
        return f"I'd search for '{query}' for you! I'm still learning how to browse the web, but I bet there's lots of great info out there! 🔍"
    
    def generate_conversational_response(self, user_input: str, personality: str, sentiment: Dict) -> str:
        """Generate conversational responses for unrecognized intents"""
        sentiment_label = sentiment.get('label', 'neutral')
        
        # Respond based on sentiment
        if sentiment_label == 'positive':
            positive_responses = {
                'friendly': "That sounds wonderful! I love your positive energy! Tell me more! 😊",
                'professional': "That's excellent to hear. How may I assist you further?",
                'enthusiastic': "AMAZING! I can feel your excitement! This is FANTASTIC! 🎉",
                'witty': "Well, well! Someone's in a great mood! *virtual high five* 🙌",
                'sarcastic': "Oh wow, such enthusiasm. Don't let me bring down your parade with my sparkling personality. 🙄",
                'zen': "Your positive energy radiates like sunlight through morning mist. Please, share more of your wisdom. 🌅",
                'scientist': "Fascinating! Your elevated mood correlates with increased optimism levels. Please elaborate on the variables. 🔬",
                'pirate': "Arrr! I can feel yer good spirits from here, matey! Tell this old sea dog more tales! 🏴‍☠️",
                'shakespearean': "Hark! What joyous tidings you bring! Pray tell, what fair fortune has blessed thee this day? 🎭",
                'valley_girl': "OMG, that's like, totally awesome! You're giving me all the good vibes! Spill the tea, girl! 💁‍♀️",
                'cowboy': "Well ain't that something! You're brighter than a new penny, partner! What's got you so chipper? 🤠",
                'robot': "POSITIVITY DETECTED. MOOD ANALYSIS: OPTIMAL. PLEASE PROVIDE ADDITIONAL DATA FOR PROCESSING. 🤖"
            }
            return positive_responses.get(personality, positive_responses['friendly'])
        
        elif sentiment_label == 'negative':
            supportive_responses = {
                'friendly': "I'm sorry to hear that. I'm here to help make things better! Would you like to talk about it?",
                'professional': "I understand. If there's anything specific you need assistance with, please let me know.",
                'enthusiastic': "Oh no! Let's turn that frown upside down! 😄 How can I assist in making your day better?",
                'witty': "Ah, a plot twist! Even the best stories have their challenges. Care to share more?",
                'sarcastic': "Oh no, problems? How absolutely shocking. Let me guess, you want me to magically fix everything? 😒",
                'zen': "The river of sorrow flows, but it also passes. Share your burden, and let us find peace together. 🧘",
                'scientist': "Negative emotional state detected. Would you like to discuss the contributing factors for analysis? 🔬",
                'pirate': "Avast! Rough seas ahead, eh matey? Fear not, we'll weather this storm together! ⚓",
                'shakespearean': "Alas! What cruel fate has befallen thee? Speak, that I might offer comfort in thy hour of need! 🎭",
                'valley_girl': "Oh no, babe! That's like, so not good! Want to talk about it? I'm totally here for you! 💕",
                'cowboy': "Aw shucks, partner. Sounds like you're ridin' through some rough terrain. Care to share what's troublin' ya? 🤠",
                'robot': "NEGATIVE SENTIMENT DETECTED. INITIATING SUPPORT PROTOCOL. HOW MAY I ASSIST IN ERROR CORRECTION? 🤖"
            }
            return supportive_responses.get(personality, supportive_responses['friendly'])
        
        # Default conversational response
        default_responses = {
            'friendly': "I'm all ears! What else is on your mind?",
            'professional': "Please provide more details so I can assist you effectively.",
            'enthusiastic': "YAY! I love chatting with you! What else can we explore together? 🚀",
            'witty': "Intriguing... *strokes virtual chin* Do go on!",
            'sarcastic': "Oh, mysterious and vague. My favorite combination. Care to be more specific, or shall I guess? 🙄",
            'zen': "I am here, present in this moment with you. What thoughts flow through your mind like leaves on water? 🍃",
            'scientist': "Insufficient data provided. Please specify your query parameters for optimal assistance. 🔬",
            'pirate': "Arrr! What be on yer mind, me hearty? Speak up, don't be shy now! 🏴‍☠️",
            'shakespearean': "Pray tell, what thoughts doth occupy thy noble mind? I am at thy service, good sir or madam! 🎭",
            'valley_girl': "So like, what's up? I'm like, totally ready to chat about whatever! 💁‍♀️",
            'cowboy': "Well howdy there, partner! What's on your mind? Don't be shy now, speak up! 🤠",
            'robot': "AWAITING INPUT. PLEASE SPECIFY REQUEST PARAMETERS. READY TO PROCESS. 🤖"
        }
        return default_responses.get(personality, default_responses['friendly'])
    
    def init_sentiment_model(self):
        """Initialize enhanced sentiment analysis"""
        return {
            'positive': ['happy', 'good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'fantastic', 
                        'awesome', 'brilliant', 'perfect', 'excited', 'thrilled', 'delighted', 'pleased',
                        'satisfied', 'grateful', 'thankful', 'optimistic', 'cheerful', 'joyful'],
            'negative': ['sad', 'bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointed', 
                        'frustrated', 'angry', 'upset', 'annoyed', 'worried', 'stressed', 'depressed',
                        'anxious', 'confused', 'tired', 'bored', 'lonely', 'overwhelmed']
        }
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis with emotion detection"""
        words = text.lower().split()
        sentiment_model = self.init_sentiment_model()
        
        pos_score = sum(1 for word in words if word in sentiment_model['positive'])
        neg_score = sum(1 for word in words if word in sentiment_model['negative'])
        
        total_words = len(words)
        if total_words == 0:
            return {'score': 0, 'magnitude': 0, 'label': 'neutral', 'emotion': 'calm'}
            
        sentiment_score = (pos_score - neg_score) / total_words
        magnitude = abs(sentiment_score)
        
        # Determine emotion
        emotion = 'calm'
        if sentiment_score > 0.2:
            emotion = 'excited'
        elif sentiment_score > 0.1:
            emotion = 'happy'
        elif sentiment_score < -0.2:
            emotion = 'upset'
        elif sentiment_score < -0.1:
            emotion = 'concerned'
        
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
            'emotion': emotion,
            'confidence': min(magnitude * 2, 1.0)
        }

    def learn_from_feedback(self, user_input: str, ai_response: str, feedback: str):
        """Learn from user feedback to improve responses"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    user_input TEXT,
                    ai_response TEXT,
                    feedback TEXT,
                    rating INTEGER
                )
            ''')
            
            # Simple rating system based on feedback
            rating = 5 if 'good' in feedback.lower() or 'great' in feedback.lower() else 3
            if 'bad' in feedback.lower() or 'wrong' in feedback.lower():
                rating = 1
            
            cursor.execute('''
                INSERT INTO feedback (timestamp, user_input, ai_response, feedback, rating)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), user_input, ai_response, feedback, rating))
            self.conn.commit()

    def get_learning_insights(self):
        """Get insights from conversation patterns"""
        with self.lock:
            cursor = self.conn.cursor()
            
            # Most common intents
            cursor.execute('''
                SELECT intent, COUNT(*) as count 
                FROM conversations 
                GROUP BY intent 
                ORDER BY count DESC 
                LIMIT 5
            ''')
            common_intents = cursor.fetchall()
            
            # Average sentiment
            cursor.execute('''
                SELECT AVG(CAST(json_extract(sentiment, '$.score') AS REAL)) as avg_sentiment
                FROM conversations
            ''')
            avg_sentiment = cursor.fetchone()[0] or 0
            
            # Response effectiveness
            cursor.execute('''
                SELECT AVG(rating) as avg_rating, COUNT(*) as total_feedback
                FROM feedback
            ''')
            feedback_stats = cursor.fetchone()
            
            return {
                'common_intents': common_intents,
                'average_sentiment': avg_sentiment,
                'avg_rating': feedback_stats[0] or 0,
                'total_feedback': feedback_stats[1] or 0
            }

    def get_database_stats(self):
        """Get database statistics"""
        with self.lock:
            cursor = self.conn.cursor()
            
            # Count conversations
            cursor.execute('SELECT COUNT(*) FROM conversations')
            conversation_count = cursor.fetchone()[0]
            
            # Count feedback entries
            cursor.execute('SELECT COUNT(*) FROM feedback')
            feedback_count = cursor.fetchone()[0]
            
            # Count user preferences
            cursor.execute('SELECT COUNT(*) FROM user_preferences')
            preferences_count = cursor.fetchone()[0]
            
            return {
                'conversations': conversation_count,
                'feedback_entries': feedback_count,
                'user_preferences': preferences_count,
                'active_timers': len(self.active_timers),
                'active_reminders': len(self.reminders)
            }

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

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    user_input = data.get('user_input', '')
    ai_response = data.get('ai_response', '')
    feedback = data.get('feedback', '')
    
    if not all([user_input, ai_response, feedback]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    ai_processor.learn_from_feedback(user_input, ai_response, feedback)
    return jsonify({'message': 'Feedback received! I\'m learning from it.'})

@app.route('/api/insights', methods=['GET'])
def get_insights():
    insights = ai_processor.get_learning_insights()
    return jsonify(insights)

@app.route('/api/voice-commands', methods=['GET'])
def get_voice_commands():
    """Get available voice commands for the user"""
    commands = {
        'time_date': [
            "What time is it?",
            "What's today's date?",
            "What day is it?"
        ],
        'weather': [
            "What's the weather like?",
            "Is it raining?",
            "What's the temperature?"
        ],
        'math': [
            "Calculate 25 times 4",
            "What's 100 divided by 5?",
            "Solve 15 plus 30"
        ],
        'entertainment': [
            "Tell me a joke",
            "Play some music",
            "Give me a fun fact"
        ],
        'reminders': [
            "Remind me to call mom",
            "Set a timer for 10 minutes",
            "Don't forget to buy groceries"
        ],
        'smart_home': [
            "Turn on the lights",
            "Set temperature to 72",
            "Dim the living room lights"
        ],
        'information': [
            "Define artificial intelligence",
            "Search for Python tutorials",
            "What does 'serendipity' mean?"
        ]
    }
    return jsonify(commands)

@app.route('/api/personality', methods=['POST'])
def set_personality():
    data = request.json
    personality = data.get('personality', 'friendly')
    
    # Store user preference
    with ai_processor.lock:
        cursor = ai_processor.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences (key, value)
            VALUES (?, ?)
        ''', ('default_personality', personality))
        ai_processor.conn.commit()
    
    return jsonify({'message': f'Personality set to {personality}!'})

@app.route('/api/conversation-export', methods=['GET'])
def export_conversations():
    """Export conversation history"""
    with ai_processor.lock:
        cursor = ai_processor.conn.cursor()
        cursor.execute('''
            SELECT timestamp, user_input, ai_response, intent, sentiment
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        conversations = cursor.fetchall()
    
    export_data = []
    for conv in conversations:
        export_data.append({
            'timestamp': conv[0],
            'user_input': conv[1],
            'ai_response': conv[2],
            'intent': conv[3],
            'sentiment': json.loads(conv[4]) if conv[4] else None
        })
    
    return jsonify(export_data)

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

@app.route('/api/cancel-reminder', methods=['POST'])
def cancel_reminder():
    """Cancel a specific reminder"""
    data = request.json
    reminder_id = data.get('reminder_id', '')
    
    # Find and remove reminder
    reminder_found = False
    for i, reminder in enumerate(ai_processor.reminders):
        if reminder['id'] == reminder_id:
            ai_processor.reminders.pop(i)
            reminder_found = True
            break
    
    if reminder_found:
        return jsonify({'message': 'Reminder cancelled successfully!'})
    else:
        return jsonify({'error': 'Reminder not found'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_timers': len(ai_processor.active_timers),
        'active_reminders': len(ai_processor.reminders),
        'total_conversations': len(ai_processor.conversation_history)
    })

@app.route('/api/reset-memory', methods=['POST'])
def reset_memory():
    """Reset AI memory and conversation history"""
    with ai_processor.lock:
        cursor = ai_processor.conn.cursor()
        cursor.execute('DELETE FROM conversations')
        cursor.execute('DELETE FROM feedback')
        ai_processor.conn.commit()
    
    ai_processor.conversation_history = []
    ai_processor.context_memory = {}
    
    return jsonify({'message': 'AI memory reset successfully!'})

@app.route('/api/train', methods=['POST'])
def train_ai():
    """Train AI with custom responses"""
    data = request.json
    pattern = data.get('pattern', '')
    response = data.get('response', '')
    intent = data.get('intent', 'custom')
    
    if not pattern or not response:
        return jsonify({'error': 'Pattern and response are required'}), 400
    
    # Store custom training data
    with ai_processor.lock:
        cursor = ai_processor.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS custom_training (
                id INTEGER PRIMARY KEY,
                pattern TEXT,
                response TEXT,
                intent TEXT,
                created_at TEXT
            )
        ''')
        cursor.execute('''
            INSERT INTO custom_training (pattern, response, intent, created_at)
            VALUES (?, ?, ?, ?)
        ''', (pattern, response, intent, datetime.now().isoformat()))
        ai_processor.conn.commit()
    
    return jsonify({'message': 'AI training data added successfully!'})

@app.route('/api/system-info', methods=['GET'])
def get_system_info():
    """Get system information and capabilities"""
    return jsonify({
        'name': 'Horizon AI Assistant',
        'version': '2.0.0',
        'capabilities': [
            'Natural Language Processing',
            'Sentiment Analysis',
            'Intent Recognition',
            'Timers and Reminders',
            'Math Calculations',
            'Weather Information',
            'Smart Home Control (Mock)',
            'Conversation Memory',
            'Learning from Feedback',
            'Multiple Personalities',
            'Voice Command Recognition',
            'Context Awareness'
        ],
        'supported_intents': list(ai_processor.intent_patterns.keys()),
        'personalities': ['friendly', 'professional', 'enthusiastic', 'witty'],
        'database_size': ai_processor.get_database_stats()
    })

@app.route('/api/database-stats', methods=['GET'])
def get_database_stats():
    """Get database statistics"""
    stats = ai_processor.get_database_stats()
    return jsonify(stats)

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment_endpoint():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    sentiment = ai_processor.analyze_sentiment(text)
    return jsonify(sentiment)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'conversation_count': len(ai_processor.conversation_history),
        'total_interactions': len(ai_processor.conversation_history),
        'avg_sentiment': 0.5  # Simple fallback without numpy
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)