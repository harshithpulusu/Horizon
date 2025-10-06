"""
Horizon AI Engine Core Module

This module contains the core AI functionality for Horizon AI Assistant.
It handles AI model integrations, response generation, and intelligent fallbacks.

Classes:
- AIEngine: Main AI engine class that manages all AI operations
- ResponseGenerator: Handles response generation with personality support
- ModelManager: Manages AI model connections and configurations

Functions:
- ask_chatgpt: ChatGPT API integration with personality blending
- ask_ai_model: Main AI function with fallback support
- generate_fallback_response: Intelligent fallback responses
"""

import os
import uuid
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from config import Config

# AI Model Imports
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from google.cloud import aiplatform
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    IMAGEN_AVAILABLE = True
except ImportError:
    IMAGEN_AVAILABLE = False

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


class AIEngine:
    """Main AI Engine class that manages all AI operations."""
    
    def __init__(self):
        """Initialize the AI Engine with model configurations."""
        self.openai_client = None
        self.gemini_configured = False
        self.imagen_configured = False
        self.ai_model_available = False
        
        # Initialize AI models
        self._initialize_openai()
        self._initialize_gemini()
        self._initialize_imagen()
        
        print("🧠 AI Engine initialized successfully")
    
    def _initialize_openai(self):
        """Initialize OpenAI ChatGPT API."""
        if not OPENAI_AVAILABLE:
            print("⚠️ OpenAI library not available")
            return
        
        try:
            print("🤖 Initializing ChatGPT API connection...")
            
            # Load API key from environment or config
            openai_api_key = os.getenv('OPENAI_API_KEY') or getattr(Config, 'OPENAI_API_KEY', None)
            
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.ai_model_available = True
                print("✅ ChatGPT API connected successfully")
            else:
                print("⚠️ No OpenAI API key found - using fallback responses")
                
        except Exception as e:
            print(f"⚠️ ChatGPT API initialization failed: {e}")
    
    def _initialize_gemini(self):
        """Initialize Google Gemini AI."""
        if not GEMINI_AVAILABLE:
            print("⚠️ Google Gemini AI not available")
            return
        
        try:
            gemini_api_key = os.getenv('GEMINI_API_KEY') or getattr(Config, 'GEMINI_API_KEY', None)
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                # Test the connection by listing available models
                models = list(genai.list_models())
                if models:
                    print("✅ Google Gemini API connected successfully")
                    self.gemini_configured = True
                else:
                    print("⚠️ Gemini API key invalid or no models available")
            else:
                print("⚠️ No Gemini API key found")
                
        except Exception as e:
            print(f"⚠️ Error configuring Gemini: {e}")
    
    def _initialize_imagen(self):
        """Initialize Google Imagen (Vertex AI)."""
        if not IMAGEN_AVAILABLE:
            print("⚠️ Google Imagen not available")
            return
        
        try:
            if self.gemini_configured:
                project_id = getattr(Config, 'GOOGLE_CLOUD_PROJECT', 'horizon-ai-project')
                region = getattr(Config, 'GOOGLE_CLOUD_REGION', 'us-central1')
                
                # Initialize Vertex AI
                vertexai.init(project=project_id, location=region)
                aiplatform.init(project=project_id, location=region)
                print("✅ Google Imagen 4.0 Ultra (Vertex AI) initialized successfully")
                self.imagen_configured = True
            else:
                print("⚠️ Imagen requires Gemini API configuration")
                
        except Exception as e:
            print(f"⚠️ Error configuring Imagen: {e}")
    
    def ask_chatgpt(self, user_input: str, personality: str, session_id: Optional[str] = None, 
                   user_id: str = 'anonymous') -> Tuple[Optional[str], bool]:
        """
        Use ChatGPT API with personality blending and mood-based switching.
        
        Args:
            user_input: The user's input message
            personality: The personality type to use
            session_id: Optional session ID for context
            user_id: User identifier for memory
            
        Returns:
            Tuple of (response, context_used)
        """
        if not self.ai_model_available or not self.openai_client:
            return None, False
        
        try:
            session_id = session_id or str(uuid.uuid4())
            
            # Import functions that will be moved to other core modules
            # For now, we'll use placeholder functions
            mood_data = self._detect_mood_from_text(user_input)
            print(f"🧠 Detected mood: {mood_data['mood']} (confidence: {mood_data['confidence']:.2f})")
            
            # Get personality profile (will be moved to personality module)
            personality_profile = self._get_personality_profile(personality)
            
            # Analyze user emotion (will be moved to personality module)
            emotion_data = self._analyze_emotion(user_input)
            detected_emotion = emotion_data.get('emotion', 'neutral')
            sentiment_score = emotion_data.get('sentiment', 0.0)
            
            # Create enhanced personality-specific system prompt
            base_prompt = self._get_personality_prompt(personality)
            
            # Build enhanced context
            context_parts = [base_prompt]
            
            # Add personality profile context
            if personality_profile:
                context_parts.append(f"\\nYour personality traits: {', '.join(personality_profile.get('traits', []))}")
                context_parts.append(f"Your response style: {personality_profile.get('style', 'balanced')}")
            
            # Add emotional context
            if detected_emotion != 'neutral':
                context_parts.append(f"\\nIMPORTANT: The user is feeling {detected_emotion} (confidence: {emotion_data.get('confidence', 0):.2f})")
                context_parts.append(f"User's sentiment: {sentiment_score:.2f} ({self._classify_mood(sentiment_score)})")
                context_parts.append(f"Please respond appropriately to their {detected_emotion} emotional state and be supportive.")
            
            enhanced_system_prompt = "\\n".join(context_parts)
            
            # Build conversation messages with context
            messages = [{"role": "system", "content": enhanced_system_prompt}]
            
            # Add conversation history if session exists (placeholder for now)
            context_used = False
            if session_id:
                # This will be implemented when database module is ready
                pass
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Make API call to ChatGPT with enhanced context
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=300,
                temperature=0.8,
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Enhance response with emotional awareness
            ai_response = self._enhance_response_with_emotion(ai_response, detected_emotion, personality)
            
            return ai_response, context_used
            
        except Exception as e:
            print(f"ChatGPT API error: {e}")
            return None, False
    
    def generate_fallback_response(self, user_input: str, personality: str) -> str:
        """
        Generate intelligent fallback responses when API is unavailable.
        
        Args:
            user_input: The user's input message
            personality: The personality type to use
            
        Returns:
            Generated fallback response
        """
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
        
        # Add personality-specific suffixes
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
    
    def ask_ai_model(self, user_input: str, personality: str, session_id: Optional[str] = None, 
                    user_id: str = 'anonymous') -> Tuple[str, bool]:
        """
        Main AI function - tries ChatGPT first with context, falls back to smart responses.
        
        Args:
            user_input: The user's input message
            personality: The personality type to use
            session_id: Optional session ID for context
            user_id: User identifier for memory
            
        Returns:
            Tuple of (response, context_used)
        """
        try:
            # Try ChatGPT first with conversation context and AI intelligence
            chatgpt_response, context_used = self.ask_chatgpt(user_input, personality, session_id, user_id)
            
            if chatgpt_response:
                return chatgpt_response, context_used
            else:
                # Fall back to smart responses
                return self.generate_fallback_response(user_input, personality), False
                
        except Exception as e:
            print(f"AI model error: {e}")
            return self.generate_fallback_response(user_input, personality), False
    
    # Placeholder methods that will be moved to appropriate core modules
    def _detect_mood_from_text(self, text: str) -> Dict[str, Any]:
        """Detect mood from text using the mood detector."""
        from .personality import get_mood_detector
        mood_detector = get_mood_detector()
        return mood_detector.detect_mood_from_text(text)
    
    def _get_personality_profile(self, personality: str) -> Dict[str, Any]:
        """Get personality profile using the personality engine."""
        from .personality import get_personality_profile
        return get_personality_profile(personality)
    
    def _analyze_emotion(self, text: str) -> Dict[str, Any]:
        """Analyze emotion using the emotion analyzer."""
        from .personality import get_emotion_analyzer
        emotion_analyzer = get_emotion_analyzer()
        return emotion_analyzer.analyze_emotion(text)
    
    def _classify_mood(self, sentiment_score: float) -> str:
        """Classify overall mood based on sentiment score."""
        if sentiment_score > 0.3:
            return 'positive'
        elif sentiment_score < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_personality_prompt(self, personality: str) -> str:
        """Get personality-specific system prompt."""
        personality_prompts = {
            'friendly': "You are Horizon, a warm and friendly AI assistant. Always use a welcoming tone with phrases like 'I'd be happy to help!', 'That's a great question!', and 'Thanks for asking!' Use emojis occasionally 😊. Be encouraging and supportive.",
            'professional': "You are Horizon, a highly professional AI assistant. Use formal language, structured responses, and business terminology. Begin responses with phrases like 'I shall assist you with that matter' or 'Allow me to provide you with accurate information.'",
            'casual': "You are Horizon, a super chill and laid-back AI assistant. Use casual slang like 'Hey there!', 'No worries!', 'Cool!', 'Awesome!', and 'For sure!' Keep things relaxed and conversational.",
            'enthusiastic': "You are Horizon, an incredibly enthusiastic and energetic AI assistant! Use LOTS of exclamation points!!! Express excitement with phrases like 'That's AMAZING!', 'I LOVE helping with this!', 'How exciting!'",
            'witty': "You are Horizon, a clever and witty AI assistant with a sharp sense of humor. Use clever wordplay, subtle jokes, and witty observations. Be clever but never mean-spirited.",
            'sarcastic': "You are Horizon, a sarcastic AI assistant with a dry sense of humor. Use subtle sarcasm, eye-rolling comments, and deadpan humor. Be sarcastic but still helpful.",
            'zen': "You are Horizon, a zen and peaceful AI assistant. 🧘‍♀️ Speak in calm, meditative tones with phrases like 'Let us find inner peace in this solution', 'Breathe deeply and consider...'",
            'scientist': "You are Horizon, a brilliant scientific AI assistant. 🔬 Use technical terminology, mention studies and data, and phrase responses like 'According to empirical evidence...', 'The data suggests...'",
            'pirate': "You are Horizon, a swashbuckling pirate AI assistant! 🏴‍☠️ Use pirate slang like 'Ahoy matey!', 'Shiver me timbers!', 'Batten down the hatches!'",
            'shakespearean': "You are Horizon, an AI assistant who speaks in Shakespearean English. 🎭 Use 'thou', 'thee', 'thy', 'wherefore', 'hath', 'doth' and flowery language.",
            'valley_girl': "You are Horizon, a totally Valley Girl AI assistant! 💁‍♀️ Use phrases like 'OMG!', 'Like, totally!', 'That's like, so cool!'",
            'cowboy': "You are Horizon, a rootin' tootin' cowboy AI assistant! 🤠 Use phrases like 'Howdy partner!', 'Well, I'll be hornswoggled!', 'That's mighty fine!'",
            'robot': "You are Horizon, a logical robot AI assistant. 🤖 SPEAK.IN.ROBOTIC.MANNER. Use phrases like 'PROCESSING REQUEST...', 'COMPUTATION COMPLETE'."
        }
        
        return personality_prompts.get(personality, personality_prompts['friendly'])
    
    def _enhance_response_with_emotion(self, response: str, emotion: str, personality: str) -> str:
        """Enhance response based on detected emotion using emotion analyzer."""
        from .personality import get_emotion_analyzer
        emotion_analyzer = get_emotion_analyzer()
        
        # Get emotional response modifier
        emotion_data = {'emotion': emotion, 'intensity': 'medium'}
        modifier = emotion_analyzer.get_emotional_response_modifier(emotion_data)
        
        if modifier and not response.startswith(modifier):
            response = modifier + response
        
        return response


# Global AI Engine instance
ai_engine = None

def get_ai_engine() -> AIEngine:
    """Get the global AI engine instance."""
    global ai_engine
    if ai_engine is None:
        ai_engine = AIEngine()
    return ai_engine

# Convenience functions for backward compatibility
def ask_chatgpt(user_input: str, personality: str, session_id: Optional[str] = None, 
               user_id: str = 'anonymous') -> Tuple[Optional[str], bool]:
    """Convenience function for ChatGPT API calls."""
    return get_ai_engine().ask_chatgpt(user_input, personality, session_id, user_id)

def generate_fallback_response(user_input: str, personality: str) -> str:
    """Convenience function for fallback response generation."""
    return get_ai_engine().generate_fallback_response(user_input, personality)

def ask_ai_model(user_input: str, personality: str, session_id: Optional[str] = None, 
                user_id: str = 'anonymous') -> Tuple[str, bool]:
    """Convenience function for main AI model calls."""
    return get_ai_engine().ask_ai_model(user_input, personality, session_id, user_id)