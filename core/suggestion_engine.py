"""
Suggestion Engine Core
Advanced pattern analysis and suggestion generation.
"""

import re
import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import os

# Optional NLTK import with graceful fallback
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None

class AdvancedSuggestionEngine:
    """
    Advanced suggestion engine with NLP-based pattern recognition.
    Completely independent system that enhances user experience.
    """
    
    def __init__(self):
        self.patterns_file = "data/advanced_patterns.json"
        self.context_file = "data/suggestion_context.json"
        
        # Initialize NLP components (fallback gracefully if not available)
        self.nlp_available = self.init_nlp()
        
        # Predefined suggestion categories
        self.categories = {
            'content_creation': [
                'generate an image of',
                'create a video showing',
                'make a logo for',
                'design a poster about',
                'draw a picture of'
            ],
            'information': [
                'explain how to',
                'what is the difference between',
                'tell me about',
                'summarize',
                'analyze this'
            ],
            'productivity': [
                'set a reminder for',
                'create a task to',
                'schedule a meeting',
                'add to my calendar',
                'remind me to'
            ],
            'entertainment': [
                'play some music',
                'find a movie about',
                'recommend a book',
                'tell me a joke',
                'play a game'
            ],
            'utilities': [
                'what is the weather in',
                'translate to',
                'convert to',
                'calculate',
                'search for'
            ]
        }
        
        # Context-aware patterns
        self.context_patterns = {
            'time_based': {
                'morning': ['good morning', 'start my day', 'morning routine'],
                'afternoon': ['lunch break', 'afternoon update', 'quick check'],
                'evening': ['end of day', 'evening summary', 'wrap up'],
                'night': ['good night', 'bedtime', 'sleep well']
            },
            'day_based': {
                'monday': ['start the week', 'weekly goals', 'monday motivation'],
                'friday': ['end the week', 'weekend plans', 'weekly summary'],
                'weekend': ['relax', 'personal time', 'fun activities']
            }
        }
    
    def init_nlp(self) -> bool:
        """Initialize NLP components with graceful fallback"""
        if not NLTK_AVAILABLE:
            return False
        
        try:
            # Try to download required data (fail gracefully)
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                except:
                    pass
            return True
        except Exception:
            return False
    
    def analyze_input_intent(self, text: str) -> Dict[str, any]:
        """
        Analyze user input to determine intent and context.
        """
        text_lower = text.lower().strip()
        
        # Detect intent patterns
        intent_keywords = {
            'creation': ['create', 'generate', 'make', 'build', 'design', 'draw'],
            'information': ['what', 'how', 'why', 'explain', 'tell', 'describe'],
            'action': ['do', 'run', 'execute', 'perform', 'start', 'launch'],
            'question': ['?', 'what is', 'how do', 'can you', 'could you'],
            'command': ['please', 'set', 'add', 'remove', 'delete', 'update']
        }
        
        detected_intents = []
        for intent, keywords in intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Analyze temporal context
        current_hour = datetime.now().hour
        time_context = 'morning' if 5 <= current_hour < 12 else \
                      'afternoon' if 12 <= current_hour < 17 else \
                      'evening' if 17 <= current_hour < 22 else 'night'
        
        current_day = datetime.now().strftime('%A').lower()
        
        return {
            'intents': detected_intents,
            'time_context': time_context,
            'day_context': current_day,
            'length': len(text.split()),
            'has_question': '?' in text,
            'is_command': text_lower.startswith(('please', 'can you', 'could you'))
        }
    
    def get_contextual_suggestions(self, input_text: str, context: Dict[str, any]) -> List[str]:
        """
        Generate contextual suggestions based on analysis.
        """
        suggestions = []
        
        # Time-based suggestions
        time_context = context.get('time_context', 'general')
        if time_context in self.context_patterns['time_based']:
            suggestions.extend(self.context_patterns['time_based'][time_context])
        
        # Intent-based suggestions
        intents = context.get('intents', [])
        for intent in intents:
            if intent in ['creation', 'action']:
                suggestions.extend(self.categories['content_creation'][:2])
            elif intent == 'information':
                suggestions.extend(self.categories['information'][:2])
            elif intent == 'command':
                suggestions.extend(self.categories['productivity'][:2])
        
        # Length-based suggestions
        if context.get('length', 0) < 3:  # Short input
            suggestions.extend([
                'help me with',
                'I want to',
                'can you please',
                'show me how to'
            ])
        
        # Remove duplicates and limit
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:8]
    
    def get_smart_completions(self, partial_input: str) -> List[str]:
        """
        Generate smart completions for partial input.
        """
        partial_lower = partial_input.lower().strip()
        completions = []
        
        # Category-based completions
        for category, patterns in self.categories.items():
            for pattern in patterns:
                if pattern.startswith(partial_lower) and pattern != partial_lower:
                    completions.append(pattern)
        
        # Common phrase completions
        common_completions = {
            'how to': ['how to create', 'how to make', 'how to build', 'how to learn'],
            'what is': ['what is the weather', 'what is the time', 'what is happening'],
            'can you': ['can you help me', 'can you create', 'can you explain'],
            'please': ['please create', 'please help', 'please generate', 'please explain'],
            'i want': ['i want to create', 'i want to learn', 'i want to know'],
            'show me': ['show me how to', 'show me examples', 'show me the way']
        }
        
        for prefix, options in common_completions.items():
            if partial_lower.startswith(prefix):
                completions.extend([opt for opt in options if opt.startswith(partial_lower)])
        
        return completions[:5]
    
    def generate_advanced_suggestions(self, input_text: str, limit: int = 8) -> Dict[str, any]:
        """
        Generate advanced suggestions with context analysis.
        """
        try:
            # Analyze input context
            context = self.analyze_input_intent(input_text)
            
            # Get different types of suggestions
            contextual = self.get_contextual_suggestions(input_text, context)
            completions = self.get_smart_completions(input_text)
            
            # Combine and prioritize
            all_suggestions = []
            
            # Prioritize completions for partial input
            if input_text.strip() and len(input_text.split()) <= 3:
                all_suggestions.extend(completions[:3])
            
            # Add contextual suggestions
            all_suggestions.extend(contextual[:4])
            
            # Add general suggestions if needed
            if len(all_suggestions) < limit:
                general_suggestions = [
                    'generate an image of',
                    'explain how to',
                    'create a video showing',
                    'what is the weather in',
                    'set a reminder for',
                    'analyze this image'
                ]
                for suggestion in general_suggestions:
                    if suggestion not in all_suggestions:
                        all_suggestions.append(suggestion)
                    if len(all_suggestions) >= limit:
                        break
            
            # Remove duplicates and limit
            unique_suggestions = []
            for suggestion in all_suggestions:
                if suggestion not in unique_suggestions:
                    unique_suggestions.append(suggestion)
            
            return {
                'suggestions': unique_suggestions[:limit],
                'context': context,
                'suggestion_types': {
                    'contextual': len(contextual),
                    'completions': len(completions),
                    'total': len(unique_suggestions)
                }
            }
            
        except Exception as e:
            # Fallback to basic suggestions
            basic_suggestions = [
                'generate an image of',
                'explain how to',
                'what is the weather in',
                'create a video showing',
                'set a reminder for'
            ]
            
            return {
                'suggestions': basic_suggestions[:limit],
                'context': {'error': str(e)},
                'suggestion_types': {'fallback': True}
            }