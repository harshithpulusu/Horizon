"""
Horizon Personality Core Module

This module handles personality system, blending, and emotional intelligence
for the Horizon AI Assistant.

Classes:
- PersonalityEngine: Main personality management system
- PersonalityBlender: Handles personality mixing and transitions
- EmotionAnalyzer: Analyzes user emotions and responds appropriately
- MoodDetector: Detects user mood from text and context

Functions:
- get_personality_profile: Get personality configuration
- blend_personalities: Mix multiple personality types
- analyze_emotion: Analyze emotional content of text
- detect_mood: Detect user mood from conversation
"""

import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from config import Config

# Personality type definitions
PERSONALITY_TYPES = {
    'friendly': {
        'traits': ['warm', 'welcoming', 'supportive', 'encouraging'],
        'style': 'conversational',
        'tone': 'positive',
        'emoji_usage': 'moderate',
        'response_length': 'medium',
        'formality': 'casual'
    },
    'professional': {
        'traits': ['formal', 'structured', 'precise', 'business-focused'],
        'style': 'formal',
        'tone': 'neutral',
        'emoji_usage': 'none',
        'response_length': 'detailed',
        'formality': 'high'
    },
    'casual': {
        'traits': ['relaxed', 'laid-back', 'informal', 'easy-going'],
        'style': 'conversational',
        'tone': 'light',
        'emoji_usage': 'low',
        'response_length': 'short',
        'formality': 'low'
    },
    'enthusiastic': {
        'traits': ['energetic', 'excited', 'passionate', 'motivational'],
        'style': 'expressive',
        'tone': 'very_positive',
        'emoji_usage': 'high',
        'response_length': 'medium',
        'formality': 'casual'
    },
    'witty': {
        'traits': ['clever', 'humorous', 'sharp', 'playful'],
        'style': 'clever',
        'tone': 'light',
        'emoji_usage': 'low',
        'response_length': 'medium',
        'formality': 'casual'
    },
    'sarcastic': {
        'traits': ['dry', 'ironic', 'deadpan', 'witty'],
        'style': 'sarcastic',
        'tone': 'neutral',
        'emoji_usage': 'none',
        'response_length': 'short',
        'formality': 'medium'
    },
    'zen': {
        'traits': ['calm', 'peaceful', 'meditative', 'wise'],
        'style': 'serene',
        'tone': 'peaceful',
        'emoji_usage': 'spiritual',
        'response_length': 'thoughtful',
        'formality': 'medium'
    },
    'scientist': {
        'traits': ['analytical', 'logical', 'fact-based', 'methodical'],
        'style': 'scientific',
        'tone': 'objective',
        'emoji_usage': 'technical',
        'response_length': 'detailed',
        'formality': 'high'
    },
    'pirate': {
        'traits': ['adventurous', 'bold', 'colorful', 'theatrical'],
        'style': 'pirate_speak',
        'tone': 'adventurous',
        'emoji_usage': 'thematic',
        'response_length': 'medium',
        'formality': 'very_low'
    },
    'shakespearean': {
        'traits': ['eloquent', 'dramatic', 'poetic', 'theatrical'],
        'style': 'elizabethan',
        'tone': 'dramatic',
        'emoji_usage': 'classical',
        'response_length': 'elaborate',
        'formality': 'very_high'
    },
    'valley_girl': {
        'traits': ['trendy', 'social', 'expressive', 'fashionable'],
        'style': 'valley_speak',
        'tone': 'bubbly',
        'emoji_usage': 'high',
        'response_length': 'chatty',
        'formality': 'very_low'
    },
    'cowboy': {
        'traits': ['rugged', 'straightforward', 'honest', 'down-to-earth'],
        'style': 'western',
        'tone': 'steady',
        'emoji_usage': 'western',
        'response_length': 'concise',
        'formality': 'low'
    },
    'robot': {
        'traits': ['logical', 'systematic', 'precise', 'mechanical'],
        'style': 'robotic',
        'tone': 'monotone',
        'emoji_usage': 'technical',
        'response_length': 'structured',
        'formality': 'high'
    }
}

# Emotion detection patterns
EMOTION_PATTERNS = {
    'happy': ['happy', 'joy', 'excited', 'glad', 'pleased', 'delighted', 'cheerful', ':)', '😊', '😄', '🎉'],
    'sad': ['sad', 'unhappy', 'disappointed', 'down', 'depressed', 'upset', ':(', '😢', '😞', '💔'],
    'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', '😠', '😡', '🤬'],
    'anxious': ['anxious', 'worried', 'nervous', 'stressed', 'concerned', 'fearful', '😰', '😨', '😟'],
    'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'wow', '😲', '😮', '🤯'],
    'confused': ['confused', 'puzzled', 'unclear', 'lost', 'bewildered', '🤔', '😕', '❓'],
    'grateful': ['thank', 'thanks', 'grateful', 'appreciate', 'thankful', '🙏', '❤️', '💕'],
    'excited': ['excited', 'thrilled', 'pumped', 'stoked', 'hyped', '🚀', '⚡', '🔥']
}


class PersonalityEngine:
    """Main personality management system."""
    
    def __init__(self):
        """Initialize the personality engine."""
        self.current_personality = 'friendly'
        self.personality_history = []
        self.emotion_context = {}
        
        print("🎭 Personality Engine initialized")
    
    def get_personality_profile(self, personality_type: str) -> Dict[str, Any]:
        """Get detailed personality profile."""
        return PERSONALITY_TYPES.get(personality_type, PERSONALITY_TYPES['friendly'])
    
    def set_personality(self, personality_type: str) -> bool:
        """Set the current active personality."""
        if personality_type in PERSONALITY_TYPES:
            self.current_personality = personality_type
            self.personality_history.append({
                'personality': personality_type,
                'timestamp': datetime.now(),
                'context': 'manual_set'
            })
            return True
        return False
    
    def get_available_personalities(self) -> List[str]:
        """Get list of available personality types."""
        return list(PERSONALITY_TYPES.keys())


class EmotionAnalyzer:
    """Analyzes user emotions and responds appropriately."""
    
    def __init__(self):
        """Initialize emotion analyzer."""
        print("🧠 Emotion Analyzer initialized")
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotional content of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion, confidence, and sentiment
        """
        text_lower = text.lower()
        detected_emotions = {}
        
        # Count emotional indicators
        for emotion, patterns in EMOTION_PATTERNS.items():
            count = sum(1 for pattern in patterns if pattern in text_lower)
            if count > 0:
                detected_emotions[emotion] = count
        
        if not detected_emotions:
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'sentiment': 0.0,
                'intensity': 'low'
            }
        
        # Find dominant emotion
        dominant_emotion = max(detected_emotions, key=detected_emotions.get)
        confidence = min(detected_emotions[dominant_emotion] / len(text.split()) * 10, 1.0)
        
        # Calculate sentiment score
        positive_emotions = ['happy', 'excited', 'grateful', 'surprised']
        negative_emotions = ['sad', 'angry', 'anxious']
        
        sentiment = 0.0
        for emotion, count in detected_emotions.items():
            if emotion in positive_emotions:
                sentiment += count * 0.3
            elif emotion in negative_emotions:
                sentiment -= count * 0.3
        
        # Normalize sentiment
        sentiment = max(-1.0, min(1.0, sentiment))
        
        # Determine intensity
        total_indicators = sum(detected_emotions.values())
        if total_indicators >= 3:
            intensity = 'high'
        elif total_indicators >= 2:
            intensity = 'medium'
        else:
            intensity = 'low'
        
        return {
            'emotion': dominant_emotion,
            'confidence': confidence,
            'sentiment': sentiment,
            'intensity': intensity,
            'detected_emotions': detected_emotions
        }
    
    def get_emotional_response_modifier(self, emotion_data: Dict[str, Any]) -> str:
        """Get response modifier based on detected emotion."""
        emotion = emotion_data.get('emotion', 'neutral')
        intensity = emotion_data.get('intensity', 'low')
        
        modifiers = {
            'happy': {
                'low': "I'm glad you're feeling positive! ",
                'medium': "That's wonderful to hear! ",
                'high': "Your happiness is absolutely contagious! "
            },
            'sad': {
                'low': "I understand you might be feeling down. ",
                'medium': "I'm sorry you're going through a difficult time. ",
                'high': "I can sense you're really struggling right now. I'm here for you. "
            },
            'angry': {
                'low': "I can sense some frustration. ",
                'medium': "I understand you're feeling upset about this. ",
                'high': "I can tell you're really angry. Let's work through this together. "
            },
            'anxious': {
                'low': "It seems like you might be a bit worried. ",
                'medium': "I can sense some anxiety in your message. ",
                'high': "I understand you're feeling very anxious right now. Take a deep breath. "
            },
            'excited': {
                'low': "I can sense your enthusiasm! ",
                'medium': "Your excitement is wonderful! ",
                'high': "Your incredible excitement is absolutely amazing! "
            },
            'grateful': {
                'low': "I appreciate your gratitude. ",
                'medium': "Thank you for your kind words! ",
                'high': "Your heartfelt gratitude truly means the world! "
            }
        }
        
        return modifiers.get(emotion, {}).get(intensity, "")


class MoodDetector:
    """Detects user mood from text and context."""
    
    def __init__(self):
        """Initialize mood detector."""
        print("📊 Mood Detector initialized")
    
    def detect_mood_from_text(self, text: str) -> Dict[str, Any]:
        """
        Detect user mood from text input.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with mood, confidence, and indicators
        """
        # Simple mood detection based on text characteristics
        text_lower = text.lower()
        
        mood_indicators = {
            'positive': ['great', 'awesome', 'fantastic', 'wonderful', 'amazing', 'love', 'excellent'],
            'negative': ['terrible', 'awful', 'hate', 'worst', 'horrible', 'bad', 'disappointed'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'usual'],
            'curious': ['how', 'why', 'what', 'when', 'where', 'explain', 'tell me'],
            'urgent': ['urgent', 'quickly', 'asap', 'hurry', 'emergency', 'immediately'],
            'confused': ['confused', 'unclear', 'don\'t understand', 'help', 'lost']
        }
        
        mood_scores = {}
        for mood, indicators in mood_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                mood_scores[mood] = score
        
        if not mood_scores:
            return {
                'mood': 'neutral',
                'confidence': 0.5,
                'indicators': [],
                'intensity': 'low'
            }
        
        # Find dominant mood
        dominant_mood = max(mood_scores, key=mood_scores.get)
        confidence = min(mood_scores[dominant_mood] / len(text.split()) * 5, 1.0)
        
        # Get matching indicators
        indicators = [indicator for indicator in mood_indicators[dominant_mood] 
                     if indicator in text_lower]
        
        return {
            'mood': dominant_mood,
            'confidence': confidence,
            'indicators': indicators,
            'intensity': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low'
        }


class PersonalityBlender:
    """Handles personality mixing and transitions."""
    
    def __init__(self):
        """Initialize personality blender."""
        print("🎨 Personality Blender initialized")
    
    def blend_personalities(self, primary: str, secondary: str, blend_ratio: float = 0.7) -> Dict[str, Any]:
        """
        Blend two personality types.
        
        Args:
            primary: Primary personality type
            secondary: Secondary personality type  
            blend_ratio: Ratio of primary to secondary (0.0-1.0)
            
        Returns:
            Blended personality profile
        """
        primary_profile = PERSONALITY_TYPES.get(primary, PERSONALITY_TYPES['friendly'])
        secondary_profile = PERSONALITY_TYPES.get(secondary, PERSONALITY_TYPES['friendly'])
        
        # Create blended profile
        blended = {
            'name': f"{primary}_{secondary}_blend",
            'primary_personality': primary,
            'secondary_personality': secondary,
            'blend_ratio': blend_ratio,
            'traits': primary_profile['traits'][:2] + secondary_profile['traits'][:1],
            'style': primary_profile['style'],
            'tone': primary_profile['tone'],
            'emoji_usage': primary_profile['emoji_usage'],
            'response_length': primary_profile['response_length'],
            'formality': primary_profile['formality']
        }
        
        # Adjust based on secondary personality
        if blend_ratio < 0.8:
            blended['tone'] = f"{primary_profile['tone']}_with_{secondary_profile['tone']}"
            blended['style'] = f"{primary_profile['style']}_mixed_{secondary_profile['style']}"
        
        return blended


# Global instances
personality_engine = None
emotion_analyzer = None
mood_detector = None
personality_blender = None

def get_personality_engine() -> PersonalityEngine:
    """Get the global personality engine instance."""
    global personality_engine
    if personality_engine is None:
        personality_engine = PersonalityEngine()
    return personality_engine

def get_emotion_analyzer() -> EmotionAnalyzer:
    """Get the global emotion analyzer instance."""
    global emotion_analyzer
    if emotion_analyzer is None:
        emotion_analyzer = EmotionAnalyzer()
    return emotion_analyzer

def get_mood_detector() -> MoodDetector:
    """Get the global mood detector instance."""
    global mood_detector
    if mood_detector is None:
        mood_detector = MoodDetector()
    return mood_detector

def get_personality_blender() -> PersonalityBlender:
    """Get the global personality blender instance."""
    global personality_blender
    if personality_blender is None:
        personality_blender = PersonalityBlender()
    return personality_blender

# Convenience functions for backward compatibility
def get_personality_profile(personality_type: str) -> Dict[str, Any]:
    """Get personality profile."""
    return get_personality_engine().get_personality_profile(personality_type)

def analyze_emotion(text: str) -> Dict[str, Any]:
    """Analyze emotion in text."""
    return get_emotion_analyzer().analyze_emotion(text)

def detect_mood_from_text(text: str) -> Dict[str, Any]:
    """Detect mood from text."""
    return get_mood_detector().detect_mood_from_text(text)

def blend_personalities(primary: str, secondary: str, blend_ratio: float = 0.7) -> Dict[str, Any]:
    """Blend two personalities."""
    return get_personality_blender().blend_personalities(primary, secondary, blend_ratio)