"""
Smart Suggestions API Routes
Safe, modular implementation that doesn't interfere with existing chat functionality.
"""

from flask import Blueprint, request, jsonify
import json
import re
from datetime import datetime
from typing import List, Dict, Any
import os

# Create blueprint for suggestions (separate from existing routes)
suggestions_bp = Blueprint('suggestions', __name__, url_prefix='/api/suggestions')

class SuggestionEngine:
    """
    Pattern-based suggestion engine that learns from user input patterns.
    Completely independent of existing chat functionality.
    """
    
    def __init__(self):
        self.patterns_file = "data/user_patterns.json"
        self.suggestions_cache = {}
        self.common_patterns = [
            "generate an image of",
            "create a video showing",
            "explain how to",
            "what is the weather in",
            "play some music",
            "set a reminder for",
            "analyze this image",
            "summarize this",
            "translate to",
            "convert to"
        ]
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Ensure data directory exists for storing patterns"""
        data_dir = os.path.dirname(self.patterns_file)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
    
    def learn_pattern(self, user_input: str) -> None:
        """
        Learn from user input patterns without affecting existing functionality.
        """
        try:
            # Load existing patterns
            patterns = self.load_patterns()
            
            # Extract meaningful patterns (3+ words)
            words = user_input.lower().strip().split()
            if len(words) >= 3:
                # Create pattern variations
                for i in range(len(words) - 2):
                    pattern = " ".join(words[i:i+3])
                    if len(pattern) > 10:  # Meaningful patterns only
                        patterns[pattern] = patterns.get(pattern, 0) + 1
            
            # Store updated patterns
            self.save_patterns(patterns)
            
        except Exception as e:
            print(f"Pattern learning error (non-critical): {e}")
    
    def load_patterns(self) -> Dict[str, int]:
        """Load user patterns from file"""
        try:
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def save_patterns(self, patterns: Dict[str, int]) -> None:
        """Save user patterns to file"""
        try:
            # Keep only top 1000 patterns to prevent file bloat
            sorted_patterns = dict(sorted(patterns.items(), 
                                        key=lambda x: x[1], reverse=True)[:1000])
            
            with open(self.patterns_file, 'w') as f:
                json.dump(sorted_patterns, f, indent=2)
        except Exception as e:
            print(f"Pattern saving error (non-critical): {e}")
    
    def generate_suggestions(self, current_input: str, limit: int = 5) -> List[str]:
        """
        Generate smart suggestions based on current input and learned patterns.
        """
        try:
            suggestions = []
            current_lower = current_input.lower().strip()
            
            # If input is empty, suggest common patterns
            if not current_lower:
                return self.common_patterns[:limit]
            
            # Load learned patterns
            patterns = self.load_patterns()
            
            # Find matching patterns
            matching_patterns = []
            
            # Exact prefix matches from learned patterns
            for pattern, frequency in patterns.items():
                if pattern.startswith(current_lower):
                    matching_patterns.append((pattern, frequency))
            
            # Sort by frequency and add to suggestions
            matching_patterns.sort(key=lambda x: x[1], reverse=True)
            for pattern, _ in matching_patterns[:limit//2]:
                if pattern not in suggestions:
                    suggestions.append(pattern)
            
            # Add common patterns that match
            for pattern in self.common_patterns:
                if len(suggestions) >= limit:
                    break
                if pattern.startswith(current_lower) and pattern not in suggestions:
                    suggestions.append(pattern)
            
            # Fuzzy matching for partial words
            if len(suggestions) < limit and len(current_lower) > 2:
                for pattern in self.common_patterns:
                    if len(suggestions) >= limit:
                        break
                    if current_lower in pattern and pattern not in suggestions:
                        suggestions.append(pattern)
            
            return suggestions[:limit]
            
        except Exception as e:
            print(f"Suggestion generation error: {e}")
            return self.common_patterns[:limit]

# Initialize suggestion engine
suggestion_engine = SuggestionEngine()

@suggestions_bp.route('/generate', methods=['POST'])
def generate_suggestions():
    """
    Generate smart suggestions based on current input.
    Completely separate from existing /chat endpoint.
    """
    try:
        data = request.get_json()
        current_input = data.get('input', '').strip()
        limit = min(data.get('limit', 5), 10)  # Max 10 suggestions
        
        # Generate suggestions
        suggestions = suggestion_engine.generate_suggestions(current_input, limit)
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'input': current_input,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'suggestions': suggestion_engine.common_patterns[:5]
        }), 500

@suggestions_bp.route('/learn', methods=['POST'])
def learn_pattern():
    """
    Learn from user input patterns to improve future suggestions.
    Safe operation that doesn't affect existing functionality.
    """
    try:
        data = request.get_json()
        user_input = data.get('input', '').strip()
        
        if user_input and len(user_input) > 5:
            # Learn pattern in background
            suggestion_engine.learn_pattern(user_input)
        
        return jsonify({
            'success': True,
            'message': 'Pattern learned successfully',
            'input_length': len(user_input)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@suggestions_bp.route('/patterns', methods=['GET'])
def get_patterns():
    """
    Get current learned patterns for debugging/analysis.
    """
    try:
        patterns = suggestion_engine.load_patterns()
        
        # Get top 20 patterns
        top_patterns = dict(sorted(patterns.items(), 
                                 key=lambda x: x[1], reverse=True)[:20])
        
        return jsonify({
            'success': True,
            'patterns': top_patterns,
            'total_patterns': len(patterns),
            'common_patterns': suggestion_engine.common_patterns
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@suggestions_bp.route('/clear', methods=['POST'])
def clear_patterns():
    """
    Clear learned patterns (reset suggestion learning).
    """
    try:
        if os.path.exists(suggestion_engine.patterns_file):
            os.remove(suggestion_engine.patterns_file)
        
        return jsonify({
            'success': True,
            'message': 'Patterns cleared successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@suggestions_bp.route('/health', methods=['GET'])
def suggestions_health():
    """
    Health check for suggestions system.
    """
    try:
        patterns = suggestion_engine.load_patterns()
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'patterns_count': len(patterns),
            'common_patterns_count': len(suggestion_engine.common_patterns),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e)
        }), 500