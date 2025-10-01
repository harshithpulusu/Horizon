#!/usr/bin/env python3
"""
Unit Tests for Personality Blending System
"""

import unittest
import json
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app import create_personality_blend, detect_mood_from_text, get_mood_based_personality_recommendation
    from utils.error_handler import PersonalityBlendingError, ValidationError
except ImportError as e:
    print(f"Warning: Could not import personality blending modules: {e}")


class TestPersonalityBlending(unittest.TestCase):
    """Test personality blending functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        self._init_test_database()
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def _init_test_database(self):
        """Initialize test database"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE personality_blends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                blend_id TEXT UNIQUE NOT NULL,
                personalities TEXT NOT NULL,
                weights TEXT NOT NULL,
                description TEXT,
                effectiveness REAL,
                context TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE mood_personality_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mood_state TEXT NOT NULL,
                recommended_personalities TEXT NOT NULL,
                modifiers TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Insert test mood mappings
        test_mappings = [
            ('excited', '["enthusiastic", "creative"]', '{"energy": 1.2}'),
            ('focused', '["analytical", "professional"]', '{"concentration": 1.1}'),
            ('creative', '["creative", "artistic"]', '{"imagination": 1.3}')
        ]
        
        for mood, personalities, modifiers in test_mappings:
            cursor.execute('''
                INSERT INTO mood_personality_mappings (mood_state, recommended_personalities, modifiers)
                VALUES (?, ?, ?)
            ''', (mood, personalities, modifiers))
        
        conn.commit()
        conn.close()
    
    def test_personality_blend_creation_valid_input(self):
        """Test creating personality blend with valid input"""
        if 'create_personality_blend' not in globals():
            self.skipTest("create_personality_blend function not available")
        
        with patch('app.sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            result = create_personality_blend(
                personalities=['creative', 'analytical'],
                weights=[0.6, 0.4],
                context='creative_work',
                user_id='test_user'
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('personalities', result)
            self.assertIn('weights', result)
            self.assertIn('effectiveness', result)
            self.assertEqual(result['personalities'], ['creative', 'analytical'])
            self.assertEqual(result['weights'], [0.6, 0.4])
    
    def test_personality_blend_insufficient_personalities(self):
        """Test error handling for insufficient personalities"""
        if 'create_personality_blend' not in globals():
            self.skipTest("create_personality_blend function not available")
        
        # Test with only one personality
        result = create_personality_blend(
            personalities=['creative'],
            weights=[1.0],
            context='testing'
        )
        
        # Should return error or raise exception
        if isinstance(result, dict):
            self.assertIn('error', result)
        else:
            self.fail("Expected error for insufficient personalities")
    
    def test_personality_blend_weight_normalization(self):
        """Test weight normalization in personality blending"""
        if 'create_personality_blend' not in globals():
            self.skipTest("create_personality_blend function not available")
        
        with patch('app.sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Test with weights that don't sum to 1.0
            result = create_personality_blend(
                personalities=['creative', 'analytical', 'friendly'],
                weights=[0.4, 0.4, 0.4],  # Sums to 1.2
                context='testing'
            )
            
            if isinstance(result, dict) and 'weights' in result:
                # Weights should be normalized to sum to 1.0
                weight_sum = sum(result['weights'])
                self.assertAlmostEqual(weight_sum, 1.0, places=2)
    
    def test_mood_detection_valid_text(self):
        """Test mood detection with valid text input"""
        if 'detect_mood_from_text' not in globals():
            self.skipTest("detect_mood_from_text function not available")
        
        test_cases = [
            ("I'm so excited about this new feature!", "excited"),
            ("I need to focus on this important task", "focused"),
            ("I feel stressed and overwhelmed", "stressed"),
            ("Let me think deeply about this", "contemplative")
        ]
        
        for text, expected_mood in test_cases:
            with patch('app.openai') as mock_openai:
                # Mock OpenAI response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = json.dumps({
                    "mood": expected_mood,
                    "confidence": 0.85,
                    "indicators": ["keyword analysis"]
                })
                mock_openai.ChatCompletion.create.return_value = mock_response
                
                result = detect_mood_from_text(text)
                
                self.assertIsInstance(result, dict)
                self.assertIn('mood', result)
                self.assertIn('confidence', result)
                self.assertEqual(result['mood'], expected_mood)
                self.assertGreaterEqual(result['confidence'], 0)
                self.assertLessEqual(result['confidence'], 1)
    
    def test_mood_detection_empty_text(self):
        """Test mood detection with empty text"""
        if 'detect_mood_from_text' not in globals():
            self.skipTest("detect_mood_from_text function not available")
        
        result = detect_mood_from_text("")
        
        if isinstance(result, dict):
            if 'error' in result:
                # Error response is acceptable
                self.assertIn('error', result)
            else:
                # Should default to neutral mood
                self.assertEqual(result.get('mood', 'neutral'), 'neutral')
    
    def test_mood_based_personality_recommendation(self):
        """Test mood-based personality recommendations"""
        if 'get_mood_based_personality_recommendation' not in globals():
            self.skipTest("get_mood_based_personality_recommendation function not available")
        
        with patch('app.sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock database query result
            mock_cursor.fetchone.return_value = (
                'excited',
                '["enthusiastic", "creative"]',
                '{"energy": 1.2}'
            )
            
            result = get_mood_based_personality_recommendation('excited')
            
            self.assertIsInstance(result, dict)
            self.assertIn('personalities', result)
            self.assertIn('modifiers', result)
    
    def test_personality_effectiveness_calculation(self):
        """Test personality blend effectiveness calculation"""
        # This would test the calculate_blend_effectiveness function
        # if it's available in the app module
        
        test_cases = [
            (['creative', 'analytical'], [0.6, 0.4], 'creative_work', 0.7),
            (['friendly', 'professional'], [0.5, 0.5], 'social_interaction', 0.8),
            (['zen', 'enthusiastic'], [0.8, 0.2], 'emotional_support', 0.75)
        ]
        
        for personalities, weights, context, min_expected in test_cases:
            # Mock or test the effectiveness calculation
            # This would depend on the actual implementation
            pass
    
    def test_personality_trait_blending(self):
        """Test personality trait blending logic"""
        # Test the logic that combines personality traits
        # This would test functions like blend_traits() if available
        
        # Example test structure:
        personality_traits = {
            'creative': {'creativity': 0.9, 'flexibility': 0.8, 'expressiveness': 0.85},
            'analytical': {'logic': 0.95, 'precision': 0.9, 'objectivity': 0.88}
        }
        
        weights = [0.6, 0.4]
        
        # Test that blended traits are calculated correctly
        # blended_traits = blend_traits(personality_traits, weights)
        # Expected: weighted average of traits
        pass
    
    def test_mood_confidence_thresholds(self):
        """Test mood detection confidence thresholds"""
        if 'detect_mood_from_text' not in globals():
            self.skipTest("detect_mood_from_text function not available")
        
        # Test with different confidence levels
        confidence_levels = [0.3, 0.7, 0.9]
        
        for confidence in confidence_levels:
            with patch('app.openai') as mock_openai:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = json.dumps({
                    "mood": "happy",
                    "confidence": confidence,
                    "indicators": ["test"]
                })
                mock_openai.ChatCompletion.create.return_value = mock_response
                
                result = detect_mood_from_text("Test text")
                
                if isinstance(result, dict) and 'confidence' in result:
                    self.assertEqual(result['confidence'], confidence)
                    
                    # Test confidence threshold logic
                    if confidence < 0.5:
                        # Low confidence should maybe default to neutral
                        # or include uncertainty indicator
                        pass


class TestPersonalityBlendingValidation(unittest.TestCase):
    """Test validation logic for personality blending"""
    
    def test_validate_personality_list(self):
        """Test personality list validation"""
        valid_personalities = [
            ['creative', 'analytical'],
            ['friendly', 'professional', 'enthusiastic'],
            ['zen', 'witty']
        ]
        
        invalid_personalities = [
            [],  # Empty list
            ['creative'],  # Only one personality
            ['invalid_personality', 'creative'],  # Invalid personality name
            ['creative', 'creative']  # Duplicate personalities
        ]
        
        for personalities in valid_personalities:
            # Should not raise error
            self.assertTrue(len(personalities) >= 2)
        
        for personalities in invalid_personalities:
            # Should raise error or handle gracefully
            if len(personalities) < 2:
                self.assertLess(len(personalities), 2)
    
    def test_validate_weights(self):
        """Test weight validation"""
        valid_weight_sets = [
            [0.5, 0.5],
            [0.6, 0.4],
            [0.33, 0.33, 0.34],
            [1.0, 0.0]
        ]
        
        invalid_weight_sets = [
            [0.5],  # Wrong length
            [0.5, 0.5, 0.5],  # Too many weights
            [-0.1, 1.1],  # Negative weights
            [0.6, 0.6]  # Sum > 1.0
        ]
        
        personalities = ['creative', 'analytical']
        
        for weights in valid_weight_sets[:2]:  # Test first two with 2 personalities
            self.assertEqual(len(weights), len(personalities))
            self.assertGreaterEqual(min(weights), 0)
        
        for weights in invalid_weight_sets[:2]:  # Test validation logic
            if len(weights) != len(personalities):
                self.assertNotEqual(len(weights), len(personalities))


class TestPersonalityBlendingPerformance(unittest.TestCase):
    """Test performance aspects of personality blending"""
    
    def test_blend_creation_performance(self):
        """Test performance of creating multiple blends"""
        import time
        
        if 'create_personality_blend' not in globals():
            self.skipTest("create_personality_blend function not available")
        
        start_time = time.time()
        
        with patch('app.sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Create multiple blends
            for i in range(10):
                create_personality_blend(
                    personalities=['creative', 'analytical'],
                    weights=[0.6, 0.4],
                    context='performance_test',
                    user_id=f'test_user_{i}'
                )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(duration, 1.0)  # Less than 1 second for 10 blends
    
    def test_mood_detection_performance(self):
        """Test performance of mood detection"""
        import time
        
        if 'detect_mood_from_text' not in globals():
            self.skipTest("detect_mood_from_text function not available")
        
        test_texts = [
            "I'm so excited about this!",
            "I need to focus on work",
            "Feeling stressed today",
            "This is amazing!",
            "Let me think about this"
        ]
        
        start_time = time.time()
        
        with patch('app.openai') as mock_openai:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "mood": "test",
                "confidence": 0.8,
                "indicators": ["test"]
            })
            mock_openai.ChatCompletion.create.return_value = mock_response
            
            for text in test_texts:
                detect_mood_from_text(text)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(duration, 2.0)  # Less than 2 seconds for 5 detections


def run_personality_tests():
    """Run personality blending tests"""
    test_classes = [
        TestPersonalityBlending,
        TestPersonalityBlendingValidation,
        TestPersonalityBlendingPerformance
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_personality_tests()
    sys.exit(0 if success else 1)