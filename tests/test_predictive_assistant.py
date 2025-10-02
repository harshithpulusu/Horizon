#!/usr/bin/env python3
"""
Comprehensive Tests for Predictive Assistance System
Tests behavioral analysis, pattern recognition, and prediction accuracy
"""

import unittest
import json
import tempfile
import os
import sqlite3
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.predictive_assistant import (
        PredictiveAssistant, PredictionResult, UserPattern,
        analyze_user_behavior, get_predictive_suggestions, provide_prediction_feedback
    )
    from utils.error_handler import ValidationError, AIServiceError
    PREDICTIVE_ASSISTANT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Predictive assistant not available for testing: {e}")
    PREDICTIVE_ASSISTANT_AVAILABLE = False

class TestPredictiveAssistant(unittest.TestCase):
    """Test core predictive assistant functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not PREDICTIVE_ASSISTANT_AVAILABLE:
            self.skipTest("Predictive assistant not available")
        
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize predictive assistant with test database
        self.assistant = PredictiveAssistant(self.temp_db_path)
        
        # Test data
        self.test_user_id = "test_user_predictive"
        self.sample_context = {
            'timestamp': datetime.now().isoformat(),
            'hour': 14,
            'day_of_week': 1,
            'location': 'office'
        }
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def _create_test_conversations(self):
        """Create test conversation data for pattern analysis"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Create conversations table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                input TEXT NOT NULL,
                response TEXT NOT NULL,
                personality TEXT,
                context_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add test conversations with patterns
        test_conversations = [
            (self.test_user_id, "Help with work project", "Sure! I can help.", "professional", "{}"),
            (self.test_user_id, "Schedule meeting", "I'll help schedule that.", "professional", "{}"),
            (self.test_user_id, "What's the weather?", "It's sunny today.", "friendly", "{}"),
            (self.test_user_id, "Need workout tips", "Here are some fitness ideas.", "friendly", "{}"),
            (self.test_user_id, "Project deadline help", "Let's organize your tasks.", "professional", "{}"),
        ]
        
        for conv in test_conversations:
            cursor.execute("""
                INSERT INTO conversations (user_id, input, response, personality, context_data)
                VALUES (?, ?, ?, ?, ?)
            """, conv)
        
        conn.commit()
        conn.close()
    
    def test_database_initialization(self):
        """Test that database tables are created correctly"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'user_patterns', 'prediction_history', 'contextual_triggers',
            'proactive_suggestions', 'temporal_patterns'
        ]
        
        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} should be created")
        
        conn.close()
    
    def test_pattern_analysis(self):
        """Test user behavioral pattern analysis"""
        self._create_test_conversations()
        
        patterns = self.assistant.analyze_user_patterns(self.test_user_id)
        
        # Should find some patterns
        self.assertGreater(len(patterns), 0, "Should discover user patterns")
        
        # Check pattern types
        pattern_types = [p.pattern_type for p in patterns]
        self.assertIn('interaction_style', pattern_types, "Should detect interaction style")
        
        # Check pattern data integrity
        for pattern in patterns:
            self.assertIsInstance(pattern, UserPattern)
            self.assertEqual(pattern.user_id, self.test_user_id)
            self.assertGreaterEqual(pattern.frequency, 0.0)
            self.assertLessEqual(pattern.frequency, 1.0)
            self.assertGreaterEqual(pattern.success_rate, 0.0)
            self.assertLessEqual(pattern.success_rate, 1.0)
    
    def test_temporal_pattern_analysis(self):
        """Test temporal pattern recognition"""
        # Create conversations at specific times
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                input TEXT NOT NULL,
                response TEXT NOT NULL,
                personality TEXT,
                context_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add conversations at 9 AM consistently
        morning_time = datetime.now().replace(hour=9, minute=0, second=0)
        for i in range(5):
            conv_time = morning_time - timedelta(days=i)
            cursor.execute("""
                INSERT INTO conversations (user_id, input, response, timestamp)
                VALUES (?, ?, ?, ?)
            """, (self.test_user_id, "Good morning", "Good morning!", conv_time.isoformat()))
        
        conn.commit()
        conn.close()
        
        patterns = self.assistant.analyze_user_patterns(self.test_user_id)
        
        # Should detect temporal pattern
        temporal_patterns = [p for p in patterns if p.pattern_type == 'temporal']
        self.assertGreater(len(temporal_patterns), 0, "Should detect temporal patterns")
        
        if temporal_patterns:
            pattern = temporal_patterns[0]
            self.assertEqual(pattern.typical_time, "09:00", "Should detect 9 AM pattern")
    
    def test_topic_pattern_analysis(self):
        """Test topic-based pattern recognition"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                input TEXT NOT NULL,
                response TEXT NOT NULL,
                personality TEXT,
                context_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add work-related conversations
        work_conversations = [
            "Help with project management",
            "Meeting scheduling assistance",
            "Work deadline organization", 
            "Task prioritization help",
            "Project status update"
        ]
        
        for conv in work_conversations:
            cursor.execute("""
                INSERT INTO conversations (user_id, input, response)
                VALUES (?, ?, ?)
            """, (self.test_user_id, conv, "I can help with that."))
        
        conn.commit()
        conn.close()
        
        patterns = self.assistant.analyze_user_patterns(self.test_user_id)
        
        # Should detect work topic pattern
        topic_patterns = [p for p in patterns if p.pattern_type == 'topic']
        work_patterns = [p for p in topic_patterns if 'topic_work' in p.context_triggers]
        
        self.assertGreater(len(work_patterns), 0, "Should detect work topic pattern")
    
    def test_prediction_generation(self):
        """Test prediction generation based on patterns"""
        self._create_test_conversations()
        
        # First analyze patterns
        self.assistant.analyze_user_patterns(self.test_user_id)
        
        # Generate predictions
        predictions = self.assistant.predict_user_needs(self.test_user_id, self.sample_context)
        
        # Should generate some predictions
        self.assertIsInstance(predictions, list)
        
        # Check prediction structure
        for pred in predictions:
            self.assertIsInstance(pred, PredictionResult)
            self.assertIn(pred.prediction_type, [
                'temporal_interaction', 'topic_assistance', 'style_adaptation',
                'work_context', 'weather_context', 'weekend_context', 'learning_context',
                'evening_context'
            ])
            self.assertGreaterEqual(pred.confidence, 0.0)
            self.assertLessEqual(pred.confidence, 1.0)
            self.assertIn(pred.urgency, ['low', 'medium', 'high', 'urgent'])
            self.assertIsInstance(pred.suggested_action, str)
            self.assertGreater(len(pred.suggested_action), 0)
    
    def test_context_based_predictions(self):
        """Test predictions based on current context"""
        # Test work context prediction
        work_context = {
            'location': 'office',
            'hour': 14,  # 2 PM
            'day_of_week': 1  # Tuesday
        }
        
        predictions = self.assistant.predict_user_needs(self.test_user_id, work_context)
        
        # Should include work context predictions
        work_predictions = [p for p in predictions if p.prediction_type == 'work_context']
        self.assertGreater(len(work_predictions), 0, "Should predict work context needs")
        
        # Test weekend context
        weekend_context = {
            'hour': 10,
            'day_of_week': 5  # Saturday
        }
        
        predictions = self.assistant.predict_user_needs(self.test_user_id, weekend_context)
        weekend_predictions = [p for p in predictions if p.prediction_type == 'weekend_context']
        self.assertGreater(len(weekend_predictions), 0, "Should predict weekend context needs")
    
    def test_feedback_learning(self):
        """Test learning from user feedback"""
        # Generate a prediction first
        predictions = self.assistant.predict_user_needs(self.test_user_id, self.sample_context)
        
        if predictions:
            prediction_type = predictions[0].prediction_type
            
            # Provide positive feedback
            self.assistant.update_prediction_feedback(
                self.test_user_id, prediction_type, True, "Very helpful!"
            )
            
            # Check that feedback was stored
            conn = sqlite3.connect(self.temp_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT was_accurate, user_feedback FROM prediction_history 
                WHERE user_id = ? AND prediction_type = ?
                ORDER BY created_at DESC LIMIT 1
            """, (self.test_user_id, prediction_type))
            
            result = cursor.fetchone()
            self.assertIsNotNone(result, "Feedback should be stored")
            self.assertTrue(result[0], "Should record positive feedback")
            self.assertEqual(result[1], "Very helpful!", "Should store feedback text")
            
            conn.close()
    
    def test_proactive_suggestions(self):
        """Test proactive suggestion retrieval"""
        # First generate some predictions
        self.assistant.predict_user_needs(self.test_user_id, self.sample_context)
        
        # Get proactive suggestions
        suggestions = self.assistant.get_proactive_suggestions(self.test_user_id)
        
        self.assertIsInstance(suggestions, list)
        
        # Check suggestion structure
        for suggestion in suggestions:
            self.assertIn('type', suggestion)
            self.assertIn('suggestion', suggestion)
            self.assertIn('confidence', suggestion)
            self.assertIn('urgency', suggestion)

class TestPredictiveAssistantAPI(unittest.TestCase):
    """Test predictive assistant API functions"""
    
    def setUp(self):
        """Set up test environment"""
        if not PREDICTIVE_ASSISTANT_AVAILABLE:
            self.skipTest("Predictive assistant not available")
        
        self.test_user_id = "api_test_user"
        self.sample_context = {
            'location': 'home',
            'weather': 'sunny',
            'time': '14:00'
        }
    
    def test_analyze_user_behavior_function(self):
        """Test analyze_user_behavior API function"""
        result = analyze_user_behavior(self.test_user_id, 30)
        
        self.assertIn('status', result)
        self.assertIn('patterns_found', result)
        self.assertIn('analysis_timeframe', result)
        
        if result['status'] == 'success':
            self.assertIsInstance(result['patterns_found'], int)
            self.assertEqual(result['analysis_timeframe'], 30)
    
    def test_get_predictive_suggestions_function(self):
        """Test get_predictive_suggestions API function"""
        result = get_predictive_suggestions(self.test_user_id, self.sample_context)
        
        self.assertIn('status', result)
        self.assertIn('predictions', result)
        self.assertIn('proactive_suggestions', result)
        self.assertIn('context_used', result)
        
        if result['status'] == 'success':
            self.assertIsInstance(result['predictions'], list)
            self.assertIsInstance(result['proactive_suggestions'], list)
            self.assertEqual(result['context_used'], self.sample_context)
    
    def test_provide_prediction_feedback_function(self):
        """Test provide_prediction_feedback API function"""
        result = provide_prediction_feedback(
            self.test_user_id, 'temporal_interaction', True, 'Great suggestion!'
        )
        
        self.assertIn('status', result)
        self.assertIn('message', result)
        
        if result['status'] == 'success':
            self.assertIn('learning_active', result)
            self.assertTrue(result['learning_active'])

class TestPredictiveAssistantPerformance(unittest.TestCase):
    """Test performance aspects of predictive assistance"""
    
    def setUp(self):
        """Set up performance test environment"""
        if not PREDICTIVE_ASSISTANT_AVAILABLE:
            self.skipTest("Predictive assistant not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        self.assistant = PredictiveAssistant(self.temp_db_path)
        self.test_user_id = "performance_test_user"
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def test_pattern_analysis_performance(self):
        """Test that pattern analysis completes within reasonable time"""
        start_time = time.time()
        
        patterns = self.assistant.analyze_user_patterns(self.test_user_id)
        
        analysis_time = time.time() - start_time
        
        # Should complete within 5 seconds even with empty data
        self.assertLess(analysis_time, 5.0, "Pattern analysis should be fast")
    
    def test_prediction_generation_performance(self):
        """Test that prediction generation is fast"""
        context = {'hour': 14, 'day_of_week': 1}
        
        start_time = time.time()
        
        predictions = self.assistant.predict_user_needs(self.test_user_id, context)
        
        prediction_time = time.time() - start_time
        
        # Should complete within 2 seconds
        self.assertLess(prediction_time, 2.0, "Prediction generation should be fast")
    
    def test_concurrent_predictions(self):
        """Test handling multiple concurrent prediction requests"""
        import threading
        
        results = []
        errors = []
        
        def make_prediction():
            try:
                context = {'hour': 14, 'day_of_week': 1}
                result = self.assistant.predict_user_needs(f"user_{threading.current_thread().ident}", context)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should handle concurrent requests without errors
        self.assertEqual(len(errors), 0, "Should handle concurrent requests")
        self.assertEqual(len(results), 5, "Should generate all predictions")

class TestPredictiveAssistantValidation(unittest.TestCase):
    """Test input validation and error handling"""
    
    def setUp(self):
        """Set up validation test environment"""
        if not PREDICTIVE_ASSISTANT_AVAILABLE:
            self.skipTest("Predictive assistant not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        self.assistant = PredictiveAssistant(self.temp_db_path)
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def test_empty_user_id_validation(self):
        """Test validation of empty user ID"""
        # Empty user ID should still work but return limited results
        patterns = self.assistant.analyze_user_patterns("")
        self.assertIsInstance(patterns, list)
        
        predictions = self.assistant.predict_user_needs("", {})
        self.assertIsInstance(predictions, list)
    
    def test_invalid_context_handling(self):
        """Test handling of invalid context data"""
        # None context should work
        predictions = self.assistant.predict_user_needs("test_user", None)
        self.assertIsInstance(predictions, list)
        
        # Invalid context values should be handled gracefully
        invalid_context = {
            'hour': 'invalid',
            'day_of_week': None,
            'malformed_data': {'nested': {'too': {'deep': True}}}
        }
        
        predictions = self.assistant.predict_user_needs("test_user", invalid_context)
        self.assertIsInstance(predictions, list)
    
    def test_database_error_handling(self):
        """Test handling of database errors"""
        # Use non-existent database path
        invalid_assistant = PredictiveAssistant("/invalid/path/db.db")
        
        # Should handle database errors gracefully
        patterns = invalid_assistant.analyze_user_patterns("test_user")
        self.assertIsInstance(patterns, list)

def run_predictive_assistant_tests():
    """Run all predictive assistant tests"""
    print("🧪 Running Predictive Assistant Tests")
    print("=" * 50)
    
    # Test suites
    test_classes = [
        TestPredictiveAssistant,
        TestPredictiveAssistantAPI,
        TestPredictiveAssistantPerformance,
        TestPredictiveAssistantValidation
    ]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        print(f"\n🔍 Running {test_class.__name__}")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_passed += result.testsRun - len(result.failures) - len(result.errors)
        total_failed += len(result.failures) + len(result.errors)
    
    print(f"\n📊 PREDICTIVE ASSISTANT TEST SUMMARY")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
    
    if total_failed == 0:
        print("🎉 All predictive assistant tests passed!")
    else:
        print(f"⚠️ {total_failed} tests failed - check implementation")
    
    return total_failed == 0

if __name__ == "__main__":
    if PREDICTIVE_ASSISTANT_AVAILABLE:
        success = run_predictive_assistant_tests()
        sys.exit(0 if success else 1)
    else:
        print("❌ Predictive assistant not available - skipping tests")
        sys.exit(1)