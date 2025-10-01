#!/usr/bin/env python3
"""
 for Horizon AI
Provides unit tests, integration tests, and test utilities
"""

import unittest
import json
import tempfile
import os
import sqlite3
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import application modules
try:
    from app import app, init_database, ask_chatgpt, create_personality_blend, detect_mood_from_text
    from utils.error_handler import HorizonError, ValidationError, DatabaseError, error_handler
    from config import Config
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    app = None

class HorizonTestCase(unittest.TestCase):
    """Base test case class with common setup and utilities"""
    
    def setUp(self):
        """Set up test environment"""
        if app:
            self.app = app
            self.app.config['TESTING'] = True
            self.client = self.app.test_client()
        
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize test database
        self._init_test_database()
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def _init_test_database(self):
        """Initialize test database with schema"""
        try:
            conn = sqlite3.connect(self.temp_db_path)
            cursor = conn.cursor()
            
            # Create test tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    personality_type TEXT DEFAULT 'general'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS personality_blends (
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
                CREATE TABLE IF NOT EXISTS user_mood_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    mood_state TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    context TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error initializing test database: {e}")
    
    def create_test_conversation(self, message: str = "Test message", response: str = "Test response") -> int:
        """Create a test conversation record"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (user_message, ai_response, session_id)
            VALUES (?, ?, ?)
        ''', (message, response, 'test_session'))
        
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return conversation_id
    
    def create_test_personality_blend(self, personalities: list = None, weights: list = None) -> str:
        """Create a test personality blend"""
        personalities = personalities or ['creative', 'analytical']
        weights = weights or [0.6, 0.4]
        
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        blend_id = f"test_blend_{int(time.time())}"
        
        cursor.execute('''
            INSERT INTO personality_blends 
            (blend_id, personalities, weights, description, effectiveness, context, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            blend_id,
            json.dumps(personalities),
            json.dumps(weights),
            "Test blend",
            0.85,
            "testing",
            "test_user"
        ))
        
        conn.commit()
        conn.close()
        
        return blend_id
    
    def assert_error_response(self, response_data: Dict, error_code: str = None):
        """Assert that response contains error with optional code check"""
        self.assertIn('error', response_data)
        self.assertTrue(response_data['error'])
        self.assertIn('message', response_data)
        self.assertIn('timestamp', response_data)
        
        if error_code:
            self.assertIn('error_code', response_data)
            self.assertEqual(response_data['error_code'], error_code)
    
    def assert_success_response(self, response_data: Dict):
        """Assert that response indicates success"""
        self.assertIn('success', response_data)
        self.assertTrue(response_data['success'])
        self.assertIn('timestamp', response_data)


class TestErrorHandling(HorizonTestCase):
    """Test error handling functionality"""
    
    def test_horizon_error_creation(self):
        """Test HorizonError creation and serialization"""
        error = HorizonError("Test error", "TEST_ERROR", {"detail": "test"})
        
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.error_code, "TEST_ERROR")
        self.assertEqual(error.details["detail"], "test")
        
        error_dict = error.to_dict()
        self.assertTrue(error_dict['error'])
        self.assertEqual(error_dict['error_code'], "TEST_ERROR")
        self.assertEqual(error_dict['message'], "Test error")
    
    def test_error_handler_decorator(self):
        """Test error_handler decorator functionality"""
        
        @error_handler("Test operation failed")
        def test_function_success():
            return "success"
        
        @error_handler("Test operation failed")
        def test_function_error():
            raise ValueError("Test error")
        
        # Test successful execution
        result = test_function_success()
        self.assertEqual(result, "success")
        
        # Test error handling
        with self.assertRaises(HorizonError) as context:
            test_function_error()
        
        self.assertIn("Invalid input provided", str(context.exception))
    
    def test_validation_functions(self):
        """Test validation utility functions"""
        from utils.error_handler import validate_required_fields, validate_field_types
        
        # Test required fields validation
        data = {"field1": "value1", "field2": "value2"}
        
        # Should not raise error
        validate_required_fields(data, ["field1", "field2"])
        
        # Should raise error
        with self.assertRaises(ValidationError):
            validate_required_fields(data, ["field1", "field3"])
        
        # Test field types validation
        data = {"name": "test", "age": 25, "active": True}
        field_types = {"name": str, "age": int, "active": bool}
        
        # Should not raise error
        validate_field_types(data, field_types)
        
        # Should raise error
        field_types["age"] = str
        with self.assertRaises(ValidationError):
            validate_field_types(data, field_types)


class TestDatabaseOperations(HorizonTestCase):
    """Test database operations"""
    
    def test_conversation_storage(self):
        """Test conversation storage and retrieval"""
        conversation_id = self.create_test_conversation("Hello", "Hi there!")
        
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM conversations WHERE id = ?', (conversation_id,))
        result = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "Hello")  # user_message
        self.assertEqual(result[2], "Hi there!")  # ai_response
    
    def test_personality_blend_storage(self):
        """Test personality blend storage and retrieval"""
        blend_id = self.create_test_personality_blend(['creative', 'analytical'], [0.7, 0.3])
        
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM personality_blends WHERE blend_id = ?', (blend_id,))
        result = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(result)
        self.assertEqual(json.loads(result[2]), ['creative', 'analytical'])
        self.assertEqual(json.loads(result[3]), [0.7, 0.3])


class TestPersonalityBlending(HorizonTestCase):
    """Test personality blending functionality"""
    
    @patch('app.sqlite3.connect')
    def test_create_personality_blend_success(self, mock_connect):
        """Test successful personality blend creation"""
        if 'create_personality_blend' not in globals():
            self.skipTest("create_personality_blend function not available")
        
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        result = create_personality_blend(
            personalities=['creative', 'analytical'],
            weights=[0.6, 0.4],
            context='testing',
            user_id='test_user'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('personalities', result)
        self.assertIn('effectiveness', result)
    
    def test_create_personality_blend_validation(self):
        """Test personality blend validation"""
        if 'create_personality_blend' not in globals():
            self.skipTest("create_personality_blend function not available")
        
        # Test with insufficient personalities
        with self.assertRaises(Exception):
            create_personality_blend(
                personalities=['creative'],  # Only one personality
                weights=[1.0],
                context='testing'
            )
    
    @patch('app.openai')
    def test_mood_detection(self, mock_openai):
        """Test mood detection functionality"""
        if 'detect_mood_from_text' not in globals():
            self.skipTest("detect_mood_from_text function not available")
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "mood": "excited",
            "confidence": 0.85,
            "indicators": ["exclamation marks", "positive language"]
        })
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        result = detect_mood_from_text("I'm so excited about this new feature!")
        
        self.assertIsInstance(result, dict)
        self.assertIn('mood', result)
        self.assertIn('confidence', result)


class TestAPIEndpoints(HorizonTestCase):
    """Test API endpoints"""
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
    
    @patch('app.ask_chatgpt')
    def test_chat_endpoint(self, mock_ask_chatgpt):
        """Test chat endpoint"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        mock_ask_chatgpt.return_value = "Hello! How can I help you?"
        
        response = self.client.post('/api/chat', 
            json={'message': 'Hello', 'session_id': 'test'})
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('response', data)
    
    def test_chat_endpoint_validation(self):
        """Test chat endpoint input validation"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        # Test missing message
        response = self.client.post('/api/chat', json={})
        self.assertEqual(response.status_code, 400)
        
        # Test empty message
        response = self.client.post('/api/chat', json={'message': ''})
        self.assertEqual(response.status_code, 400)


class TestUtilityFunctions(HorizonTestCase):
    """Test utility functions"""
    
    def test_safe_json_parse(self):
        """Test safe JSON parsing utility"""
        from utils.error_handler import safe_json_parse
        
        # Test valid JSON
        result = safe_json_parse('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})
        
        # Test invalid JSON with default
        result = safe_json_parse('invalid json', default={})
        self.assertEqual(result, {})
        
        # Test invalid JSON without default
        result = safe_json_parse('invalid json')
        self.assertIsNone(result)


class TestPerformance(HorizonTestCase):
    """Test performance and load handling"""
    
    def test_multiple_conversations(self):
        """Test handling multiple conversations"""
        start_time = time.time()
        
        # Create multiple test conversations
        for i in range(100):
            self.create_test_conversation(f"Message {i}", f"Response {i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (5 seconds for 100 records)
        self.assertLess(duration, 5.0)
        
        # Verify all conversations were created
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM conversations')
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 100)


class TestIntegration(HorizonTestCase):
    """Integration tests for complete workflows"""
    
    @patch('app.ask_chatgpt')
    @patch('app.create_personality_blend')
    def test_personality_chat_workflow(self, mock_blend, mock_chat):
        """Test complete personality-based chat workflow"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        # Mock personality blend creation
        mock_blend.return_value = {
            'personalities': ['creative', 'analytical'],
            'effectiveness': 0.85,
            'description': 'Creative-Analytical blend'
        }
        
        # Mock chat response
        mock_chat.return_value = "Creative analytical response"
        
        # Test workflow: create blend, then chat
        blend_response = self.client.post('/api/personality-blends', json={
            'personalities': ['creative', 'analytical'],
            'weights': [0.6, 0.4],
            'context': 'creative_work'
        })
        
        if blend_response.status_code == 200:
            chat_response = self.client.post('/api/chat', json={
                'message': 'Help me with a creative project',
                'session_id': 'test_integration'
            })
            
            self.assertEqual(chat_response.status_code, 200)


def run_tests():
    """Run all tests and generate report"""
    
    # Create test suite
    test_classes = [
        TestErrorHandling,
        TestDatabaseOperations,
        TestPersonalityBlending,
        TestAPIEndpoints,
        TestUtilityFunctions,
        TestPerformance,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate summary report
    print("\n" + "="*80)
    print("HORIZON AI TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\n')[-2]}")
    
    print("\n" + "="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)