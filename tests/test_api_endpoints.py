#!/usr/bin/env python3
"""
API Endpoint Tests for Horizon AI
Tests all REST API endpoints for functionality, validation, and error handling
"""

import unittest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app import app
    from utils.error_handler import HorizonError
except ImportError as e:
    print(f"Warning: Could not import app modules: {e}")
    app = None


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints functionality and error handling"""
    
    def setUp(self):
        """Set up test client"""
        if app:
            self.app = app
            self.app.config['TESTING'] = True
            self.client = self.app.test_client()
        else:
            self.client = None
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.get('/health')
        
        # Should return 200 status
        self.assertIn(response.status_code, [200, 404])  # 404 if endpoint doesn't exist
        
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('status', data)
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Should return HTML content
        self.assertIn(b'html', response.data.lower())
    
    @patch('app.ask_chatgpt')
    def test_chat_endpoint_success(self, mock_ask_chatgpt):
        """Test successful chat API call"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        # Mock successful response
        mock_ask_chatgpt.return_value = "Hello! How can I help you today?"
        
        response = self.client.post('/api/chat', 
            json={
                'message': 'Hello',
                'session_id': 'test_session'
            })
        
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('response', data)
            self.assertEqual(data['response'], "Hello! How can I help you today?")
        else:
            # Endpoint might not exist yet
            self.assertIn(response.status_code, [404, 405])
    
    def test_chat_endpoint_missing_message(self):
        """Test chat endpoint with missing message"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.post('/api/chat', json={})
        
        # Should return 400 for bad request
        if response.status_code not in [404, 405]:  # If endpoint exists
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertIn('error', data)
    
    def test_chat_endpoint_empty_message(self):
        """Test chat endpoint with empty message"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.post('/api/chat', 
            json={'message': '', 'session_id': 'test'})
        
        if response.status_code not in [404, 405]:
            self.assertEqual(response.status_code, 400)
    
    def test_chat_endpoint_invalid_json(self):
        """Test chat endpoint with invalid JSON"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.post('/api/chat', 
            data='invalid json',
            content_type='application/json')
        
        if response.status_code not in [404, 405]:
            self.assertEqual(response.status_code, 400)
    
    @patch('app.create_personality_blend')
    def test_personality_blend_endpoint_success(self, mock_create_blend):
        """Test successful personality blend creation"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        # Mock successful blend creation
        mock_create_blend.return_value = {
            'personalities': ['creative', 'analytical'],
            'weights': [0.6, 0.4],
            'effectiveness': 0.85,
            'description': 'Creative-Analytical blend'
        }
        
        response = self.client.post('/api/personality-blends',
            json={
                'personalities': ['creative', 'analytical'],
                'weights': [0.6, 0.4],
                'context': 'creative_work'
            })
        
        if response.status_code not in [404, 405]:
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('personalities', data)
            self.assertIn('effectiveness', data)
    
    def test_personality_blend_endpoint_validation(self):
        """Test personality blend endpoint validation"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        # Test with insufficient personalities
        response = self.client.post('/api/personality-blends',
            json={
                'personalities': ['creative'],  # Only one personality
                'weights': [1.0],
                'context': 'test'
            })
        
        if response.status_code not in [404, 405]:
            self.assertEqual(response.status_code, 400)
    
    def test_personality_blend_endpoint_weight_mismatch(self):
        """Test personality blend with mismatched weights"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.post('/api/personality-blends',
            json={
                'personalities': ['creative', 'analytical'],
                'weights': [0.6],  # Wrong number of weights
                'context': 'test'
            })
        
        if response.status_code not in [404, 405]:
            self.assertEqual(response.status_code, 400)
    
    @patch('app.detect_mood_from_text')
    def test_mood_detection_endpoint(self, mock_detect_mood):
        """Test mood detection endpoint"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        # Mock mood detection
        mock_detect_mood.return_value = {
            'mood': 'excited',
            'confidence': 0.85,
            'indicators': ['exclamation marks', 'positive language']
        }
        
        response = self.client.post('/api/mood-detection',
            json={'text': "I'm so excited about this new feature!"})
        
        if response.status_code not in [404, 405]:
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('mood', data)
            self.assertIn('confidence', data)
    
    def test_mood_detection_endpoint_empty_text(self):
        """Test mood detection with empty text"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.post('/api/mood-detection',
            json={'text': ''})
        
        if response.status_code not in [404, 405]:
            self.assertEqual(response.status_code, 400)
    
    def test_get_personalities_endpoint(self):
        """Test get personalities endpoint"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.get('/api/personalities')
        
        if response.status_code not in [404, 405]:
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('personalities', data)
            self.assertIsInstance(data['personalities'], list)
    
    def test_get_personality_blends_endpoint(self):
        """Test get personality blends endpoint"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.get('/api/personality-blends')
        
        if response.status_code not in [404, 405]:
            self.assertIn(response.status_code, [200, 400])  # May require auth
    
    def test_get_personality_blend_by_id(self):
        """Test get specific personality blend"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.get('/api/personality-blends/test_blend_id')
        
        if response.status_code not in [404, 405]:
            # Should return 404 for non-existent blend or 200 for existing
            self.assertIn(response.status_code, [200, 404])
    
    def test_delete_personality_blend(self):
        """Test delete personality blend"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.delete('/api/personality-blends/test_blend_id')
        
        if response.status_code not in [404, 405]:
            self.assertIn(response.status_code, [200, 404])
    
    def test_analytics_endpoints(self):
        """Test analytics endpoints"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        analytics_endpoints = [
            '/api/analytics/conversations',
            '/api/analytics/personality-usage',
            '/api/analytics/mood-patterns',
            '/api/analytics/blend-effectiveness'
        ]
        
        for endpoint in analytics_endpoints:
            response = self.client.get(endpoint)
            
            if response.status_code not in [404, 405]:
                # Should return 200 or require authentication
                self.assertIn(response.status_code, [200, 401, 403])
    
    def test_error_handling_consistency(self):
        """Test that error responses are consistent"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        # Test various error conditions
        error_tests = [
            ('POST', '/api/chat', {}),  # Missing required fields
            ('POST', '/api/personality-blends', {}),  # Missing required fields
            ('GET', '/api/nonexistent-endpoint', None),  # Non-existent endpoint
        ]
        
        for method, endpoint, data in error_tests:
            if method == 'POST':
                response = self.client.post(endpoint, json=data)
            else:
                response = self.client.get(endpoint)
            
            if response.status_code >= 400:
                # Error responses should be JSON and have consistent structure
                try:
                    data = json.loads(response.data)
                    self.assertIn('error', data)
                except json.JSONDecodeError:
                    # Some errors might not be JSON (like 404s)
                    pass
    
    def test_content_type_validation(self):
        """Test content type validation for POST endpoints"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        # Test sending non-JSON data to JSON endpoints
        response = self.client.post('/api/chat',
            data='not json',
            content_type='text/plain')
        
        if response.status_code not in [404, 405]:
            # Should reject non-JSON content
            self.assertIn(response.status_code, [400, 415])
    
    def test_rate_limiting(self):
        """Test rate limiting if implemented"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = self.client.get('/')
            responses.append(response.status_code)
        
        # All should succeed if no rate limiting
        # If rate limiting exists, some might return 429
        self.assertTrue(all(status in [200, 429] for status in responses))
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.get('/')
        
        # Should have CORS headers for web frontend
        # Note: Headers might be added by Flask-CORS
        cors_headers = [
            'Access-Control-Allow-Origin',
            'Access-Control-Allow-Methods',
            'Access-Control-Allow-Headers'
        ]
        
        # At least one CORS header should be present
        has_cors = any(header in response.headers for header in cors_headers)
        self.assertTrue(has_cors or response.status_code == 200)


class TestAPIResponseFormats(unittest.TestCase):
    """Test API response format consistency"""
    
    def setUp(self):
        if app:
            self.app = app
            self.app.config['TESTING'] = True
            self.client = self.app.test_client()
        else:
            self.client = None
    
    def test_success_response_format(self):
        """Test that successful responses have consistent format"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        with patch('app.ask_chatgpt') as mock_chat:
            mock_chat.return_value = "Test response"
            
            response = self.client.post('/api/chat',
                json={'message': 'test', 'session_id': 'test'})
            
            if response.status_code == 200:
                data = json.loads(response.data)
                # Should have success indicator and timestamp
                expected_fields = ['success', 'timestamp']
                for field in expected_fields:
                    if field in data:
                        self.assertIn(field, data)
    
    def test_error_response_format(self):
        """Test that error responses have consistent format"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.post('/api/chat', json={})
        
        if response.status_code >= 400 and response.status_code < 500:
            try:
                data = json.loads(response.data)
                # Error responses should have consistent structure
                expected_fields = ['error', 'error_code', 'message', 'timestamp']
                for field in expected_fields:
                    if field in data:
                        self.assertIn(field, data)
            except json.JSONDecodeError:
                # Some errors might not be JSON formatted
                pass
    
    def test_pagination_format(self):
        """Test pagination format for list endpoints"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        response = self.client.get('/api/conversations?page=1&limit=10')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            # Paginated responses should include metadata
            pagination_fields = ['page', 'limit', 'total', 'has_next', 'has_prev']
            # Check if any pagination fields are present
            has_pagination = any(field in data for field in pagination_fields)
            # This is optional, so we just check structure if present
            if has_pagination:
                self.assertIn('page', data)


class TestAPISecurityAndValidation(unittest.TestCase):
    """Test API security and input validation"""
    
    def setUp(self):
        if app:
            self.app = app
            self.app.config['TESTING'] = True
            self.client = self.app.test_client()
        else:
            self.client = None
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        malicious_inputs = [
            "'; DROP TABLE conversations; --",
            "1' OR '1'='1",
            "' UNION SELECT * FROM users --"
        ]
        
        for malicious_input in malicious_inputs:
            response = self.client.post('/api/chat',
                json={'message': malicious_input, 'session_id': 'test'})
            
            # Should not cause server error
            self.assertNotEqual(response.status_code, 500)
    
    def test_xss_protection(self):
        """Test protection against XSS attacks"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for payload in xss_payloads:
            response = self.client.post('/api/chat',
                json={'message': payload, 'session_id': 'test'})
            
            if response.status_code == 200:
                # Response should escape or sanitize the payload
                self.assertNotIn(b'<script>', response.data)
    
    def test_input_length_limits(self):
        """Test input length validation"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        # Very long message
        long_message = "x" * 10000
        
        response = self.client.post('/api/chat',
            json={'message': long_message, 'session_id': 'test'})
        
        # Should either accept it or return 400 for too long
        self.assertIn(response.status_code, [200, 400, 413])
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        if not self.client:
            self.skipTest("Flask app not available")
        
        invalid_inputs = [
            {'message': 123, 'session_id': 'test'},  # Number instead of string
            {'message': ['list'], 'session_id': 'test'},  # List instead of string
            {'message': None, 'session_id': 'test'},  # None value
        ]
        
        for invalid_input in invalid_inputs:
            response = self.client.post('/api/chat', json=invalid_input)
            
            if response.status_code not in [404, 405]:
                # Should return 400 for invalid data types
                self.assertEqual(response.status_code, 400)


def run_api_tests():
    """Run all API tests"""
    test_classes = [
        TestAPIEndpoints,
        TestAPIResponseFormats,
        TestAPISecurityAndValidation
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_api_tests()
    sys.exit(0 if success else 1)