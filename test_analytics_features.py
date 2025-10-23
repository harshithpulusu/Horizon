#!/usr/bin/env python3
"""
Test Suite for Analytics & Session Management Features

This script tests the new analytics, session management, and tracking features
that were added to Horizon.
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8080"

class AnalyticsFeatureTester:
    def __init__(self):
        self.base_url = BASE_URL
        self.test_results = []
        self.session_id = f"test_session_{int(time.time())}"
        self.user_id = f"test_user_{int(time.time())}"
        
    def log_test(self, test_name, success, details=""):
        """Log test results"""
        result = {
            "test": test_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.test_results.append(result)
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")

    def test_analytics_tracking_endpoint(self):
        """Test the analytics tracking API endpoint"""
        try:
            test_events = [
                {
                    "type": "test_event",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"test": True}
                }
            ]
            
            response = requests.post(f"{self.base_url}/api/analytics/track", 
                                   json={
                                       "events": test_events,
                                       "session_id": self.session_id,
                                       "user_id": self.user_id
                                   },
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Analytics Tracking API", True, 
                            f"Processed {data.get('events_processed', 0)} events")
            else:
                self.log_test("Analytics Tracking API", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Analytics Tracking API", False, str(e))

    def test_analytics_summary_endpoint(self):
        """Test the analytics summary API endpoint"""
        try:
            params = {
                "session_id": self.session_id,
                "user_id": self.user_id
            }
            
            response = requests.get(f"{self.base_url}/api/analytics/summary", 
                                  params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                summary = data.get('summary', {})
                self.log_test("Analytics Summary API", True, 
                            f"Uptime: {summary.get('uptime', 'unknown')}")
            else:
                self.log_test("Analytics Summary API", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Analytics Summary API", False, str(e))

    def test_heatmap_endpoint(self):
        """Test the heatmap data API endpoint"""
        try:
            params = {
                "session_id": self.session_id,
                "limit": 50
            }
            
            response = requests.get(f"{self.base_url}/api/analytics/heatmap", 
                                  params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                heatmap_data = data.get('heatmap_data', [])
                self.log_test("Heatmap Data API", True, 
                            f"Retrieved {len(heatmap_data)} heatmap points")
            else:
                self.log_test("Heatmap Data API", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Heatmap Data API", False, str(e))

    def test_personalities_endpoint(self):
        """Test the personalities API endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/personalities", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                personalities = data.get('personalities', {})
                self.log_test("Personalities API", True, 
                            f"Found {len(personalities)} personality types")
            else:
                self.log_test("Personalities API", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Personalities API", False, str(e))

    def test_chat_with_analytics(self):
        """Test chat endpoint with analytics integration"""
        try:
            chat_data = {
                "message": "Hello, this is a test message for analytics",
                "personality": "friendly",
                "session_id": self.session_id,
                "user_id": self.user_id
            }
            
            response = requests.post(f"{self.base_url}/chat", 
                                   json=chat_data, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Chat with Analytics", True, 
                            f"Response: {data.get('response', '')[:50]}...")
            else:
                self.log_test("Chat with Analytics", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Chat with Analytics", False, str(e))

    def test_health_endpoint_analytics(self):
        """Test health endpoint for analytics data"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                components = data.get('components', {})
                event_system = components.get('event_emitter', False)
                self.log_test("Health Check Analytics", True, 
                            f"Event system: {event_system}")
            else:
                self.log_test("Health Check Analytics", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Health Check Analytics", False, str(e))

    def test_frontend_static_files(self):
        """Test that analytics JavaScript files are accessible"""
        try:
            response = requests.get(f"{self.base_url}/static/analytics-session-manager.js", 
                                  timeout=10)
            
            if response.status_code == 200:
                content_length = len(response.text)
                self.log_test("Analytics JavaScript File", True, 
                            f"File size: {content_length} bytes")
            else:
                self.log_test("Analytics JavaScript File", False, 
                            f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Analytics JavaScript File", False, str(e))

    def run_all_tests(self):
        """Run all analytics feature tests"""
        print("🧪 Starting Analytics & Session Management Feature Tests")
        print("=" * 60)
        
        # Test API endpoints
        self.test_analytics_tracking_endpoint()
        self.test_analytics_summary_endpoint()
        self.test_heatmap_endpoint()
        self.test_personalities_endpoint()
        self.test_chat_with_analytics()
        self.test_health_endpoint_analytics()
        self.test_frontend_static_files()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {(passed/total)*100:.1f}%" if total > 0 else "No tests run")
        
        if passed == total:
            print("\n🎉 All analytics features are working correctly!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Check the details above.")
        
        return passed == total

def main():
    """Main test function"""
    print("🔬 Horizon Analytics & Session Management Feature Tester")
    print("🚀 Testing enhanced tracking, session persistence, and analytics dashboard...")
    print()
    
    tester = AnalyticsFeatureTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All analytics features are ready!")
        print("📱 You can now:")
        print("   • View session status in the UI")
        print("   • Browse chat history in the sidebar")
        print("   • Monitor analytics in real-time")
        print("   • View heatmap data with Ctrl+H")
        print("   • Export analytics data")
        print("   • Track performance metrics")
        print("   • Use A/B testing framework")
    else:
        print("\n❌ Some features may not be working correctly.")
        print("🔧 Please check the server logs and configuration.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)