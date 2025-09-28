#!/usr/bin/env python3
"""
Comprehensive Test Suite for Memory & Learning System
Tests cross-session memory persistence and user preference learning
"""

import requests
import json
import time
import uuid
from datetime import datetime
import sqlite3

BASE_URL = "http://localhost:8080"
TEST_USER_ID = "test_user_memory"
TEST_SESSION_ID = f"test_session_{int(time.time())}"

class MemoryLearningSystemTester:
    def __init__(self):
        self.base_url = BASE_URL
        self.user_id = TEST_USER_ID
        self.session_id = TEST_SESSION_ID
        self.test_results = []
        
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
    
    def test_database_schema(self):
        """Test that all required database tables exist"""
        try:
            conn = sqlite3.connect('ai_memory.db')
            cursor = conn.cursor()
            
            # Check for required tables
            required_tables = [
                'persistent_user_context',
                'conversation_memory', 
                'user_behavioral_patterns',
                'session_context_bridges',
                'adaptive_user_preferences',
                'preference_learning_feedback',
                'interaction_quality_metrics',
                'intelligent_suggestions'
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = [table for table in required_tables if table not in existing_tables]
            
            if missing_tables:
                self.log_test("Database Schema", False, f"Missing tables: {missing_tables}")
                return False
            else:
                self.log_test("Database Schema", True, f"All {len(required_tables)} tables exist")
                return True
                
        except Exception as e:
            self.log_test("Database Schema", False, f"Error: {str(e)}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def test_persistent_context_storage(self):
        """Test storing persistent context"""
        try:
            context_data = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "context_type": "personal_info",
                "context_key": "favorite_color",
                "context_value": {"color": "blue", "reason": "calming"},
                "importance_score": 0.8,
                "decay_rate": 0.02
            }
            
            response = requests.post(
                f"{self.base_url}/api/memory/context/store",
                headers={'Content-Type': 'application/json'},
                json=context_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('context_id'):
                    self.log_test("Context Storage", True, f"Context stored with ID: {data['context_id']}")
                    return data['context_id']
                else:
                    self.log_test("Context Storage", False, f"Response: {data}")
                    return None
            else:
                self.log_test("Context Storage", False, f"HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.log_test("Context Storage", False, f"Exception: {str(e)}")
            return None
    
    def test_persistent_context_retrieval(self):
        """Test retrieving persistent context"""
        try:
            # First store some context
            context_id = self.test_persistent_context_storage()
            if not context_id:
                return False
            
            # Wait a moment for database consistency
            time.sleep(0.5)
            
            # Now retrieve it
            params = {
                "user_id": self.user_id,
                "context_type": "personal_info",
                "limit": 10
            }
            
            response = requests.get(
                f"{self.base_url}/api/memory/context/retrieve",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('contexts'):
                    contexts = data['contexts']
                    found_context = any(ctx.get('context_key') == 'favorite_color' for ctx in contexts)
                    if found_context:
                        self.log_test("Context Retrieval", True, f"Retrieved {len(contexts)} contexts")
                        return True
                    else:
                        self.log_test("Context Retrieval", False, "Stored context not found in retrieval")
                        return False
                else:
                    self.log_test("Context Retrieval", False, f"Response: {data}")
                    return False
            else:
                self.log_test("Context Retrieval", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Context Retrieval", False, f"Exception: {str(e)}")
            return False
    
    def test_conversation_memory_storage(self):
        """Test storing conversation memories"""
        try:
            memory_data = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "memory_type": "important_fact",
                "memory_content": "User mentioned they work as a software developer",
                "memory_summary": "Professional information - software developer",
                "relevance_score": 0.9
            }
            
            response = requests.post(
                f"{self.base_url}/api/memory/conversation/store",
                headers={'Content-Type': 'application/json'},
                json=memory_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('memory_id'):
                    self.log_test("Conversation Memory Storage", True, f"Memory stored with ID: {data['memory_id']}")
                    return True
                else:
                    self.log_test("Conversation Memory Storage", False, f"Response: {data}")
                    return False
            else:
                self.log_test("Conversation Memory Storage", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Conversation Memory Storage", False, f"Exception: {str(e)}")
            return False
    
    def test_session_bridge_creation(self):
        """Test creating session bridges"""
        try:
            bridge_data = {
                "user_id": self.user_id,
                "current_session_id": self.session_id,
                "bridge_type": "unresolved_question",
                "bridge_data": {
                    "question": "How to optimize Python code for performance?",
                    "context": "User was asking about performance optimization techniques"
                },
                "importance_level": 4
            }
            
            response = requests.post(
                f"{self.base_url}/api/memory/bridge/create",
                headers={'Content-Type': 'application/json'},
                json=bridge_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('bridge_id'):
                    self.log_test("Session Bridge Creation", True, f"Bridge created with ID: {data['bridge_id']}")
                    return True
                else:
                    self.log_test("Session Bridge Creation", False, f"Response: {data}")
                    return False
            else:
                self.log_test("Session Bridge Creation", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Session Bridge Creation", False, f"Exception: {str(e)}")
            return False
    
    def test_adaptive_preference_learning(self):
        """Test learning user preferences"""
        try:
            preference_data = {
                "user_id": self.user_id,
                "preference_category": "communication",
                "preference_name": "response_length",
                "preference_value": {
                    "preferred_length": "medium",
                    "reasoning": "User typically asks follow-up questions for brief responses"
                },
                "learning_source": "behavioral_analysis",
                "confidence_level": 0.75
            }
            
            response = requests.post(
                f"{self.base_url}/api/preferences/adaptive/learn",
                headers={'Content-Type': 'application/json'},
                json=preference_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('preference_id'):
                    self.log_test("Adaptive Preference Learning", True, f"Preference learned with ID: {data['preference_id']}")
                    return data['preference_id']
                else:
                    self.log_test("Adaptive Preference Learning", False, f"Response: {data}")
                    return None
            else:
                self.log_test("Adaptive Preference Learning", False, f"HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.log_test("Adaptive Preference Learning", False, f"Exception: {str(e)}")
            return None
    
    def test_adaptive_preference_retrieval(self):
        """Test retrieving learned preferences"""
        try:
            # First learn a preference
            pref_id = self.test_adaptive_preference_learning()
            if not pref_id:
                return False
            
            time.sleep(0.5)  # Wait for database consistency
            
            params = {
                "user_id": self.user_id,
                "category": "communication",
                "min_confidence": 0.5
            }
            
            response = requests.get(
                f"{self.base_url}/api/preferences/adaptive/get",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('preferences'):
                    preferences = data['preferences']
                    found_pref = any(pref.get('name') == 'response_length' for pref in preferences)
                    if found_pref:
                        self.log_test("Adaptive Preference Retrieval", True, f"Retrieved {len(preferences)} preferences")
                        return True
                    else:
                        self.log_test("Adaptive Preference Retrieval", False, "Learned preference not found")
                        return False
                else:
                    self.log_test("Adaptive Preference Retrieval", False, f"Response: {data}")
                    return False
            else:
                self.log_test("Adaptive Preference Retrieval", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Adaptive Preference Retrieval", False, f"Exception: {str(e)}")
            return False
    
    def test_preference_feedback_recording(self):
        """Test recording preference feedback"""
        try:
            feedback_data = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "feedback_type": "positive",
                "interaction_context": {
                    "user_message": "Explain quantum computing",
                    "ai_response": "Quantum computing is a revolutionary technology..."
                },
                "user_feedback": "Perfect explanation, just the right amount of detail",
                "response_quality_rating": 0.95,
                "feedback_category": "response_quality"
            }
            
            response = requests.post(
                f"{self.base_url}/api/preferences/feedback",
                headers={'Content-Type': 'application/json'},
                json=feedback_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('feedback_id'):
                    self.log_test("Preference Feedback Recording", True, f"Feedback recorded with ID: {data['feedback_id']}")
                    return True
                else:
                    self.log_test("Preference Feedback Recording", False, f"Response: {data}")
                    return False
            else:
                self.log_test("Preference Feedback Recording", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Preference Feedback Recording", False, f"Exception: {str(e)}")
            return False
    
    def test_integration_workflow(self):
        """Test a complete workflow of memory and learning"""
        try:
            print("\n🔄 Testing Complete Integration Workflow...")
            
            # 1. Store user context
            context_stored = self.test_persistent_context_storage()
            
            # 2. Store conversation memory
            memory_stored = self.test_conversation_memory_storage()
            
            # 3. Learn user preference
            pref_learned = self.test_adaptive_preference_learning()
            
            # 4. Create session bridge
            bridge_created = self.test_session_bridge_creation()
            
            # 5. Record feedback
            feedback_recorded = self.test_preference_feedback_recording()
            
            all_successful = all([context_stored, memory_stored, pref_learned, bridge_created, feedback_recorded])
            
            self.log_test("Integration Workflow", all_successful, "Complete memory & learning workflow")
            return all_successful
            
        except Exception as e:
            self.log_test("Integration Workflow", False, f"Exception: {str(e)}")
            return False
    
    def test_data_persistence(self):
        """Test that data persists across sessions"""
        try:
            print("\n💾 Testing Data Persistence...")
            
            # Store data in one "session"
            original_session = self.session_id
            self.session_id = f"test_session_1_{int(time.time())}"
            
            context_stored = self.test_persistent_context_storage()
            if not context_stored:
                return False
            
            # Switch to a new "session" and try to retrieve
            self.session_id = f"test_session_2_{int(time.time())}"
            time.sleep(1)  # Ensure different timestamp
            
            context_retrieved = self.test_persistent_context_retrieval()
            
            # Restore original session
            self.session_id = original_session
            
            self.log_test("Data Persistence Across Sessions", context_retrieved, "Data persists between sessions")
            return context_retrieved
            
        except Exception as e:
            self.log_test("Data Persistence", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🧠 Starting Memory & Learning System Tests...")
        print("=" * 60)
        
        tests = [
            self.test_database_schema,
            self.test_persistent_context_storage,
            self.test_persistent_context_retrieval,
            self.test_conversation_memory_storage,
            self.test_session_bridge_creation,
            self.test_adaptive_preference_learning,
            self.test_adaptive_preference_retrieval,
            self.test_preference_feedback_recording,
            self.test_data_persistence,
            self.test_integration_workflow
        ]
        
        total_tests = len(tests)
        passed_tests = 0
        
        for test in tests:
            try:
                if test():
                    passed_tests += 1
            except Exception as e:
                print(f"❌ FAILED: {test.__name__} - Exception: {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"📊 TEST RESULTS SUMMARY")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Memory & Learning system is working correctly!")
        else:
            print("⚠️  Some tests failed. Please check the system configuration.")
        
        return passed_tests, total_tests
    
    def cleanup_test_data(self):
        """Clean up test data"""
        try:
            conn = sqlite3.connect('ai_memory.db')
            cursor = conn.cursor()
            
            # Clean up test data
            cursor.execute("DELETE FROM persistent_user_context WHERE user_id = ?", (self.user_id,))
            cursor.execute("DELETE FROM conversation_memory WHERE user_id = ?", (self.user_id,))
            cursor.execute("DELETE FROM session_context_bridges WHERE user_id = ?", (self.user_id,))
            cursor.execute("DELETE FROM adaptive_user_preferences WHERE user_id = ?", (self.user_id,))
            cursor.execute("DELETE FROM preference_learning_feedback WHERE user_id = ?", (self.user_id,))
            
            conn.commit()
            print(f"🧹 Cleaned up test data for user: {self.user_id}")
            
        except Exception as e:
            print(f"⚠️  Error cleaning up test data: {str(e)}")
        finally:
            if 'conn' in locals():
                conn.close()

def main():
    """Main test function"""
    tester = MemoryLearningSystemTester()
    
    try:
        # Run all tests
        passed, total = tester.run_all_tests()
        
        # Generate detailed report
        print(f"\n📋 DETAILED TEST REPORT")
        print("-" * 40)
        for result in tester.test_results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['test']} ({result['timestamp']})")
            if result['details']:
                print(f"   {result['details']}")
        
        return passed == total
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {str(e)}")
        return False
    finally:
        # Clean up test data
        try:
            tester.cleanup_test_data()
        except:
            pass

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)