"""
Horizon Memory System Core Module

This module handles user memory, learning, and context management
for the Horizon AI Assistant.

Classes:
- MemorySystem: Main memory management system
- UserMemory: Individual user memory management
- ContextManager: Conversation context and continuity
- LearningEngine: Adaptive learning and personalization

Functions:
- store_user_memory: Store user information in memory
- get_user_context: Get user conversation context
- learn_from_interaction: Learn from user interactions
- get_personalized_response: Get response based on user history
"""

import os
import json
import uuid
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from config import Config

# Memory system configuration
MEMORY_RETENTION_DAYS = getattr(Config, 'MEMORY_RETENTION_DAYS', 365)
CONTEXT_WINDOW_SIZE = getattr(Config, 'CONTEXT_WINDOW_SIZE', 10)
LEARNING_THRESHOLD = getattr(Config, 'LEARNING_THRESHOLD', 3)

# Memory importance scoring
MEMORY_IMPORTANCE_WEIGHTS = {
    'personal_info': 1.0,
    'preferences': 0.9,
    'goals': 0.8,
    'interests': 0.7,
    'feedback': 0.8,
    'conversation_topic': 0.6,
    'emotional_state': 0.7,
    'context': 0.5
}


class MemorySystem:
    """Main memory management system."""
    
    def __init__(self):
        """Initialize the memory system."""
        self.user_memories = defaultdict(dict)
        self.conversation_contexts = defaultdict(list)
        self.learning_patterns = defaultdict(list)
        
        print("🧠 Memory System initialized")
    
    def store_memory(self, user_id: str, memory_type: str, content: str,
                    context: str = None, importance_score: float = None) -> str:
        """
        Store a memory for a user.
        
        Args:
            user_id: User identifier
            memory_type: Type of memory (personal_info, preferences, etc.)
            content: Memory content
            context: Additional context
            importance_score: Optional importance score (0.0-1.0)
            
        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())
        
        # Calculate importance score if not provided
        if importance_score is None:
            importance_score = MEMORY_IMPORTANCE_WEIGHTS.get(memory_type, 0.5)
        
        memory = {
            'id': memory_id,
            'type': memory_type,
            'content': content,
            'context': context,
            'importance_score': importance_score,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'access_count': 0,
            'metadata': {}
        }
        
        # Store memory
        if user_id not in self.user_memories:
            self.user_memories[user_id] = {}
        
        if memory_type not in self.user_memories[user_id]:
            self.user_memories[user_id][memory_type] = []
        
        self.user_memories[user_id][memory_type].append(memory)
        
        # Maintain memory limits (keep only most important recent memories)
        self._cleanup_old_memories(user_id, memory_type)
        
        print(f"💾 Stored {memory_type} memory for user {user_id[:8]}...")
        return memory_id
    
    def get_memories(self, user_id: str, memory_type: str = None,
                    limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get user memories.
        
        Args:
            user_id: User identifier
            memory_type: Optional memory type filter
            limit: Maximum number of memories to return
            
        Returns:
            List of memories
        """
        if user_id not in self.user_memories:
            return []
        
        memories = []
        
        if memory_type:
            memories = self.user_memories[user_id].get(memory_type, [])
        else:
            # Get all memories for user
            for mem_type, mem_list in self.user_memories[user_id].items():
                memories.extend(mem_list)
        
        # Sort by importance and recency
        memories.sort(key=lambda m: (m['importance_score'], m['last_accessed']), reverse=True)
        
        # Update access information
        for memory in memories[:limit]:
            memory['last_accessed'] = datetime.now()
            memory['access_count'] += 1
        
        return memories[:limit]
    
    def _cleanup_old_memories(self, user_id: str, memory_type: str):
        """Clean up old or less important memories."""
        if user_id not in self.user_memories or memory_type not in self.user_memories[user_id]:
            return
        
        memories = self.user_memories[user_id][memory_type]
        
        # Remove memories older than retention period
        cutoff_date = datetime.now() - timedelta(days=MEMORY_RETENTION_DAYS)
        memories = [m for m in memories if m['created_at'] > cutoff_date]
        
        # Keep only top memories if too many
        max_memories = 50  # Configurable limit
        if len(memories) > max_memories:
            memories.sort(key=lambda m: (m['importance_score'], m['access_count']), reverse=True)
            memories = memories[:max_memories]
        
        self.user_memories[user_id][memory_type] = memories
    
    def search_memories(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search user memories by content.
        
        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        if user_id not in self.user_memories:
            return []
        
        query_lower = query.lower()
        matching_memories = []
        
        for memory_type, memories in self.user_memories[user_id].items():
            for memory in memories:
                content_lower = memory['content'].lower()
                context_lower = (memory['context'] or '').lower()
                
                # Simple keyword matching
                if query_lower in content_lower or query_lower in context_lower:
                    # Add relevance score
                    relevance = 0.0
                    if query_lower in content_lower:
                        relevance += 0.8
                    if query_lower in context_lower:
                        relevance += 0.3
                    
                    memory_copy = memory.copy()
                    memory_copy['relevance_score'] = relevance
                    matching_memories.append(memory_copy)
        
        # Sort by relevance and importance
        matching_memories.sort(key=lambda m: (m['relevance_score'], m['importance_score']), reverse=True)
        
        return matching_memories[:limit]


class UserMemory:
    """Individual user memory management."""
    
    def __init__(self, user_id: str, memory_system: MemorySystem):
        """Initialize user memory."""
        self.user_id = user_id
        self.memory_system = memory_system
        
    def remember_personal_info(self, info: str, context: str = None):
        """Remember personal information about the user."""
        return self.memory_system.store_memory(
            self.user_id, 'personal_info', info, context, 1.0
        )
    
    def remember_preference(self, preference: str, context: str = None):
        """Remember user preferences."""
        return self.memory_system.store_memory(
            self.user_id, 'preferences', preference, context, 0.9
        )
    
    def remember_interest(self, interest: str, context: str = None):
        """Remember user interests."""
        return self.memory_system.store_memory(
            self.user_id, 'interests', interest, context, 0.7
        )
    
    def remember_goal(self, goal: str, context: str = None):
        """Remember user goals."""
        return self.memory_system.store_memory(
            self.user_id, 'goals', goal, context, 0.8
        )
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get a summary of user's stored profile."""
        summary = {
            'personal_info': [],
            'preferences': [],
            'interests': [],
            'goals': [],
            'recent_topics': []
        }
        
        for memory_type in summary.keys():
            memories = self.memory_system.get_memories(self.user_id, memory_type, limit=5)
            summary[memory_type] = [m['content'] for m in memories]
        
        return summary


class ContextManager:
    """Conversation context and continuity management."""
    
    def __init__(self, memory_system: MemorySystem):
        """Initialize context manager."""
        self.memory_system = memory_system
        
        print("🔗 Context Manager initialized")
    
    def add_conversation_turn(self, user_id: str, session_id: str,
                           user_message: str, ai_response: str,
                           personality: str, emotion_data: Dict[str, Any] = None):
        """Add a conversation turn to context."""
        context_entry = {
            'user_message': user_message,
            'ai_response': ai_response,
            'personality': personality,
            'emotion_data': emotion_data or {},
            'timestamp': datetime.now()
        }
        
        # Store in conversation context
        if user_id not in self.memory_system.conversation_contexts:
            self.memory_system.conversation_contexts[user_id] = []
        
        self.memory_system.conversation_contexts[user_id].append(context_entry)
        
        # Maintain context window size
        if len(self.memory_system.conversation_contexts[user_id]) > CONTEXT_WINDOW_SIZE:
            self.memory_system.conversation_contexts[user_id] = \
                self.memory_system.conversation_contexts[user_id][-CONTEXT_WINDOW_SIZE:]
        
        # Extract and store interesting information
        self._extract_learnable_content(user_id, user_message, ai_response, context_entry)
    
    def get_conversation_context(self, user_id: str, limit: int = 5) -> str:
        """Get recent conversation context for user."""
        if user_id not in self.memory_system.conversation_contexts:
            return ""
        
        recent_turns = self.memory_system.conversation_contexts[user_id][-limit:]
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"User: {turn['user_message']}")
            context_parts.append(f"Assistant: {turn['ai_response']}")
        
        return "\n".join(context_parts)
    
    def _extract_learnable_content(self, user_id: str, user_message: str,
                                 ai_response: str, context_entry: Dict[str, Any]):
        """Extract learnable content from conversation."""
        message_lower = user_message.lower()
        
        # Extract personal information
        personal_indicators = ['my name is', 'i am', 'i work', 'i live', 'i study']
        for indicator in personal_indicators:
            if indicator in message_lower:
                self.memory_system.store_memory(
                    user_id, 'personal_info', user_message,
                    f"Conversation context: {ai_response[:100]}...", 0.8
                )
                break
        
        # Extract preferences
        preference_indicators = ['i like', 'i love', 'i prefer', 'i enjoy', 'i hate', 'i dislike']
        for indicator in preference_indicators:
            if indicator in message_lower:
                self.memory_system.store_memory(
                    user_id, 'preferences', user_message,
                    f"Preference expressed in conversation", 0.9
                )
                break
        
        # Extract interests
        interest_indicators = ['interested in', 'hobby', 'passion', 'fascinated by']
        for indicator in interest_indicators:
            if indicator in message_lower:
                self.memory_system.store_memory(
                    user_id, 'interests', user_message,
                    f"Interest mentioned in conversation", 0.7
                )
                break


class LearningEngine:
    """Adaptive learning and personalization."""
    
    def __init__(self, memory_system: MemorySystem):
        """Initialize learning engine."""
        self.memory_system = memory_system
        
        print("🎓 Learning Engine initialized")
    
    def learn_from_interaction(self, user_id: str, interaction_data: Dict[str, Any]):
        """
        Learn from user interaction.
        
        Args:
            user_id: User identifier
            interaction_data: Data about the interaction
        """
        # Track learning patterns
        pattern = {
            'user_message': interaction_data.get('user_message', ''),
            'ai_response': interaction_data.get('ai_response', ''),
            'personality_used': interaction_data.get('personality', 'friendly'),
            'user_satisfaction': interaction_data.get('satisfaction', 0.5),
            'response_time': interaction_data.get('response_time', 0),
            'context_helpful': interaction_data.get('context_helpful', False),
            'timestamp': datetime.now()
        }
        
        self.memory_system.learning_patterns[user_id].append(pattern)
        
        # Analyze patterns for insights
        if len(self.memory_system.learning_patterns[user_id]) >= LEARNING_THRESHOLD:
            self._analyze_learning_patterns(user_id)
    
    def _analyze_learning_patterns(self, user_id: str):
        """Analyze user patterns to improve responses."""
        patterns = self.memory_system.learning_patterns[user_id]
        
        # Analyze personality preferences
        personality_scores = defaultdict(float)
        for pattern in patterns[-10:]:  # Last 10 interactions
            personality = pattern['personality_used']
            satisfaction = pattern['user_satisfaction']
            personality_scores[personality] += satisfaction
        
        # Find preferred personality
        if personality_scores:
            best_personality = max(personality_scores, key=personality_scores.get)
            avg_satisfaction = personality_scores[best_personality] / \
                            sum(1 for p in patterns[-10:] if p['personality_used'] == best_personality)
            
            if avg_satisfaction > 0.7:  # High satisfaction threshold
                self.memory_system.store_memory(
                    user_id, 'preferences',
                    f"Prefers {best_personality} personality (satisfaction: {avg_satisfaction:.2f})",
                    "Learned from interaction patterns", 0.8
                )
    
    def get_personalized_response_hints(self, user_id: str) -> Dict[str, Any]:
        """Get hints for personalizing responses."""
        # Get user memories for personalization
        recent_memories = self.memory_system.get_memories(user_id, limit=20)
        
        hints = {
            'preferred_personality': 'friendly',
            'interests': [],
            'communication_style': 'balanced',
            'context_preferences': {},
            'emotional_state': 'neutral'
        }
        
        # Extract personalization hints from memories
        for memory in recent_memories:
            memory_type = memory['type']
            content = memory['content']
            
            if memory_type == 'preferences' and 'personality' in content.lower():
                # Extract personality preference
                for personality in ['friendly', 'professional', 'casual', 'enthusiastic']:
                    if personality in content.lower():
                        hints['preferred_personality'] = personality
                        break
            
            elif memory_type == 'interests':
                hints['interests'].append(content)
        
        return hints


# Global instances
memory_system = None
context_manager = None
learning_engine = None

def get_memory_system() -> MemorySystem:
    """Get the global memory system instance."""
    global memory_system
    if memory_system is None:
        memory_system = MemorySystem()
    return memory_system

def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global context_manager
    if context_manager is None:
        context_manager = ContextManager(get_memory_system())
    return context_manager

def get_learning_engine() -> LearningEngine:
    """Get the global learning engine instance."""
    global learning_engine
    if learning_engine is None:
        learning_engine = LearningEngine(get_memory_system())
    return learning_engine

def get_user_memory(user_id: str) -> UserMemory:
    """Get user memory instance for specific user."""
    return UserMemory(user_id, get_memory_system())

# Enhanced memory system with database integration from app.py
class DatabaseMemorySystem(MemorySystem):
    """Enhanced memory system with SQLite database integration."""
    
    def __init__(self, db_path: str = 'ai_memory.db'):
        """Initialize database memory system."""
        super().__init__()
        self.db_path = db_path
        self._initialize_database()
        print("🗄️ Database Memory System initialized")
    
    def _initialize_database(self):
        """Initialize database tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # User memory table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_identifier TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    importance_score REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    UNIQUE(user_identifier, memory_type, memory_key)
                )
            ''')
            
            # Check if conversation_context table exists and has correct schema
            cursor.execute("PRAGMA table_info(conversation_context)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'user_identifier' not in columns:
                # Drop and recreate table with correct schema
                cursor.execute('DROP TABLE IF EXISTS conversation_context')
            
            # Conversation context table with correct schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_identifier TEXT,
                    user_input TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    personality TEXT,
                    sentiment_score REAL,
                    emotion_data TEXT,
                    context_used INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    importance_score REAL DEFAULT 0.5
                )
            ''')
            
            # Learning patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_identifier TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    confidence_score REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
    
    def save_user_memory_db(self, user_id: str, memory_type: str, key: str, 
                           value: str, importance: float = 0.5) -> bool:
        """Save user memory to database (extracted from app.py)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if memory already exists
            cursor.execute('''
                SELECT id FROM user_memory 
                WHERE user_identifier = ? AND memory_type = ? AND memory_key = ?
            ''', (user_id, memory_type, key))
            
            if cursor.fetchone():
                # Update existing memory
                cursor.execute('''
                    UPDATE user_memory 
                    SET memory_value = ?, importance_score = ?, updated_at = ?, access_count = access_count + 1
                    WHERE user_identifier = ? AND memory_type = ? AND memory_key = ?
                ''', (value, importance, datetime.now().isoformat(), user_id, memory_type, key))
            else:
                # Create new memory
                cursor.execute('''
                    INSERT INTO user_memory (user_identifier, memory_type, memory_key, memory_value, importance_score, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, memory_type, key, value, importance, datetime.now().isoformat(), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error saving user memory: {e}")
            return False
    
    def retrieve_user_memory_db(self, user_id: str, memory_type: str = None, 
                               key: str = None) -> Any:
        """Retrieve user memory from database (extracted from app.py)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if key:
                cursor.execute('''
                    SELECT memory_value, importance_score FROM user_memory 
                    WHERE user_identifier = ? AND memory_type = ? AND memory_key = ?
                ''', (user_id, memory_type, key))
                result = cursor.fetchone()
                conn.close()
                return result[0] if result else None
            elif memory_type:
                cursor.execute('''
                    SELECT memory_key, memory_value, importance_score FROM user_memory 
                    WHERE user_identifier = ? AND memory_type = ?
                    ORDER BY importance_score DESC, access_count DESC
                ''', (user_id, memory_type))
            else:
                cursor.execute('''
                    SELECT memory_type, memory_key, memory_value, importance_score FROM user_memory 
                    WHERE user_identifier = ?
                    ORDER BY importance_score DESC, access_count DESC
                ''', (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            return results
            
        except Exception as e:
            print(f"❌ Error retrieving user memory: {e}")
            return []
    
    def save_conversation_db(self, user_input: str, ai_response: str, personality: str,
                           session_id: str = None, user_id: str = None,
                           sentiment_score: float = None, emotion_data: Dict = None,
                           context_used: bool = False) -> str:
        """Save conversation to database (extracted from app.py)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if not session_id:
                session_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO conversation_context 
                (session_id, user_identifier, user_input, ai_response, personality, 
                 sentiment_score, emotion_data, context_used, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, user_id, user_input, ai_response, personality,
                  sentiment_score, json.dumps(emotion_data) if emotion_data else None,
                  int(context_used), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            return session_id
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
            return session_id or str(uuid.uuid4())
    
    def build_conversation_context_db(self, session_id: str, current_input: str) -> str:
        """Build conversation context from database (extracted from app.py)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent conversation history
            cursor.execute('''
                SELECT user_input, ai_response, personality, timestamp
                FROM conversation_context 
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 5
            ''', (session_id,))
            
            conversations = cursor.fetchall()
            conn.close()
            
            if not conversations:
                return f"Current input: {current_input}"
            
            # Build context string
            context_parts = [f"Current input: {current_input}", "\nRecent conversation:"]
            
            for user_input, ai_response, personality, timestamp in reversed(conversations):
                context_parts.append(f"User: {user_input}")
                context_parts.append(f"Assistant ({personality}): {ai_response}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"❌ Error building conversation context: {e}")
            return f"Current input: {current_input}"
    
    def extract_learning_patterns_db(self, user_input: str, ai_response: str,
                                   intent: str, confidence: float) -> Dict[str, Any]:
        """Extract learning patterns from conversation (extracted from app.py)."""
        patterns = {
            'intent': intent,
            'confidence': confidence,
            'input_length': len(user_input),
            'response_length': len(ai_response),
            'keywords': [],
            'topics': []
        }
        
        try:
            # Extract keywords (simple approach)
            words = user_input.lower().split()
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            keywords = [word for word in words if len(word) > 3 and word not in common_words]
            patterns['keywords'] = keywords[:5]  # Top 5 keywords
            
            # Simple topic detection
            topics = []
            if any(word in user_input.lower() for word in ['image', 'picture', 'photo', 'generate']):
                topics.append('image_generation')
            if any(word in user_input.lower() for word in ['video', 'movie', 'animation']):
                topics.append('video_generation')
            if any(word in user_input.lower() for word in ['music', 'audio', 'sound']):
                topics.append('audio_generation')
            if any(word in user_input.lower() for word in ['help', 'how', 'what', 'question']):
                topics.append('question_answering')
            
            patterns['topics'] = topics
            
            # Save pattern to database
            self._save_learning_pattern(intent, patterns, confidence)
            
        except Exception as e:
            print(f"❌ Error extracting learning patterns: {e}")
        
        return patterns
    
    def _save_learning_pattern(self, user_id: str, pattern_data: Dict[str, Any], 
                              confidence: float) -> bool:
        """Save learning pattern to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO learning_patterns 
                (user_identifier, pattern_type, pattern_data, confidence_score, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, 'conversation_pattern', json.dumps(pattern_data), 
                  confidence, datetime.now().isoformat(), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error saving learning pattern: {e}")
            return False
    
    def get_memory_stats(self, user_id: str = None) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            if user_id:
                # User-specific stats
                cursor.execute('''
                    SELECT COUNT(*), AVG(importance_score), MAX(access_count)
                    FROM user_memory WHERE user_identifier = ?
                ''', (user_id,))
                memory_stats = cursor.fetchone()
                
                cursor.execute('''
                    SELECT COUNT(*) FROM conversation_context 
                    WHERE user_identifier = ?
                ''', (user_id,))
                conversation_count = cursor.fetchone()[0]
                
                stats = {
                    'user_id': user_id,
                    'total_memories': memory_stats[0] or 0,
                    'avg_importance': memory_stats[1] or 0.0,
                    'max_access_count': memory_stats[2] or 0,
                    'total_conversations': conversation_count,
                    'memory_types': []
                }
                
                # Get memory types
                cursor.execute('''
                    SELECT memory_type, COUNT(*) FROM user_memory 
                    WHERE user_identifier = ? GROUP BY memory_type
                ''', (user_id,))
                stats['memory_types'] = cursor.fetchall()
                
            else:
                # Global stats
                cursor.execute('SELECT COUNT(*) FROM user_memory')
                total_memories = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM conversation_context')
                total_conversations = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT user_identifier) FROM user_memory')
                unique_users = cursor.fetchone()[0]
                
                stats = {
                    'total_memories': total_memories,
                    'total_conversations': total_conversations,
                    'unique_users': unique_users,
                    'database_path': self.db_path
                }
            
            conn.close()
            return stats
            
        except Exception as e:
            print(f"❌ Error getting memory stats: {e}")
            return {'error': str(e)}


class EnhancedMemorySystem(DatabaseMemorySystem):
    """Enhanced memory system with advanced features."""
    
    def __init__(self, db_path: str = 'ai_memory.db'):
        """Initialize enhanced memory system."""
        super().__init__(db_path)
        self.predictive_cache = {}
        print("🧠✨ Enhanced Memory System initialized with predictive capabilities")
    
    def get_recent_conversations(self, user_id: str = None, session_id: str = None, 
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversations for MCP resource."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute('''
                    SELECT user_input, ai_response, personality, timestamp, sentiment_score
                    FROM conversation_context 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (session_id, limit))
            elif user_id:
                cursor.execute('''
                    SELECT user_input, ai_response, personality, timestamp, sentiment_score
                    FROM conversation_context 
                    WHERE user_identifier = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (user_id, limit))
            else:
                cursor.execute('''
                    SELECT user_input, ai_response, personality, timestamp, sentiment_score
                    FROM conversation_context 
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    'user_input': row[0],
                    'ai_response': row[1],
                    'personality': row[2],
                    'timestamp': row[3],
                    'sentiment_score': row[4]
                })
            
            conn.close()
            return conversations
            
        except Exception as e:
            print(f"❌ Error getting recent conversations: {e}")
            return []
    
    def analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user patterns for personalization."""
        try:
            patterns = {
                'preferred_personalities': {},
                'common_topics': {},
                'interaction_frequency': 0,
                'avg_sentiment': 0.0,
                'last_interaction': None
            }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Analyze personality preferences
            cursor.execute('''
                SELECT personality, COUNT(*) as usage_count, AVG(sentiment_score) as avg_sentiment
                FROM conversation_context 
                WHERE user_identifier = ? AND personality IS NOT NULL
                GROUP BY personality
                ORDER BY usage_count DESC
            ''', (user_id,))
            
            for row in cursor.fetchall():
                patterns['preferred_personalities'][row[0]] = {
                    'usage_count': row[1],
                    'avg_sentiment': row[2] or 0.0
                }
            
            # Get interaction frequency
            cursor.execute('''
                SELECT COUNT(*), AVG(sentiment_score), MAX(timestamp)
                FROM conversation_context 
                WHERE user_identifier = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            if result:
                patterns['interaction_frequency'] = result[0] or 0
                patterns['avg_sentiment'] = result[1] or 0.0
                patterns['last_interaction'] = result[2]
            
            conn.close()
            return patterns
            
        except Exception as e:
            print(f"❌ Error analyzing user patterns: {e}")
            return {}


# Global enhanced instances
enhanced_memory_system = None
database_memory_system = None

def get_enhanced_memory_system() -> EnhancedMemorySystem:
    """Get the enhanced memory system instance."""
    global enhanced_memory_system
    if enhanced_memory_system is None:
        enhanced_memory_system = EnhancedMemorySystem()
    return enhanced_memory_system

def get_database_memory_system() -> DatabaseMemorySystem:
    """Get the database memory system instance."""
    global database_memory_system
    if database_memory_system is None:
        database_memory_system = DatabaseMemorySystem()
    return database_memory_system

# Enhanced convenience functions that use database storage
def save_user_memory(user_id: str, memory_type: str, key: str, value: str, importance: float = 0.5) -> bool:
    """Save user memory using database system."""
    return get_enhanced_memory_system().save_user_memory_db(user_id, memory_type, key, value, importance)

def retrieve_user_memory(user_id: str, memory_type: str = None, key: str = None) -> Any:
    """Retrieve user memory using database system."""
    return get_enhanced_memory_system().retrieve_user_memory_db(user_id, memory_type, key)

def save_conversation(user_input: str, ai_response: str, personality: str,
                     session_id: str = None, user_id: str = None,
                     sentiment_score: float = None, emotion_data: Dict = None,
                     context_used: bool = False) -> str:
    """Save conversation using database system."""
    return get_enhanced_memory_system().save_conversation_db(
        user_input, ai_response, personality, session_id, user_id,
        sentiment_score, emotion_data, context_used
    )

def build_conversation_context(session_id: str, current_input: str) -> str:
    """Build conversation context using database system."""
    return get_enhanced_memory_system().build_conversation_context_db(session_id, current_input)

def extract_learning_patterns(user_input: str, ai_response: str, intent: str, confidence: float) -> Dict[str, Any]:
    """Extract learning patterns using enhanced system."""
    return get_enhanced_memory_system().extract_learning_patterns_db(user_input, ai_response, intent, confidence)

def get_memory_stats(user_id: str = None) -> Dict[str, Any]:
    """Get memory statistics."""
    return get_enhanced_memory_system().get_memory_stats(user_id)

# Backward compatibility function
def store_user_memory(user_id: str, memory_type: str, content: str,
                     context: str = None, importance_score: float = None) -> str:
    """Store user memory (backward compatibility)."""
    # Convert to key-value format for database storage
    key = f"content_{len(content)}"  # Simple key generation
    importance = importance_score if importance_score is not None else 0.5
    success = save_user_memory(user_id, memory_type, key, content, importance)
    return key if success else None

def get_user_context(user_id: str, limit: int = 5) -> str:
    """Get user conversation context (backward compatibility)."""
    return get_context_manager().get_conversation_context(user_id, limit)

def learn_from_interaction(user_id: str, interaction_data: Dict[str, Any]):
    """Learn from user interaction (backward compatibility)."""
    return get_learning_engine().learn_from_interaction(user_id, interaction_data)

def get_personalized_response(user_id: str) -> Dict[str, Any]:
    """Get personalized response hints (backward compatibility)."""
    return get_learning_engine().get_personalized_response_hints(user_id)