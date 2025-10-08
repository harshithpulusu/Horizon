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

# Convenience functions for backward compatibility
def store_user_memory(user_id: str, memory_type: str, content: str,
                     context: str = None, importance_score: float = None) -> str:
    """Store user memory."""
    return get_memory_system().store_memory(user_id, memory_type, content, context, importance_score)

def get_user_context(user_id: str, limit: int = 5) -> str:
    """Get user conversation context."""
    return get_context_manager().get_conversation_context(user_id, limit)

def learn_from_interaction(user_id: str, interaction_data: Dict[str, Any]):
    """Learn from user interaction."""
    return get_learning_engine().learn_from_interaction(user_id, interaction_data)

def get_personalized_response(user_id: str) -> Dict[str, Any]:
    """Get personalized response hints."""
    return get_learning_engine().get_personalized_response_hints(user_id)