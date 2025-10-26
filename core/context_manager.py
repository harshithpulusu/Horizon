"""
Context Manager - Backend support for conversation memory
Handles conversation persistence and context analysis without modifying core functionality
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid

@dataclass
class ConversationMessage:
    """Represents a single conversation message."""
    id: str
    timestamp: str
    user_message: str
    ai_response: str
    personality: str
    session_id: str
    metadata: Dict[str, Any]

class ContextManager:
    """
    Manages conversation context and memory.
    Stores conversation history and provides context analysis.
    """
    
    def __init__(self, storage_dir: str = "data/conversations"):
        self.storage_dir = storage_dir
        self.max_conversations_per_session = 50
        self.max_context_length = 5  # Number of recent messages for context
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
    
    def _get_session_file(self, session_id: str) -> str:
        """Get the file path for a session's conversation history."""
        return os.path.join(self.storage_dir, f"session_{session_id}.json")
    
    def save_conversation(self, session_id: str, user_message: str, ai_response: str, 
                         personality: str = 'friendly', metadata: Dict[str, Any] = None) -> str:
        """
        Save a conversation message to persistent storage.
        
        Args:
            session_id: Unique session identifier
            user_message: User's message
            ai_response: AI's response
            personality: Personality mode used
            metadata: Additional metadata (images, response time, etc.)
        
        Returns:
            Message ID
        """
        try:
            if metadata is None:
                metadata = {}
            
            # Create conversation message
            message = ConversationMessage(
                id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                user_message=user_message,
                ai_response=ai_response,
                personality=personality,
                session_id=session_id,
                metadata=metadata
            )
            
            # Load existing conversations
            conversations = self.load_session_conversations(session_id)
            
            # Add new message
            conversations.append(asdict(message))
            
            # Trim to max conversations
            if len(conversations) > self.max_conversations_per_session:
                conversations = conversations[-self.max_conversations_per_session:]
            
            # Save to file
            session_file = self._get_session_file(session_id)
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, ensure_ascii=False)
            
            return message.id
            
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return ""
    
    def load_session_conversations(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation messages
        """
        try:
            session_file = self._get_session_file(session_id)
            
            if not os.path.exists(session_file):
                return []
            
            with open(session_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            return conversations if isinstance(conversations, list) else []
            
        except Exception as e:
            print(f"Error loading conversations: {e}")
            return []
    
    def get_context_for_message(self, session_id: str, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Get conversation context for enhanced AI responses.
        
        Args:
            session_id: Session identifier
            include_metadata: Whether to include message metadata
            
        Returns:
            Context information for AI processing
        """
        try:
            conversations = self.load_session_conversations(session_id)
            
            if not conversations:
                return {
                    'has_context': False,
                    'message_count': 0,
                    'context_summary': 'No previous conversation history'
                }
            
            # Get recent conversations for context
            recent_conversations = conversations[-self.max_context_length:]
            
            # Analyze conversation patterns
            topics = set()
            question_count = 0
            personalities_used = set()
            total_length = 0
            
            for conv in conversations:
                user_msg = conv.get('user_message', '').lower()
                personalities_used.add(conv.get('personality', 'friendly'))
                total_length += len(user_msg) + len(conv.get('ai_response', ''))
                
                if '?' in user_msg:
                    question_count += 1
                
                # Simple topic detection
                if any(word in user_msg for word in ['image', 'picture', 'photo', 'visual']):
                    topics.add('images')
                if any(word in user_msg for word in ['time', 'timer', 'remind', 'schedule']):
                    topics.add('time_management')
                if any(word in user_msg for word in ['code', 'program', 'function', 'script']):
                    topics.add('programming')
                if any(word in user_msg for word in ['create', 'generate', 'make', 'design']):
                    topics.add('creative')
            
            # Build context summary
            context_data = {
                'has_context': True,
                'message_count': len(conversations),
                'recent_message_count': len(recent_conversations),
                'question_count': question_count,
                'topics_discussed': list(topics),
                'personalities_used': list(personalities_used),
                'avg_message_length': total_length // len(conversations) if conversations else 0,
                'last_activity': conversations[-1].get('timestamp') if conversations else None,
                'context_summary': self._build_context_summary(recent_conversations, topics),
                'session_id': session_id
            }
            
            if include_metadata:
                context_data['recent_conversations'] = recent_conversations
            
            return context_data
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return {
                'has_context': False,
                'error': str(e),
                'message_count': 0
            }
    
    def _build_context_summary(self, recent_conversations: List[Dict], topics: set) -> str:
        """Build a human-readable context summary."""
        if not recent_conversations:
            return "No recent conversation history"
        
        summary_parts = []
        
        # Message count
        msg_count = len(recent_conversations)
        summary_parts.append(f"Previous {msg_count} message{'s' if msg_count != 1 else ''}")
        
        # Topics
        if topics:
            topic_str = ', '.join(sorted(topics))
            summary_parts.append(f"discussing {topic_str}")
        
        # Recent activity
        last_msg = recent_conversations[-1]
        user_preview = last_msg.get('user_message', '')[:50]
        if len(user_preview) < len(last_msg.get('user_message', '')):
            user_preview += "..."
        
        summary_parts.append(f"Last message: \"{user_preview}\"")
        
        return "; ".join(summary_parts)
    
    def clear_session_history(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_file = self._get_session_file(session_id)
            
            if os.path.exists(session_file):
                os.remove(session_file)
            
            return True
            
        except Exception as e:
            print(f"Error clearing session history: {e}")
            return False
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get information about all stored sessions.
        
        Returns:
            List of session information
        """
        try:
            sessions = []
            
            for filename in os.listdir(self.storage_dir):
                if filename.startswith('session_') and filename.endswith('.json'):
                    session_id = filename[8:-5]  # Remove 'session_' prefix and '.json' suffix
                    
                    conversations = self.load_session_conversations(session_id)
                    if conversations:
                        sessions.append({
                            'session_id': session_id,
                            'message_count': len(conversations),
                            'first_message': conversations[0].get('timestamp') if conversations else None,
                            'last_message': conversations[-1].get('timestamp') if conversations else None,
                            'file_path': self._get_session_file(session_id)
                        })
            
            # Sort by last activity
            sessions.sort(key=lambda s: s.get('last_message', ''), reverse=True)
            
            return sessions
            
        except Exception as e:
            print(f"Error getting sessions: {e}")
            return []

# Global context manager instance
_context_manager = None

def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager