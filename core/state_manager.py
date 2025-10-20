"""
Horizon State Management Core Module

This module provides a centralized state management system for the Horizon AI Assistant.
It manages application state, user data, conversation context, and system status.

Classes:
- StateManager: Main state management system
- State: Base state container
- ConversationState: Conversation-specific state
- UserState: User-specific state
- SystemState: System-wide state

Functions:
- get_state: Get current state
- update_state: Update state values
- subscribe_to_state: Subscribe to state changes
"""

import json
import uuid
import copy
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging
from .events import EventEmitter, EventData, HorizonEventTypes, get_event_emitter

# Configure logging for state management
logger = logging.getLogger('horizon_state')


@dataclass
class BaseState:
    """Base state container with common functionality."""
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert state to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    def update_timestamp(self):
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now().isoformat()
        self.version += 1


@dataclass
class ConversationState(BaseState):
    """State container for conversation data."""
    current_message: str = ""
    ai_response: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    intent: str = "general"
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: float = 0.0
    mood: str = "neutral"
    is_processing: bool = False
    processing_stage: str = ""
    response_time: float = 0.0
    total_messages: int = 0
    
    def add_message(self, user_input: str, ai_response: str, intent: str = "general", 
                   entities: List = None, sentiment: float = 0.0):
        """Add a message to conversation history."""
        message = {
            'user_input': user_input,
            'ai_response': ai_response,
            'intent': intent,
            'entities': entities or [],
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(message)
        self.current_message = user_input
        self.ai_response = ai_response
        self.intent = intent
        self.entities = entities or []
        self.sentiment = sentiment
        self.total_messages += 1
        self.update_timestamp()
    
    def get_recent_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context."""
        return self.history[-limit:] if self.history else []
    
    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
        self.total_messages = 0
        self.update_timestamp()


@dataclass
class UserState(BaseState):
    """State container for user data."""
    user_id: str = "anonymous"
    name: str = ""
    email: str = ""
    preferences: Dict[str, Any] = field(default_factory=dict)
    personality_mode: str = "friendly"
    language: str = "en"
    timezone: str = "UTC"
    theme: str = "auto"
    
    # User behavior and learning
    learning_data: Dict[str, Any] = field(default_factory=dict)
    quick_commands_usage: Dict[str, int] = field(default_factory=dict)
    favorite_features: List[str] = field(default_factory=list)
    sessions_count: int = 0
    total_interactions: int = 0
    average_session_length: float = 0.0
    
    # User settings
    notifications_enabled: bool = True
    auto_save: bool = True
    privacy_mode: bool = False
    
    def update_preference(self, key: str, value: Any):
        """Update a user preference."""
        self.preferences[key] = value
        self.update_timestamp()
    
    def increment_command_usage(self, command: str):
        """Increment usage count for a command."""
        self.quick_commands_usage[command] = self.quick_commands_usage.get(command, 0) + 1
        self.total_interactions += 1
        self.update_timestamp()
    
    def add_favorite_feature(self, feature: str):
        """Add a feature to favorites."""
        if feature not in self.favorite_features:
            self.favorite_features.append(feature)
            self.update_timestamp()


@dataclass
class AIState(BaseState):
    """State container for AI system data."""
    current_model: str = "chatgpt"
    available_models: List[str] = field(default_factory=list)
    model_status: Dict[str, str] = field(default_factory=dict)
    api_keys_configured: Dict[str, bool] = field(default_factory=dict)
    
    # Processing state
    is_busy: bool = False
    current_task: str = ""
    queue_length: int = 0
    processing_time_avg: float = 0.0
    
    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    uptime_start: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Model-specific data
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fallback_responses_used: int = 0
    
    def set_busy(self, task: str = ""):
        """Set AI as busy with optional task description."""
        self.is_busy = True
        self.current_task = task
        self.update_timestamp()
    
    def set_idle(self):
        """Set AI as idle."""
        self.is_busy = False
        self.current_task = ""
        self.update_timestamp()
    
    def record_request(self, success: bool, processing_time: float = 0.0):
        """Record an API request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        if processing_time > 0:
            # Calculate running average
            total_time = self.processing_time_avg * (self.total_requests - 1) + processing_time
            self.processing_time_avg = total_time / self.total_requests
        
        self.update_timestamp()


@dataclass
class MediaState(BaseState):
    """State container for media generation data."""
    generation_queue: List[Dict[str, Any]] = field(default_factory=list)
    recent_generations: List[Dict[str, Any]] = field(default_factory=list)
    available_generators: List[str] = field(default_factory=list)
    
    # Generation statistics
    total_images_generated: int = 0
    total_videos_generated: int = 0
    total_audio_generated: int = 0
    total_models_generated: int = 0
    
    # Current generation status
    is_generating: bool = False
    current_generation_type: str = ""
    current_generation_prompt: str = ""
    generation_progress: float = 0.0
    
    def add_to_queue(self, generation_type: str, prompt: str, params: Dict[str, Any]):
        """Add a generation request to the queue."""
        request = {
            'id': str(uuid.uuid4()),
            'type': generation_type,
            'prompt': prompt,
            'params': params,
            'status': 'queued',
            'created_at': datetime.now().isoformat()
        }
        self.generation_queue.append(request)
        self.update_timestamp()
        return request['id']
    
    def start_generation(self, request_id: str):
        """Mark a generation as started."""
        for req in self.generation_queue:
            if req['id'] == request_id:
                req['status'] = 'generating'
                req['started_at'] = datetime.now().isoformat()
                self.is_generating = True
                self.current_generation_type = req['type']
                self.current_generation_prompt = req['prompt']
                self.update_timestamp()
                break
    
    def complete_generation(self, request_id: str, result: Dict[str, Any]):
        """Mark a generation as completed."""
        for i, req in enumerate(self.generation_queue):
            if req['id'] == request_id:
                req['status'] = 'completed'
                req['completed_at'] = datetime.now().isoformat()
                req['result'] = result
                
                # Move to recent generations
                self.recent_generations.append(req)
                del self.generation_queue[i]
                
                # Update statistics
                if req['type'] == 'image':
                    self.total_images_generated += 1
                elif req['type'] == 'video':
                    self.total_videos_generated += 1
                elif req['type'] == 'audio':
                    self.total_audio_generated += 1
                elif req['type'] == 'model':
                    self.total_models_generated += 1
                
                # Reset current generation state
                self.is_generating = False
                self.current_generation_type = ""
                self.current_generation_prompt = ""
                self.generation_progress = 0.0
                
                self.update_timestamp()
                break


@dataclass
class SystemState(BaseState):
    """State container for system-wide data."""
    app_version: str = "2.0.0"
    startup_time: str = field(default_factory=lambda: datetime.now().isoformat())
    is_healthy: bool = True
    debug_mode: bool = False
    
    # System resources
    active_sessions: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Database state
    database_connected: bool = False
    database_version: str = ""
    total_conversations_stored: int = 0
    
    # API status
    apis_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Error tracking
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    last_error: str = ""
    
    # Feature flags
    features_enabled: Dict[str, bool] = field(default_factory=lambda: {
        'image_generation': True,
        'video_generation': True,
        'audio_generation': True,
        'memory_learning': True,
        'personality_modes': True,
        'nlp_processing': True
    })
    
    def add_error(self, error: str, context: Dict[str, Any] = None):
        """Add an error to the error log."""
        error_entry = {
            'error': error,
            'context': context or {},
            'timestamp': datetime.now().isoformat(),
            'id': str(uuid.uuid4())
        }
        self.recent_errors.append(error_entry)
        self.last_error = error
        self.error_count += 1
        
        # Keep only last 50 errors
        if len(self.recent_errors) > 50:
            self.recent_errors.pop(0)
        
        self.update_timestamp()
    
    def update_api_status(self, api_name: str, status: str, details: Dict[str, Any] = None):
        """Update the status of an API."""
        self.apis_status[api_name] = {
            'status': status,
            'details': details or {},
            'last_checked': datetime.now().isoformat()
        }
        self.update_timestamp()


@dataclass
class AppState:
    """Main application state container."""
    conversation: ConversationState = field(default_factory=ConversationState)
    user: UserState = field(default_factory=UserState)
    ai: AIState = field(default_factory=AIState)
    media: MediaState = field(default_factory=MediaState)
    system: SystemState = field(default_factory=SystemState)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire app state to dictionary."""
        return {
            'conversation': self.conversation.to_dict(),
            'user': self.user.to_dict(),
            'ai': self.ai.to_dict(),
            'media': self.media.to_dict(),
            'system': self.system.to_dict()
        }
    
    def to_json(self) -> str:
        """Convert app state to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class StateManager:
    """Main state management system for Horizon."""
    
    def __init__(self, event_emitter: Optional[EventEmitter] = None):
        """Initialize the state manager."""
        self._state = AppState()
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._event_emitter = event_emitter or get_event_emitter()
        
        # State change history for debugging
        self._change_history: List[Dict[str, Any]] = []
        self._max_history = 100
        
        logger.info("🗃️ State Manager initialized")
    
    def get_state(self, state_path: Optional[str] = None) -> Union[AppState, Any]:
        """
        Get current state or specific state path.
        
        Args:
            state_path: Dot-separated path (e.g., 'user.preferences', 'conversation.history')
            
        Returns:
            State object or specific value
        """
        with self._lock:
            if state_path is None:
                return copy.deepcopy(self._state)
            
            # Navigate to specific path
            current = self._state
            for part in state_path.split('.'):
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    raise KeyError(f"State path '{state_path}' not found")
            
            return copy.deepcopy(current)
    
    def update_state(self, state_path: str, value: Any, 
                    emit_event: bool = True, source: str = "state_manager") -> bool:
        """
        Update state at specific path.
        
        Args:
            state_path: Dot-separated path to update
            value: New value to set
            emit_event: Whether to emit state change event
            source: Source of the update for event tracking
            
        Returns:
            True if successful
        """
        try:
            with self._lock:
                # Navigate to parent and update
                parts = state_path.split('.')
                current = self._state
                
                # Navigate to parent
                for part in parts[:-1]:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        raise KeyError(f"State path '{state_path}' not found")
                
                # Get old value for change tracking
                old_value = getattr(current, parts[-1], None)
                
                # Set new value
                setattr(current, parts[-1], value)
                
                # Update timestamp if the state has that method
                if hasattr(current, 'update_timestamp'):
                    current.update_timestamp()
                
                # Record change
                change_record = {
                    'path': state_path,
                    'old_value': old_value,
                    'new_value': value,
                    'timestamp': datetime.now().isoformat(),
                    'source': source
                }
                self._change_history.append(change_record)
                
                # Limit history size
                if len(self._change_history) > self._max_history:
                    self._change_history.pop(0)
                
                # Notify subscribers
                self._notify_subscribers(state_path, value, old_value)
                
                # Emit event if requested
                if emit_event:
                    self._event_emitter.emit(
                        "state_updated",
                        source,
                        {
                            'path': state_path,
                            'value': value,
                            'old_value': old_value
                        }
                    )
                
                logger.debug(f"📝 State updated: {state_path} = {value}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to update state '{state_path}': {e}")
            return False
    
    def subscribe(self, state_path: str, callback: Callable[[Any, Any], None]) -> str:
        """
        Subscribe to state changes at specific path.
        
        Args:
            state_path: Path to watch for changes
            callback: Function called when state changes (new_value, old_value)
            
        Returns:
            Subscription ID for removal
        """
        with self._lock:
            subscription_id = str(uuid.uuid4())
            self._subscribers[state_path].append((subscription_id, callback))
            logger.debug(f"👂 Subscribed to state changes: {state_path}")
            return subscription_id
    
    def unsubscribe(self, state_path: str, subscription_id: str) -> bool:
        """Remove a state subscription."""
        try:
            with self._lock:
                subscribers = self._subscribers.get(state_path, [])
                for i, (sub_id, callback) in enumerate(subscribers):
                    if sub_id == subscription_id:
                        del subscribers[i]
                        logger.debug(f"🗑️ Unsubscribed from state: {state_path}")
                        return True
                return False
        except Exception as e:
            logger.error(f"❌ Failed to unsubscribe: {e}")
            return False
    
    def _notify_subscribers(self, state_path: str, new_value: Any, old_value: Any):
        """Notify subscribers of state changes."""
        # Notify exact path subscribers
        subscribers = self._subscribers.get(state_path, [])
        for sub_id, callback in subscribers[:]:  # Copy to avoid modification during iteration
            try:
                callback(new_value, old_value)
            except Exception as e:
                logger.error(f"❌ Error in state subscriber for {state_path}: {e}")
        
        # Notify wildcard subscribers (parent paths)
        parts = state_path.split('.')
        for i in range(len(parts)):
            parent_path = '.'.join(parts[:i+1]) + '.*'
            parent_subscribers = self._subscribers.get(parent_path, [])
            for sub_id, callback in parent_subscribers[:]:
                try:
                    callback(new_value, old_value)
                except Exception as e:
                    logger.error(f"❌ Error in wildcard subscriber for {parent_path}: {e}")
    
    def get_change_history(self, state_path: Optional[str] = None, 
                          limit: int = 50) -> List[Dict[str, Any]]:
        """Get state change history."""
        with self._lock:
            if state_path:
                filtered = [
                    change for change in self._change_history 
                    if change['path'] == state_path or change['path'].startswith(state_path + '.')
                ]
                return filtered[-limit:]
            else:
                return self._change_history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get state management statistics."""
        with self._lock:
            return {
                'total_changes': len(self._change_history),
                'active_subscriptions': {
                    path: len(subs) for path, subs in self._subscribers.items()
                },
                'state_size': len(str(self._state)),
                'last_update': self._change_history[-1]['timestamp'] if self._change_history else None
            }
    
    def reset_state(self, preserve_user: bool = True) -> None:
        """Reset application state."""
        with self._lock:
            if preserve_user:
                user_state = copy.deepcopy(self._state.user)
                self._state = AppState()
                self._state.user = user_state
            else:
                self._state = AppState()
            
            self._change_history.clear()
            
            logger.info("🔄 Application state reset")
            
            # Emit reset event
            self._event_emitter.emit(
                "state_reset",
                "state_manager",
                {"preserve_user": preserve_user}
            )


# Global state manager instance
_global_state_manager: Optional[StateManager] = None

def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = StateManager()
    return _global_state_manager

def get_state(state_path: Optional[str] = None) -> Union[AppState, Any]:
    """Convenience function to get state."""
    return get_state_manager().get_state(state_path)

def update_state(state_path: str, value: Any, emit_event: bool = True, 
                source: str = "unknown") -> bool:
    """Convenience function to update state."""
    return get_state_manager().update_state(state_path, value, emit_event, source)

def subscribe_to_state(state_path: str, callback: Callable[[Any, Any], None]) -> str:
    """Convenience function to subscribe to state changes."""
    return get_state_manager().subscribe(state_path, callback)

def unsubscribe_from_state(state_path: str, subscription_id: str) -> bool:
    """Convenience function to unsubscribe from state changes."""
    return get_state_manager().unsubscribe(state_path, subscription_id)

def get_state_stats() -> Dict[str, Any]:
    """Convenience function to get state statistics."""
    return get_state_manager().get_stats()

# Initialize state management system when module is imported
def initialize_state_system():
    """Initialize the global state management system."""
    state_manager = get_state_manager()
    
    # Set initial system state
    state_manager.update_state("system.startup_time", datetime.now().isoformat(), False)
    state_manager.update_state("system.is_healthy", True, False)
    state_manager.update_state("system.active_sessions", 0, False)
    
    logger.info("🚀 Horizon State Management System initialized")
    
    # Emit initialization event
    event_emitter = get_event_emitter()
    event_emitter.emit(
        HorizonEventTypes.SYSTEM_INITIALIZED,
        "state_manager",
        {"component": "state_management", "version": "1.0"}
    )

# Auto-initialize when module is imported
if __name__ != "__main__":
    initialize_state_system()