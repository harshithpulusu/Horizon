"""
Horizon Event System Core Module

This module provides a centralized event-driven architecture for the Horizon AI Assistant.
Components can emit and listen for events, enabling loose coupling and better modularity.

Classes:
- EventEmitter: Main event management system
- EventHandler: Base class for event handlers
- EventData: Structured event data container

Functions:
- emit_event: Emit an event to all listeners
- listen_for_event: Register an event listener
- remove_listener: Remove an event listener
"""

import uuid
import json
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import weakref
import logging

# Configure logging for events
logger = logging.getLogger('horizon_events')


@dataclass
class EventData:
    """Structured container for event data."""
    event_type: str
    source: str
    data: Dict[str, Any]
    timestamp: str
    event_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: str = 'normal'  # low, normal, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event data to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert event data to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class EventHandler:
    """Base class for event handlers."""
    
    def __init__(self, name: str):
        """Initialize event handler."""
        self.name = name
        self.handled_events: List[str] = []
    
    def can_handle(self, event_type: str) -> bool:
        """Check if this handler can handle the event type."""
        return event_type in self.handled_events
    
    async def handle_event(self, event: EventData) -> Optional[Any]:
        """Handle an event. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement handle_event")
    
    def handle_event_sync(self, event: EventData) -> Optional[Any]:
        """Handle an event synchronously. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement handle_event_sync")


class EventEmitter:
    """Main event management system for Horizon."""
    
    def __init__(self):
        """Initialize the event emitter."""
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._event_history: List[EventData] = []
        self._max_history = 1000
        self._running = True
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Event statistics
        self._stats = {
            'events_emitted': 0,
            'events_handled': 0,
            'handlers_registered': 0,
            'listeners_registered': 0
        }
        
        logger.info("🔄 Event Emitter initialized")
    
    def emit(self, event_type: str, source: str, data: Dict[str, Any], 
             user_id: Optional[str] = None, session_id: Optional[str] = None,
             priority: str = 'normal') -> str:
        """
        Emit an event to all registered listeners and handlers.
        
        Args:
            event_type: Type of event (e.g., 'user_message', 'ai_response_ready')
            source: Source module/component name
            data: Event data payload
            user_id: Optional user identifier
            session_id: Optional session identifier
            priority: Event priority level
            
        Returns:
            Event ID string
        """
        if not self._running:
            return ""
        
        # Create event data
        event = EventData(
            event_type=event_type,
            source=source,
            data=data,
            timestamp=datetime.now().isoformat(),
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            priority=priority
        )
        
        with self._lock:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            # Update stats
            self._stats['events_emitted'] += 1
            
            logger.debug(f"📡 Emitting event: {event_type} from {source}")
            
            # Notify synchronous listeners first
            listeners = self._listeners.get(event_type, [])
            for listener in listeners[:]:  # Copy to avoid modification during iteration
                try:
                    if listener:  # Check if listener still exists
                        listener(event)
                        self._stats['events_handled'] += 1
                except Exception as e:
                    logger.error(f"❌ Error in event listener for {event_type}: {e}")
                    # Remove broken listeners
                    if listener in self._listeners[event_type]:
                        self._listeners[event_type].remove(listener)
            
            # Notify handlers
            handlers = self._handlers.get(event_type, [])
            for handler in handlers[:]:  # Copy to avoid modification during iteration
                try:
                    if hasattr(handler, 'handle_event_sync'):
                        handler.handle_event_sync(event)
                        self._stats['events_handled'] += 1
                except Exception as e:
                    logger.error(f"❌ Error in event handler {handler.name} for {event_type}: {e}")
        
        return event.event_id
    
    def listen(self, event_type: str, callback: Callable[[EventData], None]) -> str:
        """
        Register a listener for an event type.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
            
        Returns:
            Listener ID for removal
        """
        with self._lock:
            self._listeners[event_type].append(callback)
            listener_id = str(uuid.uuid4())
            self._stats['listeners_registered'] += 1
            
            logger.debug(f"👂 Registered listener for {event_type}")
            return listener_id
    
    def register_handler(self, event_type: str, handler: EventHandler) -> bool:
        """
        Register an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Event handler instance
            
        Returns:
            True if successful
        """
        try:
            with self._lock:
                self._handlers[event_type].append(handler)
                self._stats['handlers_registered'] += 1
                
                logger.info(f"🔧 Registered handler '{handler.name}' for {event_type}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to register handler: {e}")
            return False
    
    def remove_listener(self, event_type: str, callback: Callable) -> bool:
        """Remove a listener for an event type."""
        try:
            with self._lock:
                if callback in self._listeners[event_type]:
                    self._listeners[event_type].remove(callback)
                    logger.debug(f"🗑️ Removed listener for {event_type}")
                    return True
                return False
        except Exception as e:
            logger.error(f"❌ Error removing listener: {e}")
            return False
    
    def remove_handler(self, event_type: str, handler: EventHandler) -> bool:
        """Remove an event handler."""
        try:
            with self._lock:
                if handler in self._handlers[event_type]:
                    self._handlers[event_type].remove(handler)
                    logger.info(f"🗑️ Removed handler '{handler.name}' for {event_type}")
                    return True
                return False
        except Exception as e:
            logger.error(f"❌ Error removing handler: {e}")
            return False
    
    def get_event_history(self, event_type: Optional[str] = None, 
                         limit: int = 100) -> List[EventData]:
        """Get event history, optionally filtered by type."""
        with self._lock:
            if event_type:
                filtered = [e for e in self._event_history if e.event_type == event_type]
                return filtered[-limit:]
            else:
                return self._event_history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics."""
        with self._lock:
            return {
                **self._stats,
                'active_listeners': {
                    event_type: len(listeners) 
                    for event_type, listeners in self._listeners.items()
                },
                'active_handlers': {
                    event_type: len(handlers) 
                    for event_type, handlers in self._handlers.items()
                },
                'history_size': len(self._event_history)
            }
    
    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()
            logger.info("🧹 Event history cleared")
    
    def shutdown(self) -> None:
        """Shutdown the event system."""
        self._running = False
        with self._lock:
            self._listeners.clear()
            self._handlers.clear()
            logger.info("🛑 Event Emitter shutdown")


class HorizonEventTypes:
    """Standard event types for Horizon AI Assistant."""
    
    # User interaction events
    USER_MESSAGE_RECEIVED = "user_message_received"
    USER_SESSION_STARTED = "user_session_started"
    USER_SESSION_ENDED = "user_session_ended"
    USER_PERSONALITY_CHANGED = "user_personality_changed"
    
    # AI processing events
    AI_PROCESSING_STARTED = "ai_processing_started"
    AI_RESPONSE_GENERATED = "ai_response_generated"
    AI_RESPONSE_READY = "ai_response_ready"
    AI_ERROR_OCCURRED = "ai_error_occurred"
    AI_MODEL_SWITCHED = "ai_model_switched"
    
    # Media generation events
    MEDIA_GENERATION_REQUESTED = "media_generation_requested"
    MEDIA_GENERATION_STARTED = "media_generation_started"
    MEDIA_GENERATION_COMPLETED = "media_generation_completed"
    MEDIA_GENERATION_FAILED = "media_generation_failed"
    
    # System events
    SYSTEM_INITIALIZED = "system_initialized"
    SYSTEM_ERROR = "system_error"
    SYSTEM_SHUTDOWN = "system_shutdown"
    DATABASE_UPDATED = "database_updated"
    
    # Memory and learning events
    MEMORY_UPDATED = "memory_updated"
    PATTERN_LEARNED = "pattern_learned"
    PREFERENCE_SAVED = "preference_saved"
    
    # API and external service events
    API_REQUEST_STARTED = "api_request_started"
    API_REQUEST_COMPLETED = "api_request_completed"
    API_REQUEST_FAILED = "api_request_failed"
    
    # Web interface events
    WEB_REQUEST_RECEIVED = "web_request_received"
    WEB_RESPONSE_SENT = "web_response_sent"
    WEB_ERROR_OCCURRED = "web_error_occurred"
    
    # Timer events
    TIMER_CREATED = "timer_created"
    TIMER_STARTED = "timer_started"
    TIMER_PAUSED = "timer_paused"
    TIMER_RESUMED = "timer_resumed"
    TIMER_STOPPED = "timer_stopped"
    TIMER_COMPLETED = "timer_completed"
    TIMER_UPDATED = "timer_updated"
    TIMER_DELETED = "timer_deleted"
    
    # Reminder events
    REMINDER_CREATED = "reminder_created"
    REMINDER_UPDATED = "reminder_updated"
    REMINDER_DUE = "reminder_due"
    REMINDER_SNOOZED = "reminder_snoozed"
    REMINDER_COMPLETED = "reminder_completed"
    REMINDER_DELETED = "reminder_deleted"
    
    # Real-time communication events
    WEBSOCKET_CONNECTED = "websocket_connected"
    WEBSOCKET_DISCONNECTED = "websocket_disconnected"
    REALTIME_UPDATE_SENT = "realtime_update_sent"


# Global event emitter instance
_global_event_emitter: Optional[EventEmitter] = None

def get_event_emitter() -> EventEmitter:
    """Get the global event emitter instance."""
    global _global_event_emitter
    if _global_event_emitter is None:
        _global_event_emitter = EventEmitter()
    return _global_event_emitter

def emit_event(event_type: str, source: str, data: Dict[str, Any], 
               user_id: Optional[str] = None, session_id: Optional[str] = None,
               priority: str = 'normal') -> str:
    """Convenience function to emit an event."""
    return get_event_emitter().emit(event_type, source, data, user_id, session_id, priority)

def listen_for_event(event_type: str, callback: Callable[[EventData], None]) -> str:
    """Convenience function to listen for an event."""
    return get_event_emitter().listen(event_type, callback)

def register_event_handler(event_type: str, handler: EventHandler) -> bool:
    """Convenience function to register an event handler."""
    return get_event_emitter().register_handler(event_type, handler)

def remove_event_listener(event_type: str, callback: Callable) -> bool:
    """Convenience function to remove an event listener."""
    return get_event_emitter().remove_listener(event_type, callback)

def get_event_stats() -> Dict[str, Any]:
    """Convenience function to get event statistics."""
    return get_event_emitter().get_stats()

def get_event_history(event_type: Optional[str] = None, limit: int = 100) -> List[EventData]:
    """Convenience function to get event history."""
    return get_event_emitter().get_event_history(event_type, limit)

# Predefined event handlers for common operations

class LoggingEventHandler(EventHandler):
    """Event handler that logs all events."""
    
    def __init__(self):
        super().__init__("logging_handler")
        self.handled_events = ['*']  # Handle all events
    
    def can_handle(self, event_type: str) -> bool:
        return True  # Handle all events
    
    def handle_event_sync(self, event: EventData) -> None:
        """Log the event."""
        logger.info(f"📝 Event logged: {event.event_type} from {event.source} at {event.timestamp}")


class DebugEventHandler(EventHandler):
    """Event handler for debugging purposes."""
    
    def __init__(self):
        super().__init__("debug_handler")
        self.handled_events = ['*']
        self.debug_enabled = False
    
    def can_handle(self, event_type: str) -> bool:
        return self.debug_enabled
    
    def handle_event_sync(self, event: EventData) -> None:
        """Debug print the event."""
        if self.debug_enabled:
            print(f"🐛 DEBUG EVENT: {event.event_type}")
            print(f"   Source: {event.source}")
            print(f"   Data: {event.data}")
            print(f"   Time: {event.timestamp}")
    
    def enable_debug(self):
        """Enable debug logging."""
        self.debug_enabled = True
    
    def disable_debug(self):
        """Disable debug logging."""
        self.debug_enabled = False


# Initialize event system when module is imported
def initialize_event_system():
    """Initialize the global event system."""
    emitter = get_event_emitter()
    
    # Register default handlers
    logging_handler = LoggingEventHandler()
    debug_handler = DebugEventHandler()
    
    # Register for system events
    emitter.register_handler(HorizonEventTypes.SYSTEM_INITIALIZED, logging_handler)
    emitter.register_handler(HorizonEventTypes.SYSTEM_ERROR, logging_handler)
    
    logger.info("🚀 Horizon Event System initialized")
    
    # Emit system initialization event
    emitter.emit(
        HorizonEventTypes.SYSTEM_INITIALIZED,
        "event_system",
        {"component": "event_system", "version": "1.0"}
    )

# Auto-initialize when module is imported
if __name__ != "__main__":
    initialize_event_system()