"""
Horizon WebSocket Real-time Communication Module

This module provides WebSocket integration for real-time updates.
Enables live timer notifications, chat updates, and synchronized data.

Features:
- Real-time timer countdown updates
- Live chat with AI
- Timer completion notifications
- Multi-device synchronization
- Room-based communication
"""

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect

# Import our core systems
from .events import get_event_emitter, listen_for_event, HorizonEventTypes, EventHandler
from .database import get_timer_manager, get_reminder_manager
from .state_manager import get_state_manager, update_state

class HorizonWebSocketManager:
    """Manages WebSocket connections and real-time updates."""
    
    def __init__(self, socketio: SocketIO):
        """Initialize WebSocket manager."""
        self.socketio = socketio
        self.connected_clients = {}  # session_id -> client_info
        self.user_rooms = {}  # user_id -> list of session_ids
        self.timer_threads = {}  # timer_id -> thread
        
        # Register event handlers
        self._register_event_handlers()
        
        print("🔄 WebSocket Manager initialized")
    
    def _register_event_handlers(self):
        """Register event handlers for real-time updates."""
        event_emitter = get_event_emitter()
        
        # Timer events
        listen_for_event(HorizonEventTypes.TIMER_CREATED, self._on_timer_created)
        listen_for_event(HorizonEventTypes.TIMER_STARTED, self._on_timer_started)
        listen_for_event(HorizonEventTypes.TIMER_PAUSED, self._on_timer_paused)
        listen_for_event(HorizonEventTypes.TIMER_STOPPED, self._on_timer_stopped)
        listen_for_event(HorizonEventTypes.TIMER_COMPLETED, self._on_timer_completed)
        listen_for_event(HorizonEventTypes.TIMER_UPDATED, self._on_timer_updated)
        listen_for_event(HorizonEventTypes.TIMER_DELETED, self._on_timer_deleted)
        
        # Reminder events
        listen_for_event(HorizonEventTypes.REMINDER_CREATED, self._on_reminder_created)
        listen_for_event(HorizonEventTypes.REMINDER_DUE, self._on_reminder_due)
        listen_for_event(HorizonEventTypes.REMINDER_UPDATED, self._on_reminder_updated)
        listen_for_event(HorizonEventTypes.REMINDER_COMPLETED, self._on_reminder_completed)
        listen_for_event(HorizonEventTypes.REMINDER_DELETED, self._on_reminder_deleted)
        
        # AI events for live chat
        listen_for_event(HorizonEventTypes.AI_PROCESSING_STARTED, self._on_ai_processing_started)
        listen_for_event(HorizonEventTypes.AI_RESPONSE_GENERATED, self._on_ai_response_generated)
    
    def register_client(self, session_id: str, user_id: str, client_info: Dict[str, Any]):
        """Register a new WebSocket client."""
        self.connected_clients[session_id] = {
            'user_id': user_id,
            'connected_at': datetime.now(),
            'client_info': client_info
        }
        
        # Add to user room
        if user_id not in self.user_rooms:
            self.user_rooms[user_id] = []
        self.user_rooms[user_id].append(session_id)
        
        print(f"🔌 Client registered: {session_id} for user {user_id}")
    
    def unregister_client(self, session_id: str):
        """Unregister a WebSocket client."""
        if session_id in self.connected_clients:
            client = self.connected_clients[session_id]
            user_id = client['user_id']
            
            # Remove from user room
            if user_id in self.user_rooms:
                self.user_rooms[user_id] = [sid for sid in self.user_rooms[user_id] if sid != session_id]
                if not self.user_rooms[user_id]:
                    del self.user_rooms[user_id]
            
            del self.connected_clients[session_id]
            print(f"🔌 Client unregistered: {session_id}")
    
    def emit_to_user(self, user_id: str, event: str, data: Dict[str, Any]):
        """Emit event to all sessions for a specific user."""
        if user_id in self.user_rooms:
            for session_id in self.user_rooms[user_id]:
                self.socketio.emit(event, data, room=session_id)
    
    def emit_to_all(self, event: str, data: Dict[str, Any]):
        """Emit event to all connected clients."""
        self.socketio.emit(event, data, broadcast=True)
    
    def start_timer_countdown(self, timer_id: str, user_id: str):
        """Start real-time countdown for a timer."""
        def countdown_worker():
            timer_manager = get_timer_manager()
            
            while True:
                timer = timer_manager.get_timer(timer_id)
                if not timer or timer['status'] != 'running':
                    break
                
                # Calculate remaining time
                end_time = datetime.fromisoformat(timer['end_time']) if isinstance(timer['end_time'], str) else timer['end_time']
                now = datetime.now()
                remaining = end_time - now
                
                if remaining.total_seconds() <= 0:
                    # Timer completed
                    timer_manager.complete_timer(timer_id)
                    self.emit_to_user(user_id, 'timer_countdown', {
                        'timer_id': timer_id,
                        'remaining_seconds': 0,
                        'status': 'completed',
                        'message': f"Timer '{timer['title']}' completed!"
                    })
                    break
                
                # Send countdown update
                self.emit_to_user(user_id, 'timer_countdown', {
                    'timer_id': timer_id,
                    'remaining_seconds': int(remaining.total_seconds()),
                    'status': 'running',
                    'timer': timer
                })
                
                time.sleep(1)  # Update every second
            
            # Clean up thread reference
            if timer_id in self.timer_threads:
                del self.timer_threads[timer_id]
        
        # Start countdown thread
        if timer_id not in self.timer_threads:
            thread = threading.Thread(target=countdown_worker, daemon=True)
            thread.start()
            self.timer_threads[timer_id] = thread
    
    def stop_timer_countdown(self, timer_id: str):
        """Stop countdown for a timer."""
        if timer_id in self.timer_threads:
            # Thread will stop automatically when timer status changes
            del self.timer_threads[timer_id]
    
    # Event handlers
    def _on_timer_created(self, event_data):
        """Handle timer created event."""
        timer = event_data.data.get('timer')
        user_id = event_data.data.get('user_id')
        
        if timer and user_id:
            self.emit_to_user(user_id, 'timer_created', {
                'timer': timer,
                'message': f"Timer '{timer['title']}' created"
            })
    
    def _on_timer_started(self, event_data):
        """Handle timer started event."""
        timer = event_data.data.get('timer')
        user_id = event_data.data.get('user_id')
        
        if timer and user_id:
            self.emit_to_user(user_id, 'timer_started', {
                'timer': timer,
                'message': f"Timer '{timer['title']}' started"
            })
            
            # Start real-time countdown
            self.start_timer_countdown(timer['id'], user_id)
    
    def _on_timer_paused(self, event_data):
        """Handle timer paused event."""
        timer = event_data.data.get('timer')
        user_id = event_data.data.get('user_id')
        
        if timer and user_id:
            self.emit_to_user(user_id, 'timer_paused', {
                'timer': timer,
                'message': f"Timer '{timer['title']}' paused"
            })
            
            # Stop countdown
            self.stop_timer_countdown(timer['id'])
    
    def _on_timer_stopped(self, event_data):
        """Handle timer stopped event."""
        timer = event_data.data.get('timer')
        user_id = event_data.data.get('user_id')
        
        if timer and user_id:
            self.emit_to_user(user_id, 'timer_stopped', {
                'timer': timer,
                'message': f"Timer '{timer['title']}' stopped"
            })
            
            # Stop countdown
            self.stop_timer_countdown(timer['id'])
    
    def _on_timer_completed(self, event_data):
        """Handle timer completed event."""
        timer = event_data.data.get('timer')
        user_id = event_data.data.get('user_id')
        
        if timer and user_id:
            self.emit_to_user(user_id, 'timer_completed', {
                'timer': timer,
                'message': f"⏰ Timer '{timer['title']}' completed!",
                'notification': True
            })
            
            # Stop countdown
            self.stop_timer_countdown(timer['id'])
    
    def _on_timer_updated(self, event_data):
        """Handle timer updated event."""
        timer = event_data.data.get('timer')
        user_id = event_data.data.get('user_id')
        
        if timer and user_id:
            self.emit_to_user(user_id, 'timer_updated', {
                'timer': timer,
                'message': f"Timer '{timer['title']}' updated"
            })
    
    def _on_timer_deleted(self, event_data):
        """Handle timer deleted event."""
        timer = event_data.data.get('timer')
        user_id = event_data.data.get('user_id')
        
        if timer and user_id:
            self.emit_to_user(user_id, 'timer_deleted', {
                'timer': timer,
                'message': f"Timer '{timer['title']}' deleted"
            })
            
            # Stop countdown
            self.stop_timer_countdown(timer['id'])
    
    def _on_reminder_created(self, event_data):
        """Handle reminder created event."""
        reminder = event_data.data.get('reminder')
        user_id = event_data.data.get('user_id')
        
        if reminder and user_id:
            self.emit_to_user(user_id, 'reminder_created', {
                'reminder': reminder,
                'message': f"Reminder '{reminder['title']}' created"
            })
    
    def _on_reminder_due(self, event_data):
        """Handle reminder due event."""
        reminder = event_data.data.get('reminder')
        user_id = event_data.data.get('user_id')
        
        if reminder and user_id:
            self.emit_to_user(user_id, 'reminder_due', {
                'reminder': reminder,
                'message': f"🔔 Reminder: {reminder['title']}",
                'notification': True
            })
    
    def _on_reminder_updated(self, event_data):
        """Handle reminder updated event."""
        reminder = event_data.data.get('reminder')
        user_id = event_data.data.get('user_id')
        
        if reminder and user_id:
            self.emit_to_user(user_id, 'reminder_updated', {
                'reminder': reminder,
                'message': f"Reminder '{reminder['title']}' updated"
            })
    
    def _on_reminder_completed(self, event_data):
        """Handle reminder completed event."""
        reminder = event_data.data.get('reminder')
        user_id = event_data.data.get('user_id')
        
        if reminder and user_id:
            self.emit_to_user(user_id, 'reminder_completed', {
                'reminder': reminder,
                'message': f"Reminder '{reminder['title']}' completed"
            })
    
    def _on_reminder_deleted(self, event_data):
        """Handle reminder deleted event."""
        reminder = event_data.data.get('reminder')
        user_id = event_data.data.get('user_id')
        
        if reminder and user_id:
            self.emit_to_user(user_id, 'reminder_deleted', {
                'reminder': reminder,
                'message': f"Reminder '{reminder['title']}' deleted"
            })
    
    def _on_ai_processing_started(self, event_data):
        """Handle AI processing started event."""
        user_id = event_data.data.get('user_id')
        
        if user_id:
            self.emit_to_user(user_id, 'ai_typing', {
                'typing': True,
                'message': 'AI is thinking...'
            })
    
    def _on_ai_response_generated(self, event_data):
        """Handle AI response generated event."""
        user_id = event_data.data.get('user_id')
        response = event_data.data.get('response')
        
        if user_id and response:
            self.emit_to_user(user_id, 'ai_response', {
                'typing': False,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })


# Global WebSocket manager instance
_websocket_manager: Optional[HorizonWebSocketManager] = None

def init_websocket_manager(socketio: SocketIO) -> HorizonWebSocketManager:
    """Initialize the global WebSocket manager."""
    global _websocket_manager
    _websocket_manager = HorizonWebSocketManager(socketio)
    return _websocket_manager

def get_websocket_manager() -> Optional[HorizonWebSocketManager]:
    """Get the global WebSocket manager instance."""
    return _websocket_manager

def setup_websocket_handlers(socketio: SocketIO):
    """Setup WebSocket event handlers."""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        session_id = request.sid
        user_id = request.args.get('user_id', 'default_user')
        client_info = {
            'user_agent': request.headers.get('User-Agent', ''),
            'origin': request.headers.get('Origin', ''),
            'ip': request.remote_addr
        }
        
        ws_manager = get_websocket_manager()
        if ws_manager:
            ws_manager.register_client(session_id, user_id, client_info)
            
            # Join user-specific room
            join_room(f"user_{user_id}")
            
            # Send welcome message
            emit('connected', {
                'message': 'Connected to Horizon AI Assistant',
                'session_id': session_id,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"🔌 WebSocket client connected: {session_id}")
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        session_id = request.sid
        
        ws_manager = get_websocket_manager()
        if ws_manager:
            if session_id in ws_manager.connected_clients:
                user_id = ws_manager.connected_clients[session_id]['user_id']
                leave_room(f"user_{user_id}")
            
            ws_manager.unregister_client(session_id)
            
            print(f"🔌 WebSocket client disconnected: {session_id}")
    
    @socketio.on('join_timer_room')
    def handle_join_timer_room(data):
        """Handle joining timer-specific room."""
        timer_id = data.get('timer_id')
        if timer_id:
            join_room(f"timer_{timer_id}")
            emit('joined_timer_room', {'timer_id': timer_id})
    
    @socketio.on('leave_timer_room')
    def handle_leave_timer_room(data):
        """Handle leaving timer-specific room."""
        timer_id = data.get('timer_id')
        if timer_id:
            leave_room(f"timer_{timer_id}")
            emit('left_timer_room', {'timer_id': timer_id})
    
    @socketio.on('ping')
    def handle_ping():
        """Handle ping for connection health check."""
        emit('pong', {'timestamp': datetime.now().isoformat()})
    
    @socketio.on('get_active_timers')
    def handle_get_active_timers():
        """Handle request for active timers."""
        session_id = request.sid
        ws_manager = get_websocket_manager()
        
        if ws_manager and session_id in ws_manager.connected_clients:
            user_id = ws_manager.connected_clients[session_id]['user_id']
            timer_manager = get_timer_manager()
            active_timers = timer_manager.get_active_timers(user_id)
            
            emit('active_timers', {
                'timers': active_timers,
                'count': len(active_timers)
            })
    
    @socketio.on('get_due_reminders')
    def handle_get_due_reminders():
        """Handle request for due reminders."""
        session_id = request.sid
        ws_manager = get_websocket_manager()
        
        if ws_manager and session_id in ws_manager.connected_clients:
            user_id = ws_manager.connected_clients[session_id]['user_id']
            reminder_manager = get_reminder_manager()
            due_reminders = reminder_manager.get_due_reminders(user_id)
            
            emit('due_reminders', {
                'reminders': due_reminders,
                'count': len(due_reminders)
            })
    
    print("🔄 WebSocket handlers registered")