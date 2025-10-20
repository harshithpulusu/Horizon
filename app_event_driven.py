#!/usr/bin/env python3
"""
Horizon AI Assistant - Event-Driven Architecture Version
Advanced AI features with event-driven component architecture and centralized state management
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import os
import time
import json
import uuid
from typing import Dict, Any, Optional

# Import core modules with new architecture
from core.events import (
    get_event_emitter, emit_event, listen_for_event, 
    HorizonEventTypes, EventData, EventHandler
)
from core.state_manager import (
    get_state_manager, get_state, update_state, 
    subscribe_to_state
)
from core.ai_engine import get_ai_engine
from core.media_generator import get_enhanced_media_engine
from config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global references to core systems
event_emitter = None
state_manager = None
ai_engine = None
media_engine = None


class WebEventHandler(EventHandler):
    """Event handler for web interface events."""
    
    def __init__(self):
        super().__init__("web_event_handler")
        self.handled_events = [
            HorizonEventTypes.AI_RESPONSE_READY,
            HorizonEventTypes.MEDIA_GENERATION_COMPLETED,
            HorizonEventTypes.SYSTEM_ERROR
        ]
    
    def handle_event_sync(self, event: EventData) -> None:
        """Handle web-related events."""
        try:
            if event.event_type == HorizonEventTypes.AI_RESPONSE_READY:
                self._handle_ai_response_ready(event)
            elif event.event_type == HorizonEventTypes.MEDIA_GENERATION_COMPLETED:
                self._handle_media_completed(event)
            elif event.event_type == HorizonEventTypes.SYSTEM_ERROR:
                self._handle_system_error(event)
        except Exception as e:
            print(f"❌ Error in web event handler: {e}")
    
    def _handle_ai_response_ready(self, event: EventData):
        """Handle AI response ready events."""
        print(f"🌐 Web interface: AI response ready for session {event.session_id}")
    
    def _handle_media_completed(self, event: EventData):
        """Handle media generation completion."""
        print(f"🎨 Web interface: Media generation completed - {event.data.get('media_type')}")
    
    def _handle_system_error(self, event: EventData):
        """Handle system errors."""
        print(f"⚠️ Web interface: System error - {event.data.get('error')}")


def initialize_horizon_systems():
    """Initialize all Horizon core systems."""
    global event_emitter, state_manager, ai_engine, media_engine
    
    print("🚀 Initializing Horizon AI Assistant with Event-Driven Architecture...")
    
    # Initialize core systems (they auto-initialize when imported)
    event_emitter = get_event_emitter()
    state_manager = get_state_manager()
    ai_engine = get_ai_engine()
    media_engine = get_enhanced_media_engine()
    
    # Register web event handler
    web_handler = WebEventHandler()
    event_emitter.register_handler(HorizonEventTypes.AI_RESPONSE_READY, web_handler)
    event_emitter.register_handler(HorizonEventTypes.MEDIA_GENERATION_COMPLETED, web_handler)
    event_emitter.register_handler(HorizonEventTypes.SYSTEM_ERROR, web_handler)
    
    # Set initial system state
    update_state("system.app_version", "2.0.0", source="main_app")
    update_state("system.debug_mode", False, source="main_app")
    update_state("system.features_enabled.event_driven_architecture", True, source="main_app")
    
    # Emit system initialization complete event
    emit_event(
        HorizonEventTypes.SYSTEM_INITIALIZED,
        "main_app",
        {
            "component": "horizon_app",
            "version": "2.0.0",
            "architecture": "event_driven",
            "features": ["ai_engine", "media_generation", "state_management", "event_system"]
        }
    )
    
    print("✅ Horizon systems initialized successfully!")


def create_session_if_needed(session_id: Optional[str] = None) -> str:
    """Create a new session if needed."""
    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Update session state
        update_state("conversation.session_id", session_id, source="main_app")
        update_state("system.active_sessions", get_state("system.active_sessions") + 1, source="main_app")
        
        # Emit session started event
        emit_event(
            HorizonEventTypes.USER_SESSION_STARTED,
            "main_app",
            {"session_id": session_id},
            session_id=session_id
        )
    
    return session_id


def process_user_input_event_driven(user_input: str, personality: str = 'friendly', 
                                   session_id: Optional[str] = None, 
                                   user_id: str = 'anonymous') -> Dict[str, Any]:
    """
    Process user input using event-driven architecture.
    
    Args:
        user_input: The user's input message
        personality: The personality type to use
        session_id: Optional session ID for context
        user_id: User identifier for memory
        
    Returns:
        Dictionary with processing results
    """
    if not user_input or not user_input.strip():
        return {
            'error': 'Empty input provided',
            'success': False
        }
    
    # Create session if needed
    session_id = create_session_if_needed(session_id)
    
    # Update user state
    update_state("user.user_id", user_id, source="main_app")
    update_state("user.personality_mode", personality, source="main_app")
    update_state("user.total_interactions", get_state("user.total_interactions") + 1, source="main_app")
    
    # Emit user message event
    emit_event(
        HorizonEventTypes.USER_MESSAGE_RECEIVED,
        "web_interface",
        {
            'message': user_input,
            'personality': personality,
            'timestamp': datetime.now().isoformat()
        },
        user_id=user_id,
        session_id=session_id
    )
    
    # Process with AI engine (this will trigger AI events internally)
    start_time = time.time()
    response, context_used = ai_engine.ask_ai_model(user_input, personality, session_id, user_id)
    processing_time = time.time() - start_time
    
    # Get current conversation state
    conversation_state = get_state("conversation")
    ai_state = get_state("ai")
    
    # Add message to conversation history
    conversation_state.add_message(
        user_input, 
        response, 
        intent=recognize_intent_simple(user_input),
        sentiment=conversation_state.sentiment
    )
    update_state("conversation", conversation_state, source="main_app")
    
    # Prepare response data
    response_data = {
        'response': response,
        'timestamp': datetime.now().isoformat(),
        'personality': personality,
        'session_id': session_id,
        'user_id': user_id,
        'context_used': context_used,
        'processing_time': f"{processing_time:.2f}s",
        'ai_source': 'chatgpt' if context_used else 'fallback',
        'intent': recognize_intent_simple(user_input),
        'sentiment': conversation_state.sentiment,
        'mood': conversation_state.mood,
        'total_messages': conversation_state.total_messages,
        'ai_model': ai_state.current_model,
        'success': True
    }
    
    # Emit web response event
    emit_event(
        HorizonEventTypes.WEB_RESPONSE_SENT,
        "web_interface",
        response_data,
        user_id=user_id,
        session_id=session_id
    )
    
    return response_data


def recognize_intent_simple(text: str) -> str:
    """Simple intent recognition for basic commands."""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning']):
        return 'greeting'
    elif 'time' in text_lower:
        return 'time'
    elif 'date' in text_lower:
        return 'date'
    elif any(op in text_lower for op in ['+', '-', '*', '/', 'calculate', 'math']):
        return 'math'
    elif any(phrase in text_lower for phrase in ['generate image', 'create image', 'make image']):
        return 'image_generation'
    elif any(phrase in text_lower for phrase in ['generate logo', 'create logo', 'make logo']):
        return 'logo_generation'
    else:
        return 'general'


# Flask Routes

@app.route('/')
def home():
    """Serve the main web interface."""
    # Emit web request event
    emit_event(
        HorizonEventTypes.WEB_REQUEST_RECEIVED,
        "web_interface",
        {"endpoint": "/", "method": "GET"}
    )
    
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Enhanced chat endpoint with event-driven processing."""
    try:
        # Emit web request event
        emit_event(
            HorizonEventTypes.WEB_REQUEST_RECEIVED,
            "web_interface",
            {"endpoint": "/chat", "method": "POST"}
        )
        
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_input = data.get('message', '').strip()
        personality = data.get('personality', 'friendly')
        session_id = data.get('session_id')
        user_id = data.get('user_id', 'anonymous')
        
        if not user_input:
            return jsonify({'error': 'Empty message provided'}), 400
        
        # Process using event-driven architecture
        result = process_user_input_event_driven(user_input, personality, session_id, user_id)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
        
    except Exception as e:
        print(f"Chat API error: {e}")
        
        # Emit error event
        emit_event(
            HorizonEventTypes.WEB_ERROR_OCCURRED,
            "web_interface",
            {"error": str(e), "endpoint": "/chat"}
        )
        
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint with system state."""
    try:
        # Get current system state
        system_state = get_state("system")
        ai_state = get_state("ai")
        
        health_data = {
            'status': 'healthy' if system_state.is_healthy else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'version': system_state.app_version,
            'uptime': system_state.startup_time,
            'ai_available': len(ai_state.available_models) > 0,
            'active_sessions': system_state.active_sessions,
            'event_system': True,
            'state_management': True,
            'components': {
                'ai_engine': ai_state.current_model if ai_state.available_models else 'unavailable',
                'media_engine': 'available',
                'database': system_state.database_connected,
                'event_emitter': True,
                'state_manager': True
            }
        }
        
        return jsonify(health_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/state')
def get_app_state():
    """Get current application state (for debugging/monitoring)."""
    try:
        # Get full application state
        app_state = get_state()
        
        # Convert to dict for JSON serialization
        state_dict = app_state.to_dict()
        
        return jsonify({
            'state': state_dict,
            'timestamp': datetime.now().isoformat(),
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/events/history')
def get_event_history():
    """Get recent event history."""
    try:
        event_type = request.args.get('type')
        limit = int(request.args.get('limit', 50))
        
        # Get event history from event emitter
        history = event_emitter.get_event_history(event_type, limit)
        
        # Convert events to dict format
        history_data = [event.to_dict() for event in history]
        
        return jsonify({
            'events': history_data,
            'count': len(history_data),
            'filter': event_type,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/stats')
def get_system_stats():
    """Get comprehensive system statistics."""
    try:
        # Get stats from all systems
        event_stats = event_emitter.get_stats()
        state_stats = state_manager.get_stats()
        
        # Get state data
        system_state = get_state("system")
        ai_state = get_state("ai")
        conversation_state = get_state("conversation")
        user_state = get_state("user")
        media_state = get_state("media")
        
        stats = {
            'system': {
                'uptime': system_state.startup_time,
                'version': system_state.app_version,
                'active_sessions': system_state.active_sessions,
                'total_errors': system_state.error_count,
                'is_healthy': system_state.is_healthy
            },
            'ai': {
                'current_model': ai_state.current_model,
                'total_requests': ai_state.total_requests,
                'successful_requests': ai_state.successful_requests,
                'failed_requests': ai_state.failed_requests,
                'average_processing_time': ai_state.processing_time_avg,
                'fallback_responses_used': ai_state.fallback_responses_used
            },
            'conversation': {
                'total_messages': conversation_state.total_messages,
                'current_session': conversation_state.session_id,
                'average_sentiment': conversation_state.sentiment
            },
            'user': {
                'total_interactions': user_state.total_interactions,
                'sessions_count': user_state.sessions_count,
                'personality_mode': user_state.personality_mode
            },
            'media': {
                'total_images': media_state.total_images_generated,
                'total_videos': media_state.total_videos_generated,
                'is_generating': media_state.is_generating,
                'queue_length': len(media_state.generation_queue)
            },
            'events': event_stats,
            'state_management': state_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/personalities')
def get_personalities():
    """Get available personality types."""
    personalities = {
        'friendly': {
            'name': 'Friendly',
            'description': 'Warm, encouraging, and conversational',
            'traits': ['empathetic', 'encouraging', 'casual']
        },
        'professional': {
            'name': 'Professional', 
            'description': 'Formal, structured, and business-oriented',
            'traits': ['formal', 'precise', 'business-focused']
        },
        'casual': {
            'name': 'Casual',
            'description': 'Relaxed and laid-back',
            'traits': ['relaxed', 'informal', 'buddy-like']
        },
        'enthusiastic': {
            'name': 'Enthusiastic',
            'description': 'High-energy and excited',
            'traits': ['energetic', 'excited', 'motivational']
        }
    }
    
    return jsonify({
        'personalities': personalities,
        'default': 'friendly',
        'count': len(personalities)
    })


if __name__ == '__main__':
    print("🚀 Starting Horizon AI Assistant (Event-Driven Architecture)...")
    
    # Initialize all core systems
    initialize_horizon_systems()
    
    print("🌐 Server starting on http://0.0.0.0:8080...")
    print("📱 Local access: http://127.0.0.1:8080")
    print("🔄 Event-driven architecture active")
    print("🗃️ Centralized state management active")
    
    app.run(host='0.0.0.0', port=8080, debug=False)