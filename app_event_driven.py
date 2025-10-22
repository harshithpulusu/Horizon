#!/usr/bin/env python3
"""
Horizon AI Assistant - Event-Driven Architecture Version
Advanced AI features with event-driven component architecture and centralized state management
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
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
from core.timer_api import api_bp
from core.websocket_manager import init_websocket_manager, setup_websocket_handlers
from config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize SocketIO for WebSocket support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Register API Blueprint
app.register_blueprint(api_bp)

# Global references to core systems
event_emitter = None
state_manager = None
ai_engine = None
media_engine = None
websocket_manager = None


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
    global event_emitter, state_manager, ai_engine, media_engine, websocket_manager
    
    print("🚀 Initializing Horizon AI Assistant with Event-Driven Architecture...")
    
    # Initialize core systems (they auto-initialize when imported)
    event_emitter = get_event_emitter()
    state_manager = get_state_manager()
    ai_engine = get_ai_engine()
    media_engine = get_enhanced_media_engine()
    
    # Initialize WebSocket manager
    websocket_manager = init_websocket_manager(socketio)
    setup_websocket_handlers(socketio)
    
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
    
    # Start timing
    start_time = time.time()
    
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
    
    # Recognize intent first
    intent = recognize_intent_simple(user_input)
    
    # Handle special intents that need immediate processing
    special_response = handle_special_intents(user_input, intent)
    if special_response:
        # For special intents, return immediately
        response = special_response
        context_used = False
        processing_time = time.time() - start_time
    else:
        # Process with AI engine (this will trigger AI events internally)
        response, context_used = ai_engine.ask_ai_model(user_input, personality, session_id, user_id)
        processing_time = time.time() - start_time
    
    # Get current conversation state
    conversation_state = get_state("conversation")
    ai_state = get_state("ai")
    
    # Add message to conversation history
    conversation_state.add_message(
        user_input, 
        response, 
        intent=intent,  # Use the intent we already determined
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
        'ai_source': 'chatgpt' if context_used else ('special_intent' if special_response else 'fallback'),
        'intent': intent,  # Use the intent we already determined
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
    elif any(phrase in text_lower for phrase in [
        'generate image', 'create image', 'make image', 'generate picture', 
        'create picture', 'make picture', 'draw', 'create an image', 'generate an image',
        'make an image', 'create a picture', 'generate a picture', 'make a picture'
    ]):
        return 'image_generation'
    elif any(phrase in text_lower for phrase in [
        'generate logo', 'create logo', 'make logo', 'design logo'
    ]):
        return 'logo_generation'
    else:
        return 'general'


def handle_special_intents(user_input: str, intent: str) -> Optional[str]:
    """Handle special intents that need immediate processing."""
    if intent == 'image_generation':
        return handle_image_generation_event_driven(user_input)
    elif intent == 'logo_generation':
        return handle_logo_generation_event_driven(user_input)
    elif intent == 'time':
        return f"The current time is {datetime.now().strftime('%I:%M %p')}"
    elif intent == 'date':
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"
    elif intent == 'math':
        return handle_math_simple(user_input)
    return None


def handle_image_generation_event_driven(user_input: str) -> str:
    """Handle image generation using the event-driven media engine."""
    try:
        # Extract image description
        prompt = extract_image_prompt(user_input)
        print(f"🎨 Processing image generation request: {prompt}")
        
        # Get media engine instance
        from core.media_generator import get_enhanced_media_engine
        media_engine = get_enhanced_media_engine()
        
        # Use media engine to generate image
        result = media_engine.generate_media('image', prompt)
        
        if result.get('success'):
            image_url = result.get('image_url') or result.get('url')
            if image_url:
                return f"🎨 I've created your image! Here it is: <img src='{image_url}' alt='{prompt}' style='max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;' /><br>📝 Prompt: {prompt}"
            else:
                return f"🎨 Image generated successfully! File: {result.get('filename', 'unknown')}<br>📝 Prompt: {prompt}"
        else:
            error_msg = result.get('error', 'Unknown error')
            return f"🎨 I had trouble generating that image. Error: {error_msg}"
            
    except Exception as e:
        print(f"❌ Image generation error: {e}")
        return f"🎨 I encountered an error while generating your image: {str(e)}"


def handle_logo_generation_event_driven(user_input: str) -> str:
    """Handle logo generation using the event-driven media engine."""
    try:
        # Extract brand/company name
        prompt = extract_logo_prompt(user_input)
        enhanced_prompt = f"professional minimalist logo design for {prompt}, clean vector style, modern typography, simple geometric shapes, flat design, corporate logo"
        
        print(f"🎨 Processing logo generation request: {enhanced_prompt}")
        
        # Get media engine instance
        from core.media_generator import get_enhanced_media_engine
        media_engine = get_enhanced_media_engine()
        
        # Use media engine to generate logo
        result = media_engine.generate_media('image', enhanced_prompt)
        
        if result.get('success'):
            image_url = result.get('image_url') or result.get('url')
            if image_url:
                return f"🎨 Your professional logo is ready! Here it is: <img src='{image_url}' alt='Logo for {prompt}' style='max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; background: white; padding: 20px;' /><br>📝 Logo for: {prompt}<br>💡 This is a professional logo design with clean, modern aesthetics!"
            else:
                return f"🎨 Logo generated successfully! File: {result.get('filename', 'unknown')}<br>📝 Logo for: {prompt}"
        else:
            error_msg = result.get('error', 'Unknown error')
            return f"🎨 I had trouble generating that logo. Error: {error_msg}"
            
    except Exception as e:
        print(f"❌ Logo generation error: {e}")
        return f"🎨 I encountered an error while generating your logo: {str(e)}"


def extract_image_prompt(user_input: str) -> str:
    """Extract image description from user input."""
    import re
    prompt = user_input
    
    # Remove trigger words to extract the actual description
    for word in ['generate', 'create', 'make', 'draw', 'image', 'picture', 'an', 'a', 'of']:
        prompt = re.sub(r'\b' + word + r'\b', '', prompt, flags=re.IGNORECASE)
    
    prompt = re.sub(r'\s+', ' ', prompt).strip()  # Clean up extra spaces
    
    if not prompt or len(prompt) < 3:
        prompt = "a beautiful landscape"
    
    return prompt


def extract_logo_prompt(user_input: str) -> str:
    """Extract brand/company name from logo request."""
    import re
    prompt = user_input
    
    # Remove trigger words
    for word in ['generate', 'create', 'make', 'design', 'logo', 'for', 'a', 'an', 'the']:
        prompt = re.sub(r'\b' + word + r'\b', '', prompt, flags=re.IGNORECASE)
    
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    
    if not prompt or len(prompt) < 2:
        prompt = "modern tech company"
    
    return prompt


def handle_math_simple(user_input: str) -> str:
    """Handle basic math operations."""
    import re
    try:
        # Extract numbers and operations
        expression = re.sub(r'[^0-9+\-*/().\s]', '', user_input)
        if expression.strip():
            result = eval(expression.strip())
            return f"The answer is: {result}"
        else:
            return "I couldn't understand the math problem. Please try again with a clear expression like '5 + 3' or '10 * 2'."
    except Exception:
        return "I had trouble calculating that. Please make sure your math expression is valid."


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
    print("🔄 WebSocket real-time updates active")
    print("⏱️ Timer/Reminder API endpoints available")
    
    # Use SocketIO run instead of app.run for WebSocket support
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)