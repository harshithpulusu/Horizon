"""
Horizon Timer and Reminder API Routes

RESTful endpoints for timer and reminder CRUD operations.
Integrates with the event-driven architecture and WebSocket real-time updates.

Routes:
- /api/timers - Timer CRUD operations
- /api/reminders - Reminder CRUD operations
- /api/timers/active - Get active timers
- /api/reminders/due - Get due reminders
"""

import json
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from typing import Dict, List, Any, Optional

# Import our database managers and event system
from .database import get_timer_manager, get_reminder_manager
from .events import get_event_emitter, emit_event, HorizonEventTypes
from .state_manager import get_state_manager, update_state

# Create Blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Helper functions
def get_user_id() -> str:
    """Get current user ID from request context."""
    # For now, return a default user ID
    # In production, this would extract from JWT token or session
    return request.headers.get('X-User-ID', 'default_user')

def emit_timer_event(event_type: str, timer_data: Dict[str, Any]):
    """Emit timer-related event."""
    emit_event(event_type, "timer_api", {
        'timer': timer_data,
        'user_id': get_user_id(),
        'timestamp': datetime.now().isoformat()
    })

def emit_reminder_event(event_type: str, reminder_data: Dict[str, Any]):
    """Emit reminder-related event."""
    emit_event(event_type, "reminder_api", {
        'reminder': reminder_data,
        'user_id': get_user_id(),
        'timestamp': datetime.now().isoformat()
    })

def validate_timer_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate timer creation/update data."""
    required_fields = ['title', 'duration_seconds']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    if not isinstance(data['duration_seconds'], int) or data['duration_seconds'] <= 0:
        return False, "duration_seconds must be a positive integer"
    
    return True, ""

def validate_reminder_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate reminder creation/update data."""
    required_fields = ['title', 'reminder_time']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate reminder_time format
    try:
        if isinstance(data['reminder_time'], str):
            datetime.fromisoformat(data['reminder_time'].replace('Z', '+00:00'))
    except ValueError:
        return False, "reminder_time must be a valid ISO format datetime"
    
    return True, ""

# Timer API Endpoints
@api_bp.route('/timers', methods=['GET'])
def get_timers():
    """Get all timers for the current user."""
    try:
        user_id = get_user_id()
        timer_manager = get_timer_manager()
        
        status = request.args.get('status')
        timers = timer_manager.get_user_timers(user_id, status)
        
        return jsonify({
            'success': True,
            'timers': timers,
            'count': len(timers)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/timers/<timer_id>', methods=['GET'])
def get_timer(timer_id: str):
    """Get a specific timer by ID."""
    try:
        timer_manager = get_timer_manager()
        timer = timer_manager.get_timer(timer_id)
        
        if not timer:
            return jsonify({
                'success': False,
                'error': 'Timer not found'
            }), 404
        
        return jsonify({
            'success': True,
            'timer': timer
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/timers', methods=['POST'])
def create_timer():
    """Create a new timer."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate data
        is_valid, error_msg = validate_timer_data(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        user_id = get_user_id()
        timer_manager = get_timer_manager()
        
        # Create timer
        timer_id = timer_manager.create_timer(
            user_id=user_id,
            title=data['title'],
            duration_seconds=data['duration_seconds'],
            description=data.get('description'),
            timer_type=data.get('timer_type', 'general'),
            auto_start=data.get('auto_start', False),
            metadata=data.get('metadata', {})
        )
        
        # Get the created timer
        timer = timer_manager.get_timer(timer_id)
        
        # Emit event
        emit_timer_event(HorizonEventTypes.TIMER_CREATED, timer)
        
        # Auto-start if requested
        if data.get('auto_start'):
            timer_manager.start_timer(timer_id)
            timer = timer_manager.get_timer(timer_id)  # Get updated timer
            emit_timer_event(HorizonEventTypes.TIMER_STARTED, timer)
        
        return jsonify({
            'success': True,
            'timer': timer,
            'timer_id': timer_id
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/timers/<timer_id>', methods=['PUT'])
def update_timer(timer_id: str):
    """Update a timer."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        timer_manager = get_timer_manager()
        
        # Check if timer exists
        if not timer_manager.get_timer(timer_id):
            return jsonify({
                'success': False,
                'error': 'Timer not found'
            }), 404
        
        # Update timer
        success = timer_manager.update_timer(timer_id, **data)
        
        if success:
            timer = timer_manager.get_timer(timer_id)
            emit_timer_event(HorizonEventTypes.TIMER_UPDATED, timer)
            
            return jsonify({
                'success': True,
                'timer': timer
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update timer'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/timers/<timer_id>', methods=['DELETE'])
def delete_timer(timer_id: str):
    """Delete a timer."""
    try:
        timer_manager = get_timer_manager()
        
        # Get timer before deletion for event
        timer = timer_manager.get_timer(timer_id)
        if not timer:
            return jsonify({
                'success': False,
                'error': 'Timer not found'
            }), 404
        
        success = timer_manager.delete_timer(timer_id)
        
        if success:
            emit_timer_event(HorizonEventTypes.TIMER_DELETED, timer)
            
            return jsonify({
                'success': True,
                'message': 'Timer deleted successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to delete timer'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Timer Control Endpoints
@api_bp.route('/timers/<timer_id>/start', methods=['POST'])
def start_timer(timer_id: str):
    """Start a timer."""
    try:
        timer_manager = get_timer_manager()
        success = timer_manager.start_timer(timer_id)
        
        if success:
            timer = timer_manager.get_timer(timer_id)
            emit_timer_event(HorizonEventTypes.TIMER_STARTED, timer)
            
            return jsonify({
                'success': True,
                'timer': timer,
                'message': 'Timer started'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start timer'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/timers/<timer_id>/pause', methods=['POST'])
def pause_timer(timer_id: str):
    """Pause a timer."""
    try:
        timer_manager = get_timer_manager()
        success = timer_manager.pause_timer(timer_id)
        
        if success:
            timer = timer_manager.get_timer(timer_id)
            emit_timer_event(HorizonEventTypes.TIMER_PAUSED, timer)
            
            return jsonify({
                'success': True,
                'timer': timer,
                'message': 'Timer paused'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to pause timer'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/timers/<timer_id>/stop', methods=['POST'])
def stop_timer(timer_id: str):
    """Stop a timer."""
    try:
        timer_manager = get_timer_manager()
        success = timer_manager.stop_timer(timer_id)
        
        if success:
            timer = timer_manager.get_timer(timer_id)
            emit_timer_event(HorizonEventTypes.TIMER_STOPPED, timer)
            
            return jsonify({
                'success': True,
                'timer': timer,
                'message': 'Timer stopped'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to stop timer'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/timers/<timer_id>/complete', methods=['POST'])
def complete_timer(timer_id: str):
    """Mark a timer as completed."""
    try:
        timer_manager = get_timer_manager()
        success = timer_manager.complete_timer(timer_id)
        
        if success:
            timer = timer_manager.get_timer(timer_id)
            emit_timer_event(HorizonEventTypes.TIMER_COMPLETED, timer)
            
            return jsonify({
                'success': True,
                'timer': timer,
                'message': 'Timer completed'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to complete timer'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/timers/active', methods=['GET'])
def get_active_timers():
    """Get all active timers for the current user."""
    try:
        user_id = get_user_id()
        timer_manager = get_timer_manager()
        
        active_timers = timer_manager.get_active_timers(user_id)
        
        return jsonify({
            'success': True,
            'timers': active_timers,
            'count': len(active_timers)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Reminder API Endpoints
@api_bp.route('/reminders', methods=['GET'])
def get_reminders():
    """Get all reminders for the current user."""
    try:
        user_id = get_user_id()
        reminder_manager = get_reminder_manager()
        
        status = request.args.get('status')
        reminders = reminder_manager.get_user_reminders(user_id, status)
        
        return jsonify({
            'success': True,
            'reminders': reminders,
            'count': len(reminders)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/reminders/<reminder_id>', methods=['GET'])
def get_reminder(reminder_id: str):
    """Get a specific reminder by ID."""
    try:
        reminder_manager = get_reminder_manager()
        reminder = reminder_manager.get_reminder(reminder_id)
        
        if not reminder:
            return jsonify({
                'success': False,
                'error': 'Reminder not found'
            }), 404
        
        return jsonify({
            'success': True,
            'reminder': reminder
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/reminders', methods=['POST'])
def create_reminder():
    """Create a new reminder."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate data
        is_valid, error_msg = validate_reminder_data(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        user_id = get_user_id()
        reminder_manager = get_reminder_manager()
        
        # Parse reminder time
        reminder_time = data['reminder_time']
        if isinstance(reminder_time, str):
            reminder_time = datetime.fromisoformat(reminder_time.replace('Z', '+00:00'))
        
        # Create reminder
        reminder_id = reminder_manager.create_reminder(
            user_id=user_id,
            title=data['title'],
            reminder_time=reminder_time,
            description=data.get('description'),
            priority=data.get('priority', 'medium'),
            category=data.get('category', 'general'),
            recurring_pattern=data.get('recurring_pattern'),
            metadata=data.get('metadata', {})
        )
        
        # Get the created reminder
        reminder = reminder_manager.get_reminder(reminder_id)
        
        # Emit event
        emit_reminder_event(HorizonEventTypes.REMINDER_CREATED, reminder)
        
        return jsonify({
            'success': True,
            'reminder': reminder,
            'reminder_id': reminder_id
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/reminders/<reminder_id>', methods=['PUT'])
def update_reminder(reminder_id: str):
    """Update a reminder."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        reminder_manager = get_reminder_manager()
        
        # Check if reminder exists
        if not reminder_manager.get_reminder(reminder_id):
            return jsonify({
                'success': False,
                'error': 'Reminder not found'
            }), 404
        
        # Parse reminder_time if provided
        if 'reminder_time' in data and isinstance(data['reminder_time'], str):
            data['reminder_time'] = datetime.fromisoformat(data['reminder_time'].replace('Z', '+00:00'))
        
        # Update reminder
        success = reminder_manager.update_reminder(reminder_id, **data)
        
        if success:
            reminder = reminder_manager.get_reminder(reminder_id)
            emit_reminder_event(HorizonEventTypes.REMINDER_UPDATED, reminder)
            
            return jsonify({
                'success': True,
                'reminder': reminder
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update reminder'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/reminders/<reminder_id>', methods=['DELETE'])
def delete_reminder(reminder_id: str):
    """Delete a reminder."""
    try:
        reminder_manager = get_reminder_manager()
        
        # Get reminder before deletion for event
        reminder = reminder_manager.get_reminder(reminder_id)
        if not reminder:
            return jsonify({
                'success': False,
                'error': 'Reminder not found'
            }), 404
        
        success = reminder_manager.delete_reminder(reminder_id)
        
        if success:
            emit_reminder_event(HorizonEventTypes.REMINDER_DELETED, reminder)
            
            return jsonify({
                'success': True,
                'message': 'Reminder deleted successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to delete reminder'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Reminder Control Endpoints
@api_bp.route('/reminders/<reminder_id>/snooze', methods=['POST'])
def snooze_reminder(reminder_id: str):
    """Snooze a reminder."""
    try:
        data = request.get_json() or {}
        minutes = data.get('minutes', 10)
        
        reminder_manager = get_reminder_manager()
        success = reminder_manager.snooze_reminder(reminder_id, minutes)
        
        if success:
            reminder = reminder_manager.get_reminder(reminder_id)
            emit_reminder_event(HorizonEventTypes.REMINDER_SNOOZED, reminder)
            
            return jsonify({
                'success': True,
                'reminder': reminder,
                'message': f'Reminder snoozed for {minutes} minutes'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to snooze reminder'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/reminders/<reminder_id>/complete', methods=['POST'])
def complete_reminder(reminder_id: str):
    """Mark a reminder as completed."""
    try:
        reminder_manager = get_reminder_manager()
        success = reminder_manager.complete_reminder(reminder_id)
        
        if success:
            reminder = reminder_manager.get_reminder(reminder_id)
            emit_reminder_event(HorizonEventTypes.REMINDER_COMPLETED, reminder)
            
            return jsonify({
                'success': True,
                'reminder': reminder,
                'message': 'Reminder completed'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to complete reminder'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/reminders/due', methods=['GET'])
def get_due_reminders():
    """Get all due reminders for the current user."""
    try:
        user_id = get_user_id()
        reminder_manager = get_reminder_manager()
        
        due_reminders = reminder_manager.get_due_reminders(user_id)
        
        return jsonify({
            'success': True,
            'reminders': due_reminders,
            'count': len(due_reminders)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Health check endpoint
@api_bp.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        'success': True,
        'message': 'Timer/Reminder API is healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200