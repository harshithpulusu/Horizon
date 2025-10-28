"""
Calendar Integration API Routes
Safe, modular implementation that enhances existing timer/reminder system without interference.
"""

from flask import Blueprint, request, jsonify
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

# Create blueprint for calendar integration (separate from existing routes)
calendar_bp = Blueprint('calendar', __name__, url_prefix='/api/calendar')

class CalendarManager:
    """
    Google Calendar integration manager that works alongside existing timer system.
    Completely independent of existing timer/reminder functionality.
    """
    
    def __init__(self):
        self.credentials_file = "data/calendar_credentials.json"
        self.token_file = "data/calendar_token.json"
        self.sync_log_file = "data/calendar_sync_log.json"
        
        # Calendar service instance (lazy-loaded)
        self.service = None
        self.authenticated = False
        
        # Sync configuration
        self.sync_config = {
            'calendar_id': 'primary',  # Use primary calendar
            'timezone': 'UTC',
            'sync_enabled': False,
            'auto_sync_interval': 300,  # 5 minutes
            'event_prefix': '[Horizon AI] ',
            'max_sync_items': 100
        }
        
        # Ensure data directory exists
        self.ensure_data_directory()
        
        # Load configuration
        self.load_config()
        
        # Initialize Google Calendar API (with fallback)
        self.init_calendar_service()
    
    def ensure_data_directory(self):
        """Ensure data directory exists for storing calendar data"""
        directories = ['data', 'data/tokens', 'data/calendar_events']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    def load_config(self):
        """Load calendar configuration from file"""
        config_file = "data/calendar_config.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.sync_config.update(saved_config)
        except Exception as e:
            print(f"Config loading error (non-critical): {e}")
    
    def save_config(self):
        """Save calendar configuration to file"""
        config_file = "data/calendar_config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self.sync_config, f, indent=2)
        except Exception as e:
            print(f"Config saving error (non-critical): {e}")
    
    def init_calendar_service(self):
        """Initialize Google Calendar API service with graceful fallback"""
        try:
            # Try to import Google Calendar API libraries
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            
            # Calendar API scope
            SCOPES = ['https://www.googleapis.com/auth/calendar']
            
            creds = None
            
            # Load existing token
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
            
            # If no valid credentials, handle OAuth flow
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        print(f"Token refresh failed: {e}")
                        creds = None
                
                # Need new authentication
                if not creds and os.path.exists(self.credentials_file):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, SCOPES)
                    # For now, we'll skip the interactive flow and just mark as unavailable
                    print("Calendar authentication required - will be available after OAuth setup")
                    return False
            
            # Save credentials
            if creds:
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
                
                # Build the service
                self.service = build('calendar', 'v3', credentials=creds)
                self.authenticated = True
                print("✅ Google Calendar API connected successfully")
                return True
            
        except ImportError:
            print("⚠️ Google Calendar API libraries not installed - calendar sync unavailable")
        except Exception as e:
            print(f"⚠️ Calendar service initialization error: {e}")
        
        return False
    
    def get_auth_url(self) -> Optional[str]:
        """Get OAuth authorization URL for calendar access"""
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            
            if not os.path.exists(self.credentials_file):
                return None
            
            SCOPES = ['https://www.googleapis.com/auth/calendar']
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file, SCOPES)
            
            # Get authorization URL
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'  # For manual copy-paste
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            return auth_url
            
        except Exception as e:
            print(f"Auth URL generation error: {e}")
            return None
    
    def complete_auth(self, auth_code: str) -> bool:
        """Complete OAuth flow with authorization code"""
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            
            SCOPES = ['https://www.googleapis.com/auth/calendar']
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file, SCOPES)
            
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
            flow.fetch_token(code=auth_code)
            
            creds = flow.credentials
            
            # Save credentials
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
            
            # Initialize service
            self.service = build('calendar', 'v3', credentials=creds)
            self.authenticated = True
            
            print("✅ Calendar authentication completed successfully")
            return True
            
        except Exception as e:
            print(f"Auth completion error: {e}")
            return False
    
    def create_calendar_event(self, timer_data: Dict[str, Any]) -> Optional[str]:
        """Create calendar event from timer data (enhances existing timer system)"""
        if not self.authenticated or not self.service:
            return None
        
        try:
            # Extract timer information
            title = timer_data.get('title', 'Horizon AI Reminder')
            description = timer_data.get('description', '')
            start_time = timer_data.get('datetime')  # ISO format datetime
            duration_minutes = timer_data.get('duration', 30)  # Default 30 minutes
            
            if not start_time:
                return None
            
            # Parse start time
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = start_dt + timedelta(minutes=duration_minutes)
            
            # Create event
            event = {
                'summary': f"{self.sync_config['event_prefix']}{title}",
                'description': description,
                'start': {
                    'dateTime': start_dt.isoformat(),
                    'timeZone': self.sync_config['timezone'],
                },
                'end': {
                    'dateTime': end_dt.isoformat(),
                    'timeZone': self.sync_config['timezone'],
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'popup', 'minutes': 10},
                    ],
                },
            }
            
            # Insert event
            created_event = self.service.events().insert(
                calendarId=self.sync_config['calendar_id'],
                body=event
            ).execute()
            
            # Log sync
            self.log_sync_action('create', created_event['id'], timer_data)
            
            return created_event['id']
            
        except Exception as e:
            print(f"Calendar event creation error: {e}")
            return None
    
    def update_calendar_event(self, event_id: str, timer_data: Dict[str, Any]) -> bool:
        """Update existing calendar event"""
        if not self.authenticated or not self.service:
            return False
        
        try:
            # Get existing event
            event = self.service.events().get(
                calendarId=self.sync_config['calendar_id'],
                eventId=event_id
            ).execute()
            
            # Update event data
            title = timer_data.get('title', 'Horizon AI Reminder')
            description = timer_data.get('description', '')
            start_time = timer_data.get('datetime')
            duration_minutes = timer_data.get('duration', 30)
            
            if start_time:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = start_dt + timedelta(minutes=duration_minutes)
                
                event['summary'] = f"{self.sync_config['event_prefix']}{title}"
                event['description'] = description
                event['start']['dateTime'] = start_dt.isoformat()
                event['end']['dateTime'] = end_dt.isoformat()
            
            # Update event
            updated_event = self.service.events().update(
                calendarId=self.sync_config['calendar_id'],
                eventId=event_id,
                body=event
            ).execute()
            
            # Log sync
            self.log_sync_action('update', event_id, timer_data)
            
            return True
            
        except Exception as e:
            print(f"Calendar event update error: {e}")
            return False
    
    def delete_calendar_event(self, event_id: str) -> bool:
        """Delete calendar event"""
        if not self.authenticated or not self.service:
            return False
        
        try:
            self.service.events().delete(
                calendarId=self.sync_config['calendar_id'],
                eventId=event_id
            ).execute()
            
            # Log sync
            self.log_sync_action('delete', event_id, {})
            
            return True
            
        except Exception as e:
            print(f"Calendar event deletion error: {e}")
            return False
    
    def get_upcoming_events(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get upcoming calendar events"""
        if not self.authenticated or not self.service:
            return []
        
        try:
            # Calculate time range
            now = datetime.utcnow()
            time_min = now.isoformat() + 'Z'
            time_max = (now + timedelta(days=days)).isoformat() + 'Z'
            
            # Get events
            events_result = self.service.events().list(
                calendarId=self.sync_config['calendar_id'],
                timeMin=time_min,
                timeMax=time_max,
                maxResults=self.sync_config['max_sync_items'],
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            # Filter Horizon AI events
            horizon_events = []
            for event in events:
                if event.get('summary', '').startswith(self.sync_config['event_prefix']):
                    horizon_events.append({
                        'id': event['id'],
                        'title': event.get('summary', '').replace(self.sync_config['event_prefix'], ''),
                        'description': event.get('description', ''),
                        'start': event['start'].get('dateTime', event['start'].get('date')),
                        'end': event['end'].get('dateTime', event['end'].get('date')),
                        'status': event.get('status', 'confirmed')
                    })
            
            return horizon_events
            
        except Exception as e:
            print(f"Calendar events fetch error: {e}")
            return []
    
    def sync_timer_to_calendar(self, timer_id: str, timer_data: Dict[str, Any]) -> Optional[str]:
        """Sync a single timer to calendar (enhances existing timer without modification)"""
        if not self.sync_config['sync_enabled']:
            return None
        
        try:
            # Check if timer already has calendar event
            sync_log = self.load_sync_log()
            existing_event_id = sync_log.get('timers', {}).get(timer_id, {}).get('calendar_event_id')
            
            if existing_event_id:
                # Update existing event
                success = self.update_calendar_event(existing_event_id, timer_data)
                return existing_event_id if success else None
            else:
                # Create new event
                event_id = self.create_calendar_event(timer_data)
                if event_id:
                    # Record sync mapping
                    if 'timers' not in sync_log:
                        sync_log['timers'] = {}
                    sync_log['timers'][timer_id] = {
                        'calendar_event_id': event_id,
                        'synced_at': datetime.now().isoformat()
                    }
                    self.save_sync_log(sync_log)
                return event_id
                
        except Exception as e:
            print(f"Timer sync error: {e}")
            return None
    
    def log_sync_action(self, action: str, event_id: str, data: Dict[str, Any]):
        """Log sync actions for debugging and tracking"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'event_id': event_id,
                'data': data
            }
            
            # Load existing log
            sync_log = []
            if os.path.exists(self.sync_log_file):
                with open(self.sync_log_file, 'r') as f:
                    sync_log = json.load(f)
            
            # Add new entry
            sync_log.append(log_entry)
            
            # Keep only last 1000 entries
            sync_log = sync_log[-1000:]
            
            # Save log
            with open(self.sync_log_file, 'w') as f:
                json.dump(sync_log, f, indent=2)
                
        except Exception as e:
            print(f"Sync logging error: {e}")
    
    def load_sync_log(self) -> Dict[str, Any]:
        """Load sync mapping log"""
        sync_map_file = "data/calendar_sync_map.json"
        try:
            if os.path.exists(sync_map_file):
                with open(sync_map_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def save_sync_log(self, sync_log: Dict[str, Any]):
        """Save sync mapping log"""
        sync_map_file = "data/calendar_sync_map.json"
        try:
            with open(sync_map_file, 'w') as f:
                json.dump(sync_log, f, indent=2)
        except Exception as e:
            print(f"Sync map saving error: {e}")
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get calendar sync status and statistics"""
        sync_log = self.load_sync_log()
        
        return {
            'authenticated': self.authenticated,
            'sync_enabled': self.sync_config['sync_enabled'],
            'calendar_id': self.sync_config['calendar_id'],
            'synced_timers': len(sync_log.get('timers', {})),
            'last_sync': sync_log.get('last_sync'),
            'config': self.sync_config
        }

# Initialize calendar manager
calendar_manager = CalendarManager()

@calendar_bp.route('/status', methods=['GET'])
def get_calendar_status():
    """
    Get calendar integration status.
    Completely separate from existing timer endpoints.
    """
    try:
        status = calendar_manager.get_sync_status()
        
        return jsonify({
            'success': True,
            'calendar_status': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@calendar_bp.route('/auth/url', methods=['GET'])
def get_auth_url():
    """Get OAuth authorization URL for calendar access"""
    try:
        auth_url = calendar_manager.get_auth_url()
        
        if auth_url:
            return jsonify({
                'success': True,
                'auth_url': auth_url,
                'message': 'Visit this URL to authorize calendar access'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Calendar credentials file not found or invalid'
            }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@calendar_bp.route('/auth/complete', methods=['POST'])
def complete_auth():
    """Complete OAuth flow with authorization code"""
    try:
        data = request.get_json()
        auth_code = data.get('code')
        
        if not auth_code:
            return jsonify({
                'success': False,
                'error': 'Authorization code is required'
            }), 400
        
        success = calendar_manager.complete_auth(auth_code)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Calendar authorization completed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Authorization failed'
            }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@calendar_bp.route('/sync/timer', methods=['POST'])
def sync_timer():
    """
    Sync a timer to calendar (enhances existing timer system).
    Does not modify existing timer functionality.
    """
    try:
        data = request.get_json()
        timer_id = data.get('timer_id')
        timer_data = data.get('timer_data', {})
        
        if not timer_id or not timer_data:
            return jsonify({
                'success': False,
                'error': 'Timer ID and data are required'
            }), 400
        
        event_id = calendar_manager.sync_timer_to_calendar(timer_id, timer_data)
        
        if event_id:
            return jsonify({
                'success': True,
                'event_id': event_id,
                'message': 'Timer synced to calendar successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Calendar sync failed or disabled'
            }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@calendar_bp.route('/events', methods=['GET'])
def get_calendar_events():
    """Get upcoming calendar events"""
    try:
        days = int(request.args.get('days', 7))
        events = calendar_manager.get_upcoming_events(days)
        
        return jsonify({
            'success': True,
            'events': events,
            'count': len(events),
            'days': days
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@calendar_bp.route('/config', methods=['GET', 'POST'])
def calendar_config():
    """Get or update calendar configuration"""
    try:
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'config': calendar_manager.sync_config
            })
        
        else:  # POST
            data = request.get_json()
            
            # Update configuration
            if 'sync_enabled' in data:
                calendar_manager.sync_config['sync_enabled'] = bool(data['sync_enabled'])
            if 'calendar_id' in data:
                calendar_manager.sync_config['calendar_id'] = data['calendar_id']
            if 'timezone' in data:
                calendar_manager.sync_config['timezone'] = data['timezone']
            if 'event_prefix' in data:
                calendar_manager.sync_config['event_prefix'] = data['event_prefix']
            
            # Save configuration
            calendar_manager.save_config()
            
            return jsonify({
                'success': True,
                'config': calendar_manager.sync_config,
                'message': 'Configuration updated successfully'
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@calendar_bp.route('/health', methods=['GET'])
def calendar_health():
    """
    Health check for calendar integration system.
    """
    try:
        status = calendar_manager.get_sync_status()
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'authenticated': status['authenticated'],
            'sync_enabled': status['sync_enabled'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e)
        }), 500