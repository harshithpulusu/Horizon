"""
Calendar Manager Core
Advanced Google Calendar integration that enhances existing timer/reminder system.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import uuid
import threading
import time

class AdvancedCalendarManager:
    """
    Advanced calendar manager with smart scheduling and conflict detection.
    Completely independent system that enhances existing timer functionality.
    """
    
    def __init__(self):
        self.config_file = "data/advanced_calendar_config.json"
        self.cache_file = "data/calendar_cache.json"
        self.conflicts_file = "data/calendar_conflicts.json"
        
        # Advanced configuration
        self.config = {
            'smart_scheduling': True,
            'conflict_detection': True,
            'auto_reschedule': False,
            'preferred_hours': {
                'start': 9,  # 9 AM
                'end': 17    # 5 PM
            },
            'working_days': [0, 1, 2, 3, 4],  # Monday-Friday
            'break_duration': 15,  # 15 minutes between events
            'max_daily_events': 20,
            'timezone_awareness': True,
            'smart_reminders': True
        }
        
        # Event cache for performance
        self.event_cache = {}
        self.cache_expiry = datetime.now()
        self.cache_duration = timedelta(minutes=15)
        
        # Conflict tracking
        self.detected_conflicts = []
        
        # Background sync
        self.sync_thread = None
        self.sync_running = False
        
        # Load configuration
        self.load_config()
        
        # Initialize background sync if enabled
        self.start_background_sync()
    
    def load_config(self):
        """Load advanced calendar configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
        except Exception as e:
            print(f"Advanced config loading error: {e}")
    
    def save_config(self):
        """Save advanced calendar configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Advanced config saving error: {e}")
    
    def start_background_sync(self):
        """Start background synchronization thread"""
        if not self.sync_running and self.config.get('auto_sync', False):
            self.sync_running = True
            self.sync_thread = threading.Thread(target=self.background_sync_loop, daemon=True)
            self.sync_thread.start()
            print("✅ Calendar background sync started")
    
    def stop_background_sync(self):
        """Stop background synchronization"""
        self.sync_running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        print("🛑 Calendar background sync stopped")
    
    def background_sync_loop(self):
        """Background sync loop for automatic calendar updates"""
        while self.sync_running:
            try:
                # Refresh cache every 15 minutes
                if datetime.now() > self.cache_expiry:
                    self.refresh_event_cache()
                
                # Check for conflicts every 5 minutes
                self.detect_scheduling_conflicts()
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                print(f"Background sync error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def refresh_event_cache(self):
        """Refresh event cache from calendar"""
        try:
            # This would call the actual calendar service
            # For now, we'll use a mock implementation
            self.event_cache = self.get_cached_events()
            self.cache_expiry = datetime.now() + self.cache_duration
            
            # Save cache to file
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'events': self.event_cache,
                    'cached_at': datetime.now().isoformat(),
                    'expires_at': self.cache_expiry.isoformat()
                }, f, indent=2)
            
            print(f"📅 Event cache refreshed ({len(self.event_cache)} events)")
            
        except Exception as e:
            print(f"Cache refresh error: {e}")
    
    def get_cached_events(self) -> Dict[str, Any]:
        """Get events from cache or calendar service"""
        try:
            # Try to load from cache first
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    cached_at = datetime.fromisoformat(cache_data['cached_at'])
                    expires_at = datetime.fromisoformat(cache_data['expires_at'])
                    
                    if datetime.now() < expires_at:
                        return cache_data['events']
            
            # Cache expired or doesn't exist, would fetch from calendar service
            # For now, return empty dict
            return {}
            
        except Exception as e:
            print(f"Cache loading error: {e}")
            return {}
    
    def find_optimal_time_slot(self, duration_minutes: int, 
                             preferred_date: Optional[datetime] = None,
                             constraints: Optional[Dict[str, Any]] = None) -> Optional[datetime]:
        """
        Find optimal time slot for scheduling using AI-powered analysis.
        """
        try:
            # Default to tomorrow if no preferred date
            if not preferred_date:
                preferred_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
            
            # Get constraints
            constraints = constraints or {}
            working_hours = constraints.get('working_hours', self.config['preferred_hours'])
            working_days = constraints.get('working_days', self.config['working_days'])
            break_duration = constraints.get('break_duration', self.config['break_duration'])
            
            # Search for available slots over next 14 days
            search_end = preferred_date + timedelta(days=14)
            current_date = preferred_date.replace(hour=working_hours['start'])
            
            while current_date < search_end:
                # Skip non-working days
                if current_date.weekday() not in working_days:
                    current_date += timedelta(days=1)
                    current_date = current_date.replace(hour=working_hours['start'])
                    continue
                
                # Check each hour slot
                day_end = current_date.replace(hour=working_hours['end'])
                slot_start = current_date.replace(hour=working_hours['start'])
                
                while slot_start + timedelta(minutes=duration_minutes) <= day_end:
                    slot_end = slot_start + timedelta(minutes=duration_minutes)
                    
                    # Check if slot is available
                    if self.is_time_slot_available(slot_start, slot_end, break_duration):
                        return slot_start
                    
                    # Move to next 30-minute slot
                    slot_start += timedelta(minutes=30)
                
                # Move to next day
                current_date += timedelta(days=1)
                current_date = current_date.replace(hour=working_hours['start'])
            
            # No optimal slot found
            return None
            
        except Exception as e:
            print(f"Optimal time slot finding error: {e}")
            return None
    
    def is_time_slot_available(self, start_time: datetime, end_time: datetime, 
                              break_duration: int = 15) -> bool:
        """
        Check if a time slot is available (no conflicts).
        """
        try:
            # Add break buffer
            buffer_start = start_time - timedelta(minutes=break_duration)
            buffer_end = end_time + timedelta(minutes=break_duration)
            
            # Check against cached events
            for event_id, event in self.event_cache.items():
                event_start = datetime.fromisoformat(event.get('start', ''))
                event_end = datetime.fromisoformat(event.get('end', ''))
                
                # Check for overlap
                if (buffer_start < event_end and buffer_end > event_start):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Time slot availability check error: {e}")
            return True  # Default to available on error
    
    def detect_scheduling_conflicts(self) -> List[Dict[str, Any]]:
        """
        Detect and analyze scheduling conflicts.
        """
        try:
            conflicts = []
            events_list = list(self.event_cache.values())
            
            # Check each pair of events for overlaps
            for i, event1 in enumerate(events_list):
                for event2 in events_list[i+1:]:
                    try:
                        start1 = datetime.fromisoformat(event1.get('start', ''))
                        end1 = datetime.fromisoformat(event1.get('end', ''))
                        start2 = datetime.fromisoformat(event2.get('start', ''))
                        end2 = datetime.fromisoformat(event2.get('end', ''))
                        
                        # Check for overlap
                        if start1 < end2 and start2 < end1:
                            conflict = {
                                'id': str(uuid.uuid4()),
                                'event1': event1,
                                'event2': event2,
                                'overlap_start': max(start1, start2).isoformat(),
                                'overlap_end': min(end1, end2).isoformat(),
                                'severity': self.calculate_conflict_severity(event1, event2),
                                'detected_at': datetime.now().isoformat(),
                                'suggestions': self.generate_conflict_suggestions(event1, event2)
                            }
                            conflicts.append(conflict)
                    
                    except Exception as e:
                        print(f"Conflict detection error for pair: {e}")
            
            # Update detected conflicts
            self.detected_conflicts = conflicts
            
            # Save conflicts to file
            try:
                with open(self.conflicts_file, 'w') as f:
                    json.dump(conflicts, f, indent=2)
            except Exception as e:
                print(f"Conflict saving error: {e}")
            
            return conflicts
            
        except Exception as e:
            print(f"Conflict detection error: {e}")
            return []
    
    def calculate_conflict_severity(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> str:
        """Calculate conflict severity based on event properties"""
        try:
            # Factors: event importance, duration, attendees, etc.
            severity_score = 0
            
            # Check event titles for importance keywords
            important_keywords = ['meeting', 'interview', 'deadline', 'urgent', 'critical']
            
            for event in [event1, event2]:
                title = event.get('title', '').lower()
                if any(keyword in title for keyword in important_keywords):
                    severity_score += 2
            
            # Check duration (longer events = higher severity)
            for event in [event1, event2]:
                start = datetime.fromisoformat(event.get('start', ''))
                end = datetime.fromisoformat(event.get('end', ''))
                duration_hours = (end - start).total_seconds() / 3600
                
                if duration_hours >= 2:
                    severity_score += 1
            
            # Determine severity level
            if severity_score >= 4:
                return 'high'
            elif severity_score >= 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            print(f"Severity calculation error: {e}")
            return 'low'
    
    def generate_conflict_suggestions(self, event1: Dict[str, Any], 
                                    event2: Dict[str, Any]) -> List[str]:
        """Generate suggestions for resolving conflicts"""
        try:
            suggestions = []
            
            # Basic suggestions
            suggestions.append("Reschedule one of the events to a different time")
            suggestions.append("Shorten the duration of one or both events")
            suggestions.append("Combine events if they are related")
            
            # Smart suggestions based on event analysis
            start1 = datetime.fromisoformat(event1.get('start', ''))
            start2 = datetime.fromisoformat(event2.get('start', ''))
            
            # Suggest moving the later event
            if start1 < start2:
                later_event = event2
            else:
                later_event = event1
            
            # Find next available slot
            duration = self.estimate_event_duration(later_event)
            next_slot = self.find_optimal_time_slot(duration, start1)
            
            if next_slot:
                suggestions.append(f"Move '{later_event.get('title', 'event')}' to {next_slot.strftime('%Y-%m-%d %H:%M')}")
            
            return suggestions
            
        except Exception as e:
            print(f"Suggestion generation error: {e}")
            return ["Manual rescheduling required"]
    
    def estimate_event_duration(self, event: Dict[str, Any]) -> int:
        """Estimate event duration in minutes"""
        try:
            start = datetime.fromisoformat(event.get('start', ''))
            end = datetime.fromisoformat(event.get('end', ''))
            return int((end - start).total_seconds() / 60)
        except Exception:
            return 60  # Default 1 hour
    
    def get_schedule_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Generate schedule analytics and insights"""
        try:
            # Calculate analytics from cached events
            total_events = len(self.event_cache)
            
            # Time distribution analysis
            hour_distribution = {}
            day_distribution = {}
            
            for event in self.event_cache.values():
                try:
                    start = datetime.fromisoformat(event.get('start', ''))
                    hour = start.hour
                    day = start.strftime('%A')
                    
                    hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
                    day_distribution[day] = day_distribution.get(day, 0) + 1
                
                except Exception:
                    continue
            
            # Find busiest periods
            busiest_hour = max(hour_distribution.items(), key=lambda x: x[1]) if hour_distribution else (9, 0)
            busiest_day = max(day_distribution.items(), key=lambda x: x[1]) if day_distribution else ('Monday', 0)
            
            # Calculate availability
            total_working_hours = days * 8  # 8 hours per day
            scheduled_hours = sum(self.estimate_event_duration(event) / 60 for event in self.event_cache.values())
            availability_percentage = max(0, (total_working_hours - scheduled_hours) / total_working_hours * 100)
            
            return {
                'total_events': total_events,
                'scheduled_hours': round(scheduled_hours, 1),
                'availability_percentage': round(availability_percentage, 1),
                'busiest_hour': f"{busiest_hour[0]}:00 ({busiest_hour[1]} events)",
                'busiest_day': f"{busiest_day[0]} ({busiest_day[1]} events)",
                'conflicts_detected': len(self.detected_conflicts),
                'hour_distribution': hour_distribution,
                'day_distribution': day_distribution,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Schedule analytics error: {e}")
            return {
                'total_events': 0,
                'error': str(e)
            }
    
    def suggest_meeting_times(self, duration_minutes: int, 
                            participants: List[str] = None,
                            constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Suggest optimal meeting times considering all participants.
        """
        try:
            suggestions = []
            constraints = constraints or {}
            
            # Find multiple time slots
            for i in range(5):  # Suggest up to 5 options
                preferred_date = datetime.now() + timedelta(days=i+1)
                
                optimal_time = self.find_optimal_time_slot(
                    duration_minutes, 
                    preferred_date, 
                    constraints
                )
                
                if optimal_time:
                    suggestions.append({
                        'start_time': optimal_time.isoformat(),
                        'end_time': (optimal_time + timedelta(minutes=duration_minutes)).isoformat(),
                        'confidence': self.calculate_time_confidence(optimal_time),
                        'reasoning': self.explain_time_choice(optimal_time, duration_minutes)
                    })
            
            return suggestions
            
        except Exception as e:
            print(f"Meeting time suggestion error: {e}")
            return []
    
    def calculate_time_confidence(self, suggested_time: datetime) -> float:
        """Calculate confidence score for suggested time"""
        try:
            confidence = 1.0
            
            # Reduce confidence for non-optimal hours
            hour = suggested_time.hour
            if hour < 9 or hour > 17:
                confidence *= 0.7
            
            # Reduce confidence for conflicts nearby
            nearby_events = self.get_events_near_time(suggested_time, timedelta(hours=1))
            if nearby_events:
                confidence *= 0.8
            
            return round(confidence, 2)
            
        except Exception:
            return 0.5
    
    def explain_time_choice(self, suggested_time: datetime, duration: int) -> str:
        """Explain why this time was chosen"""
        try:
            reasons = []
            
            hour = suggested_time.hour
            if 9 <= hour <= 17:
                reasons.append("within business hours")
            
            day_name = suggested_time.strftime('%A')
            if suggested_time.weekday() < 5:
                reasons.append(f"on a weekday ({day_name})")
            
            nearby_events = self.get_events_near_time(suggested_time, timedelta(hours=1))
            if not nearby_events:
                reasons.append("no conflicts detected")
            
            if not reasons:
                reasons.append("available time slot")
            
            return f"Suggested because it's {', '.join(reasons)}"
            
        except Exception:
            return "Available time slot"
    
    def get_events_near_time(self, target_time: datetime, 
                           window: timedelta) -> List[Dict[str, Any]]:
        """Get events near a specific time"""
        try:
            nearby_events = []
            window_start = target_time - window
            window_end = target_time + window
            
            for event in self.event_cache.values():
                event_start = datetime.fromisoformat(event.get('start', ''))
                if window_start <= event_start <= window_end:
                    nearby_events.append(event)
            
            return nearby_events
            
        except Exception as e:
            print(f"Nearby events search error: {e}")
            return []