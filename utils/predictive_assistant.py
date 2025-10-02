#!/usr/bin/env python3
"""
Predictive Assistance System for Horizon AI
AI that anticipates user needs based on patterns, context, and behavior

This module implements:
- Behavioral pattern analysis
- Contextual prediction
- Proactive suggestions
- Need anticipation algorithms
- Smart recommendation engine
- Temporal pattern recognition
"""

import sqlite3
import json
import re
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict

# Try to import ML libraries for advanced predictions
try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML libraries not available - using statistical methods")

# Import error handling if available
try:
    from utils.error_handler import (
        HorizonError, AIServiceError, ValidationError,
        log_error_with_context, safe_db_operation
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False
    class HorizonError(Exception):
        pass
    class AIServiceError(HorizonError):
        pass
    class ValidationError(HorizonError):
        pass
    def log_error_with_context(msg, context):
        print(f"Error: {msg}, Context: {context}")
    def safe_db_operation(func):
        return func

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Structured prediction result"""
    prediction_type: str
    confidence: float
    suggested_action: str
    context: Dict[str, Any]
    reasoning: str
    urgency: str  # 'low', 'medium', 'high', 'urgent'
    timestamp: str
    expires_at: Optional[str] = None

@dataclass
class UserPattern:
    """User behavioral pattern data"""
    user_id: str
    pattern_type: str
    frequency: float
    last_occurrence: str
    typical_time: str
    context_triggers: List[str]
    success_rate: float

class PredictiveAssistant:
    """Core predictive assistance engine"""
    
    def __init__(self, db_path: str = "ai_memory.db"):
        self.db_path = db_path
        self.prediction_cache = {}
        self.pattern_cache = {}
        self.last_cache_update = 0
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize database tables
        self._init_prediction_tables()
        
        # Load ML models if available
        if ML_AVAILABLE:
            self.scaler = StandardScaler()
            self.clusterer = None
            self._load_or_train_models()
    
    def _init_prediction_tables(self):
        """Initialize database tables for predictive assistance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # User behavioral patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    frequency REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    last_occurrence TEXT,
                    first_seen TEXT,
                    success_rate REAL DEFAULT 0.0,
                    context_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Prediction history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    prediction_data TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    was_accurate BOOLEAN,
                    user_feedback TEXT,
                    context_at_time TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                )
            """)
            
            # Contextual triggers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contextual_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_type TEXT NOT NULL,
                    trigger_conditions TEXT NOT NULL,
                    associated_patterns TEXT NOT NULL,
                    activation_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Proactive suggestions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS proactive_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    suggestion_type TEXT NOT NULL,
                    suggestion_data TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'pending',
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    acted_upon_at TIMESTAMP
                )
            """)
            
            # Temporal patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temporal_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    time_pattern TEXT NOT NULL,
                    activity_type TEXT NOT NULL,
                    frequency_data TEXT NOT NULL,
                    seasonal_data TEXT,
                    next_predicted TIMESTAMP,
                    accuracy_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("✅ Predictive assistance database tables initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            if ERROR_HANDLING_AVAILABLE:
                log_error_with_context("Predictive DB init failed", {"error": str(e)})
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _load_or_train_models(self):
        """Load existing ML models or train new ones"""
        if not ML_AVAILABLE:
            return
        
        try:
            # Try to load existing models
            # For now, we'll create simple clustering for user behavior
            self.clusterer = KMeans(n_clusters=5, random_state=42)
            logger.info("✅ ML models initialized for predictive assistance")
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")
    
    def analyze_user_patterns(self, user_id: str, timeframe_days: int = 30) -> List[UserPattern]:
        """Analyze user behavioral patterns over a timeframe"""
        patterns = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get conversation patterns
            cursor.execute("""
                SELECT 
                    strftime('%H', timestamp) as hour,
                    strftime('%w', timestamp) as day_of_week,
                    input, response, personality,
                    context_data
                FROM conversations 
                WHERE user_id = ? 
                AND datetime(timestamp) > datetime('now', '-{} days')
                ORDER BY timestamp DESC
            """.format(timeframe_days), (user_id,))
            
            conversations = cursor.fetchall()
            
            if conversations:
                # Analyze temporal patterns
                temporal_pattern = self._analyze_temporal_patterns(conversations)
                if temporal_pattern:
                    patterns.append(temporal_pattern)
                
                # Analyze topic patterns
                topic_patterns = self._analyze_topic_patterns(conversations)
                patterns.extend(topic_patterns)
                
                # Analyze interaction style patterns
                style_pattern = self._analyze_interaction_style(conversations)
                if style_pattern:
                    patterns.append(style_pattern)
            
            # Store patterns in database
            self._store_patterns(user_id, patterns)
            
        except Exception as e:
            logger.error(f"Pattern analysis failed for user {user_id}: {e}")
            if ERROR_HANDLING_AVAILABLE:
                log_error_with_context("Pattern analysis failed", 
                                     {"user_id": user_id, "error": str(e)})
        finally:
            if 'conn' in locals():
                conn.close()
        
        return patterns
    
    def _analyze_temporal_patterns(self, conversations: List[Tuple]) -> Optional[UserPattern]:
        """Analyze when user typically interacts"""
        if not conversations:
            return None
        
        hours = [int(conv[0]) for conv in conversations if conv[0]]
        days = [int(conv[1]) for conv in conversations if conv[1]]
        
        if not hours:
            return None
        
        # Find most common interaction times
        hour_counts = Counter(hours)
        day_counts = Counter(days)
        
        most_common_hour = hour_counts.most_common(1)[0][0]
        most_common_day = day_counts.most_common(1)[0][0]
        
        # Calculate frequency
        total_conversations = len(conversations)
        peak_hour_frequency = hour_counts[most_common_hour] / total_conversations
        
        return UserPattern(
            user_id="",  # Will be filled by caller
            pattern_type="temporal",
            frequency=peak_hour_frequency,
            last_occurrence=datetime.now().isoformat(),
            typical_time=f"{most_common_hour:02d}:00",
            context_triggers=[f"day_{most_common_day}", f"hour_{most_common_hour}"],
            success_rate=0.85  # Default, will be updated based on feedback
        )
    
    def _analyze_topic_patterns(self, conversations: List[Tuple]) -> List[UserPattern]:
        """Analyze what topics user frequently discusses"""
        patterns = []
        
        # Extract topics from conversations
        topics = defaultdict(int)
        
        for conv in conversations:
            input_text = conv[2] if len(conv) > 2 else ""
            response_text = conv[3] if len(conv) > 3 else ""
            
            # Simple topic extraction (can be enhanced with NLP)
            text = f"{input_text} {response_text}".lower()
            
            # Common topic keywords
            topic_keywords = {
                'work': ['work', 'job', 'meeting', 'project', 'task', 'deadline'],
                'health': ['health', 'exercise', 'fitness', 'diet', 'sleep'],
                'entertainment': ['movie', 'music', 'game', 'show', 'fun'],
                'learning': ['learn', 'study', 'course', 'tutorial', 'education'],
                'technology': ['tech', 'software', 'app', 'computer', 'ai'],
                'planning': ['plan', 'schedule', 'organize', 'remind', 'calendar']
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in text for keyword in keywords):
                    topics[topic] += 1
        
        # Create patterns for frequent topics
        total_conversations = len(conversations)
        for topic, count in topics.items():
            if count > 2:  # At least 3 mentions
                frequency = count / total_conversations
                patterns.append(UserPattern(
                    user_id="",
                    pattern_type="topic",
                    frequency=frequency,
                    last_occurrence=datetime.now().isoformat(),
                    typical_time="",
                    context_triggers=[f"topic_{topic}"],
                    success_rate=0.8
                ))
        
        return patterns
    
    def _analyze_interaction_style(self, conversations: List[Tuple]) -> Optional[UserPattern]:
        """Analyze user's interaction style and preferences"""
        if not conversations:
            return None
        
        personalities_used = []
        response_lengths = []
        
        for conv in conversations:
            if len(conv) > 4 and conv[4]:  # personality
                personalities_used.append(conv[4])
            
            if len(conv) > 3 and conv[3]:  # response
                response_lengths.append(len(conv[3]))
        
        if personalities_used:
            most_common_personality = Counter(personalities_used).most_common(1)[0][0]
            personality_frequency = personalities_used.count(most_common_personality) / len(personalities_used)
            
            avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
            
            return UserPattern(
                user_id="",
                pattern_type="interaction_style",
                frequency=personality_frequency,
                last_occurrence=datetime.now().isoformat(),
                typical_time="",
                context_triggers=[f"personality_{most_common_personality}", 
                                f"length_{avg_response_length:.0f}"],
                success_rate=0.9
            )
        
        return None
    
    def _store_patterns(self, user_id: str, patterns: List[UserPattern]):
        """Store discovered patterns in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for pattern in patterns:
                pattern.user_id = user_id
                
                # Check if pattern already exists
                cursor.execute("""
                    SELECT id FROM user_patterns 
                    WHERE user_id = ? AND pattern_type = ?
                """, (user_id, pattern.pattern_type))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing pattern
                    cursor.execute("""
                        UPDATE user_patterns 
                        SET pattern_data = ?, frequency = ?, last_occurrence = ?,
                            success_rate = ?, context_data = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (
                        json.dumps(asdict(pattern)),
                        pattern.frequency,
                        pattern.last_occurrence,
                        pattern.success_rate,
                        json.dumps(pattern.context_triggers),
                        existing[0]
                    ))
                else:
                    # Insert new pattern
                    cursor.execute("""
                        INSERT INTO user_patterns 
                        (user_id, pattern_type, pattern_data, frequency, 
                         last_occurrence, first_seen, success_rate, context_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id,
                        pattern.pattern_type,
                        json.dumps(asdict(pattern)),
                        pattern.frequency,
                        pattern.last_occurrence,
                        datetime.now().isoformat(),
                        pattern.success_rate,
                        json.dumps(pattern.context_triggers)
                    ))
            
            conn.commit()
            logger.info(f"✅ Stored {len(patterns)} patterns for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to store patterns: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def predict_user_needs(self, user_id: str, current_context: Dict[str, Any]) -> List[PredictionResult]:
        """Generate predictions about user needs based on patterns and context"""
        predictions = []
        
        try:
            # Get user patterns
            patterns = self._get_user_patterns(user_id)
            
            # Current time context
            now = datetime.now()
            current_hour = now.hour
            current_day = now.weekday()
            
            # Analyze each pattern for predictions
            for pattern_data in patterns:
                pattern = json.loads(pattern_data['pattern_data'])
                
                if pattern['pattern_type'] == 'temporal':
                    prediction = self._predict_temporal_needs(pattern, current_hour, current_day)
                    if prediction:
                        predictions.append(prediction)
                
                elif pattern['pattern_type'] == 'topic':
                    prediction = self._predict_topic_needs(pattern, current_context)
                    if prediction:
                        predictions.append(prediction)
                
                elif pattern['pattern_type'] == 'interaction_style':
                    prediction = self._predict_style_needs(pattern, current_context)
                    if prediction:
                        predictions.append(prediction)
            
            # Context-based predictions
            context_predictions = self._predict_from_context(user_id, current_context)
            predictions.extend(context_predictions)
            
            # Sort by confidence and urgency
            predictions.sort(key=lambda x: (
                -x.confidence,
                {'urgent': 4, 'high': 3, 'medium': 2, 'low': 1}[x.urgency]
            ))
            
            # Store predictions
            self._store_predictions(user_id, predictions)
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            if ERROR_HANDLING_AVAILABLE:
                log_error_with_context("Prediction failed", 
                                     {"user_id": user_id, "error": str(e)})
        
        return predictions[:5]  # Return top 5 predictions
    
    def _get_user_patterns(self, user_id: str) -> List[Dict]:
        """Retrieve user patterns from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pattern_type, pattern_data, frequency, success_rate, context_data
                FROM user_patterns 
                WHERE user_id = ? 
                ORDER BY frequency DESC, success_rate DESC
            """, (user_id,))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'pattern_type': row[0],
                    'pattern_data': row[1],
                    'frequency': row[2],
                    'success_rate': row[3],
                    'context_data': row[4]
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get patterns: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _predict_temporal_needs(self, pattern: Dict, current_hour: int, current_day: int) -> Optional[PredictionResult]:
        """Predict needs based on temporal patterns"""
        try:
            pattern_hour = int(pattern.get('typical_time', '0:0').split(':')[0])
            
            # If it's close to user's typical interaction time
            hour_diff = abs(current_hour - pattern_hour)
            if hour_diff <= 1 or hour_diff >= 23:  # Account for day boundary
                
                confidence = pattern.get('frequency', 0.5) * 0.8  # Temporal predictions are generally reliable
                
                suggestions = {
                    'morning': "Good morning! Ready to start your day? I can help with planning or quick tasks.",
                    'afternoon': "How's your day going? Need help with any work or projects?",
                    'evening': "Evening! Time to unwind? I can suggest entertainment or help wrap up the day.",
                    'night': "Working late? I can help you stay productive or suggest when to take a break."
                }
                
                time_period = 'morning' if 5 <= current_hour < 12 else \
                             'afternoon' if 12 <= current_hour < 17 else \
                             'evening' if 17 <= current_hour < 22 else 'night'
                
                return PredictionResult(
                    prediction_type="temporal_interaction",
                    confidence=confidence,
                    suggested_action=suggestions[time_period],
                    context={"time_period": time_period, "typical_hour": pattern_hour},
                    reasoning=f"User typically interacts around {pattern_hour}:00",
                    urgency="medium",
                    timestamp=datetime.now().isoformat(),
                    expires_at=(datetime.now() + timedelta(hours=2)).isoformat()
                )
        
        except Exception as e:
            logger.error(f"Temporal prediction failed: {e}")
        
        return None
    
    def _predict_topic_needs(self, pattern: Dict, current_context: Dict) -> Optional[PredictionResult]:
        """Predict needs based on topic patterns"""
        try:
            triggers = pattern.get('context_triggers', [])
            topic = triggers[0].replace('topic_', '') if triggers else None
            
            if not topic:
                return None
            
            frequency = pattern.get('frequency', 0.5)
            
            # Topic-specific suggestions
            topic_suggestions = {
                'work': "Need help with work tasks? I can assist with planning, writing, or problem-solving.",
                'health': "Time for a health check-in? I can help track fitness goals or suggest wellness tips.",
                'entertainment': "Looking for something fun? I can recommend movies, music, or games.",
                'learning': "Ready to learn something new? I can find courses, tutorials, or explain concepts.",
                'technology': "Tech questions? I can help troubleshoot, explain concepts, or suggest tools.",
                'planning': "Need to organize something? I can help with schedules, reminders, or planning."
            }
            
            if topic in topic_suggestions and frequency > 0.3:
                return PredictionResult(
                    prediction_type="topic_assistance",
                    confidence=frequency * 0.7,
                    suggested_action=topic_suggestions[topic],
                    context={"topic": topic, "frequency": frequency},
                    reasoning=f"User frequently discusses {topic} (frequency: {frequency:.2f})",
                    urgency="low" if frequency < 0.5 else "medium",
                    timestamp=datetime.now().isoformat(),
                    expires_at=(datetime.now() + timedelta(hours=4)).isoformat()
                )
        
        except Exception as e:
            logger.error(f"Topic prediction failed: {e}")
        
        return None
    
    def _predict_style_needs(self, pattern: Dict, current_context: Dict) -> Optional[PredictionResult]:
        """Predict needs based on interaction style"""
        try:
            triggers = pattern.get('context_triggers', [])
            personality = None
            
            for trigger in triggers:
                if trigger.startswith('personality_'):
                    personality = trigger.replace('personality_', '')
                    break
            
            if not personality:
                return None
            
            frequency = pattern.get('frequency', 0.5)
            
            # Style-based suggestions
            style_suggestions = {
                'friendly': "I notice you enjoy friendly conversations! Want to chat about something interesting?",
                'professional': "Ready for focused, professional assistance? I can help with business tasks.",
                'creative': "Time for some creative inspiration? I can help brainstorm or generate ideas.",
                'analytical': "Looking for data-driven insights? I can help analyze or explain complex topics.",
                'casual': "Want to keep things casual? I'm here for relaxed conversation and help."
            }
            
            if personality in style_suggestions and frequency > 0.4:
                return PredictionResult(
                    prediction_type="style_adaptation",
                    confidence=frequency * 0.6,
                    suggested_action=style_suggestions[personality],
                    context={"preferred_style": personality, "frequency": frequency},
                    reasoning=f"User prefers {personality} interaction style",
                    urgency="low",
                    timestamp=datetime.now().isoformat()
                )
        
        except Exception as e:
            logger.error(f"Style prediction failed: {e}")
        
        return None
    
    def _predict_from_context(self, user_id: str, context: Dict) -> List[PredictionResult]:
        """Generate predictions based on current context"""
        predictions = []
        
        try:
            # Time-based context predictions
            now = datetime.now()
            
            if context.get('location') == 'office' and 9 <= now.hour <= 17:
                predictions.append(PredictionResult(
                    prediction_type="work_context",
                    confidence=0.7,
                    suggested_action="In work mode? I can help with productivity, writing, or quick research.",
                    context={"location": "office", "work_hours": True},
                    reasoning="User is at office during work hours",
                    urgency="medium",
                    timestamp=now.isoformat()
                ))
            
            if context.get('weather') == 'rainy':
                predictions.append(PredictionResult(
                    prediction_type="weather_context",
                    confidence=0.6,
                    suggested_action="Rainy day? Perfect for indoor activities! I can suggest entertainment or learning.",
                    context={"weather": "rainy"},
                    reasoning="Rainy weather often changes activity preferences",
                    urgency="low",
                    timestamp=now.isoformat()
                ))
            
            # Weekend context
            if now.weekday() >= 5:  # Saturday or Sunday
                predictions.append(PredictionResult(
                    prediction_type="weekend_context",
                    confidence=0.8,
                    suggested_action="Weekend vibes! Want help planning fun activities or personal projects?",
                    context={"day_type": "weekend"},
                    reasoning="Weekend patterns differ from weekdays",
                    urgency="low",
                    timestamp=now.isoformat()
                ))
        
        except Exception as e:
            logger.error(f"Context prediction failed: {e}")
        
        return predictions
    
    def _store_predictions(self, user_id: str, predictions: List[PredictionResult]):
        """Store predictions in database for tracking accuracy"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for pred in predictions:
                cursor.execute("""
                    INSERT INTO prediction_history 
                    (user_id, prediction_type, prediction_data, confidence, context_at_time)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id,
                    pred.prediction_type,
                    json.dumps(asdict(pred)),
                    pred.confidence,
                    json.dumps(pred.context)
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_proactive_suggestions(self, user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get proactive suggestions for the user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent predictions that haven't expired
            cursor.execute("""
                SELECT prediction_data, confidence 
                FROM prediction_history 
                WHERE user_id = ? 
                AND datetime(created_at) > datetime('now', '-1 hour')
                AND was_accurate IS NULL
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            suggestions = []
            for row in cursor.fetchall():
                try:
                    pred_data = json.loads(row[0])
                    suggestions.append({
                        'type': pred_data.get('prediction_type'),
                        'suggestion': pred_data.get('suggested_action'),
                        'confidence': row[1],
                        'reasoning': pred_data.get('reasoning'),
                        'urgency': pred_data.get('urgency', 'low')
                    })
                except json.JSONDecodeError:
                    continue
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def update_prediction_feedback(self, user_id: str, prediction_type: str, was_accurate: bool, feedback: str = ""):
        """Update prediction accuracy based on user feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE prediction_history 
                SET was_accurate = ?, user_feedback = ?, resolved_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND prediction_type = ? 
                AND was_accurate IS NULL
                ORDER BY created_at DESC
                LIMIT 1
            """, (was_accurate, feedback, user_id, prediction_type))
            
            conn.commit()
            
            # Update pattern success rates based on feedback
            self._update_pattern_success_rates(user_id, prediction_type, was_accurate)
            
        except Exception as e:
            logger.error(f"Failed to update feedback: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _update_pattern_success_rates(self, user_id: str, prediction_type: str, was_accurate: bool):
        """Update pattern success rates based on prediction feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get related patterns
            cursor.execute("""
                SELECT id, success_rate FROM user_patterns 
                WHERE user_id = ? 
                AND pattern_type LIKE ?
            """, (user_id, f"%{prediction_type.split('_')[0]}%"))
            
            patterns = cursor.fetchall()
            
            for pattern_id, current_rate in patterns:
                # Update success rate using exponential moving average
                alpha = 0.3  # Learning rate
                new_accuracy = 1.0 if was_accurate else 0.0
                new_rate = alpha * new_accuracy + (1 - alpha) * current_rate
                
                cursor.execute("""
                    UPDATE user_patterns 
                    SET success_rate = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_rate, pattern_id))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to update pattern success rates: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

# Global instance
predictive_assistant = PredictiveAssistant()

def analyze_user_behavior(user_id: str, timeframe_days: int = 30) -> Dict[str, Any]:
    """Analyze user behavior and return patterns"""
    try:
        patterns = predictive_assistant.analyze_user_patterns(user_id, timeframe_days)
        return {
            'status': 'success',
            'patterns_found': len(patterns),
            'patterns': [asdict(p) for p in patterns],
            'analysis_timeframe': timeframe_days
        }
    except Exception as e:
        logger.error(f"Behavior analysis failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'patterns_found': 0
        }

def get_predictive_suggestions(user_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get predictive suggestions for user"""
    try:
        if context is None:
            context = {}
        
        predictions = predictive_assistant.predict_user_needs(user_id, context)
        suggestions = predictive_assistant.get_proactive_suggestions(user_id)
        
        return {
            'status': 'success',
            'predictions': [asdict(p) for p in predictions],
            'proactive_suggestions': suggestions,
            'total_suggestions': len(predictions) + len(suggestions),
            'context_used': context
        }
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'predictions': [],
            'proactive_suggestions': []
        }

def provide_prediction_feedback(user_id: str, prediction_type: str, was_helpful: bool, feedback: str = "") -> Dict[str, Any]:
    """Provide feedback on prediction accuracy"""
    try:
        predictive_assistant.update_prediction_feedback(user_id, prediction_type, was_helpful, feedback)
        return {
            'status': 'success',
            'message': 'Feedback recorded successfully',
            'learning_active': True
        }
    except Exception as e:
        logger.error(f"Feedback update failed: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == "__main__":
    # Test the predictive assistance system
    print("🔮 Horizon AI Predictive Assistance System")
    print("=" * 50)
    
    # Initialize
    assistant = PredictiveAssistant()
    
    # Test pattern analysis
    test_user = "test_user_123"
    patterns = assistant.analyze_user_patterns(test_user)
    print(f"✅ Found {len(patterns)} patterns for {test_user}")
    
    # Test predictions
    context = {
        'time': datetime.now().isoformat(),
        'location': 'home',
        'weather': 'sunny'
    }
    predictions = assistant.predict_user_needs(test_user, context)
    print(f"✅ Generated {len(predictions)} predictions")
    
    for pred in predictions:
        print(f"  📝 {pred.prediction_type}: {pred.suggested_action} (confidence: {pred.confidence:.2f})")
    
    print("\n🎉 Predictive assistance system ready!")