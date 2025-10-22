"""
Horizon Database Core Module

This module handles all database operations for the Horizon AI Assistant.
It provides a unified interface for data storage and retrieval.

Classes:
- DatabaseManager: Main database management system
- UserManager: User data and profile management
- ConversationManager: Conversation history and context
- MemoryManager: AI memory and learning data
- AnalyticsManager: Usage analytics and insights

Functions:
- get_database_connection: Get database connection
- init_database: Initialize database with tables
- backup_database: Create database backup
- restore_database: Restore from backup
"""

import os
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from threading import Lock
from config import Config

# Database configuration
DATABASE_PATH = getattr(Config, 'DATABASE_PATH', 'ai_memory.db')
BACKUP_PATH = getattr(Config, 'BACKUP_PATH', 'backups/')

# Thread-safe database lock
db_lock = Lock()

# Database schema version
SCHEMA_VERSION = "2.0"

class DatabaseManager:
    """Main database management system."""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize database manager."""
        self.db_path = db_path
        self.connection = None
        self._ensure_database_exists()
        print(f"🗄️ Database Manager initialized: {db_path}")
    
    def _ensure_database_exists(self):
        """Ensure database file exists and is properly initialized."""
        if not os.path.exists(self.db_path):
            self.init_database()
        else:
            # Check if database needs updates
            self._check_schema_version()
    
    def _check_schema_version(self):
        """Check and update database schema if needed."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM metadata WHERE key = 'schema_version'")
                result = cursor.fetchone()
                
                if not result or result[0] != SCHEMA_VERSION:
                    print(f"📊 Updating database schema to version {SCHEMA_VERSION}")
                    self._update_schema(conn)
        except sqlite3.OperationalError:
            # Table doesn't exist, initialize database
            self.init_database()
    
    def _update_schema(self, conn: sqlite3.Connection):
        """Update database schema to current version."""
        cursor = conn.cursor()
        
        # Update schema version
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES ('schema_version', ?, ?)
        """, (SCHEMA_VERSION, datetime.now()))
        
        conn.commit()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn
    
    def init_database(self):
        """Initialize database with all required tables."""
        print("🚀 Initializing Horizon database...")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    email TEXT UNIQUE,
                    preferences TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_conversations INTEGER DEFAULT 0,
                    favorite_personality TEXT DEFAULT 'friendly'
                )
            """)
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    personality TEXT DEFAULT 'friendly',
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    user_id TEXT,
                    role TEXT CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    personality TEXT,
                    emotion_data TEXT,
                    mood_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # User memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    memory_type TEXT,
                    content TEXT NOT NULL,
                    context TEXT,
                    importance_score REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Personality profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_profiles (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    personality_type TEXT,
                    custom_traits TEXT,
                    usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    event_type TEXT,
                    event_data TEXT,
                    session_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Media generation history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS media_history (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    media_type TEXT,
                    prompt TEXT,
                    file_path TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Timer and Reminder tables for productivity features
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timers (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    duration_seconds INTEGER NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'created',
                    timer_type TEXT DEFAULT 'general',
                    notification_sound TEXT,
                    auto_start BOOLEAN DEFAULT 0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    reminder_time TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'active',
                    priority TEXT DEFAULT 'medium',
                    category TEXT DEFAULT 'general',
                    recurring_pattern TEXT,
                    notification_sent BOOLEAN DEFAULT 0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Insert schema version
            cursor.execute("""
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES ('schema_version', ?, ?)
            """, (SCHEMA_VERSION, datetime.now()))
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_memories_user_id ON user_memories(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_media_history_user_id ON media_history(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timers_user_id ON timers(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timers_status ON timers(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminders_user_id ON reminders(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminders_reminder_time ON reminders(reminder_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status)")
            
            conn.commit()
            print("✅ Database initialized successfully")
    
    def backup_database(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of the database."""
        if backup_name is None:
            backup_name = f"horizon_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        backup_path = os.path.join(BACKUP_PATH, backup_name)
        os.makedirs(BACKUP_PATH, exist_ok=True)
        
        # Create backup using SQL BACKUP command
        with self.get_connection() as source:
            with sqlite3.connect(backup_path) as backup:
                source.backup(backup)
        
        print(f"📦 Database backup created: {backup_path}")
        return backup_path
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Table counts
            tables = ['users', 'conversations', 'messages', 'user_memories', 
                     'personality_profiles', 'analytics', 'media_history']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Database size
            cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
            result = cursor.fetchone()
            stats['database_size_bytes'] = result[0] if result else 0
            
            # Schema version
            cursor.execute("SELECT value FROM metadata WHERE key = 'schema_version'")
            result = cursor.fetchone()
            stats['schema_version'] = result[0] if result else 'unknown'
            
            return stats


class UserManager:
    """User data and profile management."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize user manager."""
        self.db = db_manager
        print("👤 User Manager initialized")
    
    def create_user(self, username: str, email: str = None, 
                   preferences: Dict[str, Any] = None) -> str:
        """Create a new user."""
        user_id = str(uuid.uuid4())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (id, username, email, preferences)
                VALUES (?, ?, ?, ?)
            """, (user_id, username, email, json.dumps(preferences or {})))
            conn.commit()
        
        print(f"✅ User created: {username} ({user_id})")
        return user_id
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def update_user_activity(self, user_id: str):
        """Update user's last activity timestamp."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET last_active = ? WHERE id = ?
            """, (datetime.now(), user_id))
            conn.commit()


class ConversationManager:
    """Conversation history and context management."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize conversation manager."""
        self.db = db_manager
        print("💬 Conversation Manager initialized")
    
    def create_conversation(self, user_id: str, title: str = None, 
                          personality: str = 'friendly') -> str:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        title = title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (id, user_id, title, personality)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, user_id, title, personality))
            conn.commit()
        
        return conversation_id
    
    def add_message(self, conversation_id: str, user_id: str, role: str, 
                   content: str, personality: str = None, 
                   emotion_data: Dict[str, Any] = None,
                   mood_data: Dict[str, Any] = None) -> str:
        """Add a message to a conversation."""
        message_id = str(uuid.uuid4())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (id, conversation_id, user_id, role, content, 
                                    personality, emotion_data, mood_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (message_id, conversation_id, user_id, role, content, 
                  personality, json.dumps(emotion_data or {}), 
                  json.dumps(mood_data or {})))
            
            # Update conversation stats
            cursor.execute("""
                UPDATE conversations 
                SET last_message_at = ?, message_count = message_count + 1
                WHERE id = ?
            """, (datetime.now(), conversation_id))
            
            conn.commit()
        
        return message_id
    
    def get_conversation_history(self, conversation_id: str, 
                               limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation message history."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (conversation_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]


class MemoryManager:
    """AI memory and learning data management."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize memory manager."""
        self.db = db_manager
        print("🧠 Memory Manager initialized")
    
    def store_memory(self, user_id: str, memory_type: str, content: str,
                    context: str = None, importance_score: float = 0.5) -> str:
        """Store a user memory."""
        memory_id = str(uuid.uuid4())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_memories (id, user_id, memory_type, content, 
                                         context, importance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (memory_id, user_id, memory_type, content, context, importance_score))
            conn.commit()
        
        return memory_id
    
    def get_user_memories(self, user_id: str, memory_type: str = None,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """Get user memories."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            if memory_type:
                cursor.execute("""
                    SELECT * FROM user_memories 
                    WHERE user_id = ? AND memory_type = ?
                    ORDER BY importance_score DESC, last_accessed DESC
                    LIMIT ?
                """, (user_id, memory_type, limit))
            else:
                cursor.execute("""
                    SELECT * FROM user_memories 
                    WHERE user_id = ?
                    ORDER BY importance_score DESC, last_accessed DESC
                    LIMIT ?
                """, (user_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]


class AnalyticsManager:
    """Usage analytics and insights management."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize analytics manager."""
        self.db = db_manager
        print("📊 Analytics Manager initialized")
    
    def log_event(self, user_id: str, event_type: str, event_data: Dict[str, Any],
                 session_id: str = None) -> str:
        """Log an analytics event."""
        event_id = str(uuid.uuid4())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analytics (id, user_id, event_type, event_data, session_id)
                VALUES (?, ?, ?, ?, ?)
            """, (event_id, user_id, event_type, json.dumps(event_data), session_id))
            conn.commit()
        
        return event_id
    
    def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user analytics for specified time period."""
        since_date = datetime.now() - timedelta(days=days)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_type, COUNT(*) as count
                FROM analytics 
                WHERE user_id = ? AND created_at >= ?
                GROUP BY event_type
                ORDER BY count DESC
            """, (user_id, since_date))
            
            events = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get total activity
            cursor.execute("""
                SELECT COUNT(*) FROM analytics 
                WHERE user_id = ? AND created_at >= ?
            """, (user_id, since_date))
            
            total_events = cursor.fetchone()[0]
            
            return {
                'total_events': total_events,
                'event_breakdown': events,
                'period_days': days,
                'most_common_event': max(events, key=events.get) if events else None
            }


class TimerManager:
    """Timer management system with CRUD operations."""
    
    def __init__(self, database_manager: DatabaseManager):
        """Initialize timer manager."""
        self.db_manager = database_manager
        print("⏱️ Timer Manager initialized")
    
    def create_timer(self, user_id: str, title: str, duration_seconds: int, 
                    description: str = None, timer_type: str = "general",
                    auto_start: bool = False, metadata: Dict[str, Any] = None) -> str:
        """Create a new timer."""
        timer_id = str(uuid.uuid4())
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO timers (id, user_id, title, description, duration_seconds, 
                                  status, timer_type, auto_start, metadata)
                VALUES (?, ?, ?, ?, ?, 'created', ?, ?, ?)
            """, (timer_id, user_id, title, description, duration_seconds, 
                  timer_type, auto_start, json.dumps(metadata or {})))
            conn.commit()
        
        return timer_id
    
    def get_timer(self, timer_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific timer by ID."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM timers WHERE id = ?", (timer_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(cursor, row)
            return None
    
    def get_user_timers(self, user_id: str, status: str = None) -> List[Dict[str, Any]]:
        """Get all timers for a user, optionally filtered by status."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT * FROM timers WHERE user_id = ? AND status = ?
                    ORDER BY created_at DESC
                """, (user_id, status))
            else:
                cursor.execute("""
                    SELECT * FROM timers WHERE user_id = ?
                    ORDER BY created_at DESC
                """, (user_id,))
            
            return [self._row_to_dict(cursor, row) for row in cursor.fetchall()]
    
    def update_timer(self, timer_id: str, **kwargs) -> bool:
        """Update timer fields."""
        if not kwargs:
            return False
        
        # Add updated_at timestamp
        kwargs['updated_at'] = datetime.now()
        
        # Build dynamic UPDATE query
        set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        values = list(kwargs.values()) + [timer_id]
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE timers SET {set_clause} WHERE id = ?", values)
            conn.commit()
            
            return cursor.rowcount > 0
    
    def start_timer(self, timer_id: str) -> bool:
        """Start a timer."""
        now = datetime.now()
        timer = self.get_timer(timer_id)
        
        if not timer:
            return False
        
        duration = timedelta(seconds=timer['duration_seconds'])
        end_time = now + duration
        
        return self.update_timer(timer_id, 
                               status='running',
                               start_time=now,
                               end_time=end_time)
    
    def pause_timer(self, timer_id: str) -> bool:
        """Pause a running timer."""
        return self.update_timer(timer_id, status='paused')
    
    def stop_timer(self, timer_id: str) -> bool:
        """Stop a timer."""
        return self.update_timer(timer_id, status='stopped')
    
    def complete_timer(self, timer_id: str) -> bool:
        """Mark a timer as completed."""
        return self.update_timer(timer_id, status='completed')
    
    def delete_timer(self, timer_id: str) -> bool:
        """Delete a timer."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM timers WHERE id = ?", (timer_id,))
            conn.commit()
            
            return cursor.rowcount > 0
    
    def get_active_timers(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active (running/paused) timers for a user."""
        return self.get_user_timers(user_id, status='running') + \
               self.get_user_timers(user_id, status='paused')
    
    def _row_to_dict(self, cursor, row) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        columns = [desc[0] for desc in cursor.description]
        result = dict(zip(columns, row))
        
        # Parse JSON metadata
        if result.get('metadata'):
            try:
                result['metadata'] = json.loads(result['metadata'])
            except:
                result['metadata'] = {}
        
        return result


class ReminderManager:
    """Reminder management system with CRUD operations."""
    
    def __init__(self, database_manager: DatabaseManager):
        """Initialize reminder manager."""
        self.db_manager = database_manager
        print("🔔 Reminder Manager initialized")
    
    def create_reminder(self, user_id: str, title: str, reminder_time: datetime,
                       description: str = None, priority: str = "medium",
                       category: str = "general", recurring_pattern: str = None,
                       metadata: Dict[str, Any] = None) -> str:
        """Create a new reminder."""
        reminder_id = str(uuid.uuid4())
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reminders (id, user_id, title, description, reminder_time,
                                     priority, category, recurring_pattern, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (reminder_id, user_id, title, description, reminder_time,
                  priority, category, recurring_pattern, json.dumps(metadata or {})))
            conn.commit()
        
        return reminder_id
    
    def get_reminder(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific reminder by ID."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(cursor, row)
            return None
    
    def get_user_reminders(self, user_id: str, status: str = None) -> List[Dict[str, Any]]:
        """Get all reminders for a user, optionally filtered by status."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT * FROM reminders WHERE user_id = ? AND status = ?
                    ORDER BY reminder_time ASC
                """, (user_id, status))
            else:
                cursor.execute("""
                    SELECT * FROM reminders WHERE user_id = ?
                    ORDER BY reminder_time ASC
                """, (user_id,))
            
            return [self._row_to_dict(cursor, row) for row in cursor.fetchall()]
    
    def get_due_reminders(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get reminders that are due now."""
        now = datetime.now()
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("""
                    SELECT * FROM reminders 
                    WHERE user_id = ? AND reminder_time <= ? AND status = 'active'
                    ORDER BY reminder_time ASC
                """, (user_id, now))
            else:
                cursor.execute("""
                    SELECT * FROM reminders 
                    WHERE reminder_time <= ? AND status = 'active'
                    ORDER BY reminder_time ASC
                """, (now,))
            
            return [self._row_to_dict(cursor, row) for row in cursor.fetchall()]
    
    def update_reminder(self, reminder_id: str, **kwargs) -> bool:
        """Update reminder fields."""
        if not kwargs:
            return False
        
        # Add updated_at timestamp
        kwargs['updated_at'] = datetime.now()
        
        # Build dynamic UPDATE query
        set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        values = list(kwargs.values()) + [reminder_id]
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE reminders SET {set_clause} WHERE id = ?", values)
            conn.commit()
            
            return cursor.rowcount > 0
    
    def snooze_reminder(self, reminder_id: str, minutes: int = 10) -> bool:
        """Snooze a reminder by specified minutes."""
        reminder = self.get_reminder(reminder_id)
        if not reminder:
            return False
        
        current_time = datetime.fromisoformat(reminder['reminder_time'].replace('Z', '+00:00')) \
                      if isinstance(reminder['reminder_time'], str) else reminder['reminder_time']
        new_time = current_time + timedelta(minutes=minutes)
        
        return self.update_reminder(reminder_id, reminder_time=new_time)
    
    def complete_reminder(self, reminder_id: str) -> bool:
        """Mark a reminder as completed."""
        return self.update_reminder(reminder_id, status='completed', notification_sent=True)
    
    def delete_reminder(self, reminder_id: str) -> bool:
        """Delete a reminder."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            conn.commit()
            
            return cursor.rowcount > 0
    
    def _row_to_dict(self, cursor, row) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        columns = [desc[0] for desc in cursor.description]
        result = dict(zip(columns, row))
        
        # Parse JSON metadata
        if result.get('metadata'):
            try:
                result['metadata'] = json.loads(result['metadata'])
            except:
                result['metadata'] = {}
        
        return result


# Global instances
database_manager = None
user_manager = None
conversation_manager = None
memory_manager = None
analytics_manager = None
timer_manager = None
reminder_manager = None

def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global database_manager
    if database_manager is None:
        database_manager = DatabaseManager()
    return database_manager

def get_user_manager() -> UserManager:
    """Get the global user manager instance."""
    global user_manager
    if user_manager is None:
        user_manager = UserManager(get_database_manager())
    return user_manager

def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager instance."""
    global conversation_manager
    if conversation_manager is None:
        conversation_manager = ConversationManager(get_database_manager())
    return conversation_manager

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global memory_manager
    if memory_manager is None:
        memory_manager = MemoryManager(get_database_manager())
    return memory_manager

def get_analytics_manager() -> AnalyticsManager:
    """Get the global analytics manager instance."""
    global analytics_manager
    if analytics_manager is None:
        analytics_manager = AnalyticsManager(get_database_manager())
    return analytics_manager

def get_timer_manager() -> TimerManager:
    """Get the global timer manager instance."""
    global timer_manager
    if timer_manager is None:
        timer_manager = TimerManager(get_database_manager())
    return timer_manager

def get_reminder_manager() -> ReminderManager:
    """Get the global reminder manager instance."""
    global reminder_manager
    if reminder_manager is None:
        reminder_manager = ReminderManager(get_database_manager())
    return reminder_manager

# Convenience functions for backward compatibility
def get_database_connection() -> sqlite3.Connection:
    """Get database connection."""
    return get_database_manager().get_connection()

def init_database():
    """Initialize database."""
    return get_database_manager().init_database()

def backup_database(backup_name: Optional[str] = None) -> str:
    """Create database backup."""
    return get_database_manager().backup_database(backup_name)