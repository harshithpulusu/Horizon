#!/usr/bin/env python3
"""
Enterprise Security Database Schema for Horizon AI
Comprehensive security-focused database tables and operations
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
import secrets
from utils.security_manager import EncryptionManager

class SecurityDatabase:
    """Manages security-related database operations"""
    
    def __init__(self, db_path: str = "horizon_security.db"):
        self.db_path = db_path
        self.encryption_manager = EncryptionManager()
        self.init_security_tables()
    
    def get_connection(self):
        """Get database connection with security settings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def init_security_tables(self):
        """Initialize all security-related database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Users table with security features
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    email_encrypted TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    is_active BOOLEAN DEFAULT 1,
                    email_verified BOOLEAN DEFAULT 0,
                    two_factor_enabled BOOLEAN DEFAULT 0,
                    two_factor_secret TEXT,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP NULL,
                    last_login TIMESTAMP NULL,
                    last_password_change TIMESTAMP NOT NULL,
                    password_history TEXT, -- JSON array of previous password hashes
                    security_questions TEXT, -- JSON array of security Q&A
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by_ip TEXT,
                    last_login_ip TEXT,
                    account_flags TEXT -- JSON object for various security flags
                )
            ''')
            
            # User sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    access_token_hash TEXT NOT NULL,
                    refresh_token_hash TEXT,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    device_fingerprint TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    logout_reason TEXT,
                    security_flags TEXT, -- JSON object
                    FOREIGN KEY (user_id) REFERENCES security_users (user_id)
                )
            ''')
            
            # Security events log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL, -- login, logout, failed_login, password_change, etc.
                    user_id TEXT,
                    session_id TEXT,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    event_data TEXT, -- JSON object with event details
                    risk_score INTEGER DEFAULT 0,
                    threat_level TEXT DEFAULT 'LOW', -- LOW, MEDIUM, HIGH, CRITICAL
                    action_taken TEXT, -- ALLOW, BLOCK, MONITOR, ALERT
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT 0,
                    resolution_notes TEXT,
                    FOREIGN KEY (user_id) REFERENCES security_users (user_id),
                    FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
                )
            ''')
            
            # Rate limiting table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rate_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT NOT NULL, -- IP address, user_id, or custom identifier
                    endpoint TEXT NOT NULL,
                    request_count INTEGER DEFAULT 1,
                    window_start TIMESTAMP NOT NULL,
                    window_end TIMESTAMP NOT NULL,
                    blocked BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # IP blacklist/whitelist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ip_security (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip_address TEXT NOT NULL,
                    ip_range TEXT, -- CIDR notation for ranges
                    list_type TEXT NOT NULL, -- BLACKLIST, WHITELIST, GREYLIST
                    reason TEXT NOT NULL,
                    added_by TEXT,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    incident_count INTEGER DEFAULT 0,
                    last_incident TIMESTAMP
                )
            ''')
            
            # API keys and tokens
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    api_key_hash TEXT NOT NULL,
                    api_secret_hash TEXT,
                    key_name TEXT NOT NULL,
                    permissions TEXT, -- JSON array of allowed operations
                    rate_limit_tier TEXT DEFAULT 'STANDARD',
                    is_active BOOLEAN DEFAULT 1,
                    last_used TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    revoked_at TIMESTAMP,
                    revocation_reason TEXT,
                    ip_restrictions TEXT, -- JSON array of allowed IPs
                    FOREIGN KEY (user_id) REFERENCES security_users (user_id)
                )
            ''')
            
            # Security policies
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_policies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_name TEXT UNIQUE NOT NULL,
                    policy_type TEXT NOT NULL, -- PASSWORD, SESSION, RATE_LIMIT, etc.
                    policy_config TEXT NOT NULL, -- JSON configuration
                    is_active BOOLEAN DEFAULT 1,
                    priority INTEGER DEFAULT 0,
                    applies_to_roles TEXT, -- JSON array of applicable roles
                    created_by TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    effective_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    effective_until TIMESTAMP
                )
            ''')
            
            # Audit trail
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    old_values TEXT, -- JSON object of previous values
                    new_values TEXT, -- JSON object of new values
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    session_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    additional_data TEXT, -- JSON object for extra context
                    FOREIGN KEY (user_id) REFERENCES security_users (user_id)
                )
            ''')
            
            # Threat intelligence
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threat_intelligence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    threat_type TEXT NOT NULL, -- MALWARE, PHISHING, BRUTEFORCE, etc.
                    indicator_type TEXT NOT NULL, -- IP, DOMAIN, URL, HASH, etc.
                    indicator_value TEXT NOT NULL,
                    threat_level TEXT NOT NULL, -- LOW, MEDIUM, HIGH, CRITICAL
                    confidence_score REAL DEFAULT 0.5,
                    source TEXT NOT NULL,
                    description TEXT,
                    tags TEXT, -- JSON array of tags
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    false_positive BOOLEAN DEFAULT 0,
                    action_taken TEXT -- BLOCK, MONITOR, ALERT
                )
            ''')
            
            # Security alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL, -- LOW, MEDIUM, HIGH, CRITICAL
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    user_id TEXT,
                    ip_address TEXT,
                    event_data TEXT, -- JSON object with related data
                    status TEXT DEFAULT 'OPEN', -- OPEN, INVESTIGATING, RESOLVED, FALSE_POSITIVE
                    assigned_to TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    resolution_notes TEXT,
                    escalated BOOLEAN DEFAULT 0,
                    escalation_reason TEXT,
                    related_alerts TEXT, -- JSON array of related alert IDs
                    FOREIGN KEY (user_id) REFERENCES security_users (user_id)
                )
            ''')
            
            # Compliance and regulations tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regulation_type TEXT NOT NULL, -- GDPR, CCPA, HIPAA, SOX, etc.
                    event_type TEXT NOT NULL, -- DATA_ACCESS, DATA_DELETION, CONSENT, etc.
                    user_id TEXT,
                    data_subject_id TEXT,
                    legal_basis TEXT,
                    purpose TEXT NOT NULL,
                    data_categories TEXT, -- JSON array of data types
                    retention_period INTEGER, -- days
                    consent_given BOOLEAN DEFAULT 0,
                    consent_withdrawn BOOLEAN DEFAULT 0,
                    processing_details TEXT, -- JSON object
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    compliance_officer TEXT,
                    notes TEXT,
                    FOREIGN KEY (user_id) REFERENCES security_users (user_id)
                )
            ''')
            
            # Data encryption keys management
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS encryption_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT UNIQUE NOT NULL,
                    key_type TEXT NOT NULL, -- AES, RSA, FERNET, etc.
                    key_purpose TEXT NOT NULL, -- DATABASE, SESSION, API, etc.
                    key_hash TEXT NOT NULL, -- Hash of the key for identification
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    rotated_at TIMESTAMP,
                    rotation_reason TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    created_by TEXT NOT NULL
                )
            ''')
            
            # Create indexes for performance
            self._create_security_indexes(cursor)
            
            # Insert default security policies
            self._insert_default_policies(cursor)
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _create_security_indexes(self, cursor):
        """Create database indexes for security tables"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_security_users_username ON security_users(username)",
            "CREATE INDEX IF NOT EXISTS idx_security_users_email ON security_users(email_encrypted)",
            "CREATE INDEX IF NOT EXISTS idx_security_users_user_id ON security_users(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security_events(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_security_events_ip_address ON security_events(ip_address)",
            "CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_security_events_threat_level ON security_events(threat_level)",
            "CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier)",
            "CREATE INDEX IF NOT EXISTS idx_rate_limits_endpoint ON rate_limits(endpoint)",
            "CREATE INDEX IF NOT EXISTS idx_ip_security_ip_address ON ip_security(ip_address)",
            "CREATE INDEX IF NOT EXISTS idx_api_credentials_user_id ON api_credentials(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_trail_user_id ON audit_trail(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp ON audit_trail(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_threat_intelligence_indicator ON threat_intelligence(indicator_value)",
            "CREATE INDEX IF NOT EXISTS idx_security_alerts_status ON security_alerts(status)",
            "CREATE INDEX IF NOT EXISTS idx_security_alerts_severity ON security_alerts(severity)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    def _insert_default_policies(self, cursor):
        """Insert default security policies"""
        default_policies = [
            {
                'policy_name': 'DEFAULT_PASSWORD_POLICY',
                'policy_type': 'PASSWORD',
                'policy_config': json.dumps({
                    'min_length': 12,
                    'require_uppercase': True,
                    'require_lowercase': True,
                    'require_numbers': True,
                    'require_special_chars': True,
                    'password_history_count': 5,
                    'max_age_days': 90
                }),
                'applies_to_roles': json.dumps(['user', 'admin', 'moderator']),
                'created_by': 'SYSTEM'
            },
            {
                'policy_name': 'DEFAULT_SESSION_POLICY',
                'policy_type': 'SESSION',
                'policy_config': json.dumps({
                    'max_duration_hours': 24,
                    'idle_timeout_minutes': 30,
                    'max_concurrent_sessions': 5,
                    'require_fresh_auth_for_sensitive': True
                }),
                'applies_to_roles': json.dumps(['user', 'admin', 'moderator']),
                'created_by': 'SYSTEM'
            },
            {
                'policy_name': 'DEFAULT_RATE_LIMIT_POLICY',
                'policy_type': 'RATE_LIMIT',
                'policy_config': json.dumps({
                    'requests_per_minute': 60,
                    'requests_per_hour': 1000,
                    'requests_per_day': 10000,
                    'burst_allowance': 10
                }),
                'applies_to_roles': json.dumps(['user']),
                'created_by': 'SYSTEM'
            },
            {
                'policy_name': 'ADMIN_RATE_LIMIT_POLICY',
                'policy_type': 'RATE_LIMIT',
                'policy_config': json.dumps({
                    'requests_per_minute': 120,
                    'requests_per_hour': 5000,
                    'requests_per_day': 50000,
                    'burst_allowance': 20
                }),
                'applies_to_roles': json.dumps(['admin', 'moderator']),
                'created_by': 'SYSTEM'
            }
        ]
        
        for policy in default_policies:
            cursor.execute('''
                INSERT OR IGNORE INTO security_policies 
                (policy_name, policy_type, policy_config, applies_to_roles, created_by)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                policy['policy_name'],
                policy['policy_type'], 
                policy['policy_config'],
                policy['applies_to_roles'],
                policy['created_by']
            ))
    
    def create_user(self, user_data: Dict) -> str:
        """Create a new user with security measures"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Generate salt and hash password
            salt = secrets.token_hex(32)
            password_hash = hashlib.pbkdf2_hmac('sha256', 
                                              user_data['password'].encode('utf-8'), 
                                              salt.encode('utf-8'), 
                                              100000)
            
            # Encrypt email
            encrypted_email = self.encryption_manager.encrypt(user_data['email'])
            
            cursor.execute('''
                INSERT INTO security_users 
                (user_id, username, email_encrypted, password_hash, salt, role, 
                 last_password_change, password_history, created_by_ip)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_data['user_id'],
                user_data['username'],
                encrypted_email,
                password_hash.hex(),
                salt,
                user_data.get('role', 'user'),
                datetime.now().isoformat(),
                json.dumps([password_hash.hex()]),
                user_data.get('ip_address', 'unknown')
            ))
            
            # Log user creation event
            self.log_security_event(
                event_type='USER_CREATED',
                user_id=user_data['user_id'],
                ip_address=user_data.get('ip_address', 'unknown'),
                event_data={'username': user_data['username'], 'role': user_data.get('role', 'user')}
            )
            
            conn.commit()
            return user_data['user_id']
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def create_session(self, session_data: Dict) -> str:
        """Create a new user session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO user_sessions 
                (session_id, user_id, access_token_hash, refresh_token_hash, 
                 ip_address, user_agent, device_fingerprint, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_data['session_id'],
                session_data['user_id'],
                hashlib.sha256(session_data['access_token'].encode()).hexdigest(),
                hashlib.sha256(session_data.get('refresh_token', '').encode()).hexdigest(),
                session_data['ip_address'],
                session_data.get('user_agent', ''),
                session_data.get('device_fingerprint', ''),
                session_data['expires_at']
            ))
            
            conn.commit()
            return session_data['session_id']
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def log_security_event(self, event_type: str, user_id: str = None, ip_address: str = None, 
                          event_data: Dict = None, risk_score: int = 0, threat_level: str = 'LOW'):
        """Log a security event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO security_events 
                (event_type, user_id, ip_address, event_data, risk_score, threat_level)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                event_type,
                user_id,
                ip_address,
                json.dumps(event_data) if event_data else None,
                risk_score,
                threat_level
            ))
            
            conn.commit()
            
        except Exception as e:
            print(f"Error logging security event: {e}")
        finally:
            conn.close()
    
    def create_security_alert(self, alert_data: Dict) -> int:
        """Create a security alert"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO security_alerts 
                (alert_type, severity, title, description, user_id, ip_address, event_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert_data['alert_type'],
                alert_data['severity'],
                alert_data['title'],
                alert_data['description'],
                alert_data.get('user_id'),
                alert_data.get('ip_address'),
                json.dumps(alert_data.get('event_data', {}))
            ))
            
            alert_id = cursor.lastrowid
            conn.commit()
            return alert_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM security_users WHERE username = ? AND is_active = 1
            ''', (username,))
            
            row = cursor.fetchone()
            if row:
                user_data = dict(row)
                # Decrypt email
                user_data['email'] = self.encryption_manager.decrypt(user_data['email_encrypted'])
                return user_data
            return None
            
        finally:
            conn.close()
    
    def update_failed_login_attempts(self, username: str, increment: bool = True):
        """Update failed login attempts for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if increment:
                cursor.execute('''
                    UPDATE security_users 
                    SET failed_login_attempts = failed_login_attempts + 1
                    WHERE username = ?
                ''', (username,))
            else:
                cursor.execute('''
                    UPDATE security_users 
                    SET failed_login_attempts = 0, locked_until = NULL
                    WHERE username = ?
                ''', (username,))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_security_metrics(self) -> Dict:
        """Get security metrics for monitoring"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            metrics = {}
            
            # Active sessions
            cursor.execute('SELECT COUNT(*) FROM user_sessions WHERE is_active = 1')
            metrics['active_sessions'] = cursor.fetchone()[0]
            
            # Recent security events (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM security_events 
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            metrics['events_24h'] = cursor.fetchone()[0]
            
            # Open security alerts
            cursor.execute('SELECT COUNT(*) FROM security_alerts WHERE status = "OPEN"')
            metrics['open_alerts'] = cursor.fetchone()[0]
            
            # Blocked IPs
            cursor.execute('''
                SELECT COUNT(*) FROM ip_security 
                WHERE list_type = "BLACKLIST" AND is_active = 1
            ''')
            metrics['blocked_ips'] = cursor.fetchone()[0]
            
            # Failed login attempts (last hour)
            cursor.execute('''
                SELECT COUNT(*) FROM security_events 
                WHERE event_type = "FAILED_LOGIN" 
                AND timestamp > datetime('now', '-1 hour')
            ''')
            metrics['failed_logins_1h'] = cursor.fetchone()[0]
            
            return metrics
            
        finally:
            conn.close()

# Global security database instance
security_db = SecurityDatabase()