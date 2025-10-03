#!/usr/bin/env python3
"""
Enterprise-Grade Security Manager for Horizon AI
Provides comprehensive security features including authentication, authorization,
encryption, rate limiting, and threat detection.
"""

import hashlib
import secrets
import jwt
import bcrypt
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from functools import wraps
import re
import ipaddress
from flask import request, jsonify, g
import redis
import sqlite3
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

# Configure security logging
security_logger = logging.getLogger('horizon_security')
security_logger.setLevel(logging.INFO)
handler = logging.FileHandler('horizon_security.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
security_logger.addHandler(handler)

class SecurityConfig:
    """Enterprise security configuration"""
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_HOURS = 24
    JWT_REFRESH_EXPIRATION_DAYS = 30
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = 60
    RATE_LIMIT_PER_HOUR = 1000
    RATE_LIMIT_PER_DAY = 10000
    
    # Password Policy
    MIN_PASSWORD_LENGTH = 12
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_NUMBERS = True
    REQUIRE_SPECIAL_CHARS = True
    PASSWORD_HISTORY_COUNT = 5
    
    # Session Security
    SESSION_TIMEOUT_MINUTES = 30
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15
    
    # Encryption
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())
    
    # Security Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }
    
    # Threat Detection
    MAX_FAILED_REQUESTS = 10
    SUSPICIOUS_PATTERNS = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'(union|select|drop|insert|update|delete)\s+',
        r'\.\./',
        r'%00',
        r'<iframe',
        r'eval\s*\(',
        r'document\.cookie'
    ]

class EncryptionManager:
    """Handles encryption and decryption of sensitive data"""
    
    def __init__(self):
        key = SecurityConfig.ENCRYPTION_KEY.encode() if isinstance(SecurityConfig.ENCRYPTION_KEY, str) else SecurityConfig.ENCRYPTION_KEY
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

class RateLimiter:
    """Advanced rate limiting with Redis backend"""
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_available = True
        except:
            self.redis_available = False
            self.in_memory_store = {}
    
    def is_allowed(self, identifier: str, limit: int, window: int) -> Tuple[bool, Dict]:
        """Check if request is within rate limits"""
        current_time = int(time.time())
        window_start = current_time - window
        
        if self.redis_available:
            return self._redis_rate_limit(identifier, limit, window, current_time, window_start)
        else:
            return self._memory_rate_limit(identifier, limit, window, current_time, window_start)
    
    def _redis_rate_limit(self, identifier: str, limit: int, window: int, current_time: int, window_start: int) -> Tuple[bool, Dict]:
        """Redis-based rate limiting"""
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(identifier, 0, window_start)
        
        # Count current requests
        pipe.zcard(identifier)
        
        # Add current request
        pipe.zadd(identifier, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(identifier, window)
        
        results = pipe.execute()
        current_count = results[1] + 1
        
        remaining = max(0, limit - current_count)
        reset_time = current_time + window
        
        return current_count <= limit, {
            'limit': limit,
            'remaining': remaining,
            'reset': reset_time,
            'retry_after': reset_time - current_time if current_count > limit else 0
        }
    
    def _memory_rate_limit(self, identifier: str, limit: int, window: int, current_time: int, window_start: int) -> Tuple[bool, Dict]:
        """In-memory rate limiting fallback"""
        if identifier not in self.in_memory_store:
            self.in_memory_store[identifier] = []
        
        # Remove old entries
        self.in_memory_store[identifier] = [
            timestamp for timestamp in self.in_memory_store[identifier]
            if timestamp > window_start
        ]
        
        # Add current request
        self.in_memory_store[identifier].append(current_time)
        
        current_count = len(self.in_memory_store[identifier])
        remaining = max(0, limit - current_count)
        reset_time = current_time + window
        
        return current_count <= limit, {
            'limit': limit,
            'remaining': remaining,
            'reset': reset_time,
            'retry_after': reset_time - current_time if current_count > limit else 0
        }

class ThreatDetector:
    """Advanced threat detection and prevention"""
    
    def __init__(self):
        self.suspicious_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SecurityConfig.SUSPICIOUS_PATTERNS]
        self.blocked_ips = set()
        self.suspicious_activities = {}
    
    def analyze_request(self, request_data: Dict) -> Dict[str, Any]:
        """Analyze request for threats"""
        threats = []
        risk_score = 0
        
        # Check for malicious patterns
        for key, value in request_data.items():
            if isinstance(value, str):
                for pattern in self.suspicious_patterns:
                    if pattern.search(value):
                        threats.append(f"Suspicious pattern in {key}: {pattern.pattern}")
                        risk_score += 10
        
        # Check request size
        if len(str(request_data)) > 100000:  # 100KB
            threats.append("Unusually large request")
            risk_score += 5
        
        # Determine threat level
        if risk_score >= 20:
            threat_level = "HIGH"
        elif risk_score >= 10:
            threat_level = "MEDIUM"
        elif risk_score > 0:
            threat_level = "LOW"
        else:
            threat_level = "NONE"
        
        return {
            'threat_level': threat_level,
            'risk_score': risk_score,
            'threats': threats,
            'action': 'BLOCK' if threat_level == 'HIGH' else 'MONITOR'
        }
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips
    
    def block_ip(self, ip_address: str, reason: str = "Security violation"):
        """Block IP address"""
        self.blocked_ips.add(ip_address)
        security_logger.warning(f"IP blocked: {ip_address} - Reason: {reason}")
    
    def track_suspicious_activity(self, ip_address: str, activity: str):
        """Track suspicious activities per IP"""
        if ip_address not in self.suspicious_activities:
            self.suspicious_activities[ip_address] = []
        
        self.suspicious_activities[ip_address].append({
            'activity': activity,
            'timestamp': datetime.now().isoformat()
        })
        
        # Auto-block after too many suspicious activities
        if len(self.suspicious_activities[ip_address]) >= SecurityConfig.MAX_FAILED_REQUESTS:
            self.block_ip(ip_address, "Multiple suspicious activities")

class AuthenticationManager:
    """Enterprise authentication and authorization"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.rate_limiter = RateLimiter()
        self.threat_detector = ThreatDetector()
        self.failed_attempts = {}
        self.active_sessions = {}
    
    def create_user(self, username: str, password: str, email: str, role: str = "user") -> Dict[str, Any]:
        """Create new user with security validations"""
        # Validate password strength
        password_validation = self.validate_password(password)
        if not password_validation['valid']:
            return {'success': False, 'error': 'Password does not meet security requirements', 'details': password_validation}
        
        # Hash password
        password_hash = self.encryption_manager.hash_password(password)
        
        # Generate user ID
        user_id = secrets.token_urlsafe(16)
        
        # Create user record (this would typically go to database)
        user_data = {
            'user_id': user_id,
            'username': username,
            'password_hash': password_hash,
            'email': self.encryption_manager.encrypt(email),
            'role': role,
            'created_at': datetime.now().isoformat(),
            'is_active': True,
            'failed_login_attempts': 0,
            'last_login': None,
            'password_changed_at': datetime.now().isoformat(),
            'two_factor_enabled': False
        }
        
        security_logger.info(f"User created: {username} (ID: {user_id})")
        return {'success': True, 'user_id': user_id, 'user_data': user_data}
    
    def authenticate_user(self, username: str, password: str, ip_address: str) -> Dict[str, Any]:
        """Authenticate user with security checks"""
        
        # Check if IP is blocked
        if self.threat_detector.is_ip_blocked(ip_address):
            return {'success': False, 'error': 'Access denied', 'code': 'IP_BLOCKED'}
        
        # Rate limiting for login attempts
        allowed, rate_info = self.rate_limiter.is_allowed(f"login:{ip_address}", 5, 300)  # 5 attempts per 5 minutes
        if not allowed:
            return {'success': False, 'error': 'Too many login attempts', 'retry_after': rate_info['retry_after']}
        
        # Check account lockout
        if username in self.failed_attempts:
            attempts_data = self.failed_attempts[username]
            if attempts_data['count'] >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
                lockout_end = attempts_data['locked_until']
                if datetime.now() < lockout_end:
                    return {'success': False, 'error': 'Account locked', 'unlock_time': lockout_end.isoformat()}
                else:
                    # Reset after lockout period
                    del self.failed_attempts[username]
        
        # This would typically fetch from database
        # For demo, we'll simulate user lookup
        user_data = self._get_user_by_username(username)
        if not user_data:
            self._record_failed_attempt(username, ip_address)
            return {'success': False, 'error': 'Invalid credentials'}
        
        # Verify password
        if not self.encryption_manager.verify_password(password, user_data['password_hash']):
            self._record_failed_attempt(username, ip_address)
            return {'success': False, 'error': 'Invalid credentials'}
        
        # Generate JWT tokens
        access_token = self._generate_access_token(user_data)
        refresh_token = self._generate_refresh_token(user_data)
        
        # Create session
        session_id = self._create_session(user_data, ip_address)
        
        # Clear failed attempts
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        security_logger.info(f"User authenticated: {username} from {ip_address}")
        
        return {
            'success': True,
            'access_token': access_token,
            'refresh_token': refresh_token,
            'session_id': session_id,
            'user': {
                'user_id': user_data['user_id'],
                'username': user_data['username'],
                'role': user_data['role']
            }
        }
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, SecurityConfig.JWT_SECRET_KEY, algorithms=[SecurityConfig.JWT_ALGORITHM])
            
            # Check if token is expired
            if datetime.fromtimestamp(payload['exp']) < datetime.now():
                return {'valid': False, 'error': 'Token expired'}
            
            # Check if session is still active
            if payload.get('session_id') and payload['session_id'] not in self.active_sessions:
                return {'valid': False, 'error': 'Session invalid'}
            
            return {'valid': True, 'payload': payload}
            
        except jwt.InvalidTokenError as e:
            return {'valid': False, 'error': str(e)}
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password against security policy"""
        issues = []
        
        if len(password) < SecurityConfig.MIN_PASSWORD_LENGTH:
            issues.append(f"Password must be at least {SecurityConfig.MIN_PASSWORD_LENGTH} characters")
        
        if SecurityConfig.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        if SecurityConfig.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        if SecurityConfig.REQUIRE_NUMBERS and not re.search(r'\d', password):
            issues.append("Password must contain at least one number")
        
        if SecurityConfig.REQUIRE_SPECIAL_CHARS and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character")
        
        # Check for common weak passwords
        weak_patterns = ['password', '123456', 'qwerty', 'admin', 'letmein']
        if any(weak in password.lower() for weak in weak_patterns):
            issues.append("Password contains common weak patterns")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'strength': self._calculate_password_strength(password)
        }
    
    def _calculate_password_strength(self, password: str) -> str:
        """Calculate password strength"""
        score = 0
        
        # Length bonus
        score += min(len(password) * 2, 20)
        
        # Character variety
        if re.search(r'[a-z]', password):
            score += 5
        if re.search(r'[A-Z]', password):
            score += 5
        if re.search(r'\d', password):
            score += 5
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 10
        
        # Uniqueness
        unique_chars = len(set(password))
        score += unique_chars
        
        if score >= 40:
            return "STRONG"
        elif score >= 25:
            return "MEDIUM"
        else:
            return "WEAK"
    
    def _record_failed_attempt(self, username: str, ip_address: str):
        """Record failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = {'count': 0, 'locked_until': None}
        
        self.failed_attempts[username]['count'] += 1
        
        if self.failed_attempts[username]['count'] >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
            lockout_until = datetime.now() + timedelta(minutes=SecurityConfig.LOCKOUT_DURATION_MINUTES)
            self.failed_attempts[username]['locked_until'] = lockout_until
            security_logger.warning(f"Account locked: {username} from {ip_address}")
        
        self.threat_detector.track_suspicious_activity(ip_address, f"Failed login for {username}")
    
    def _get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user data by username (placeholder - would query database)"""
        # This would typically query your database
        # For demo purposes, return None (no user found)
        return None
    
    def _generate_access_token(self, user_data: Dict) -> str:
        """Generate JWT access token"""
        payload = {
            'user_id': user_data['user_id'],
            'username': user_data['username'],
            'role': user_data['role'],
            'iat': datetime.now(),
            'exp': datetime.now() + timedelta(hours=SecurityConfig.JWT_EXPIRATION_HOURS),
            'type': 'access'
        }
        return jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
    
    def _generate_refresh_token(self, user_data: Dict) -> str:
        """Generate JWT refresh token"""
        payload = {
            'user_id': user_data['user_id'],
            'iat': datetime.now(),
            'exp': datetime.now() + timedelta(days=SecurityConfig.JWT_REFRESH_EXPIRATION_DAYS),
            'type': 'refresh'
        }
        return jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
    
    def _create_session(self, user_data: Dict, ip_address: str) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            'user_id': user_data['user_id'],
            'username': user_data['username'],
            'ip_address': ip_address,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'expires_at': datetime.now() + timedelta(minutes=SecurityConfig.SESSION_TIMEOUT_MINUTES)
        }
        return session_id

class SecurityMiddleware:
    """Flask middleware for security enforcement"""
    
    def __init__(self, app=None):
        self.app = app
        self.auth_manager = AuthenticationManager()
        self.rate_limiter = RateLimiter()
        self.threat_detector = ThreatDetector()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize security middleware with Flask app"""
        
        @app.before_request
        def security_check():
            """Run security checks before each request"""
            
            # Add security headers
            @app.after_request
            def add_security_headers(response):
                for header, value in SecurityConfig.SECURITY_HEADERS.items():
                    response.headers[header] = value
                return response
            
            # Get client IP
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            if client_ip:
                client_ip = client_ip.split(',')[0].strip()
            
            # Check if IP is blocked
            if self.threat_detector.is_ip_blocked(client_ip):
                security_logger.warning(f"Blocked IP attempted access: {client_ip}")
                return jsonify({'error': 'Access denied'}), 403
            
            # Rate limiting
            endpoint = request.endpoint or 'unknown'
            allowed, rate_info = self.rate_limiter.is_allowed(
                f"global:{client_ip}", 
                SecurityConfig.RATE_LIMIT_PER_MINUTE, 
                60
            )
            
            if not allowed:
                security_logger.warning(f"Rate limit exceeded: {client_ip} on {endpoint}")
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': rate_info['retry_after']
                }), 429
            
            # Threat detection for POST requests
            if request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    request_data = request.get_json() or {}
                    request_data.update(request.form.to_dict())
                    request_data.update({k: v for k, v in request.args.items()})
                    
                    threat_analysis = self.threat_detector.analyze_request(request_data)
                    
                    if threat_analysis['action'] == 'BLOCK':
                        security_logger.error(f"Threat detected from {client_ip}: {threat_analysis['threats']}")
                        self.threat_detector.block_ip(client_ip, "Malicious request detected")
                        return jsonify({'error': 'Request blocked for security reasons'}), 403
                    
                    elif threat_analysis['threat_level'] != 'NONE':
                        security_logger.warning(f"Suspicious request from {client_ip}: {threat_analysis['threats']}")
                        
                except Exception as e:
                    security_logger.error(f"Error in threat detection: {e}")
        
        # Authentication decorator
        def require_auth(roles=None):
            """Decorator to require authentication"""
            def decorator(f):
                @wraps(f)
                def decorated_function(*args, **kwargs):
                    token = request.headers.get('Authorization')
                    if not token:
                        return jsonify({'error': 'Authentication required'}), 401
                    
                    if token.startswith('Bearer '):
                        token = token[7:]
                    
                    validation = self.auth_manager.validate_token(token)
                    if not validation['valid']:
                        return jsonify({'error': validation['error']}), 401
                    
                    # Check role authorization
                    if roles:
                        user_role = validation['payload'].get('role')
                        if user_role not in roles:
                            return jsonify({'error': 'Insufficient permissions'}), 403
                    
                    # Store user info in Flask's g object
                    g.current_user = validation['payload']
                    
                    return f(*args, **kwargs)
                return decorated_function
            return decorator
        
        # Make decorator available to routes
        app.require_auth = require_auth

# Global security manager instance
security_manager = SecurityMiddleware()