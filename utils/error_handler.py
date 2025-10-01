#!/usr/bin/env python3
"""
Standardized Error Handling System for Horizon AI
Provides consistent error handling, logging, and response formatting
"""

import logging
import traceback
import functools
from datetime import datetime
from typing import Any, Dict, Optional, Union, Callable
from flask import jsonify, request
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('horizon_errors.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class HorizonError(Exception):
    """Base exception class for Horizon AI application"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        self.message = message
        self.error_code = error_code or 'UNKNOWN_ERROR'
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict:
        """Convert error to dictionary for JSON responses"""
        return {
            'error': True,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp
        }

class AIServiceError(HorizonError):
    """Error related to AI service operations (ChatGPT, Gemini, etc.)"""
    pass

class DatabaseError(HorizonError):
    """Error related to database operations"""
    pass

class ValidationError(HorizonError):
    """Error related to input validation"""
    pass

class AuthenticationError(HorizonError):
    """Error related to authentication and authorization"""
    pass

class PersonalityBlendingError(HorizonError):
    """Error related to personality blending operations"""
    pass

class FileOperationError(HorizonError):
    """Error related to file operations (image, video, audio)"""
    pass

class NetworkError(HorizonError):
    """Error related to network operations"""
    pass

def log_error(error: Exception, context: Dict = None) -> None:
    """Log error with context information"""
    context_info = context or {}
    
    # Add request context if available
    try:
        if request:
            context_info.update({
                'endpoint': request.endpoint,
                'method': request.method,
                'url': request.url,
                'user_agent': request.headers.get('User-Agent'),
                'ip_address': request.remote_addr
            })
    except RuntimeError:
        # Outside of request context
        pass
    
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context_info,
        'traceback': traceback.format_exc()
    }
    
    logger.error(f"Horizon AI Error: {json.dumps(error_info, indent=2)}")

def handle_error(error: Exception, default_message: str = "An unexpected error occurred") -> Dict:
    """Handle error and return standardized response"""
    
    if isinstance(error, HorizonError):
        log_error(error)
        return error.to_dict()
    
    # Handle standard exceptions
    error_mapping = {
        ValueError: ('VALIDATION_ERROR', 'Invalid input provided'),
        TypeError: ('TYPE_ERROR', 'Invalid data type'),
        KeyError: ('KEY_ERROR', 'Required field missing'),
        FileNotFoundError: ('FILE_NOT_FOUND', 'Requested file not found'),
        PermissionError: ('PERMISSION_ERROR', 'Insufficient permissions'),
        ConnectionError: ('CONNECTION_ERROR', 'Network connection failed'),
        TimeoutError: ('TIMEOUT_ERROR', 'Operation timed out')
    }
    
    error_code, message = error_mapping.get(type(error), ('UNKNOWN_ERROR', default_message))
    
    horizon_error = HorizonError(
        message=message,
        error_code=error_code,
        details={'original_error': str(error)}
    )
    
    log_error(error, {'handled_as': error_code})
    return horizon_error.to_dict()

def error_handler(default_message: str = "Operation failed"):
    """Decorator for automatic error handling in functions"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_response = handle_error(e, default_message)
                # For Flask routes, return JSON response
                if hasattr(func, '__name__') and func.__name__.endswith('_endpoint'):
                    return jsonify(error_response), 500
                # For regular functions, raise HorizonError
                raise HorizonError(
                    message=error_response['message'],
                    error_code=error_response['error_code'],
                    details=error_response['details']
                )
        return wrapper
    return decorator

def api_error_handler(func: Callable) -> Callable:
    """Decorator specifically for API endpoints"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # Ensure successful responses have consistent format
            if isinstance(result, dict) and 'error' not in result:
                result['success'] = True
                result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except HorizonError as e:
            log_error(e)
            return jsonify(e.to_dict()), 500
            
        except Exception as e:
            error_response = handle_error(e, "API operation failed")
            return jsonify(error_response), 500
    
    return wrapper

def validate_required_fields(data: Dict, required_fields: list) -> None:
    """Validate that required fields are present in data"""
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    
    if missing_fields:
        raise ValidationError(
            message=f"Missing required fields: {', '.join(missing_fields)}",
            error_code='MISSING_REQUIRED_FIELDS',
            details={'missing_fields': missing_fields, 'provided_fields': list(data.keys())}
        )

def validate_field_types(data: Dict, field_types: Dict[str, type]) -> None:
    """Validate that fields have correct types"""
    type_errors = []
    
    for field, expected_type in field_types.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], expected_type):
                type_errors.append({
                    'field': field,
                    'expected_type': expected_type.__name__,
                    'actual_type': type(data[field]).__name__,
                    'value': str(data[field])
                })
    
    if type_errors:
        raise ValidationError(
            message="Invalid field types provided",
            error_code='INVALID_FIELD_TYPES',
            details={'type_errors': type_errors}
        )

def safe_json_parse(json_string: str, default: Any = None) -> Any:
    """Safely parse JSON string with error handling"""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        log_error(e, {'json_string': json_string[:100]})  # Log first 100 chars
        return default

def safe_db_operation(operation: Callable, *args, **kwargs) -> Any:
    """Safely execute database operations with error handling"""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        raise DatabaseError(
            message="Database operation failed",
            error_code='DB_OPERATION_FAILED',
            details={
                'operation': operation.__name__ if hasattr(operation, '__name__') else str(operation),
                'error': str(e)
            }
        )

class ErrorMetrics:
    """Track error metrics for monitoring and debugging"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
    
    def record_error(self, error_code: str, error_type: str):
        """Record an error occurrence"""
        key = f"{error_code}:{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        self.error_history.append({
            'error_code': error_code,
            'error_type': error_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
    
    def get_error_summary(self) -> Dict:
        """Get summary of error metrics"""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts,
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }

# Global error metrics instance
error_metrics = ErrorMetrics()

# Update error handling to use metrics
original_log_error = log_error

def log_error_with_metrics(error: Exception, context: Dict = None) -> None:
    """Enhanced log_error that also tracks metrics"""
    original_log_error(error, context)
    
    error_code = getattr(error, 'error_code', 'UNKNOWN_ERROR')
    error_type = type(error).__name__
    error_metrics.record_error(error_code, error_type)

# Replace the original log_error function
log_error = log_error_with_metrics