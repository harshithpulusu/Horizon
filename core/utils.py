"""
Horizon Core Utils Module

This module contains shared utilities and helper functions
for the Horizon AI Assistant core modules.

Functions:
- setup_logging: Configure logging for core modules
- validate_config: Validate configuration settings
- format_response: Format AI responses consistently
- sanitize_input: Sanitize user input
- generate_unique_id: Generate unique identifiers
- measure_time: Measure execution time
"""

import os
import re
import uuid
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from functools import wraps
from config import Config

# Logging configuration
LOG_LEVEL = getattr(Config, 'LOG_LEVEL', 'INFO')
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Input sanitization patterns
UNSAFE_PATTERNS = [
    r'<script.*?>.*?</script>',
    r'javascript:',
    r'on\w+\s*=',
    r'<.*?>',  # Basic HTML tag removal
]

# Response formatting settings
MAX_RESPONSE_LENGTH = 1000
DEFAULT_TRUNCATION_SUFFIX = "..."


class CoreLogger:
    """Centralized logging for core modules."""
    
    def __init__(self, name: str = "HorizonCore"):
        """Initialize logger."""
        self.logger = logging.getLogger(name)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Avoid duplicate handlers
        if self.logger.handlers:
            return
        
        # Set log level
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        try:
            log_dir = getattr(Config, 'LOG_DIR', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(
                os.path.join(log_dir, 'horizon_core.log')
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.warning(f"Could not setup file logging: {e}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_api_keys() -> Dict[str, bool]:
        """Validate API key configurations."""
        api_keys = {
            'openai': bool(os.getenv('OPENAI_API_KEY') or getattr(Config, 'OPENAI_API_KEY', None)),
            'gemini': bool(os.getenv('GEMINI_API_KEY') or getattr(Config, 'GEMINI_API_KEY', None)),
            'google_cloud_project': bool(getattr(Config, 'GOOGLE_CLOUD_PROJECT', None)),
            'replicate': bool(os.getenv('REPLICATE_API_TOKEN') or getattr(Config, 'REPLICATE_API_TOKEN', None))
        }
        
        return api_keys
    
    @staticmethod
    def validate_directories() -> Dict[str, bool]:
        """Validate required directories exist."""
        directories = {
            'static': os.path.exists('static'),
            'templates': os.path.exists('templates'),
            'core': os.path.exists('core'),
            'logs': True  # Will be created if needed
        }
        
        # Check media directories
        media_dirs = [
            'static/generated_images',
            'static/generated_videos',
            'static/generated_audio',
            'static/generated_music',
            'static/generated_3d_models'
        ]
        
        for media_dir in media_dirs:
            directories[media_dir] = os.path.exists(media_dir)
        
        return directories
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information for diagnostics."""
        info = {
            'python_version': None,
            'platform': None,
            'memory_available': None,
            'disk_space': None
        }
        
        try:
            import sys
            import platform
            import psutil
            
            info['python_version'] = sys.version
            info['platform'] = platform.platform()
            info['memory_available'] = psutil.virtual_memory().available
            info['disk_space'] = psutil.disk_usage('.').free
        except ImportError:
            pass
        except Exception:
            pass
        
        return info


class InputSanitizer:
    """Input sanitization and validation."""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000) -> str:
        """Sanitize text input."""
        if not isinstance(text, str):
            return ""
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove unsafe patterns
        for pattern in UNSAFE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """Validate user ID format."""
        if not isinstance(user_id, str):
            return False
        
        # Check if it's a valid UUID or alphanumeric string
        try:
            uuid.UUID(user_id)
            return True
        except ValueError:
            # Allow alphanumeric strings with limited length
            return bool(re.match(r'^[a-zA-Z0-9_-]{1,50}$', user_id))
    
    @staticmethod
    def validate_personality(personality: str) -> bool:
        """Validate personality type."""
        valid_personalities = [
            'friendly', 'professional', 'casual', 'enthusiastic',
            'witty', 'sarcastic', 'zen', 'scientist', 'pirate',
            'shakespearean', 'valley_girl', 'cowboy', 'robot'
        ]
        
        return personality in valid_personalities


class ResponseFormatter:
    """Response formatting utilities."""
    
    @staticmethod
    def format_ai_response(response: str, personality: str = 'friendly',
                          max_length: int = None) -> str:
        """Format AI response consistently."""
        if not response:
            return ""
        
        # Apply max length if specified
        if max_length and len(response) > max_length:
            response = response[:max_length - len(DEFAULT_TRUNCATION_SUFFIX)] + DEFAULT_TRUNCATION_SUFFIX
        
        # Clean up formatting
        response = response.strip()
        
        # Remove excessive line breaks
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        return response
    
    @staticmethod
    def format_error_response(error_message: str, personality: str = 'friendly') -> str:
        """Format error responses in a user-friendly way."""
        personality_prefixes = {
            'friendly': "I'm sorry, but ",
            'professional': "I apologize, but ",
            'casual': "Oops, ",
            'enthusiastic': "Oh no! ",
            'witty': "Well, this is awkward... ",
            'sarcastic': "Fantastic... ",
            'zen': "In the spirit of acceptance, ",
            'scientist': "According to error analysis, ",
            'pirate': "Shiver me timbers! ",
            'shakespearean': "Alas! ",
            'valley_girl': "OMG, like, ",
            'cowboy': "Well, partner, ",
            'robot': "ERROR: "
        }
        
        prefix = personality_prefixes.get(personality, "I'm sorry, but ")
        
        # Generic error message without technical details
        generic_message = "I encountered an issue and couldn't complete that request. Please try again or rephrase your question."
        
        return prefix + generic_message
    
    @staticmethod
    def extract_key_points(text: str, max_points: int = 3) -> List[str]:
        """Extract key points from text."""
        if not text:
            return []
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Return first few sentences as key points
        return sentences[:max_points]


class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    @staticmethod
    def measure_time(func: Callable) -> Callable:
        """Decorator to measure function execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            logger = CoreLogger(f"Performance.{func.__name__}")
            logger.debug(f"Execution time: {execution_time:.4f} seconds")
            
            return result
        return wrapper
    
    @staticmethod
    def profile_memory_usage():
        """Get current memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss,  # Resident Set Size
                'vms': memory_info.vms,  # Virtual Memory Size
                'memory_percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}


class DataProcessor:
    """Data processing utilities."""
    
    @staticmethod
    def serialize_for_storage(data: Any) -> str:
        """Serialize data for database storage."""
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except Exception:
            return "{}"
    
    @staticmethod
    def deserialize_from_storage(data_str: str) -> Any:
        """Deserialize data from database storage."""
        try:
            return json.loads(data_str) if data_str else {}
        except Exception:
            return {}
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        if not text:
            return []
        
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this',
            'that', 'these', 'those', 'was', 'were', 'been', 'have', 'has',
            'had', 'will', 'would', 'should', 'could', 'can', 'may', 'might'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_keywords[:max_keywords]]


# Global instances
core_logger = CoreLogger()
config_validator = ConfigValidator()
input_sanitizer = InputSanitizer()
response_formatter = ResponseFormatter()
performance_monitor = PerformanceMonitor()
data_processor = DataProcessor()

# Convenience functions
def setup_logging(name: str = "HorizonCore") -> CoreLogger:
    """Setup logging for a module."""
    return CoreLogger(name)

def validate_config() -> Dict[str, Any]:
    """Validate overall configuration."""
    return {
        'api_keys': config_validator.validate_api_keys(),
        'directories': config_validator.validate_directories(),
        'system_info': config_validator.get_system_info()
    }

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input."""
    return input_sanitizer.sanitize_text(text, max_length)

def format_response(response: str, personality: str = 'friendly',
                   max_length: int = None) -> str:
    """Format AI response."""
    return response_formatter.format_ai_response(response, personality, max_length)

def generate_unique_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())

def measure_time(func: Callable) -> Callable:
    """Measure function execution time."""
    return performance_monitor.measure_time(func)

def log_info(message: str):
    """Log info message."""
    core_logger.info(message)

def log_warning(message: str):
    """Log warning message."""
    core_logger.warning(message)

def log_error(message: str):
    """Log error message."""
    core_logger.error(message)

def log_debug(message: str):
    """Log debug message."""
    core_logger.debug(message)