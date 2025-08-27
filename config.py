# Configuration for Horizon AI Assistant

# API Keys (add your own)
OPENAI_API_KEY = "your-openai-api-key-here"
NEWS_API_KEY = "your-news-api-key-here"

# Wake Word Detection
WAKE_WORDS = ["hey horizon", "horizon", "hey assistant", "assistant"]
WAKE_WORD_SENSITIVITY = 0.7  # Sensitivity threshold (0.0 - 1.0)

# Features
ENABLE_SPEECH_RECOGNITION = True
ENABLE_TEXT_TO_SPEECH = True
ENABLE_LEARNING = True
ENABLE_TIMERS = True
ENABLE_REMINDERS = True

# Default Settings
DEFAULT_PERSONALITY = "friendly"
MAX_CONVERSATION_HISTORY = 100
AUTO_SAVE_INTERVAL = 300  # seconds

# Voice Settings
VOICE_RATE = 200  # words per minute
VOICE_VOLUME = 0.9

# AI Settings
CONFIDENCE_THRESHOLD = 0.3
SENTIMENT_THRESHOLD = 0.1
LEARNING_RATE = 0.01

# Database Settings
DATABASE_PATH = "ai_memory.db"
BACKUP_INTERVAL = 3600  # seconds
