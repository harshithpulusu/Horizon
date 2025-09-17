# Configuration for Horizon AI Assistant
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Core AI APIs
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyAFl8KrjvNuafGJc03CZqr8Cqzdki9z2AA')
    
    # Google Cloud Configuration for Imagen
    GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT', 'horizon-ai-project')
    GOOGLE_CLOUD_REGION = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
    
    # 🎵 Music & Audio APIs
    SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', '')
    SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', '')
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '')
    SUNO_API_KEY = os.getenv('SUNO_API_KEY', '')
    MUSICGEN_API_KEY = os.getenv('MUSICGEN_API_KEY', '')
    
    # 🎼 Professional Music Generation APIs
    REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN', '')
    STABILITY_API_KEY = os.getenv('STABILITY_API_KEY', '')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
    
    # 🌤️ Weather & Location APIs
    OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', '')
    WEATHERAPI_KEY = os.getenv('WEATHERAPI_KEY', '')
    MAPBOX_API_KEY = os.getenv('MAPBOX_API_KEY', '')
    
    # 📰 News & Information APIs
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    NEWSDATA_API_KEY = os.getenv('NEWSDATA_API_KEY', '')
    
    # 🏠 Smart Home & IoT APIs
    PHILIPS_HUE_API_KEY = os.getenv('PHILIPS_HUE_API_KEY', '')
    NEST_API_KEY = os.getenv('NEST_API_KEY', '')
    SMARTTHINGS_API_KEY = os.getenv('SMARTTHINGS_API_KEY', '')
    
    # 📈 Finance & Crypto APIs
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    COINMARKETCAP_API_KEY = os.getenv('COINMARKETCAP_API_KEY', '')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
    
    # 🛒 E-commerce & Shopping APIs
    AMAZON_API_KEY = os.getenv('AMAZON_API_KEY', '')
    EBAY_API_KEY = os.getenv('EBAY_API_KEY', '')
    
    # 🚗 Transportation APIs
    UBER_API_KEY = os.getenv('UBER_API_KEY', '')
    LYFT_API_KEY = os.getenv('LYFT_API_KEY', '')
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')
    
    # 📱 Social Media APIs
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
    INSTAGRAM_API_KEY = os.getenv('INSTAGRAM_API_KEY', '')
    
    # 🎮 Gaming & Entertainment APIs
    STEAM_API_KEY = os.getenv('STEAM_API_KEY', '')
    TWITCH_API_KEY = os.getenv('TWITCH_API_KEY', '')
    DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN', '')
    
    # 🧠 Advanced AI APIs
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')  # Claude AI
    COHERE_API_KEY = os.getenv('COHERE_API_KEY', '')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
    DEEPAI_API_KEY = os.getenv('DEEPAI_API_KEY', '')
    
    # 🎬 Video Generation APIs
    RUNWAY_API_KEY = os.getenv('RUNWAY_API_KEY', '')  # Runway ML for cinematic videos
    PIKA_API_KEY = os.getenv('PIKA_API_KEY', '')  # Pika Labs alternative
    STABLE_VIDEO_API_KEY = os.getenv('STABLE_VIDEO_API_KEY', '')  # Stability AI
    
    # 🎨 Visual AI & Design APIs
    MIDJOURNEY_API_KEY = os.getenv('MIDJOURNEY_API_KEY', '')  # Avatar & design generation
    LEONARDO_AI_API_KEY = os.getenv('LEONARDO_AI_API_KEY', '')  # Character consistency
    REMOVE_BG_API_KEY = os.getenv('REMOVE_BG_API_KEY', '')  # Background removal
    UPSCAYL_API_KEY = os.getenv('UPSCAYL_API_KEY', '')  # Image upscaling
    TRIPO_API_KEY = os.getenv('TRIPO_API_KEY', '')  # 3D model generation
    MESHY_API_KEY = os.getenv('MESHY_API_KEY', '')  # 3D object creation
    LOOKA_API_KEY = os.getenv('LOOKA_API_KEY', '')  # Logo generation
    BRANDMARK_API_KEY = os.getenv('BRANDMARK_API_KEY', '')  # Brand design
    
    # 🗣️ Voice & Speech APIs
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '')  # Premium TTS
    AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY', '')
    GOOGLE_SPEECH_API_KEY = os.getenv('GOOGLE_SPEECH_API_KEY', '')
    
    # 📚 Knowledge & Search APIs
    WOLFRAM_ALPHA_API_KEY = os.getenv('WOLFRAM_ALPHA_API_KEY', '')
    GOOGLE_SEARCH_API_KEY = os.getenv('GOOGLE_SEARCH_API_KEY', '')
    BING_SEARCH_API_KEY = os.getenv('BING_SEARCH_API_KEY', '')
    
    # 🔍 Computer Vision APIs
    GOOGLE_VISION_API_KEY = os.getenv('GOOGLE_VISION_API_KEY', '')
    AZURE_VISION_API_KEY = os.getenv('AZURE_VISION_API_KEY', '')
    CLARIFAI_API_KEY = os.getenv('CLARIFAI_API_KEY', '')
    
    # 📧 Communication APIs
    SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY', '')
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
    
    # 📊 Analytics & Monitoring APIs
    MIXPANEL_API_KEY = os.getenv('MIXPANEL_API_KEY', '')
    AMPLITUDE_API_KEY = os.getenv('AMPLITUDE_API_KEY', '')

    # Wake Word Detection
    WAKE_WORDS = ["hey horizon", "horizon", "hey assistant", "assistant"]
    WAKE_WORD_SENSITIVITY = 0.7  # Sensitivity threshold (0.0 - 1.0)

    # Features
    ENABLE_SPEECH_RECOGNITION = True
    ENABLE_TEXT_TO_SPEECH = False
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
