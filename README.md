# 🌟 Horizon AI Assistant

An advanced voice-activated AI assistant built with Python Flask and JavaScript, designed to be as close to Siri/Alexa as possible with local processing capabilities.

## ✨ Features

### 🎤 Voice Interaction
- **Speech Recognition**: Real-time voice input processing
- **Text-to-Speech**: Natural voice responses
- **Wake Word Detection**: "Hey Horizon" always-listening mode 🌟
- **Multi-language Support**: Expandable language system

### 🧠 Advanced AI Capabilities
- **Intent Recognition**: Advanced pattern matching and ML-like scoring
- **Sentiment Analysis**: Emotion detection in user input
- **Context Awareness**: Remembers conversation history
- **Learning System**: Improves responses based on user feedback
- **Multiple Personalities**: Friendly, Professional, Enthusiastic, Witty

### ⚡ Core Skills
- **Time & Date**: Current time, date, timezone information
- **Mathematics**: Calculator with complex expression support
- **Weather**: Real-time weather information with OpenWeatherMap API 🌤️
- **Timers & Reminders**: Scheduled notifications and alerts
- **Smart Home**: Device control simulation
- **Entertainment**: Jokes, trivia, music control
- **Information**: Web search, definitions, translations
- **Email & Calendar**: Basic productivity features

### 💾 Data & Memory
- **SQLite Database**: Persistent conversation storage
- **User Preferences**: Customizable settings
- **Feedback System**: Continuous learning from user interactions
- **Export/Import**: Conversation history management
- **Analytics**: Usage statistics and insights

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modern web browser with microphone support
- Internet connection (for some features)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/harshithpulusu/Horizon.git
cd Horizon
```

2. **Run the startup script:**
```bash
./start.sh
```

3. **Open your browser:**
   - Navigate to `http://localhost:5000`
   - Allow microphone access when prompted
   - Start talking to Horizon!

### Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🎯 Usage Examples

### Voice Commands
- **"What time is it?"** - Get current time
- **"Set a timer for 5 minutes"** - Create countdown timer
- **"Remind me to call mom in 1 hour"** - Schedule reminder
- **"What's 25 times 4?"** - Mathematical calculations
- **"Tell me a joke"** - Entertainment
- **"What's the weather like?"** - Real weather information 🌤️
- **"What's the weather in Tokyo?"** - Weather for specific cities
- **"Turn on the lights"** - Smart home control
- **"Search for Python tutorials"** - Web search

### Wake Word Activation 🌟
Enable "Always Listening Mode" in the app:
- **"Hey Horizon"** - Activate assistant
- **"Horizon"** - Quick activation
- **"Hey Assistant"** - Alternative wake phrase
- **"Assistant"** - Short activation

### Text Input
You can also type commands instead of using voice input.

### Personality Modes
Switch between different AI personalities:
- **Friendly** 😊: Warm and conversational
- **Professional** 👔: Business-like and efficient
- **Enthusiastic** 🚀: Energetic and excited
- **Witty** 😏: Humorous and clever

## 🔧 Configuration

### API Keys Setup

#### 🌤️ Weather API (OpenWeatherMap)
1. **Get your free API key:**
   - Visit [OpenWeatherMap](https://openweathermap.org/api)
   - Sign up for a free account
   - Go to API Keys section
   - Copy your API key

2. **Add to config.py:**
```python
WEATHER_API_KEY = "your-actual-api-key-here"
```

#### Other API Keys
Add your API keys to `config.py`:
```python
OPENAI_API_KEY = "your-openai-api-key"
NEWS_API_KEY = "your-news-api-key"
```

### New Features Configuration
Enable/disable features in `config.py`:
```python
ENABLE_SPEECH_RECOGNITION = True
ENABLE_TEXT_TO_SPEECH = True
ENABLE_LEARNING = True
ENABLE_TIMERS = True

# Wake Word Detection
WAKE_WORDS = ["hey horizon", "horizon", "hey assistant", "assistant"]
WAKE_WORD_SENSITIVITY = 0.7  # Sensitivity threshold (0.0 - 1.0)

# Weather Settings
DEFAULT_WEATHER_LOCATION = "New York"  # Default city for weather
DEFAULT_WEATHER_UNITS = "imperial"     # imperial, metric, kelvin
```

## 📊 API Endpoints

- `POST /api/process` - Process user input
- `GET /api/voice-commands` - Get available commands
- `POST /api/feedback` - Submit feedback
- `GET /api/insights` - Get learning insights
- `GET /api/timers-reminders` - Active timers/reminders
- `POST /api/personality` - Change AI personality
- `GET /api/health` - System health check

## 🏗️ Architecture

```
Horizon AI Assistant
├── Frontend (HTML/CSS/JavaScript)
│   ├── Speech Recognition
│   ├── Text-to-Speech
│   └── Real-time UI Updates
├── Backend (Python Flask)
│   ├── Intent Recognition
│   ├── Sentiment Analysis
│   ├── Skill Processing
│   └── Database Management
└── Data Layer (SQLite)
    ├── Conversations
    ├── User Preferences
    └── Feedback
```

## 🔮 Advanced Features

### Learning System
Horizon learns from user interactions:
- Feedback-based improvement
- Conversation pattern analysis
- Personalized responses
- Adaptive personality

### Context Awareness
- Remembers previous conversations
- Maintains topic continuity
- Reference resolution
- Follow-up question handling

### Extensible Skills
Easy to add new capabilities:
```python
def my_custom_skill(self, user_input, entities, personality):
    # Your custom logic here
    return "Custom response"

# Register the skill
self.skills['custom'] = self.my_custom_skill
```

## 🛠️ Development

### Project Structure
```
Horizon/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── start.sh             # Startup script
├── templates/
│   └── index.html       # Main UI template
├── static/
│   └── enhanced_ai.js   # Frontend JavaScript
└── ai_memory.db         # SQLite database (auto-created)
```

### Adding New Skills
1. Define the skill function in `AdvancedAIProcessor`
2. Add intent patterns to `init_intent_patterns()`
3. Register the skill in `init_skills()`
4. Test with voice or text input

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

**Microphone not working:**
- Check browser permissions
- Ensure HTTPS or localhost
- Try different browsers

**Dependencies failing:**
- Update pip: `pip install --upgrade pip`
- Use virtual environment
- Check Python version (3.8+)

**Database errors:**
- Delete `ai_memory.db` to reset
- Check file permissions
- Ensure SQLite is available

### Debug Mode
Run with debug enabled:
```bash
export FLASK_DEBUG=1
python app.py
```

## 📈 Performance

### Optimization Tips
- Use virtual environment
- Enable browser caching
- Optimize database queries
- Use background processing for long tasks

### Monitoring
- Check `/api/health` endpoint
- Monitor conversation counts
- Track response times
- Review error logs

## 🔒 Security

### Privacy
- All processing happens locally
- No data sent to external servers (except configured APIs)
- Conversation data stored locally
- User control over data retention

### Best Practices
- Use HTTPS in production
- Secure API keys
- Regular database backups
- Input validation and sanitization

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 📱 Browser Compatibility

- ✅ Chrome 60+
- ✅ Firefox 55+
- ✅ Safari 11+
- ✅ Edge 79+
- ❌ Internet Explorer (not supported)

## 🆕 What's New

### Version 2.1.0 - Siri/Alexa Features! 🚀
- **🌟 Wake Word Detection**: "Hey Horizon" always-listening mode
- **🌤️ Real Weather API**: Live weather data from OpenWeatherMap
- **🎯 Enhanced Voice Recognition**: Improved speech processing
- **🔧 Better Error Handling**: Graceful API failures with fallbacks
- **📍 Location-Based Weather**: Weather for any city worldwide
- **⚙️ Advanced Configuration**: More customization options

### Version 2.0.0
- Advanced intent recognition
- Real timer and reminder system
- Enhanced sentiment analysis
- Learning from feedback
- Multiple personalities
- Improved UI/UX
- Better error handling
- Comprehensive API

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and code of conduct.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for AI inspiration
- Flask community for the web framework
- Web Speech API for voice capabilities
- SQLite for local database
- All contributors and testers

## 📞 Support

- 📧 Email: harshithpulusu@example.com
- 🐛 Issues: GitHub Issues page
- 💬 Discussions: GitHub Discussions
- 📚 Documentation: Wiki pages

---

**Made with ❤️ by Harshith Pulusu**

*Horizon AI Assistant - Bringing the future of voice AI to your local machine!*
