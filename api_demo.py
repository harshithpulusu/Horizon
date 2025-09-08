#!/usr/bin/env python3
"""
🚀 Enhanced API Integration Demo for Horizon AI Assistant
Shows the power of multiple API integrations working together
"""

import requests
import json
from config import Config

class EnhancedAPIController:
    """Demonstrates powerful API integrations"""
    
    def __init__(self):
        self.available_apis = self.check_available_apis()
        print(f"🚀 Available APIs: {len(self.available_apis)}")
        for api in self.available_apis:
            print(f"  ✅ {api}")
    
    def check_available_apis(self):
        """Check which APIs are configured"""
        apis = []
        
        if Config.OPENAI_API_KEY:
            apis.append("OpenAI (GPT + DALL-E)")
        if Config.OPENWEATHERMAP_API_KEY:
            apis.append("Weather")
        if Config.SPOTIFY_CLIENT_ID:
            apis.append("Spotify Music")
        if Config.ELEVENLABS_API_KEY:
            apis.append("ElevenLabs Voice")
        if Config.YOUTUBE_API_KEY:
            apis.append("YouTube")
        if Config.WOLFRAM_ALPHA_API_KEY:
            apis.append("Wolfram Alpha")
        if Config.NEWS_API_KEY:
            apis.append("News")
        if Config.GOOGLE_MAPS_API_KEY:
            apis.append("Google Maps")
            
        return apis
    
    def get_weather(self, city="New York"):
        """Enhanced weather with multiple data points"""
        if not Config.OPENWEATHERMAP_API_KEY:
            return "❌ OpenWeatherMap API key not configured"
        
        try:
            # Current weather
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": city,
                "appid": Config.OPENWEATHERMAP_API_KEY,
                "units": "metric"
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                temp = data['main']['temp']
                feels_like = data['main']['feels_like']
                humidity = data['main']['humidity']
                description = data['weather'][0]['description']
                wind_speed = data['wind']['speed']
                
                weather_report = f"""🌤️ Weather in {city}:
🌡️ Temperature: {temp}°C (feels like {feels_like}°C)
☁️ Conditions: {description.title()}
💧 Humidity: {humidity}%
💨 Wind Speed: {wind_speed} m/s"""
                
                return weather_report
            else:
                return f"❌ Weather data not found for {city}"
                
        except Exception as e:
            return f"❌ Weather API error: {str(e)}"
    
    def get_news(self, topic="technology", count=3):
        """Get latest news on any topic"""
        if not Config.NEWS_API_KEY:
            return "❌ News API key not configured"
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": topic,
                "apiKey": Config.NEWS_API_KEY,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": count
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                articles = data['articles']
                news_summary = f"📰 Latest {topic} news:\n\n"
                
                for i, article in enumerate(articles, 1):
                    title = article['title']
                    source = article['source']['name']
                    news_summary += f"{i}. **{title}**\n   Source: {source}\n\n"
                
                return news_summary
            else:
                return f"❌ News API error: {data.get('message', 'Unknown error')}"
                
        except Exception as e:
            return f"❌ News API error: {str(e)}"
    
    def search_youtube(self, query, max_results=3):
        """Search YouTube videos"""
        if not Config.YOUTUBE_API_KEY:
            return "❌ YouTube API key not configured"
        
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "key": Config.YOUTUBE_API_KEY,
                "type": "video",
                "maxResults": max_results
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                videos = data['items']
                results = f"🎬 YouTube search for '{query}':\n\n"
                
                for i, video in enumerate(videos, 1):
                    title = video['snippet']['title']
                    channel = video['snippet']['channelTitle']
                    video_id = video['id']['videoId']
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    
                    results += f"{i}. **{title}**\n   Channel: {channel}\n   URL: {url}\n\n"
                
                return results
            else:
                return f"❌ YouTube API error: {data.get('error', {}).get('message', 'Unknown error')}"
                
        except Exception as e:
            return f"❌ YouTube API error: {str(e)}"
    
    def calculate_wolfram(self, query):
        """Wolfram Alpha computational intelligence"""
        if not Config.WOLFRAM_ALPHA_API_KEY:
            return "❌ Wolfram Alpha API key not configured"
        
        try:
            import wolframalpha
            client = wolframalpha.Client(Config.WOLFRAM_ALPHA_API_KEY)
            res = client.query(query)
            
            if res.success:
                answer = next(res.results).text
                return f"🧮 Wolfram Alpha: {query}\n💡 Answer: {answer}"
            else:
                return f"❌ Wolfram Alpha couldn't process: {query}"
                
        except Exception as e:
            return f"❌ Wolfram Alpha error: {str(e)}"
    
    def get_stock_price(self, symbol):
        """Get stock price (using free API)"""
        try:
            url = f"https://api.exchangerate-api.com/v4/latest/USD"  # Free alternative
            response = requests.get(url)
            
            # For demo - you'd use Alpha Vantage or other stock APIs
            return f"📈 Stock APIs require API keys. Available: Alpha Vantage, Polygon, Yahoo Finance"
            
        except Exception as e:
            return f"❌ Stock API error: {str(e)}"
    
    def demo_all_features(self):
        """Demonstrate all available API features"""
        print("🚀 Horizon AI Assistant - API Integration Demo\n")
        
        # Weather demo
        print("1. 🌤️ Weather Information:")
        print(self.get_weather("London"))
        print("\n" + "="*50 + "\n")
        
        # News demo
        print("2. 📰 Latest News:")
        print(self.get_news("artificial intelligence", 2))
        print("\n" + "="*50 + "\n")
        
        # YouTube demo
        print("3. 🎬 YouTube Search:")
        print(self.search_youtube("AI tutorial", 2))
        print("\n" + "="*50 + "\n")
        
        # Calculation demo
        print("4. 🧮 Computational Intelligence:")
        print(self.calculate_wolfram("solve x^2 + 5x + 6 = 0"))
        print("\n" + "="*50 + "\n")
        
        # Integration suggestions
        print("5. 🔮 Available Integrations:")
        suggestions = [
            "🎵 Spotify - 'Play jazz music', 'Skip song'",
            "🗣️ ElevenLabs - Ultra-realistic voice synthesis",
            "🏠 Smart Home - Control Philips Hue lights",
            "🚗 Transportation - Book Uber/Lyft rides",
            "💰 Finance - Real-time stock prices",
            "📱 Social Media - Twitter/Instagram integration",
            "🎮 Gaming - Steam library management",
            "📧 Communication - Send emails/SMS via Twilio"
        ]
        
        for suggestion in suggestions:
            print(f"  {suggestion}")

def demonstrate_api_power():
    """Show the power of API integrations"""
    controller = EnhancedAPIController()
    
    print("""
🎯 Voice Commands You Can Add:

🌤️ WEATHER:
- "What's the weather in Tokyo?"
- "Will it rain tomorrow?"
- "Check temperature in London"

📰 NEWS:
- "Show me tech news"
- "Latest sports updates"
- "AI news today"

🎬 YOUTUBE:
- "Find cooking tutorials"
- "Search for cat videos"
- "Show me Python programming videos"

🧮 CALCULATIONS:
- "Solve 2x + 5 = 15"
- "What's the derivative of x squared?"
- "Convert 100 USD to EUR"

🎵 MUSIC (with Spotify):
- "Play some jazz"
- "Skip this song"
- "What's playing?"

🗣️ VOICE (with ElevenLabs):
- Ultra-realistic text-to-speech
- Voice cloning capabilities
- Emotion control

Ready to transform your AI assistant? Add API keys to your .env file!
    """)
    
    # Run the demo
    controller.demo_all_features()

if __name__ == "__main__":
    demonstrate_api_power()
