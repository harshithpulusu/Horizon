# 🌤️ Weather API Setup Guide

## Getting Your Free OpenWeatherMap API Key

### Step 1: Sign Up
1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Click "Sign Up" to create a free account
3. Fill in your details and verify your email

### Step 2: Get Your API Key
1. Log in to your OpenWeatherMap account
2. Go to the [API Keys section](https://home.openweathermap.org/api_keys)
3. Your default API key will be shown
4. Copy this API key (it looks like: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`)

### Step 3: Add to Horizon
1. Open `config.py` in your Horizon project
2. Replace this line:
   ```python
   WEATHER_API_KEY = "your-openweathermap-api-key-here"
   ```
   with:
   ```python
   WEATHER_API_KEY = "b022593a8152ffc60a41f2015b2a0e3f"  # Your actual key
   ```

### Step 4: Test It!
1. Restart your Horizon app
2. Try saying: "What's the weather like?"
3. Or try: "What's the weather in Paris?"

## Free Tier Limits
- **1,000 API calls per day** (more than enough for personal use)
- **60 calls per minute**
- Access to current weather data for any location

## Troubleshooting

**❌ "Invalid API key" error:**
- Make sure you copied the entire key
- Check for extra spaces or characters
- Wait a few minutes after creating the key (can take up to 10 minutes to activate)

**❌ "Location not found" error:**
- Try using full city names: "New York City" instead of "NYC"
- Include country codes: "London, UK" or "Paris, France"

**❌ Still getting mock weather:**
- Double-check your API key in `config.py`
- Restart the Flask app after making changes
- Check the terminal for any error messages

## Example Weather Commands

✅ **Working examples:**
- "What's the weather?"
- "Weather in Tokyo"
- "How's the weather in London, UK?"
- "Tell me the weather forecast"
- "Is it raining in Seattle?"

🌟 **Pro tip:** Enable "Always Listening Mode" and just say "Hey Horizon, what's the weather like?" for hands-free weather updates!

---
Need help? Check the main README.md or create an issue on GitHub!
