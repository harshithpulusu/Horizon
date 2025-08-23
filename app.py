from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import json
from datetime import datetime
import random

app = Flask(__name__)
CORS(app)

class AIProcessor:
    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {}
        self.sentiment_model = self.init_sentiment_model()
        
    def init_sentiment_model(self):
        # Simple sentiment analysis using keywords
        return {
            'positive': ['happy', 'good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'fantastic', 'awesome'],
            'negative': ['sad', 'bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointed', 'frustrated', 'angry']
        }
    
    def analyze_sentiment(self, text):
        words = text.lower().split()
        pos_score = sum(1 for word in words if word in self.sentiment_model['positive'])
        neg_score = sum(1 for word in words if word in self.sentiment_model['negative'])
        
        total_words = len(words)
        if total_words == 0:
            return {'score': 0, 'magnitude': 0, 'label': 'neutral'}
            
        sentiment_score = (pos_score - neg_score) / total_words
        magnitude = abs(sentiment_score)
        
        if sentiment_score > 0.1:
            label = 'positive'
        elif sentiment_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
            
        return {
            'score': sentiment_score,
            'magnitude': magnitude,
            'label': label
        }
    
    def generate_response(self, user_input, personality='friendly'):
        # Add to conversation history
        self.conversation_history.append({
            'input': user_input,
            'timestamp': datetime.now().isoformat(),
            'sentiment': self.analyze_sentiment(user_input)
        })
        
        # Generate response based on input and personality
        response = self.get_contextual_response(user_input, personality)
        
        return {
            'response': response,
            'sentiment_analysis': self.analyze_sentiment(user_input),
            'conversation_count': len(self.conversation_history),
            'confidence': random.uniform(0.7, 0.95)
        }
    
    def get_contextual_response(self, user_input, personality):
        lower_input = user_input.lower()
        
        # Math operations
        if any(op in lower_input for op in ['+', '-', '*', '/', 'calculate', 'math']):
            return self.handle_math(user_input)
        
        # Time queries
        if 'time' in lower_input:
            current_time = datetime.now().strftime('%I:%M %p')
            return f"The current time is {current_time}"
        
        # Greeting responses
        if any(greeting in lower_input for greeting in ['hello', 'hi', 'hey']):
            return self.get_personality_greeting(personality)
        
        # Weather (mock response)
        if 'weather' in lower_input:
            return "I'd love to help with weather, but I need access to weather APIs. It's a beautiful day for coding though!"
        
        # Default contextual response
        return self.get_personality_default(personality)
    
    def handle_math(self, input_text):
        import re
        # Simple math parser
        pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, input_text)
        
        if match:
            num1, operator, num2 = match.groups()
            num1, num2 = float(num1), float(num2)
            
            operations = {
                '+': num1 + num2,
                '-': num1 - num2,
                '*': num1 * num2,
                '/': num1 / num2 if num2 != 0 else 'undefined'
            }
            
            result = operations.get(operator, 'unknown operation')
            return f"{num1} {operator} {num2} = {result}"
        
        return "I can help with basic math! Try something like '5 + 3' or '10 * 2'"
    
    def get_personality_greeting(self, personality):
        greetings = {
            'friendly': "Hello there! I'm so excited to chat with you today! 😊",
            'professional': "Good day. How may I assist you efficiently?",
            'enthusiastic': "WOW! Hello there! This is going to be AMAZING! 🚀",
            'witty': "Well, well, well... look who decided to talk to an AI today! 😏"
        }
        return greetings.get(personality, greetings['friendly'])
    
    def get_personality_default(self, personality):
        defaults = {
            'friendly': "That's really interesting! I love learning about new things from you!",
            'professional': "I understand. How would you like to proceed with this matter?",
            'enthusiastic': "OH MY GOODNESS, that's INCREDIBLE! Tell me MORE!",
            'witty': "Hmm, interesting... *adjusts virtual glasses* Do go on!"
        }
        return defaults.get(personality, defaults['friendly'])

# Initialize AI processor
ai_processor = AIProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_input():
    data = request.json
    user_input = data.get('input', '')
    personality = data.get('personality', 'friendly')
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    result = ai_processor.generate_response(user_input, personality)
    return jsonify(result)

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment_endpoint():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    sentiment = ai_processor.analyze_sentiment(text)
    return jsonify(sentiment)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'conversation_count': len(ai_processor.conversation_history),
        'total_interactions': len(ai_processor.conversation_history),
        'avg_sentiment': np.mean([conv['sentiment']['score'] 
                                for conv in ai_processor.conversation_history]) if ai_processor.conversation_history else 0
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)