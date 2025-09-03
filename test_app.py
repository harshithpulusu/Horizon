#!/usr/bin/env python3
"""
Simple test version of Horizon to debug issues
"""

from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Template error: {str(e)}"

@app.route('/test')
def test():
    return "Test route works!"

if __name__ == '__main__':
    print("🌐 Starting test server on http://127.0.0.1:4000...")
    app.run(host='127.0.0.1', port=4000, debug=True)
