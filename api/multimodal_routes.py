"""
Multimodal API Routes - Separate from existing chat functionality
Handles image uploads, context management, and enhanced chat without touching core /chat endpoint
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import base64
import uuid
from datetime import datetime
import json
from typing import Dict, Any, List, Optional

# Create blueprint for multimodal functionality
multimodal_bp = Blueprint('multimodal', __name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_image_content(filename: str, file_size: int) -> Dict[str, Any]:
    """
    Basic image analysis based on filename and metadata.
    Can be enhanced with AI vision models later.
    """
    analysis = {
        'filename': filename,
        'size_kb': round(file_size / 1024, 1),
        'analysis_timestamp': datetime.now().isoformat(),
        'content_type': 'image'
    }
    
    # Simple content detection based on filename
    filename_lower = filename.lower()
    
    if any(keyword in filename_lower for keyword in ['chart', 'graph', 'plot', 'data']):
        analysis.update({
            'category': 'data_visualization',
            'description': 'This appears to be a data visualization, chart, or graph',
            'suggested_analysis': 'I can help analyze the data trends, patterns, or insights shown in this visualization.'
        })
    elif any(keyword in filename_lower for keyword in ['screenshot', 'screen', 'capture']):
        analysis.update({
            'category': 'screenshot',
            'description': 'This appears to be a screenshot or screen capture',
            'suggested_analysis': 'I can help explain what\'s shown in this screenshot or assist with any questions about it.'
        })
    elif any(keyword in filename_lower for keyword in ['code', 'programming', 'script']):
        analysis.update({
            'category': 'code_image',
            'description': 'This appears to be an image of code or programming content',
            'suggested_analysis': 'I can help review the code, explain functionality, or suggest improvements.'
        })
    elif any(keyword in filename_lower for keyword in ['diagram', 'flowchart', 'architecture']):
        analysis.update({
            'category': 'diagram',
            'description': 'This appears to be a diagram, flowchart, or architectural drawing',
            'suggested_analysis': 'I can help explain the flow, structure, or relationships shown in this diagram.'
        })
    else:
        analysis.update({
            'category': 'general_image',
            'description': 'General image content',
            'suggested_analysis': 'I can describe what I see in this image and answer questions about it.'
        })
    
    return analysis

@multimodal_bp.route('/upload', methods=['POST'])
def upload_images():
    """
    Handle image uploads for multimodal chat.
    Separate endpoint from existing /chat functionality.
    """
    try:
        # Check if files are present
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No images selected'}), 400
        
        # Process uploaded images
        uploaded_images = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Secure filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                
                # Check file size
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > MAX_FILE_SIZE:
                    return jsonify({
                        'error': f'File {filename} is too large. Maximum size is 10MB.'
                    }), 400
                
                # Create upload directory if it doesn't exist
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                
                # Save file
                file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(file_path)
                
                # Analyze image content
                analysis = analyze_image_content(filename, file_size)
                
                # Add file info
                image_info = {
                    'id': str(uuid.uuid4()),
                    'original_filename': filename,
                    'stored_filename': unique_filename,
                    'file_path': file_path,
                    'file_size': file_size,
                    'upload_timestamp': datetime.now().isoformat(),
                    'analysis': analysis
                }
                
                uploaded_images.append(image_info)
        
        return jsonify({
            'success': True,
            'images_uploaded': len(uploaded_images),
            'images': uploaded_images,
            'message': f'Successfully uploaded {len(uploaded_images)} image(s)'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Upload failed: {str(e)}',
            'success': False
        }), 500

@multimodal_bp.route('/analyze-context', methods=['POST'])
def analyze_conversation_context():
    """
    Analyze conversation context for enhanced responses.
    Works alongside existing chat without replacing it.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        conversation_history = data.get('conversation_history', [])
        current_message = data.get('current_message', '')
        
        # Basic context analysis
        context_analysis = {
            'total_messages': len(conversation_history),
            'conversation_length': sum(len(msg.get('content', '')) for msg in conversation_history),
            'analysis_timestamp': datetime.now().isoformat(),
            'context_available': len(conversation_history) > 0
        }
        
        # Analyze conversation patterns
        if conversation_history:
            recent_topics = []
            question_count = 0
            
            for msg in conversation_history[-5:]:  # Last 5 messages
                content = msg.get('content', '').lower()
                if '?' in content:
                    question_count += 1
                
                # Simple topic extraction (can be enhanced with NLP)
                if any(word in content for word in ['image', 'picture', 'photo']):
                    recent_topics.append('images')
                if any(word in content for word in ['time', 'timer', 'remind']):
                    recent_topics.append('time_management')
                if any(word in content for word in ['code', 'program', 'function']):
                    recent_topics.append('programming')
            
            context_analysis.update({
                'recent_topics': list(set(recent_topics)),
                'question_count': question_count,
                'conversation_active': len(conversation_history) >= 3,
                'context_summary': f"Conversation with {len(conversation_history)} messages covering topics: {', '.join(set(recent_topics)) if recent_topics else 'general chat'}"
            })
        
        return jsonify({
            'success': True,
            'context_analysis': context_analysis
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Context analysis failed: {str(e)}',
            'success': False
        }), 500

@multimodal_bp.route('/enhanced-chat', methods=['POST'])
def enhanced_chat():
    """
    Enhanced chat endpoint that combines text, images, and context.
    Optional endpoint that doesn't replace existing /chat functionality.
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data.get('message', '').strip()
        images = data.get('images', [])
        context = data.get('context', {})
        personality = data.get('personality', 'friendly')
        
        if not message:
            return jsonify({'error': 'Empty message provided'}), 400
        
        # Build enhanced prompt
        enhanced_prompt = message
        
        # Add image context if available
        if images:
            enhanced_prompt += f"\n\n[Images provided: {len(images)} image(s)]"
            for i, img in enumerate(images):
                analysis = img.get('analysis', {})
                enhanced_prompt += f"\nImage {i+1}: {analysis.get('description', 'Image uploaded')} ({analysis.get('category', 'general')})"
        
        # Add conversation context if available
        if context.get('context_available'):
            context_summary = context.get('context_summary', '')
            if context_summary:
                enhanced_prompt += f"\n\n[Context: {context_summary}]"
        
        # Prepare response data
        response_data = {
            'success': True,
            'enhanced_prompt': enhanced_prompt,
            'original_message': message,
            'multimodal_info': {
                'has_images': len(images) > 0,
                'image_count': len(images),
                'has_context': context.get('context_available', False),
                'personality': personality
            },
            'processing_timestamp': datetime.now().isoformat(),
            'ready_for_ai': True,
            'message': 'Enhanced prompt prepared for AI processing'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': f'Enhanced chat processing failed: {str(e)}',
            'success': False
        }), 500

@multimodal_bp.route('/health', methods=['GET'])
def multimodal_health():
    """Health check for multimodal API endpoints."""
    return jsonify({
        'status': 'healthy',
        'service': 'multimodal_api',
        'endpoints': [
            '/upload - Handle image uploads',
            '/analyze-context - Analyze conversation context',
            '/enhanced-chat - Process multimodal messages',
            '/health - Service health check'
        ],
        'timestamp': datetime.now().isoformat()
    })