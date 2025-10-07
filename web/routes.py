"""
Horizon Web Routes

This module contains all the web routes for the Horizon AI Assistant.
All routes use the shared core module for business logic.
"""

import os
import uuid
import json
from datetime import datetime
from flask import (
    request, jsonify, render_template, session, 
    redirect, url_for, flash, send_file
)

# Import core modules
from core import (
    ask_ai_model, get_personality_engine, get_database_manager,
    get_memory_system, get_media_engine, get_user_memory, get_ai_engine,
    store_user_memory, generate_image, generate_video, 
    generate_audio, sanitize_input, log_info, log_error
)

def register_routes(app):
    """Register all routes with the Flask application."""
    
    @app.route('/')
    def index():
        """Main page route."""
        try:
            # Initialize session if needed
            if 'user_id' not in session:
                session['user_id'] = str(uuid.uuid4())
                session['conversation_id'] = str(uuid.uuid4())
                session.permanent = True
            
            # Get user's preferred personality from memory
            user_memory = get_user_memory(session['user_id'])
            profile = user_memory.get_profile_summary()
            
            # Default personality or use remembered preference
            default_personality = 'friendly'
            if profile['preferences']:
                for pref in profile['preferences']:
                    if 'personality' in pref.lower():
                        # Extract personality from preference text
                        personality_engine = get_personality_engine()
                        available_personalities = personality_engine.get_available_personalities()
                        for personality in available_personalities:
                            if personality in pref.lower():
                                default_personality = personality
                                break
            
            log_info(f"User {session['user_id'][:8]} accessed main page")
            
            return render_template('index.html', 
                                 current_personality=default_personality,
                                 user_profile=profile)
        
        except Exception as e:
            log_error(f"Error in index route: {e}")
            return render_template('index.html', 
                                 current_personality='friendly',
                                 user_profile={})
    
    @app.route('/chat', methods=['POST'])
    def chat():
        """Handle chat messages."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            user_message = data.get('message', '').strip()
            personality = data.get('personality', 'friendly')
            
            if not user_message:
                return jsonify({'error': 'Empty message'}), 400
            
            # Sanitize input
            user_message = sanitize_input(user_message, max_length=1000)
            
            # Get user session info
            user_id = session.get('user_id', 'anonymous')
            conversation_id = session.get('conversation_id')
            
            # Generate AI response using core module
            ai_response, context_used = ask_ai_model(
                user_message, personality, conversation_id, user_id
            )
            
            # Store conversation in database using core module
            try:
                db_manager = get_database_manager()
                conversation_manager = db_manager.get_conversation_manager()
                
                if conversation_id:
                    # Add user message
                    conversation_manager.add_message(
                        conversation_id, user_id, 'user', user_message, personality
                    )
                    
                    # Add AI response
                    conversation_manager.add_message(
                        conversation_id, user_id, 'assistant', ai_response, personality
                    )
            except Exception as e:
                log_error(f"Error saving conversation: {e}")
            
            # Learn from interaction using memory system
            try:
                memory_system = get_memory_system()
                learning_engine = memory_system.get_learning_engine()
                
                interaction_data = {
                    'user_message': user_message,
                    'ai_response': ai_response,
                    'personality': personality,
                    'context_helpful': context_used,
                    'satisfaction': 0.8  # Default positive feedback
                }
                
                learning_engine.learn_from_interaction(user_id, interaction_data)
            except Exception as e:
                log_error(f"Error in learning system: {e}")
            
            log_info(f"Chat response generated for user {user_id[:8]}")
            
            return jsonify({
                'response': ai_response,
                'personality': personality,
                'context_used': context_used,
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as e:
            log_error(f"Error in chat route: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.route('/generate_image', methods=['POST'])
    def generate_image_route():
        """Handle image generation requests."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            prompt = data.get('prompt', '').strip()
            if not prompt:
                return jsonify({'error': 'Empty prompt'}), 400
            
            # Sanitize prompt
            prompt = sanitize_input(prompt, max_length=500)
            
            # Generate image using core module
            result = generate_image(prompt, data.get('params', {}))
            
            # Store generation history
            try:
                user_id = session.get('user_id', 'anonymous')
                db_manager = get_database_manager()
                
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO media_history (id, user_id, media_type, prompt, 
                                                 file_path, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (str(uuid.uuid4()), user_id, 'image', prompt,
                          result.get('filepath', ''), json.dumps(result)))
                    conn.commit()
            except Exception as e:
                log_error(f"Error saving media history: {e}")
            
            log_info(f"Image generated for user {session.get('user_id', 'anonymous')[:8]}")
            
            return jsonify(result)
        
        except Exception as e:
            log_error(f"Error in image generation route: {e}")
            return jsonify({'error': 'Image generation failed'}), 500
    
    @app.route('/generate_video', methods=['POST'])
    def generate_video_route():
        """Handle video generation requests."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            prompt = data.get('prompt', '').strip()
            if not prompt:
                return jsonify({'error': 'Empty prompt'}), 400
            
            prompt = sanitize_input(prompt, max_length=500)
            result = generate_video(prompt, data.get('params', {}))
            
            # Store generation history
            try:
                user_id = session.get('user_id', 'anonymous')
                db_manager = get_database_manager()
                
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO media_history (id, user_id, media_type, prompt, 
                                                 file_path, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (str(uuid.uuid4()), user_id, 'video', prompt,
                          result.get('filepath', ''), json.dumps(result)))
                    conn.commit()
            except Exception as e:
                log_error(f"Error saving media history: {e}")
            
            log_info(f"Video generated for user {session.get('user_id', 'anonymous')[:8]}")
            
            return jsonify(result)
        
        except Exception as e:
            log_error(f"Error in video generation route: {e}")
            return jsonify({'error': 'Video generation failed'}), 500
    
    @app.route('/generate_audio', methods=['POST'])
    def generate_audio_route():
        """Handle audio generation requests."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            prompt = data.get('prompt', '').strip()
            if not prompt:
                return jsonify({'error': 'Empty prompt'}), 400
            
            prompt = sanitize_input(prompt, max_length=500)
            result = generate_audio(prompt, data.get('params', {}))
            
            # Store generation history
            try:
                user_id = session.get('user_id', 'anonymous')
                db_manager = get_database_manager()
                
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO media_history (id, user_id, media_type, prompt, 
                                                 file_path, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (str(uuid.uuid4()), user_id, 'audio', prompt,
                          result.get('filepath', ''), json.dumps(result)))
                    conn.commit()
            except Exception as e:
                log_error(f"Error saving media history: {e}")
            
            log_info(f"Audio generated for user {session.get('user_id', 'anonymous')[:8]}")
            
            return jsonify(result)
        
        except Exception as e:
            log_error(f"Error in audio generation route: {e}")
            return jsonify({'error': 'Audio generation failed'}), 500
    
    @app.route('/personality/<personality_type>')
    def set_personality(personality_type):
        """Set user's preferred personality."""
        try:
            # Validate personality type
            personality_engine = get_personality_engine()
            available_personalities = personality_engine.get_available_personalities()
            
            if personality_type not in available_personalities:
                flash(f'Invalid personality type: {personality_type}', 'error')
                return redirect(url_for('index'))
            
            # Store preference in memory
            user_id = session.get('user_id', 'anonymous')
            store_user_memory(
                user_id, 'preferences', 
                f"Prefers {personality_type} personality",
                f"Set via web interface on {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            flash(f'Personality set to {personality_type}', 'success')
            log_info(f"User {user_id[:8]} set personality to {personality_type}")
            
            return redirect(url_for('index'))
        
        except Exception as e:
            log_error(f"Error setting personality: {e}")
            flash('Error setting personality', 'error')
            return redirect(url_for('index'))
    
    @app.route('/history')
    def conversation_history():
        """Show conversation history."""
        try:
            user_id = session.get('user_id', 'anonymous')
            conversation_id = session.get('conversation_id')
            
            if not conversation_id:
                return render_template('history.html', messages=[])
            
            # Get conversation history from database
            db_manager = get_database_manager()
            conversation_manager = db_manager.get_conversation_manager()
            
            messages = conversation_manager.get_conversation_history(conversation_id, limit=100)
            
            # Reverse to show oldest first
            messages.reverse()
            
            log_info(f"History accessed by user {user_id[:8]}")
            
            return render_template('history.html', messages=messages)
        
        except Exception as e:
            log_error(f"Error loading history: {e}")
            return render_template('history.html', messages=[], error=str(e))
    
    @app.route('/profile')
    def user_profile():
        """Show user profile and memories."""
        try:
            user_id = session.get('user_id', 'anonymous')
            
            # Get user memory profile
            user_memory = get_user_memory(user_id)
            profile = user_memory.get_profile_summary()
            
            # Get database stats
            db_manager = get_database_manager()
            stats = db_manager.get_database_stats()
            
            log_info(f"Profile accessed by user {user_id[:8]}")
            
            return render_template('profile.html', 
                                 profile=profile, 
                                 user_id=user_id[:8],
                                 stats=stats)
        
        except Exception as e:
            log_error(f"Error loading profile: {e}")
            return render_template('profile.html', 
                                 profile={}, 
                                 user_id='unknown',
                                 error=str(e))
    
    @app.route('/api/status')
    def api_status():
        """API endpoint for system status."""
        try:
            # Get system status from core modules
            ai_engine = get_ai_engine()
            db_manager = get_database_manager()
            media_engine = get_media_engine()
            
            status = {
                'status': 'operational',
                'timestamp': datetime.now().isoformat(),
                'core_modules': {
                    'ai_engine': ai_engine.ai_model_available,
                    'database': True,  # If we get here, DB is working
                    'media_generation': len(media_engine.get_available_generators()) > 0
                },
                'database_stats': db_manager.get_database_stats(),
                'available_generators': media_engine.get_available_generators()
            }
            
            return jsonify(status)
        
        except Exception as e:
            log_error(f"Error getting status: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @app.route('/new_conversation')
    def new_conversation():
        """Start a new conversation."""
        try:
            # Generate new conversation ID
            session['conversation_id'] = str(uuid.uuid4())
            
            user_id = session.get('user_id', 'anonymous')
            log_info(f"New conversation started by user {user_id[:8]}")
            
            flash('New conversation started', 'info')
            return redirect(url_for('index'))
        
        except Exception as e:
            log_error(f"Error starting new conversation: {e}")
            flash('Error starting new conversation', 'error')
            return redirect(url_for('index'))
    
    return app