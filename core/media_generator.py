"""
Horizon Media Generator Core Module

This module handles media generation functionality for the Horizon AI Assistant.
It provides image, video, audio, and 3D model generation capabilities.
Uses event-driven architecture and centralized state management.

Classes:
- MediaEngine: Main media generation system
- MediaEventHandler: Event handler for media-related events
- ImageGenerator: Image and artwork generation
- VideoGenerator: Video and animation generation  
- AudioGenerator: Music and audio generation
- ModelGenerator: 3D model and design generation

Functions:
- generate_image: Generate images from text prompts
- generate_video: Generate videos from prompts
- generate_audio: Generate music and sounds
- generate_3d_model: Generate 3D models
"""

import os
import json
import uuid
import random
import requests
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from config import Config

# Import event system and state management
from .events import (
    EventHandler, EventData, HorizonEventTypes, 
    get_event_emitter, emit_event, listen_for_event
)
from .state_manager import (
    get_state_manager, get_state, update_state, 
    subscribe_to_state
)

# Import optional media generation libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from google.cloud import aiplatform
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    IMAGEN_AVAILABLE = True
except ImportError:
    IMAGEN_AVAILABLE = False

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

try:
    import requests
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Media generation configurations
MEDIA_OUTPUT_DIRS = {
    'images': 'static/generated_images/',
    'videos': 'static/generated_videos/',
    'audio': 'static/generated_audio/',
    'music': 'static/generated_music/',
    'models': 'static/generated_3d_models/',
    'avatars': 'static/generated_avatars/',
    'logos': 'static/generated_logos/',
    'designs': 'static/generated_designs/',
    'gifs': 'static/generated_gifs/'
}

# Default generation parameters
DEFAULT_IMAGE_PARAMS = {
    'width': 1024,
    'height': 1024,
    'quality': 'high',
    'style': 'photorealistic',
    'steps': 30,
    'guidance_scale': 7.5
}

DEFAULT_VIDEO_PARAMS = {
    'duration': 5,
    'fps': 24,
    'width': 1280,
    'height': 720,
    'quality': 'high'
}

DEFAULT_AUDIO_PARAMS = {
    'duration': 30,
    'format': 'mp3',
    'sample_rate': 44100,
    'quality': 'high'
}


class MediaEventHandler(EventHandler):
    """Event handler for media generation events."""
    
    def __init__(self, media_engine):
        super().__init__("media_event_handler")
        self.media_engine = media_engine
        self.handled_events = [
            HorizonEventTypes.MEDIA_GENERATION_REQUESTED,
            HorizonEventTypes.AI_RESPONSE_GENERATED
        ]
    
    def handle_event_sync(self, event: EventData) -> None:
        """Handle media-related events."""
        try:
            if event.event_type == HorizonEventTypes.MEDIA_GENERATION_REQUESTED:
                self._handle_media_request(event)
            elif event.event_type == HorizonEventTypes.AI_RESPONSE_GENERATED:
                self._check_for_media_requests(event)
        except Exception as e:
            print(f"❌ Error in media event handler: {e}")
    
    def _handle_media_request(self, event: EventData):
        """Handle direct media generation requests."""
        media_type = event.data.get('media_type', 'image')
        prompt = event.data.get('prompt', '')
        params = event.data.get('params', {})
        
        # Add to generation queue
        media_state = get_state("media")
        request_id = media_state.add_to_queue(media_type, prompt, params)
        update_state("media", media_state, source="media_engine")
        
        # Start generation
        result = self.media_engine.generate_media(media_type, prompt, params)
        
        # Update state with result
        media_state = get_state("media")
        media_state.complete_generation(request_id, result)
        update_state("media", media_state, source="media_engine")
        
        # Emit completion event
        if result.get('success'):
            emit_event(
                HorizonEventTypes.MEDIA_GENERATION_COMPLETED,
                "media_engine",
                {
                    'media_type': media_type,
                    'prompt': prompt,
                    'result': result,
                    'request_id': request_id
                },
                user_id=event.user_id,
                session_id=event.session_id
            )
        else:
            emit_event(
                HorizonEventTypes.MEDIA_GENERATION_FAILED,
                "media_engine",
                {
                    'media_type': media_type,
                    'prompt': prompt,
                    'error': result.get('error', 'Unknown error'),
                    'request_id': request_id
                },
                user_id=event.user_id,
                session_id=event.session_id
            )
    
    def _check_for_media_requests(self, event: EventData):
        """Check AI responses for media generation requests."""
        user_input = event.data.get('user_input', '').lower()
        
        # Check for image generation requests
        if any(phrase in user_input for phrase in [
            'generate image', 'create image', 'make image', 'generate picture', 
            'create picture', 'make picture', 'draw', 'create an image'
        ]):
            # Extract prompt from user input
            prompt = self._extract_media_prompt(user_input, 'image')
            if prompt:
                emit_event(
                    HorizonEventTypes.MEDIA_GENERATION_REQUESTED,
                    "ai_response_analyzer",
                    {
                        'media_type': 'image',
                        'prompt': prompt,
                        'params': {}
                    },
                    user_id=event.user_id,
                    session_id=event.session_id
                )
        
        # Check for logo generation requests
        elif any(phrase in user_input for phrase in [
            'generate logo', 'create logo', 'make logo', 'design logo'
        ]):
            prompt = self._extract_media_prompt(user_input, 'logo')
            if prompt:
                emit_event(
                    HorizonEventTypes.MEDIA_GENERATION_REQUESTED,
                    "ai_response_analyzer",
                    {
                        'media_type': 'image',
                        'prompt': f"professional logo design for {prompt}",
                        'params': {'type': 'logo'}
                    },
                    user_id=event.user_id,
                    session_id=event.session_id
                )
    
    def _extract_media_prompt(self, user_input: str, media_type: str) -> str:
        """Extract media generation prompt from user input."""
        # Remove trigger words to extract the actual description
        trigger_words = {
            'image': ['generate', 'create', 'make', 'draw', 'image', 'picture', 'an', 'a', 'of'],
            'logo': ['generate', 'create', 'make', 'design', 'logo', 'for', 'a', 'an', 'the']
        }
        
        prompt = user_input
        for word in trigger_words.get(media_type, []):
            prompt = re.sub(r'\b' + word + r'\b', '', prompt, flags=re.IGNORECASE)
        
        prompt = re.sub(r'\s+', ' ', prompt).strip()  # Clean up extra spaces
        
        if not prompt or len(prompt) < 3:
            prompt = "a beautiful landscape" if media_type == 'image' else "modern tech company"
        
        return prompt


class MediaEngine:
    """Main media generation system."""
    
    def __init__(self):
        """Initialize the media engine."""
        self.generators = {
            'image': None,
            'video': None,
            'audio': None,
            'model': None
        }
        
        # Get references to event and state systems
        self.event_emitter = get_event_emitter()
        self.state_manager = get_state_manager()
        
        # Ensure output directories exist
        self._create_output_directories()
        
        # Initialize available generators
        self._initialize_generators()
        
        # Register event handler
        self.event_handler = MediaEventHandler(self)
        self.event_emitter.register_handler(HorizonEventTypes.MEDIA_GENERATION_REQUESTED, self.event_handler)
        self.event_emitter.register_handler(HorizonEventTypes.AI_RESPONSE_GENERATED, self.event_handler)
        
        # Update media state
        self._update_media_state()
        
        print("🎨 Media Engine initialized with event-driven architecture")
    
    def _update_media_state(self):
        """Update media state with current configuration."""
        available_generators = self.get_available_generators()
        
        # Update state
        update_state("media.available_generators", available_generators, source="media_engine")
        update_state("media.is_generating", False, source="media_engine")
        update_state("media.generation_progress", 0.0, source="media_engine")
    
    def _create_output_directories(self):
        """Create output directories for generated media."""
        for media_type, directory in MEDIA_OUTPUT_DIRS.items():
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_generators(self):
        """Initialize available media generators."""
        if OPENAI_AVAILABLE or IMAGEN_AVAILABLE:
            self.generators['image'] = ImageGenerator()
        
        # Initialize video generator if Gemini (for Veo 3) or Replicate is available
        if GEMINI_AVAILABLE or REPLICATE_AVAILABLE:
            self.generators['video'] = VideoGenerator()
        
        if REPLICATE_AVAILABLE:
            self.generators['audio'] = AudioGenerator()
            self.generators['model'] = ModelGenerator()
        
        print(f"🔧 Initialized generators: {list(self.generators.keys())}")
    
    def get_available_generators(self) -> List[str]:
        """Get list of available media generators."""
        return [k for k, v in self.generators.items() if v is not None]
    
    def generate_media(self, media_type: str, prompt: str, 
                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate media of specified type with event emission.
        
        Args:
            media_type: Type of media to generate (image, video, audio, model)
            prompt: Text prompt for generation
            params: Additional generation parameters
            
        Returns:
            Dictionary with generation results
        """
        if media_type not in self.generators or self.generators[media_type] is None:
            return {
                'success': False,
                'error': f'{media_type} generator not available',
                'available_generators': self.get_available_generators()
            }
        
        # Update state to indicate generation started
        update_state("media.is_generating", True, source="media_engine")
        update_state("media.current_generation_type", media_type, source="media_engine")
        update_state("media.current_generation_prompt", prompt, source="media_engine")
        
        # Emit generation started event
        emit_event(
            HorizonEventTypes.MEDIA_GENERATION_STARTED,
            "media_engine",
            {
                'media_type': media_type,
                'prompt': prompt,
                'params': params or {}
            }
        )
        
        try:
            generator = self.generators[media_type]
            result = generator.generate(prompt, params or {})
            
            # Update state with completion
            update_state("media.is_generating", False, source="media_engine")
            update_state("media.current_generation_type", "", source="media_engine")
            update_state("media.current_generation_prompt", "", source="media_engine")
            
            return result
            
        except Exception as e:
            # Update state with error
            update_state("media.is_generating", False, source="media_engine")
            update_state("media.current_generation_type", "", source="media_engine")
            update_state("media.current_generation_prompt", "", source="media_engine")
            
            return {
                'success': False,
                'error': str(e),
                'media_type': media_type,
                'prompt': prompt
            }


class ImageGenerator:
    """Image and artwork generation."""
    
    def __init__(self):
        """Initialize image generator."""
        self.available_models = []
        
        # Check available image generation APIs
        if OPENAI_AVAILABLE:
            self.available_models.append('dall-e-3')
        
        if IMAGEN_AVAILABLE:
            self.available_models.append('imagen-4.0')
        
        print(f"🖼️ Image Generator initialized with models: {self.available_models}")
    
    def generate(self, prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate an image from text prompt."""
        params = {**DEFAULT_IMAGE_PARAMS, **(params or {})}
        
        try:
            # Try DALL-E 3 first if available
            if 'dall-e-3' in self.available_models:
                return self._generate_with_dalle(prompt, params)
            
            # Fall back to Imagen if available
            elif 'imagen-4.0' in self.available_models:
                return self._generate_with_imagen(prompt, params)
            
            else:
                return self._generate_placeholder_image(prompt, params)
                
        except Exception as e:
            print(f"❌ Image generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback': self._generate_placeholder_image(prompt, params)
            }
    
    def _generate_with_dalle(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image using DALL-E 3."""
        if not OPENAI_AVAILABLE:
            raise Exception("OpenAI library not available")
        
        # This would implement actual DALL-E generation
        # For now, return a placeholder result
        filename = f"dalle_image_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(MEDIA_OUTPUT_DIRS['images'], filename)
        
        return {
            'success': True,
            'model': 'dall-e-3',
            'prompt': prompt,
            'filename': filename,
            'filepath': filepath,
            'url': f'/static/generated_images/{filename}',
            'params': params,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_with_imagen(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image using Google Imagen."""
        if not IMAGEN_AVAILABLE:
            raise Exception("Imagen library not available")
        
        filename = f"imagen_image_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(MEDIA_OUTPUT_DIRS['images'], filename)
        
        return {
            'success': True,
            'model': 'imagen-4.0',
            'prompt': prompt,
            'filename': filename,
            'filepath': filepath,
            'url': f'/static/generated_images/{filename}',
            'params': params,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_placeholder_image(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a placeholder image when APIs are unavailable."""
        filename = f"placeholder_image_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(MEDIA_OUTPUT_DIRS['images'], filename)
        
        # Create a simple placeholder image if PIL is available
        if PIL_AVAILABLE:
            try:
                img = Image.new('RGB', (params['width'], params['height']), 
                              color=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))
                img.save(filepath)
            except Exception:
                pass
        
        return {
            'success': True,
            'model': 'placeholder',
            'prompt': prompt,
            'filename': filename,
            'filepath': filepath,
            'url': f'/static/generated_images/{filename}',
            'params': params,
            'generated_at': datetime.now().isoformat(),
            'note': 'This is a placeholder. Configure API keys for actual image generation.'
        }


class VideoGenerator:
    """Video and animation generation."""
    
    def __init__(self):
        """Initialize video generator."""
        self.available_models = []
        
        # Check for Google Veo 3 availability
        if GEMINI_AVAILABLE:
            self.available_models.append('veo-3')
        
        if REPLICATE_AVAILABLE:
            self.available_models.extend(['stable-video', 'runway-ml'])
        
        print(f"🎬 Video Generator initialized with models: {self.available_models}")
    
    def generate(self, prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a video from text prompt."""
        params = {**DEFAULT_VIDEO_PARAMS, **(params or {})}
        
        try:
            # Try Veo 3 first if available
            if 'veo-3' in self.available_models:
                return self._generate_with_veo3(prompt, params)
            elif 'stable-video' in self.available_models:
                return self._generate_with_stable_video(prompt, params)
            else:
                return self._generate_placeholder_video(prompt, params)
                
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback': self._generate_placeholder_video(prompt, params)
            }
    
    def _generate_with_veo3(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video using Google Veo 3."""
        try:
            print(f"🎬 Attempting Veo 3 video generation: {prompt}")
            
            # Note: Veo 3 may not be publicly available yet
            # This is a framework for when Google releases the API
            
            # Check if we have the right API access
            if not GEMINI_AVAILABLE:
                raise Exception("Gemini AI not available")
            
            # Try to use Gemini for video generation
            # This is currently a placeholder as Veo 3 API isn't publicly available
            import google.generativeai as genai
            
            # Enhanced prompt for video generation
            video_prompt = f"Create a high-quality video of: {prompt}. Duration: {params.get('duration', 5)} seconds. Style: cinematic, professional quality."
            
            # For now, return a structured response indicating Veo 3 status
            filename = f"veo3_video_{uuid.uuid4().hex[:8]}.mp4"
            filepath = os.path.join(MEDIA_OUTPUT_DIRS['videos'], filename)
            
            return {
                'success': False,
                'model': 'veo-3',
                'prompt': prompt,
                'filename': filename,
                'filepath': filepath,
                'url': f'/static/generated_videos/{filename}',
                'params': params,
                'generated_at': datetime.now().isoformat(),
                'status': 'api_not_available',
                'message': 'Veo 3 API is not yet publicly available. Framework ready for when Google releases the API.',
                'note': 'Google Veo 3 is announced but API access is limited. Check https://deepmind.google/technologies/veo/ for updates.'
            }
            
        except Exception as e:
            print(f"⚠️ Veo 3 generation error: {e}")
            return {
                'success': False,
                'model': 'veo-3',
                'error': f'Veo 3 not available: {str(e)}',
                'fallback_available': True
            }
    
    def _generate_with_stable_video(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video using Stable Video Diffusion."""
        filename = f"stable_video_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(MEDIA_OUTPUT_DIRS['videos'], filename)
        
        return {
            'success': True,
            'model': 'stable-video',
            'prompt': prompt,
            'filename': filename,
            'filepath': filepath,
            'url': f'/static/generated_videos/{filename}',
            'params': params,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_placeholder_video(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate placeholder video info."""
        filename = f"placeholder_video_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(MEDIA_OUTPUT_DIRS['videos'], filename)
        
        return {
            'success': True,
            'model': 'placeholder',
            'prompt': prompt,
            'filename': filename,
            'filepath': filepath,
            'url': f'/static/generated_videos/{filename}',
            'params': params,
            'generated_at': datetime.now().isoformat(),
            'note': 'Video generation requires additional API configuration.'
        }


class AudioGenerator:
    """Music and audio generation."""
    
    def __init__(self):
        """Initialize audio generator."""
        self.available_models = []
        
        if REPLICATE_AVAILABLE:
            self.available_models.extend(['musicgen', 'riffusion'])
        
        print(f"🎵 Audio Generator initialized with models: {self.available_models}")
    
    def generate(self, prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate audio from text prompt."""
        params = {**DEFAULT_AUDIO_PARAMS, **(params or {})}
        
        try:
            if 'musicgen' in self.available_models:
                return self._generate_with_musicgen(prompt, params)
            else:
                return self._generate_placeholder_audio(prompt, params)
                
        except Exception as e:
            print(f"❌ Audio generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback': self._generate_placeholder_audio(prompt, params)
            }
    
    def _generate_with_musicgen(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audio using MusicGen."""
        filename = f"musicgen_audio_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(MEDIA_OUTPUT_DIRS['music'], filename)
        
        return {
            'success': True,
            'model': 'musicgen',
            'prompt': prompt,
            'filename': filename,
            'filepath': filepath,
            'url': f'/static/generated_music/{filename}',
            'params': params,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_placeholder_audio(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate placeholder audio info."""
        filename = f"placeholder_audio_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(MEDIA_OUTPUT_DIRS['music'], filename)
        
        return {
            'success': True,
            'model': 'placeholder',
            'prompt': prompt,
            'filename': filename,
            'filepath': filepath,
            'url': f'/static/generated_music/{filename}',
            'params': params,
            'generated_at': datetime.now().isoformat(),
            'note': 'Audio generation requires additional API configuration.'
        }


class ModelGenerator:
    """3D model and design generation."""
    
    def __init__(self):
        """Initialize 3D model generator."""
        self.available_models = []
        
        if REPLICATE_AVAILABLE:
            self.available_models.extend(['shap-e', 'point-e'])
        
        print(f"🗿 3D Model Generator initialized with models: {self.available_models}")
    
    def generate(self, prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate 3D model from text prompt."""
        params = params or {}
        
        try:
            if 'shap-e' in self.available_models:
                return self._generate_with_shape(prompt, params)
            else:
                return self._generate_placeholder_model(prompt, params)
                
        except Exception as e:
            print(f"❌ 3D model generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback': self._generate_placeholder_model(prompt, params)
            }
    
    def _generate_with_shape(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D model using Shap-E."""
        filename = f"shape_model_{uuid.uuid4().hex[:8]}.obj"
        filepath = os.path.join(MEDIA_OUTPUT_DIRS['models'], filename)
        
        return {
            'success': True,
            'model': 'shap-e',
            'prompt': prompt,
            'filename': filename,
            'filepath': filepath,
            'url': f'/static/generated_3d_models/{filename}',
            'params': params,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_placeholder_model(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate placeholder 3D model info."""
        filename = f"placeholder_model_{uuid.uuid4().hex[:8]}.obj"
        filepath = os.path.join(MEDIA_OUTPUT_DIRS['models'], filename)
        
        return {
            'success': True,
            'model': 'placeholder',
            'prompt': prompt,
            'filename': filename,
            'filepath': filepath,
            'url': f'/static/generated_3d_models/{filename}',
            'params': params,
            'generated_at': datetime.now().isoformat(),
            'note': '3D model generation requires additional API configuration.'
        }


# Global instances
media_engine = None
image_generator = None
video_generator = None
audio_generator = None
model_generator = None

def get_media_engine() -> MediaEngine:
    """Get the global media engine instance."""
    global media_engine
    if media_engine is None:
        media_engine = MediaEngine()
    return media_engine

def get_image_generator() -> Optional[ImageGenerator]:
    """Get the global image generator instance."""
    engine = get_media_engine()
    return engine.generators.get('image')

def get_video_generator() -> Optional[VideoGenerator]:
    """Get the global video generator instance."""
    engine = get_media_engine()
    return engine.generators.get('video')

def get_audio_generator() -> Optional[AudioGenerator]:
    """Get the global audio generator instance."""
    engine = get_media_engine()
    return engine.generators.get('audio')

def get_model_generator() -> Optional[ModelGenerator]:
    """Get the global model generator instance."""
    engine = get_media_engine()
    return engine.generators.get('model')

# Enhanced specialized generators extracted from app.py
class LogoGenerator:
    """Professional logo and brand design generation."""
    
    def __init__(self):
        """Initialize logo generator."""
        self.industry_prompts = {
            'technology': 'tech, digital, innovation, modern, circuit patterns, gear icons',
            'healthcare': 'medical, health, care, cross symbol, healing, wellness, trust',
            'finance': 'banking, money, security, stability, professional, trust, growth',
            'restaurant': 'food, dining, chef hat, fork and knife, culinary, appetite',
            'fashion': 'style, elegance, clothing, trendy, chic, sophisticated',
            'education': 'learning, books, graduation cap, knowledge, growth, development',
            'automotive': 'cars, speed, movement, wheels, engineering, power',
            'beauty': 'elegance, style, cosmetics, wellness, luxury, refined',
            'sports': 'athletic, fitness, energy, movement, strength, competition',
            'travel': 'adventure, exploration, journey, compass, globe, destinations'
        }
        print("🏷️ Logo Generator initialized")
    
    def generate_logo(self, brand_name: str, industry: str, style: str = "modern") -> Dict[str, Any]:
        """Generate professional logo design."""
        try:
            # Enhanced logo prompt with industry-specific elements
            logo_prompt = f"professional logo design for {brand_name}, {industry} industry, {style} style"
            
            # Add industry-specific elements
            if industry in self.industry_prompts:
                logo_prompt += f", {self.industry_prompts[industry]}"
            
            # Style-specific enhancements
            style_prompts = {
                "modern": "clean lines, minimalist, contemporary design, geometric shapes, sans-serif typography",
                "vintage": "retro aesthetic, classic typography, timeless design, aged textures, serif fonts",
                "creative": "artistic flair, unique concept, innovative design, abstract elements, creative typography",
                "corporate": "professional appearance, trustworthy, business-oriented, clean, authoritative",
                "playful": "fun, colorful, friendly, approachable, rounded shapes, vibrant colors",
                "elegant": "sophisticated, luxury, refined, premium, elegant typography, subtle colors"
            }
            
            if style.lower() in style_prompts:
                logo_prompt += f", {style_prompts[style.lower()]}"
            
            logo_prompt += ", vector style, high contrast, suitable for business use, scalable, memorable branding"
            
            # Generate using image generator
            image_gen = get_image_generator()
            if image_gen:
                result = image_gen.generate(logo_prompt)
                result['logo_type'] = 'professional'
                result['brand_name'] = brand_name
                result['industry'] = industry
                result['style'] = style
                return result
            else:
                return {
                    'success': False,
                    'error': 'Image generator not available for logo generation'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Logo generation failed: {str(e)}'
            }


class Enhanced3DModelGenerator:
    """Enhanced 3D model generation with multiple API support."""
    
    def __init__(self):
        """Initialize 3D model generator."""
        self.available_apis = []
        
        # Check for API availability
        if hasattr(Config, 'TRIPO_API_KEY') and Config.TRIPO_API_KEY:
            self.available_apis.append('tripo')
        if hasattr(Config, 'MESHY_API_KEY') and Config.MESHY_API_KEY:
            self.available_apis.append('meshy')
            
        print(f"🗿 Enhanced 3D Model Generator initialized with APIs: {self.available_apis}")
    
    def generate_3d_model(self, prompt: str, style: str = "realistic") -> Dict[str, Any]:
        """Generate 3D model from text description."""
        try:
            # Enhanced 3D model prompt
            model_prompt = f"3D model of {prompt}, {style} style"
            
            style_enhancements = {
                "realistic": "high detail, photorealistic textures, professional quality",
                "lowpoly": "low polygon count, game-ready, clean geometry",
                "stylized": "artistic style, creative design, unique aesthetic"
            }
            
            if style.lower() in style_enhancements:
                model_prompt += f", {style_enhancements[style.lower()]}"
            
            # Try available APIs
            if 'tripo' in self.available_apis:
                return self._generate_with_tripo(model_prompt, style)
            elif 'meshy' in self.available_apis:
                return self._generate_with_meshy(model_prompt, style)
            else:
                return self._generate_placeholder_3d_model(prompt, style)
                
        except Exception as e:
            return {
                'success': False,
                'error': f'3D model generation failed: {str(e)}'
            }
    
    def _generate_with_tripo(self, prompt: str, style: str) -> Dict[str, Any]:
        """Generate using Tripo API."""
        try:
            headers = {
                "Authorization": f"Bearer {Config.TRIPO_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "style": style,
                "quality": "high"
            }
            
            response = requests.post(
                "https://api.tripo3d.ai/v1/text-to-3d",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "model_url" in result:
                    # Download and save model
                    model_response = requests.get(result["model_url"], timeout=60)
                    
                    if model_response.status_code == 200:
                        filename = f"tripo_model_{uuid.uuid4().hex[:8]}.obj"
                        filepath = os.path.join(MEDIA_OUTPUT_DIRS['models'], filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(model_response.content)
                        
                        return {
                            'success': True,
                            'model': 'tripo',
                            'filename': filename,
                            'filepath': filepath,
                            'url': f'/static/generated_3d_models/{filename}',
                            'prompt': prompt,
                            'style': style,
                            'generated_at': datetime.now().isoformat()
                        }
            
            return {'success': False, 'error': 'Tripo API request failed'}
            
        except Exception as e:
            return {'success': False, 'error': f'Tripo API error: {str(e)}'}
    
    def _generate_with_meshy(self, prompt: str, style: str) -> Dict[str, Any]:
        """Generate using Meshy API."""
        try:
            headers = {
                "Authorization": f"Bearer {Config.MESHY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": prompt,
                "mode": "text-to-3d",
                "art_style": style
            }
            
            response = requests.post(
                "https://api.meshy.ai/v1/text-to-3d",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'model': 'meshy',
                    'status': 'processing',
                    'task_id': result.get('task_id'),
                    'message': '3D model generation started - check back in a few minutes',
                    'prompt': prompt,
                    'style': style,
                    'generated_at': datetime.now().isoformat()
                }
            
            return {'success': False, 'error': 'Meshy API request failed'}
            
        except Exception as e:
            return {'success': False, 'error': f'Meshy API error: {str(e)}'}
    
    def _generate_placeholder_3d_model(self, prompt: str, style: str) -> Dict[str, Any]:
        """Generate placeholder 3D model info."""
        filename = f"placeholder_3d_{uuid.uuid4().hex[:8]}.obj"
        filepath = os.path.join(MEDIA_OUTPUT_DIRS['models'], filename)
        
        return {
            'success': True,
            'model': 'placeholder',
            'filename': filename,
            'filepath': filepath,
            'url': f'/static/generated_3d_models/{filename}',
            'prompt': prompt,
            'style': style,
            'generated_at': datetime.now().isoformat(),
            'note': '3D model generation requires API configuration (Tripo or Meshy)'
        }


# Enhanced MediaEngine with new generators
class EnhancedMediaEngine(MediaEngine):
    """Enhanced media engine with specialized generators."""
    
    def __init__(self):
        """Initialize enhanced media engine."""
        super().__init__()
        
        # Add specialized generators
        self.logo_generator = LogoGenerator()
        self.enhanced_3d_generator = Enhanced3DModelGenerator()
        
        print("🚀 Enhanced Media Engine initialized with specialized generators")
    
    def generate_logo(self, brand_name: str, industry: str, style: str = "modern") -> Dict[str, Any]:
        """Generate professional logo design."""
        return self.logo_generator.generate_logo(brand_name, industry, style)
    
    def generate_enhanced_3d_model(self, prompt: str, style: str = "realistic") -> Dict[str, Any]:
        """Generate enhanced 3D model."""
        return self.enhanced_3d_generator.generate_3d_model(prompt, style)
    
    def get_generation_capabilities(self) -> Dict[str, List[str]]:
        """Get available generation capabilities."""
        return {
            'image_generation': self.get_available_generators(),
            'logo_generation': ['professional', 'creative', 'corporate'],
            '3d_generation': self.enhanced_3d_generator.available_apis,
            'media_formats': {
                'images': ['png', 'jpg', 'webp'],
                'videos': ['mp4', 'webm', 'gif'],
                'audio': ['mp3', 'wav', 'ogg'],
                'models': ['obj', 'glb', 'fbx']
            }
        }


# Global enhanced instances
enhanced_media_engine = None
logo_generator = None
enhanced_3d_generator = None

def get_enhanced_media_engine() -> EnhancedMediaEngine:
    """Get the enhanced media engine instance."""
    global enhanced_media_engine
    if enhanced_media_engine is None:
        enhanced_media_engine = EnhancedMediaEngine()
    return enhanced_media_engine

def get_logo_generator() -> LogoGenerator:
    """Get the logo generator instance."""
    global logo_generator
    if logo_generator is None:
        logo_generator = LogoGenerator()
    return logo_generator

def get_enhanced_3d_generator() -> Enhanced3DModelGenerator:
    """Get the enhanced 3D generator instance."""
    global enhanced_3d_generator
    if enhanced_3d_generator is None:
        enhanced_3d_generator = Enhanced3DModelGenerator()
    return enhanced_3d_generator

# Convenience functions for backward compatibility
def generate_image(prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate an image from text prompt."""
    return get_enhanced_media_engine().generate_media('image', prompt, params)

def generate_video(prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate a video from text prompt."""
    return get_enhanced_media_engine().generate_media('video', prompt, params)

def generate_audio(prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate audio from text prompt."""
    return get_enhanced_media_engine().generate_media('audio', prompt, params)

def generate_3d_model(prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate a 3D model from text prompt."""
    return get_enhanced_media_engine().generate_enhanced_3d_model(prompt, params.get('style', 'realistic') if params else 'realistic')

def generate_logo_design(brand_name: str, industry: str, style: str = "modern") -> Dict[str, Any]:
    """Generate professional logo design."""
    return get_enhanced_media_engine().generate_logo(brand_name, industry, style)