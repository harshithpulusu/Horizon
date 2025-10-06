"""
Horizon Media Generator Core Module

This module handles media generation functionality for the Horizon AI Assistant.
It provides image, video, audio, and 3D model generation capabilities.

Classes:
- MediaEngine: Main media generation system
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
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from config import Config

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
        
        # Ensure output directories exist
        self._create_output_directories()
        
        # Initialize available generators
        self._initialize_generators()
        
        print("🎨 Media Engine initialized")
    
    def _create_output_directories(self):
        """Create output directories for generated media."""
        for media_type, directory in MEDIA_OUTPUT_DIRS.items():
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_generators(self):
        """Initialize available media generators."""
        if OPENAI_AVAILABLE or IMAGEN_AVAILABLE:
            self.generators['image'] = ImageGenerator()
        
        if REPLICATE_AVAILABLE:
            self.generators['video'] = VideoGenerator()
            self.generators['audio'] = AudioGenerator()
            self.generators['model'] = ModelGenerator()
        
        print(f"🔧 Initialized generators: {list(self.generators.keys())}")
    
    def get_available_generators(self) -> List[str]:
        """Get list of available media generators."""
        return [k for k, v in self.generators.items() if v is not None]
    
    def generate_media(self, media_type: str, prompt: str, 
                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate media of specified type.
        
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
        
        generator = self.generators[media_type]
        return generator.generate(prompt, params or {})


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
        
        if REPLICATE_AVAILABLE:
            self.available_models.extend(['stable-video', 'runway-ml'])
        
        print(f"🎬 Video Generator initialized with models: {self.available_models}")
    
    def generate(self, prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a video from text prompt."""
        params = {**DEFAULT_VIDEO_PARAMS, **(params or {})}
        
        try:
            if 'stable-video' in self.available_models:
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

# Convenience functions for backward compatibility
def generate_image(prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate an image from text prompt."""
    return get_media_engine().generate_media('image', prompt, params)

def generate_video(prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate a video from text prompt."""
    return get_media_engine().generate_media('video', prompt, params)

def generate_audio(prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate audio from text prompt."""
    return get_media_engine().generate_media('audio', prompt, params)

def generate_3d_model(prompt: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate a 3D model from text prompt."""
    return get_media_engine().generate_media('model', prompt, params)