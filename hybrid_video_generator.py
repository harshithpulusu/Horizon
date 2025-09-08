#!/usr/bin/env python3
"""
🎬 Hybrid Video Generation System for Horizon AI Assistant
Combines DALL-E (fast/cheap) with Runway ML (cinematic/premium)
"""

import requests
import time
import uuid
import os
import io
from PIL import Image
from config import Config

class HybridVideoGenerator:
    """Hybrid video generation with DALL-E and Runway ML options"""
    
    def __init__(self):
        self.dalle_available = bool(Config.OPENAI_API_KEY)
        self.runway_available = bool(Config.RUNWAY_API_KEY)
        self.setup_status()
    
    def setup_status(self):
        """Check available video generation methods"""
        print("🎬 Hybrid Video Generation System Initialized:")
        print(f"  ✅ DALL-E System: {'Available' if self.dalle_available else 'Not configured'}")
        print(f"  🎭 Runway ML: {'Available' if self.runway_available else 'Not configured'}")
        
        if not self.dalle_available and not self.runway_available:
            print("  ⚠️ No video generation systems available")
        elif self.dalle_available and self.runway_available:
            print("  🚀 Full hybrid system ready!")
    
    def determine_best_method(self, prompt, user_preference=None):
        """Intelligently choose between DALL-E and Runway based on prompt and preference"""
        
        # User explicit preference
        if user_preference:
            if user_preference.lower() in ['runway', 'cinematic', 'premium', 'realistic']:
                return 'runway' if self.runway_available else 'dalle'
            elif user_preference.lower() in ['dalle', 'fast', 'quick', 'cheap']:
                return 'dalle' if self.dalle_available else 'runway'
        
        # Analyze prompt for cinematic keywords
        cinematic_keywords = [
            'cinematic', 'movie', 'film', 'dramatic', 'realistic', 'professional',
            'motion', 'movement', 'action', 'flowing', 'dynamic', 'smooth',
            'hollywood', 'epic', 'scene', 'sequence', 'camera', 'shot'
        ]
        
        prompt_lower = prompt.lower()
        
        # Check for cinematic keywords
        if any(keyword in prompt_lower for keyword in cinematic_keywords):
            return 'runway' if self.runway_available else 'dalle'
        
        # Check for complex scenes that benefit from true video
        complex_keywords = [
            'dancing', 'running', 'flying', 'swimming', 'walking', 'moving',
            'jumping', 'rotating', 'spinning', 'flowing', 'waves', 'fire',
            'smoke', 'water', 'wind', 'explosion', 'growing', 'transforming'
        ]
        
        if any(keyword in prompt_lower for keyword in complex_keywords):
            return 'runway' if self.runway_available else 'dalle'
        
        # Default to DALL-E for static or simple scenes
        return 'dalle' if self.dalle_available else 'runway'
    
    def generate_video(self, prompt, method="auto", quality="high", duration=5):
        """Main video generation function with hybrid approach"""
        
        # Determine method
        if method == "auto":
            chosen_method = self.determine_best_method(prompt)
        else:
            chosen_method = method
        
        # Validate method availability
        if chosen_method == 'runway' and not self.runway_available:
            print("⚠️ Runway ML not available, falling back to DALL-E")
            chosen_method = 'dalle'
        elif chosen_method == 'dalle' and not self.dalle_available:
            print("⚠️ DALL-E not available, falling back to Runway ML")
            chosen_method = 'runway'
        
        # Generate video based on chosen method
        print(f"🎬 Generating video using {chosen_method.upper()} method...")
        
        if chosen_method == 'runway':
            return self.generate_runway_video(prompt, duration, quality)
        else:
            return self.generate_dalle_video(prompt, duration, quality)
    
    def generate_runway_video(self, prompt, duration=5, quality="high"):
        """Generate video using Runway ML for cinematic quality"""
        
        try:
            # Enhanced prompt for Runway ML
            runway_prompt = self.enhance_prompt_for_runway(prompt, quality)
            
            # Runway ML API call
            headers = {
                "Authorization": f"Bearer {Config.RUNWAY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Video generation request
            generation_data = {
                "text_prompt": runway_prompt,
                "duration": min(duration, 10),  # Runway has limits
                "ratio": "16:9",
                "watermark": False,
                "enhance_prompt": True,
                "seed": None  # Random seed for variety
            }
            
            print(f"🎭 Runway ML: Creating cinematic video...")
            print(f"📝 Enhanced prompt: {runway_prompt}")
            
            # Start generation
            response = requests.post(
                "https://api.runwayml.com/v1/generate",
                headers=headers,
                json=generation_data
            )
            
            if response.status_code == 200:
                task_id = response.json()["id"]
                print(f"🔄 Runway generation started (ID: {task_id})")
                
                # Poll for completion
                video_url = self.poll_runway_completion(task_id, headers)
                
                if video_url:
                    # Download and save video
                    video_filename = self.download_runway_video(video_url)
                    return video_filename, None
                else:
                    return None, "Runway ML generation timed out"
            else:
                error_msg = response.json().get('error', 'Unknown Runway error')
                print(f"❌ Runway ML error: {error_msg}")
                return None, f"Runway ML error: {error_msg}"
                
        except Exception as e:
            print(f"❌ Runway ML generation failed: {e}")
            # Fallback to DALL-E if available
            if self.dalle_available:
                print("🔄 Falling back to DALL-E system...")
                return self.generate_dalle_video(prompt, duration, quality)
            return None, f"Runway ML error: {str(e)}"
    
    def enhance_prompt_for_runway(self, prompt, quality):
        """Enhance prompt specifically for Runway ML"""
        
        quality_enhancers = {
            "quick": "simple, clean",
            "standard": "detailed, good lighting",
            "high": "cinematic, professional lighting, high detail, 4K quality",
            "ultra": "cinematic masterpiece, dramatic lighting, ultra-detailed, professional cinematography, epic scene"
        }
        
        enhancer = quality_enhancers.get(quality, quality_enhancers["high"])
        
        # Add cinematic elements
        enhanced = f"{prompt}, {enhancer}, smooth motion, realistic movement"
        
        # Add style hints based on content
        if any(word in prompt.lower() for word in ['nature', 'landscape', 'outdoor']):
            enhanced += ", natural lighting, outdoor cinematography"
        elif any(word in prompt.lower() for word in ['person', 'people', 'character']):
            enhanced += ", portrait lighting, character focus"
        elif any(word in prompt.lower() for word in ['abstract', 'artistic']):
            enhanced += ", artistic style, creative cinematography"
        
        return enhanced
    
    def poll_runway_completion(self, task_id, headers, max_wait=300):
        """Poll Runway ML for video completion"""
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                status_response = requests.get(
                    f"https://api.runwayml.com/v1/tasks/{task_id}",
                    headers=headers
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status")
                    
                    if status == "SUCCEEDED":
                        video_url = status_data.get("output", {}).get("video_url")
                        print("✅ Runway ML generation completed!")
                        return video_url
                    elif status == "FAILED":
                        error = status_data.get("error", "Unknown error")
                        print(f"❌ Runway ML generation failed: {error}")
                        return None
                    else:
                        print(f"🔄 Runway ML status: {status}...")
                        time.sleep(10)  # Wait 10 seconds before next check
                else:
                    print(f"⚠️ Status check failed: {status_response.status_code}")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"⚠️ Error checking status: {e}")
                time.sleep(5)
        
        print("⏰ Runway ML generation timed out")
        return None
    
    def download_runway_video(self, video_url):
        """Download Runway ML generated video"""
        
        try:
            video_response = requests.get(video_url)
            
            if video_response.status_code == 200:
                # Generate unique filename
                video_id = str(uuid.uuid4())
                video_filename = f"runway_{video_id}.mp4"
                
                # Ensure videos directory exists
                videos_dir = "static/generated_videos"
                os.makedirs(videos_dir, exist_ok=True)
                
                video_path = os.path.join(videos_dir, video_filename)
                
                # Save video file
                with open(video_path, 'wb') as f:
                    f.write(video_response.content)
                
                print(f"✅ Runway video saved: {video_filename}")
                return video_filename
            else:
                print(f"❌ Failed to download video: {video_response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading video: {e}")
            return None
    
    def generate_dalle_video(self, prompt, duration, quality):
        """Generate video using existing DALL-E system"""
        
        # Import the existing DALL-E video generation function
        from app import generate_text_video
        
        print(f"🎨 DALL-E: Creating animated video...")
        return generate_text_video(prompt, duration, 30, quality)
    
    def get_generation_info(self, method="auto", prompt=""):
        """Get information about generation method and costs"""
        
        if method == "auto":
            chosen_method = self.determine_best_method(prompt)
        else:
            chosen_method = method
        
        info = {
            'dalle': {
                'name': 'DALL-E Animated',
                'speed': 'Fast (30-60 seconds)',
                'cost': 'Low (~$0.04 per video)',
                'quality': 'High-quality animated slideshow',
                'best_for': 'Static scenes, portraits, landscapes, quick generation'
            },
            'runway': {
                'name': 'Runway ML Cinematic',
                'speed': 'Slow (2-5 minutes)',
                'cost': 'High (~$0.50-2.00 per video)',
                'quality': 'Professional cinematic with realistic motion',
                'best_for': 'Action, movement, cinematic scenes, realistic motion'
            }
        }
        
        return info.get(chosen_method, info['dalle'])

# Global hybrid video generator
hybrid_video_generator = HybridVideoGenerator()

def generate_hybrid_video(prompt, method="auto", quality="high", duration=5):
    """Main function for hybrid video generation"""
    return hybrid_video_generator.generate_video(prompt, method, quality, duration)

def analyze_video_prompt(prompt):
    """Analyze prompt and recommend best generation method"""
    method = hybrid_video_generator.determine_best_method(prompt)
    info = hybrid_video_generator.get_generation_info(method, prompt)
    
    return {
        'recommended_method': method,
        'info': info,
        'reasoning': f"Recommended {method.upper()} because: {info['best_for']}"
    }

if __name__ == "__main__":
    # Test the hybrid system
    print("🎬 Testing Hybrid Video Generation System\n")
    
    test_prompts = [
        "a cute cat sitting in a garden",
        "cinematic shot of a dragon flying over mountains",
        "a person dancing in the rain",
        "abstract colorful patterns",
        "dramatic sunset over the ocean"
    ]
    
    for prompt in test_prompts:
        analysis = analyze_video_prompt(prompt)
        print(f"📝 Prompt: {prompt}")
        print(f"🎯 Recommended: {analysis['recommended_method'].upper()}")
        print(f"💡 Reasoning: {analysis['reasoning']}")
        print("-" * 50)
