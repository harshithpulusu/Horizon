#!/usr/bin/env python3
"""
Veo 3 Video Generation Test Script

Tests the availability and functionality of Google Veo 3 video generation
through the Horizon AI system.
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

def print_banner():
    """Print test banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                          🎬 Veo 3 Test Suite 🎬                             ║
    ║                      Google Video Generation Testing                         ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_veo3_availability():
    """Check if Veo 3 is available."""
    print("🔍 Checking Veo 3 availability...")
    
    try:
        from core import get_enhanced_media_engine
        
        # Get media engine
        media_engine = get_enhanced_media_engine()
        video_gen = media_engine.generators.get('video')
        
        if video_gen is None:
            print("❌ Video generator not available")
            return False
        
        print(f"✅ Video generator initialized")
        print(f"📋 Available models: {video_gen.available_models}")
        
        # Check if Veo 3 is in available models
        if 'veo-3' in video_gen.available_models:
            print("✅ Veo 3 detected in available models")
            return True
        else:
            print("⚠️ Veo 3 not in available models")
            return False
            
    except Exception as e:
        print(f"❌ Error checking Veo 3 availability: {e}")
        return False

def test_veo3_generation():
    """Test Veo 3 video generation."""
    print("🎬 Testing Veo 3 video generation...")
    
    try:
        from core import generate_video
        
        # Test prompt
        test_prompt = "A beautiful sunset over a calm ocean with gentle waves"
        test_params = {
            'duration': 5,
            'fps': 24,
            'quality': 'high'
        }
        
        print(f"📝 Test prompt: '{test_prompt}'")
        print(f"⚙️ Parameters: {test_params}")
        
        # Generate video
        result = generate_video(test_prompt, test_params)
        
        print("📊 Generation Result:")
        print(json.dumps(result, indent=2))
        
        if result.get('success'):
            print("✅ Video generation successful!")
            print(f"📁 File: {result.get('filename')}")
            print(f"🔗 URL: {result.get('url')}")
            return True
        else:
            print("⚠️ Video generation failed or limited")
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            if result.get('status') == 'api_not_available':
                print("💡 This is expected - Veo 3 API is not yet publicly available")
                print(f"📋 Status: {result.get('message')}")
                print(f"ℹ️ Note: {result.get('note')}")
                return "limited"
            
            return False
            
    except Exception as e:
        print(f"❌ Error testing Veo 3 generation: {e}")
        return False

def test_video_fallbacks():
    """Test video generation fallbacks."""
    print("🔄 Testing video generation fallbacks...")
    
    try:
        from core import get_enhanced_media_engine
        
        media_engine = get_enhanced_media_engine()
        video_gen = media_engine.generators.get('video')
        
        if video_gen is None:
            print("❌ Video generator not available")
            return False
        
        # Test with a simple prompt
        test_prompt = "A cat playing in a garden"
        result = video_gen.generate(test_prompt)
        
        print("📊 Fallback Result:")
        print(json.dumps(result, indent=2))
        
        if result.get('success') or result.get('fallback_available'):
            print("✅ Fallback system working")
            return True
        else:
            print("⚠️ Fallback system issues")
            return False
            
    except Exception as e:
        print(f"❌ Error testing fallbacks: {e}")
        return False

def check_google_api_status():
    """Check Google API configuration."""
    print("🔧 Checking Google API configuration...")
    
    try:
        from config import Config
        
        # Check Gemini API key
        if hasattr(Config, 'GEMINI_API_KEY') and Config.GEMINI_API_KEY:
            print("✅ Gemini API key configured")
        else:
            print("⚠️ Gemini API key not configured")
        
        # Try to import and initialize Gemini
        try:
            import google.generativeai as genai
            print("✅ Google Generative AI library available")
            
            if hasattr(Config, 'GEMINI_API_KEY') and Config.GEMINI_API_KEY:
                genai.configure(api_key=Config.GEMINI_API_KEY)
                print("✅ Gemini API configured successfully")
                
                # List available models
                models = []
                try:
                    for model in genai.list_models():
                        models.append(model.name)
                    print(f"📋 Available Gemini models: {len(models)}")
                    
                    # Check for video-related models
                    video_models = [m for m in models if 'video' in m.lower() or 'veo' in m.lower()]
                    if video_models:
                        print(f"🎬 Video-related models: {video_models}")
                    else:
                        print("⚠️ No video-specific models found in current API")
                        
                except Exception as e:
                    print(f"⚠️ Could not list models: {e}")
                
            else:
                print("⚠️ Cannot configure Gemini - no API key")
                
        except ImportError:
            print("❌ Google Generative AI library not available")
            print("💡 Install with: pip install google-generativeai")
            
    except Exception as e:
        print(f"❌ Error checking Google API: {e}")

def print_veo3_info():
    """Print information about Veo 3."""
    info = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                              📋 Veo 3 Information                            ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  Google Veo 3 is Google's latest video generation AI model announced        ║
    ║  in December 2024. It can create high-quality videos from text prompts.     ║
    ║                                                                              ║
    ║  Current Status:                                                             ║
    ║  ├─ Announced: ✅ Yes (December 2024)                                       ║
    ║  ├─ Public API: ❌ Not yet available                                         ║
    ║  ├─ Limited Access: ⚠️ Possible for select partners                         ║
    ║  └─ Timeline: 🔮 Expected in 2025                                            ║
    ║                                                                              ║
    ║  Capabilities:                                                               ║
    ║  ├─ High-quality video generation                                            ║
    ║  ├─ Text-to-video synthesis                                                  ║
    ║  ├─ Various video lengths and formats                                        ║
    ║  └─ Advanced motion and scene understanding                                  ║
    ║                                                                              ║
    ║  For Updates:                                                                ║
    ║  └─ https://deepmind.google/technologies/veo/                                ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(info)

def main():
    """Main test function."""
    print_banner()
    
    # Run tests
    tests = [
        ("API Configuration", check_google_api_status),
        ("Veo 3 Availability", check_veo3_availability),
        ("Video Generation", test_veo3_generation),
        ("Fallback Systems", test_video_fallbacks),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"🧪 Running: {test_name}")
        print('='*80)
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 VEO 3 TEST SUMMARY")
    print('='*80)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result == "limited":
            status = "⚠️ LIMITED"
        else:
            status = "❌ FAIL"
        print(f"{status} {test_name}")
    
    print('='*80)
    
    # Print Veo 3 information
    print_veo3_info()
    
    # Overall assessment
    if results.get("Veo 3 Availability") and results.get("Video Generation") == "limited":
        print("🎯 RESULT: Veo 3 framework ready, waiting for Google API release")
    elif any(results.values()):
        print("⚠️ RESULT: Partial video generation capabilities available")
    else:
        print("❌ RESULT: Video generation not currently functional")

if __name__ == "__main__":
    main()