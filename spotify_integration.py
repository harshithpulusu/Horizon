#!/usr/bin/env python3
"""
🎵 Spotify Integration for Horizon AI Assistant
Add music control capabilities with voice commands
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from config import Config

class SpotifyController:
    def __init__(self):
        self.sp = None
        self.setup_spotify()
    
    def setup_spotify(self):
        """Initialize Spotify connection"""
        try:
            if Config.SPOTIFY_CLIENT_ID and Config.SPOTIFY_CLIENT_SECRET:
                self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                    client_id=Config.SPOTIFY_CLIENT_ID,
                    client_secret=Config.SPOTIFY_CLIENT_SECRET,
                    redirect_uri="http://localhost:8080/callback",
                    scope="user-modify-playback-state user-read-playback-state user-library-read playlist-modify-public playlist-modify-private"
                ))
                print("🎵 Spotify integration ready!")
                return True
        except Exception as e:
            print(f"⚠️ Spotify setup failed: {e}")
            return False
    
    def search_and_play(self, query):
        """Search for music and play it"""
        try:
            if not self.sp:
                return "❌ Spotify not connected. Please set up your API keys."
            
            # Search for tracks
            results = self.sp.search(q=query, type='track', limit=1)
            
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                track_uri = track['uri']
                track_name = track['name']
                artist_name = track['artists'][0]['name']
                
                # Play the track
                try:
                    self.sp.start_playback(uris=[track_uri])
                    return f"🎵 Now playing: '{track_name}' by {artist_name}"
                except Exception as e:
                    if "No active device" in str(e):
                        return f"🎵 Found '{track_name}' by {artist_name}. Please open Spotify on a device to start playback."
                    return f"❌ Playback error: {str(e)}"
            else:
                return f"❌ No music found for '{query}'"
                
        except Exception as e:
            return f"❌ Spotify error: {str(e)}"
    
    def control_playback(self, action):
        """Control music playback"""
        try:
            if not self.sp:
                return "❌ Spotify not connected."
            
            if action == "pause":
                self.sp.pause_playback()
                return "⏸️ Music paused"
            elif action == "resume" or action == "play":
                self.sp.start_playback()
                return "▶️ Music resumed"
            elif action == "skip" or action == "next":
                self.sp.next_track()
                return "⏭️ Skipped to next track"
            elif action == "previous":
                self.sp.previous_track()
                return "⏮️ Previous track"
            else:
                return f"❌ Unknown playback action: {action}"
                
        except Exception as e:
            if "No active device" in str(e):
                return "🎵 Please open Spotify on a device first."
            return f"❌ Playback control error: {str(e)}"
    
    def get_current_track(self):
        """Get information about currently playing track"""
        try:
            if not self.sp:
                return "❌ Spotify not connected."
            
            current = self.sp.current_playback()
            
            if current and current['is_playing']:
                track = current['item']
                track_name = track['name']
                artist_name = track['artists'][0]['name']
                album_name = track['album']['name']
                progress = current['progress_ms'] // 1000
                duration = track['duration_ms'] // 1000
                
                progress_min = progress // 60
                progress_sec = progress % 60
                duration_min = duration // 60
                duration_sec = duration % 60
                
                return f"🎼 Now playing: '{track_name}' by {artist_name}\n📀 Album: {album_name}\n⏱️ {progress_min}:{progress_sec:02d} / {duration_min}:{duration_sec:02d}"
            else:
                return "🔇 No music currently playing"
                
        except Exception as e:
            return f"❌ Error getting track info: {str(e)}"
    
    def set_volume(self, volume):
        """Set playback volume (0-10)"""
        try:
            if not self.sp:
                return "❌ Spotify not connected."
            
            # Convert 1-10 scale to 0-100
            volume_percent = max(0, min(100, int(volume) * 10))
            self.sp.volume(volume_percent)
            return f"🔊 Volume set to {volume}/10"
            
        except Exception as e:
            return f"❌ Volume control error: {str(e)}"

# Global Spotify controller instance
spotify_controller = SpotifyController()

def handle_spotify_command(command):
    """Handle Spotify voice commands"""
    command = command.lower()
    
    # Play music commands
    if command.startswith("play "):
        query = command[5:]  # Remove "play "
        return spotify_controller.search_and_play(query)
    
    # Playback control
    elif "skip" in command or "next" in command:
        return spotify_controller.control_playback("skip")
    elif "previous" in command:
        return spotify_controller.control_playback("previous")
    elif "pause" in command:
        return spotify_controller.control_playback("pause")
    elif "resume" in command or ("play" in command and "music" in command):
        return spotify_controller.control_playback("resume")
    
    # Volume control
    elif command.startswith("volume "):
        try:
            volume = command.split("volume ")[1]
            return spotify_controller.set_volume(volume)
        except:
            return "❌ Please specify volume 1-10 (e.g., 'volume 5')"
    
    # Track info
    elif "what's playing" in command or "current song" in command:
        return spotify_controller.get_current_track()
    
    else:
        return "🎵 Spotify commands: play [song], skip, pause, resume, volume [1-10], what's playing"

if __name__ == "__main__":
    # Test the integration
    print("🎵 Testing Spotify Integration...")
    print(handle_spotify_command("what's playing"))
