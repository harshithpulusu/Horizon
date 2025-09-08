# 🎵 Spotify Integration for Horizon AI Assistant

## 🚀 Quick Setup

### 1. Get Spotify API Credentials
1. Go to https://developer.spotify.com/dashboard
2. Click "Create App"
3. Fill in app details:
   - App name: "Horizon AI Assistant"
   - Description: "AI assistant with music control"
   - Redirect URI: `http://localhost:8080/callback`
4. Copy Client ID and Client Secret

### 2. Add to .env file
```bash
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

### 3. Install Spotify Package
```bash
pip install spotipy
```

## 🎯 Voice Commands Enabled

- **"Play [song/artist/genre]"** - Play music
- **"Skip song"** - Next track
- **"Previous song"** - Previous track
- **"Pause music"** - Pause playback
- **"Resume music"** - Resume playback
- **"Volume [1-10]"** - Set volume
- **"Create playlist [name]"** - Create new playlist
- **"What's playing?"** - Current track info
- **"Play my liked songs"** - Play saved tracks
- **"Shuffle on/off"** - Toggle shuffle

## 🛠️ Implementation

The integration includes:
- ✅ Search and play tracks
- ✅ Playback control (play/pause/skip)
- ✅ Volume control
- ✅ Playlist management
- ✅ Current track information
- ✅ User's saved music access

## 🔗 Authorization Flow

1. User says "Connect Spotify"
2. System opens browser for Spotify login
3. User authorizes the app
4. System saves access token
5. Full music control enabled!

## 🎮 Example Usage

**User**: "Play some jazz music"
**Assistant**: "🎵 Playing jazz music from Spotify! Currently playing 'Take Five' by Dave Brubeck."

**User**: "Skip this song"
**Assistant**: "⏭️ Skipped to next track: 'Blue Moon' by Ella Fitzgerald."

**User**: "What's playing?"
**Assistant**: "🎼 Now playing: 'Fly Me to the Moon' by Frank Sinatra from the album 'Sinatra at the Sands'."
