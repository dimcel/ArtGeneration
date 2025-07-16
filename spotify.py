import requests
import base64
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()

def get_spotify_token() -> Optional[str]:

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Missing Spotify credentials in .env file")
        return None
    
    # Encode credentials
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    
    # Get token
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {credentials}"},
        data={"grant_type": "client_credentials"}
    )
    
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print(f"Spotify auth failed: {response.status_code}")
        return None

def search_spotify_track(artist: str, track_name: str) -> Optional[Dict[str, Any]]:
    """
    Search for a track on Spotify and return all available data
    Note: Spotify has restricted access to audio-features and audio-analysis endpoints
    """
    token = get_spotify_token()
    if not token:
        return None
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Search for track
    search_url = "https://api.spotify.com/v1/search"
    params = {
        "q": f"artist:{artist} track:{track_name}",
        "type": "track",
        "limit": 1
    }
    
    response = requests.get(search_url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Search failed: {response.status_code}")
        return None
    
    data = response.json()
    if not data["tracks"]["items"]:
        print("Track not found on Spotify")
        return None
    
    track = data["tracks"]["items"][0]
    artist_id = track["artists"][0]["id"]
    
    # Build result with track metadata
    result = {
        # Basic track info  
        "title": track["name"],
        "artist": track["artists"][0]["name"],
        "album": track["album"]["name"],
        "duration": track["duration_ms"] / 1000,  # Convert to seconds
        "popularity": track["popularity"],
        "release_date": track["album"]["release_date"],
        "spotify_url": track["external_urls"]["spotify"],
        "cover_image_url": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
    }
    
    # Get artist info for genres (this still works!)
    print("Getting artist info...")
    artist_url = f"https://api.spotify.com/v1/artists/{artist_id}"
    artist_response = requests.get(artist_url, headers=headers)
    
    if artist_response.status_code == 200:
        artist_info = artist_response.json()
        print("Artist info available!")
        result.update({
            "genres": artist_info.get("genres", []),
            "artist_popularity": artist_info.get("popularity"),
            "artist_followers": artist_info.get("followers", {}).get("total"),
        })
    else:
        print(f"Artist info not available (status: {artist_response.status_code})")
    
    return result


if __name__ == "__main__":

    artist = "The Stranglers"
    track = "Golden Brown"
    
    print(f"Searching for: '{track}' by '{artist}'")
    result = search_spotify_track(artist, track)
    
    if result:
        print("\nFound track! Spotify data:")
        for key, value in result.items():
            if value is not None:
                print(f"  {key}: {value}")
    else:
        print("Could not get Spotify data")