#pip install spotipy textblob pandas matplotlib
import spotipy
from spotipy.oauth2 import SpotifyOAuth

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

# Spotify API credentials
CLIENT_ID = 'dc6892c9948e460fab75dac643f228fb'
CLIENT_SECRET = 'c6fa985da3984defbc258d8674ea9f53'
REDIRECT_URI = 'http://localhost:8888/callback/'

# Autentisering
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope='playlist-read-private'
))

# Funktion för att hämta låtar och egenskaper
def get_playlist_features(playlist_id):
    results = sp.playlist_items(playlist_id)
    tracks = []
    for item in results['items']:
        track = item['track']
        features = sp.audio_features(track['id'])[0]
        if features:
            tracks.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'danceability': features['danceability'],
                'energy': features['energy'],
                'key': features['key'],
                'loudness': features['loudness'],
                'mode': features['mode'],
                'speechiness': features['speechiness'],
                'acousticness': features['acousticness'],
                'instrumentalness': features['instrumentalness'],
                'liveness': features['liveness'],
                'valence': features['valence'],
                'tempo': features['tempo']
            })
    return pd.DataFrame(tracks)

# Exempel: Hämta data från en spellista
playlist_id = '4kMWgzMNKW8B0EbvyUfIHZ'
data = get_playlist_features(playlist_id)
print(data.head())
