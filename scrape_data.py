import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

playlists = sp.featured_playlists()
playlist_id = []
while playlists:
    for i, playlist in enumerate(playlists["items"]):
        print(f"{i + 1 + playlists['offset']:4d} {playlist['uri']} {playlist['name']}")
        playlist_id.append(playlist["uri"][-22:])
    if playlists["next"]:
        playlists = sp.next(playlists)
    else:
        playlists = None

playlist_songs = {}
def getTrackIDs(playlist_id):
    ids = []
    playlist = sp.user_playlist("spotify", playlist_id)
    for item in playlist["tracks"]["items"][:50]:
        track = item["track"]
        ids.append(track["id"])
    playlist_songs[playlist_id] = ids
    return



def getTrackFeatures(track_id):
  meta = sp.track(track_id)
  features = sp.audio_features(track_id)

  # meta
  track_id = track_id
  name = meta['name']
  album = meta['album']['name']
  artist = meta['album']['artists'][0]['name']
  release_date = meta['album']['release_date']
  length = meta['duration_ms']
  popularity = meta['popularity']

  # features
  acousticness = features[0]['acousticness']
  danceability = features[0]['danceability']
  energy = features[0]['energy']
  instrumentalness = features[0]['instrumentalness']
  liveness = features[0]['liveness']
  loudness = features[0]['loudness']
  speechiness = features[0]['speechiness']
  tempo = features[0]['tempo']
  time_signature = features[0]['time_signature']

  track = [track_id, name, album, artist, release_date, length, popularity, danceability, acousticness, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature]
  return track


