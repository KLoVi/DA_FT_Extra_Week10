import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import random
import config
import pickle
from IPython.display import IFrame


import spotipy
import pandas as pd
import json
from spotipy.oauth2 import SpotifyClientCredentials



#Initialize SpotiPy with user credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= config.client_id,
                                                           client_secret= config.client_secret))


# Load the model from the pickle file
with open('kmeans_15.pkl', 'rb') as file:
    kmeans_15 = pickle.load(file)
    
with open('minmax.pkl', 'rb') as file:
    minmax = pickle.load(file)

url_1 = "/Users/karollvivianalopezvillegas/GitHub/DA_FT_Extra_Week10/Week_10/Mini-project_Spotify/Billboard_100.csv"

df_b = pd.read_csv(url_1)
bb_100 = df_b['song'].tolist()
clusters = "/Users/karollvivianalopezvillegas/GitHub/DA_FT_Extra_Week10/Week_10/Mini-project_Spotify/My_Spotify_selection_WITH CLUSTERS_15.csv" 

ds = pd.read_csv(clusters)


def song_recommender(score_corte=85):
    
    def play_song(track_id):
        return IFrame(src="https://open.spotify.com/embed/track/"+track_id,
           width="320",
           height="80",
           frameborder="0",
           allowtransparency="true",
           allow="encrypted-media",
          )
    
    name =  st.text_input("Please enter the name of a song:").strip()

    coincidence = process.extractOne(name, bb_100, score_cutoff=score_corte)
    
    results_name = sp.search(q= name,limit=1)
    track_id_name = results_name["tracks"]["items"][0]["id"]
    
    
    st.display(play_song(track_id_name))
    
    
    if coincidence:
        # Encontró una coincidencia, obtener el nombre de la canción
        matched_song = coincidence[0]
        
        # Filtrar el DataFrame para encontrar la canción y el artista
        song_info = df_b[df_b['song'] == matched_song]

        # Recomendar una canción aleatoria del DataFrame
        recommended_song = df_b.sample(n=1).iloc[0]
        
        # Encontrar id de la canción recomendada:
        results = sp.search(q=recommended_song,limit=1)
        track_id = results["tracks"]["items"][0]["id"]
        
    
        st.write(f"Your meant this song name, right?: {matched_song}")
        st.write(f"Your song is popular! Here's another popular song: {recommended_song['song']}, Artist: {recommended_song['artist']}")
    
    else:
        # No encontró una coincidencia
        results = sp.search(q = name,limit=1)
        track_id = results["tracks"]["items"][0]["id"]
        name_features = sp.audio_features(track_id)
        df_name = pd.DataFrame(name_features)
        X_name = df_name[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_ms', 'time_signature']]
        X_name_scaled = minmax.fit_transform(X_name)
        cluster_sample = kmeans_15.predict(X_name_scaled)
        recommended_song = ds[ ds["cluster"] == cluster_sample.item() ].sample()
        track_id_recom = recommended_song["id"]
        track_id = track_id_recom.item()
        st.write(f"Your recommended song is: ")
        
    return st.play_song(track_id)



# Set the title of the app
st.title("Song recommender")

# Create a text input for the user's name
st.write("Welcome to this simple recommender, Ironhacker!")
name = st.text_input("Enter your name:")

# Create a button that the user can click
if st.button("Submit"):
    # Display:
    st.write(f"Hello, {name}!")
    
    song_recommender()


# Add a footer message
st.write("Thank you for using this simple app!")