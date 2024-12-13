import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Load the dataset (replace this with the correct file path)
file_path = 'tcc_ceds_music.csv'  # Replace with your actual dataset file path
music_data = pd.read_csv(file_path)

# Define a function to classify songs into moods
def classify_mood(row):
    if row['valence'] < 0.3 and row['energy'] > 0.7:
        return "Angry"
    elif row['valence'] > 0.7 and row['energy'] > 0.6:
        return "Happy"
    elif row['valence'] < 0.4 and row['energy'] < 0.5:
        return "Sad"
    elif row['danceability'] > 0.7 and row['energy'] > 0.5:
        return "Energetic"
    else:
        return "Calm"

# Apply the mood classification to the dataset
music_data['mood'] = music_data.apply(classify_mood, axis=1)

# Function to generate a playlist based on the mood
def generate_playlist(mood, num_songs=10):
    # Filter songs matching the desired mood
    filtered_songs = music_data[music_data['mood'] == mood]
    if len(filtered_songs) < num_songs:
        num_songs = len(filtered_songs)
    
    return filtered_songs[['track_name', 'artist_name']].sample(num_songs)

# Function to update the UI with the playlist
def display_playlist(mood):
    # Generate playlist for the selected mood
    playlist = generate_playlist(mood, num_songs=10)

    # Clear previous results
    playlist_text.delete(1.0, tk.END)

    # Display the playlist in the text widget
    playlist_text.insert(tk.END, f"Playlist for mood: {mood}\n")
    for _, song in playlist.iterrows():
        playlist_text.insert(tk.END, f"- {song['track_name']} by {song['artist_name']}\n")

# Create the main window
window = tk.Tk()
window.title("Mood-based Playlist Generator")

# Set window size
window.geometry("400x400")

# Create mood buttons
button_happy = tk.Button(window, text="Happy", width=20, command=lambda: display_playlist("Happy"))
button_happy.pack(pady=10)

button_angry = tk.Button(window, text="Angry", width=20, command=lambda: display_playlist("Angry"))
button_angry.pack(pady=10)

button_sad = tk.Button(window, text="Sad", width=20, command=lambda: display_playlist("Sad"))
button_sad.pack(pady=10)

button_calm = tk.Button(window, text="Calm", width=20, command=lambda: display_playlist("Calm"))
button_calm.pack(pady=10)

# Create a text widget to display the playlist
playlist_text = tk.Text(window, height=10, width=40)
playlist_text.pack(pady=20)

# Run the application
window.mainloop()
