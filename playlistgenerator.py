# import pandas as pd
# import tkinter as tk
# from tkinter import ttk

# # Load the dataset (replace this with the correct file path)
# file_path = 'tcc_ceds_music.csv'  # Replace with your actual dataset file path
# music_data = pd.read_csv(file_path)

# # Define a function to classify songs into moods
# def classify_mood(row):
#     if row['valence'] < 0.3 and row['energy'] > 0.7:
#         return "Angry"
#     elif row['valence'] > 0.7 and row['energy'] > 0.6:
#         return "Happy"
#     elif row['valence'] < 0.4 and row['energy'] < 0.5:
#         return "Sad"
#     elif row['danceability'] > 0.7 and row['energy'] > 0.5:
#         return "Energetic"
#     else:
#         return "Calm"

# # Apply the mood classification to the dataset
# music_data['mood'] = music_data.apply(classify_mood, axis=1)

# # Function to generate a playlist based on the mood
# def generate_playlist(mood, num_songs=10):
#     # Filter songs matching the desired mood
#     filtered_songs = music_data[music_data['mood'] == mood]
#     if len(filtered_songs) < num_songs:
#         num_songs = len(filtered_songs)  # Use all songs if less than 'num_songs'
    
#     # Return the songs, no random sample, so all of them show up
#     return filtered_songs[['track_name', 'artist_name']].head(num_songs)  # Use .head to limit if fewer than num_songs

# # Function to update the UI with the playlist
# def display_playlist(mood):
#     # Generate playlist for the selected mood
#     playlist = generate_playlist(mood, num_songs=10)

#     # Clear previous results in playlist_frame
#     for widget in playlist_frame.winfo_children():
#         widget.destroy()

#     # Add title above playlist display
#     title_label.config(text=f"Playlist for mood: {mood}")

#     # Display the playlist in a more styled format (Spotify-like)
#     for _, song in playlist.iterrows():
#         # Create label for the song title with a larger font size
#         title_label = ttk.Label(playlist_frame,
#                                text=song['track_name'],
#                                font=("Helvetica", 18, "bold"),  # Larger font for title
#                                anchor="w",  # Align left
#                                background="#282828",  # Dark background for the songs
#                                foreground="white",
#                                padding=(10, 5))
#         title_label.pack(fill="x", pady=(5, 0))  # Padding between songs

#         # Create label for the artist name with a smaller font size
#         artist_label = ttk.Label(playlist_frame,
#                                  text=song['artist_name'],
#                                  font=("Helvetica", 14),  # Smaller font for artist
#                                  anchor="w",  # Align left
#                                  background="#282828",  # Dark background for the songs
#                                  foreground="white",
#                                  padding=(10, 5))
#         artist_label.pack(fill="x", pady=(0, 10))  # Padding between song and next artist

# # Create the main window
# window = tk.Tk()
# window.title("Mood-based Playlist Generator")

# # Set window size and background color
# window.geometry("800x600")  # Set a fixed size
# window.configure(bg="black")  # Set the background color of the entire window

# # Create a Style object for ttk widgets
# style = ttk.Style()

# # Configure custom styles for buttons
# style.configure("TButton",
#                 font=("Helvetica", 16, "bold"),
#                 foreground="white",
#                 background="#1DB954",  # Spotify green color
#                 width=12,  # Set width to fit the button size better
#                 height=2,
#                 padding=(20, 10),  # Increase padding for larger buttons
#                 relief="solid",  # Button border style
#                 borderwidth=2,  # Border width to give more visibility to button edges
#                 cursor='hand1',
#                 highlightthickness=0)  # Remove highlight border

# # Create mood buttons using ttk.Button with consistent packing
# button_frame = tk.Frame(window, bg="black")  # Create a frame to hold buttons
# button_frame.pack(pady=10, fill="x")

# # Create mood buttons using ttk.Button and pack them
# button_happy = ttk.Button(button_frame, text="Happy", style="TButton", command=lambda: display_playlist("Happy"))
# button_happy.pack(side="left", padx=5, expand=True)

# button_angry = ttk.Button(button_frame, text="Angry", style="TButton", command=lambda: display_playlist("Angry"))
# button_angry.pack(side="left", padx=5, expand=True)

# button_sad = ttk.Button(button_frame, text="Sad", style="TButton", command=lambda: display_playlist("Sad"))
# button_sad.pack(side="left", padx=5, expand=True)

# button_calm = ttk.Button(button_frame, text="Calm", style="TButton", command=lambda: display_playlist("Calm"))
# button_calm.pack(side="left", padx=5, expand=True)

# # Create a label for the title above the playlist (large white title)
# title_label = ttk.Label(window,
#                         background="black",  # Dark background for the title
#                         text="",  # Initial placeholder text
#                         font=("Helvetica", 24, "bold"),
#                         foreground="white",  # White text color
#                         padding=5)
# title_label.pack(fill="x", pady=20)  # Title above the playlist, with padding

# # Create a frame to hold the playlist items (this ensures better organization)
# playlist_frame = tk.Frame(window, bg="black", padx=20, pady=20)  # Set background to black
# playlist_frame.pack(pady=20, fill="both", expand=True)

# # Run the application
# window.mainloop()

import pandas as pd
import tkinter as tk
from tkinter import ttk

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
        num_songs = len(filtered_songs)  # Use all songs if fewer than 'num_songs'
    
    # Return the songs, no random sample, so all of them show up
    return filtered_songs[['track_name', 'artist_name']].head(num_songs)  # Use .head to limit if fewer than num_songs

# Function to update the UI with the playlist
def display_playlist(mood):
    # Generate playlist for the selected mood
    playlist = generate_playlist(mood, num_songs=10)

    # Clear previous results in playlist_frame
    for widget in playlist_frame.winfo_children():
        widget.destroy()

    # Add title above playlist display (use global title_label)
    title_label.config(text=f"Playlist for mood: {mood}")

    # Display the playlist in a more styled format (Spotify-like)
    for _, song in playlist.iterrows():
        # Create label for the song title with a larger font size
        song_title_label = ttk.Label(playlist_frame,
                                     text=song['track_name'],
                                     font=("Helvetica", 16),  # Larger font for title
                                     anchor="w",  # Align left
                                     background="#282828",  # Dark background for the songs
                                     foreground="white",
                                     padding=(1, 1))
        song_title_label.pack(fill="x", pady=(1, 0))  # Padding between songs

        # Create label for the artist name with a smaller font size
        artist_label = ttk.Label(playlist_frame,
                                 text=song['artist_name'],
                                 font=("Helvetica", 12),  # Smaller font for artist
                                 anchor="w",  # Align left
                                 background="#101010",  # Dark background for the songs
                                 foreground="grey",
                                 padding=(1, 1))
        artist_label.pack(fill="x", pady=(0, 1))  # Padding between song and next artist

# Create the main window
window = tk.Tk()
window.title("Mood-based Playlist Generator")

# Set window size and background color
window.geometry("800x600")  # Set a fixed size
window.configure(bg="black")  # Set the background color of the entire window

# Create a Style object for ttk widgets
style = ttk.Style()

# Configure custom styles for buttons (Spotify-like)
style.configure("TButton",
                font=("Helvetica", 16, "bold"),
                foreground="white",
                background="#1DB954",  # Spotify green color
                width=12,  # Set width to fit the button size better
                height=3,  # Make buttons larger
                padding=(20, 10),  # Increase padding for larger buttons
                relief="flat",  # Make button edges flat for a modern look
                borderwidth=1,  # Border width to give more visibility to button edges
                cursor='hand1',
                highlightthickness=0,  # Remove highlight border
                anchor="center")

# Create mood buttons using ttk.Button with consistent packing
button_frame = tk.Frame(window, bg="black")  # Create a frame to hold buttons
button_frame.pack(pady=10, fill="x")

# Create mood buttons using ttk.Button and pack them
button_happy = ttk.Button(button_frame, text="Happy", style="TButton", command=lambda: display_playlist("Happy"))
button_happy.pack(side="left", padx=10, expand=True)

button_angry = ttk.Button(button_frame, text="Angry", style="TButton", command=lambda: display_playlist("Angry"))
button_angry.pack(side="left", padx=10, expand=True)

button_sad = ttk.Button(button_frame, text="Sad", style="TButton", command=lambda: display_playlist("Sad"))
button_sad.pack(side="left", padx=10, expand=True)

button_calm = ttk.Button(button_frame, text="Calm", style="TButton", command=lambda: display_playlist("Calm"))
button_calm.pack(side="left", padx=10, expand=True)

# Create a label for the title above the playlist (large white title)
title_label = ttk.Label(window,
                        background="black",  # Dark background for the title
                        text="",  # Initial placeholder text
                        font=("Helvetica", 24, "bold"),
                        foreground="white",  # White text color
                        padding=5)
title_label.pack(fill="x", pady=20)  # Title above the playlist, with padding

# Create a frame to hold the playlist items (this ensures better organization)
playlist_frame = tk.Frame(window, bg="black", padx=20, pady=20)  # Set background to black
playlist_frame.pack(pady=20, fill="both", expand=True)

# Run the application
window.mainloop()
