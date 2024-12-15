import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

# Step 1: Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Step 2: Load the dataset
file_path = 'tcc_ceds_music.csv'  # Replace with your actual dataset file path
music_data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(music_data.head())  # Debug: Display dataset

# Step 3: TF-IDF Vectorization for Song Metadata
def vectorize_song_metadata():
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    song_metadata = music_data['track_name'] + ' ' + music_data['artist_name']
    X = vectorizer.fit_transform(song_metadata).toarray()
    print("TF-IDF vectorization completed!")  # Debug
    return vectorizer, X

# Step 4: Train and Save Random Forest Model
def train_and_save_random_forest(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Random Forest model trained!")  # Debug
    
    # Evaluate Model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")  # Debug
    
    # Save Model and Vectorizer
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Model and vectorizer saved!")  # Debug

# Check if model and vectorizer exist; if not, train and save
if not os.path.exists('random_forest_model.pkl') or not os.path.exists('tfidf_vectorizer.pkl'):
    vectorizer, X = vectorize_song_metadata()
    train_and_save_random_forest(X, music_data['mood'])

# Step 5: Load Trained Model and Vectorizer
rf_model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
print("Model and vectorizer loaded!")  # Debug

# Step 6: Assign Predicted Moods to Songs
def assign_song_moods():
    song_metadata = music_data['track_name'] + ' ' + music_data['artist_name']
    features = vectorizer.transform(song_metadata).toarray()
    music_data['predicted_mood'] = rf_model.predict(features)
    print("Moods assigned to songs!")  # Debug

assign_song_moods()

# Step 7: Detect User Mood Using Sentiment Analysis
def detect_user_mood(user_input):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(user_input)
    print(f"Sentiment scores: {sentiment}")  # Debug
    if sentiment['compound'] >= 0.5:
        return "Happy"
    elif sentiment['compound'] <= -0.5:
        return "Angry"
    elif sentiment['compound'] > -0.5 and sentiment['compound'] < 0:
        return "Sad"
    else:
        return "Calm"

# Step 8: Generate Playlist Based on Mood
def generate_playlist(mood, num_songs=10):
    print(f"Generating playlist for mood: {mood}")  # Debug
    filtered_songs = music_data[music_data['predicted_mood'] == mood]
    if len(filtered_songs) < num_songs:
        num_songs = len(filtered_songs)
    return filtered_songs[['track_name', 'artist_name']].sample(num_songs)

# Step 9: Display Playlist in GUI
def display_playlist():
    user_input = mood_entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showerror("Error", "Please enter a mood description!")
        return
    
    # Detect Mood from User Input
    user_mood = detect_user_mood(user_input)
    print(f"Detected mood: {user_mood}")  # Debug
    
    # Generate Playlist
    playlist = generate_playlist(user_mood, num_songs=10)
    
    # Display Playlist in Text Widget
    playlist_text.delete(1.0, tk.END)
    playlist_text.insert(tk.END, f"Playlist for mood: {user_mood}\n")
    for _, song in playlist.iterrows():
        playlist_text.insert(tk.END, f"- {song['track_name']} by {song['artist_name']}\n")

# Step 10: Create GUI
window = tk.Tk()
window.title("Mood-based Playlist Generator")

# Set Window Size
window.geometry("400x500")

# GUI Elements
label = tk.Label(window, text="Describe your mood:")
label.pack(pady=10)

mood_entry = tk.Text(window, height=5, width=40)
mood_entry.pack(pady=10)

generate_button = tk.Button(window, text="Generate Playlist", command=display_playlist)
generate_button.pack(pady=10)

playlist_text = tk.Text(window, height=15, width=40)
playlist_text.pack(pady=20)

# Run the Application
window.mainloop()
