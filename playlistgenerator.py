import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
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

# Step 3: Define a function to classify songs into moods (including "Angry," "Sad," "Energetic," "Calm")
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

# Step 4: Apply the updated mood classification
music_data['mood'] = music_data.apply(classify_mood, axis=1)

# Display the updated mood distribution
print("\nMood Distribution:")
print(music_data['mood'].value_counts())

# Step 5: Prepare Features (Lyrics + Audio Features)
def prepare_features():
    # Combine lyrics for TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    text_features = vectorizer.fit_transform(music_data['lyrics']).toarray()

    # Scale audio features (valence, energy, danceability, etc.)
    numerical_features = ['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness']
    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(music_data[numerical_features])

    # Combine TF-IDF text features with scaled numerical features
    X = pd.concat([pd.DataFrame(text_features), pd.DataFrame(scaled_numerical_features)], axis=1).values
    print("Features prepared!")  # Debug
    return vectorizer, scaler, X

# Step 6: Train and Save Random Forest Model (Updated with Confusion Matrix)
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
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Model, Vectorizer, and Scaler
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    print("Model, vectorizer, and scaler saved!")  # Debug

# Check if model, vectorizer, and scaler exist; if not, train and save
if not os.path.exists('random_forest_model.pkl') or not os.path.exists('tfidf_vectorizer.pkl') or not os.path.exists('feature_scaler.pkl'):
    vectorizer, scaler, X = prepare_features()
    train_and_save_random_forest(X, music_data['mood'])

# Step 7: Load Trained Model, Vectorizer, and Scaler
rf_model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('feature_scaler.pkl')
print("Model, vectorizer, and scaler loaded!")  # Debug

# Step 8: Assign Predicted Moods to Songs
def assign_song_moods():
    # Transform features using the loaded vectorizer and scaler
    text_features = vectorizer.transform(music_data['lyrics']).toarray()
    numerical_features = ['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness']
    scaled_numerical_features = scaler.transform(music_data[numerical_features])

    # Combine text and numerical features
    combined_features = pd.concat([pd.DataFrame(text_features), pd.DataFrame(scaled_numerical_features)], axis=1).values
    music_data['predicted_mood'] = rf_model.predict(combined_features)
    print("Moods assigned to songs!")  # Debug

assign_song_moods()

# Step 9: Detect User Mood Using Sentiment Analysis
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

# Step 10: Generate Playlist Based on Mood
def generate_playlist(mood, num_songs=10):
    print(f"Generating playlist for mood: {mood}")  # Debug
    filtered_songs = music_data[music_data['predicted_mood'] == mood]
    if len(filtered_songs) < num_songs:
        num_songs = len(filtered_songs)
    return filtered_songs[['track_name', 'artist_name']].sample(num_songs)

# Step 11: Display Playlist in GUI
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
    playlist_text.delete("1.0", tk.END)
    playlist_text.insert(tk.END, f"Playlist for mood: {user_mood}\n")
    for _, song in playlist.iterrows():
        playlist_text.insert(tk.END, f"- {song['track_name']} by {song['artist_name']}\n")

# Step 12: Create GUI
window = tk.Tk()
window.title("Mood-based Playlist Generator")

# Set Window Size
window.geometry("500x600")

from tkinter import PhotoImage

from PIL import Image, ImageTk

# Load and Resize the Image
image_path = "playlisttext.png"  # Replace with the correct path to your image
try:
    # Open the image
    image = Image.open(image_path)
    # Resize the image (set width and height)
    resized_image = image.resize((350, 45))  # Set desired width=300 and height=100
    # Convert to a PhotoImage
    tk_image = ImageTk.PhotoImage(resized_image)
except Exception as e:
    print(f"Error loading image: {e}")
    tk_image = None  # Fallback if the image fails to load

# Display the resized image in the Label
if tk_image:
    label = tk.Label(window, image=tk_image)
    label.image = tk_image  # Keep a reference to avoid garbage collection
    label.pack(pady=10)
else:
    # Fallback text if the image doesn't load
    label = tk.Label(window, text="Describe your mood:", font=("Lexend Deca", 14, "bold"))
    label.pack(pady=10)

    
# GUI Elements
#label = tk.Label(window, text="Describe your mood:", font=("Lexend Deca", 14, "bold"))
#label.pack(pady=10)

mood_entry = tk.Text(window, height=5, width=50)
mood_entry.pack(pady=10)



# Customize Button to Match the Style in the Image
def style_button(button):
    button.configure(

        fg="white",  # White text color
        font=("Lexend Deca", 13, "bold"),  # Bold font
        relief="flat",  # Flat relief for a modern look
        borderwidth=0,  # Remove button border
        height=2,  # Button height
        width=20  # Button width
    )
    button.bind("<Enter>", lambda e: button.configure(bg="#1AAE48"))  # Darker green on hover
    button.bind("<Leave>", lambda e: button.configure(bg="#1DB954"))  # Default green on leave

# Create Generate Playlist Button with Rounded Edges
def rounded_button(master, **kwargs):
    button = tk.Button(master, **kwargs)
    style_button(button)
    button.config(highlightthickness=0, padx=10, pady=5, borderwidth=0)
    return button

generate_button = rounded_button(window, text="Generate Playlist", command=display_playlist)
generate_button.pack(pady=10)

playlist_text = tk.Text(window, height=20, width=50)
playlist_text.pack(pady=20)

# Run the Application
window.mainloop()
