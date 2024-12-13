import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'tcc_ceds_music.csv'  # Replace with the actual path to your dataset
music_data = pd.read_csv(file_path)

# Display basic information about the dataset
#print(music_data.info())
#print(music_data.head())

# Define a function to classify songs into moods (including "Angry")
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

# Apply the updated mood classification to create initial mood labels
music_data['mood'] = music_data.apply(classify_mood, axis=1)

# Display the first few rows to verify the mood column
print(music_data[['track_name', 'artist_name', 'mood']].head())

# Prepare features and labels
text_features = 'lyrics'  # Lyrics column for TF-IDF
numerical_features = ['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness']
X_text = music_data[text_features]
X_numerical = music_data[numerical_features]
y = music_data['mood']  # Target variable (moods)

# Combine numerical and text features using a pipeline
tfidf = TfidfVectorizer(max_features=500)  # Extract top 500 words
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('text', tfidf, text_features)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(music_data, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and Random Forest Classifier
clf = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict moods for new songs
def predict_mood(new_songs):
    predictions = clf.predict(new_songs)
    new_songs['predicted_mood'] = predictions
    return new_songs

# Example: Predict moods for new songs
new_songs = pd.DataFrame({
    'valence': [0.8, 0.2, 0.5],
    'energy': [0.7, 0.9, 0.6],
    'danceability': [0.9, 0.6, 0.7],
    'acousticness': [0.2, 0.8, 0.5],
    'instrumentalness': [0.1, 0.9, 0.0],
    'lyrics': ["I feel so happy and alive", "Angry and frustrated times", "Let the rhythm take over"]
})
predicted_songs = predict_mood(new_songs)
print("\nPredicted Moods for New Songs:")
print(predicted_songs)

# Generate a playlist based on predicted moods
def generate_playlist(mood, num_songs=10):
    # Filter songs matching the desired mood
    filtered_songs = music_data[music_data['mood'] == mood]
    if len(filtered_songs) < num_songs:
        print(f"Not enough songs available for the mood '{mood}'. Showing {len(filtered_songs)} songs instead.")
        num_songs = len(filtered_songs)
    return filtered_songs[['track_name', 'artist_name']].sample(num_songs)

# Example: Generate a playlist for a specific mood
user_mood = "Angry"  # Replace with user input for mood
playlist = generate_playlist(user_mood, num_songs=10)

# Print the playlist
print(f"\nPlaylist for mood '{user_mood}':")
print(playlist)
