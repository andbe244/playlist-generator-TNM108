import os
import pandas as pd

# Step 1: Load the Dataset
def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' not found.")
    
    print("Loading dataset...")
    data = pd.read_csv(file_path)

    # Filter relevant columns for simplicity
    if not all(col in data.columns for col in ["track_name", "artist_name", "valence", "energy"]):
        raise ValueError("Dataset must include 'track_name', 'artist_name', 'valence', and 'energy' columns.")

    print(f"Dataset loaded successfully with {len(data)} songs.")
    return data[["track_name", "artist_name", "valence", "energy"]].to_dict(orient="records")

# Step 2: Define mood-to-feature mapping
mood_to_features = {
    "happy": {"valence": [0.7, 1.0], "energy": [0.6, 1.0]},
    "chill": {"valence": [0.4, 0.7], "energy": [0.2, 0.5]},
    "sad": {"valence": [0.0, 0.4], "energy": [0.0, 0.3]}
}

# Step 3: Define the recommendation algorithm
def recommend_songs(mood, songs, mood_to_features):
    features = mood_to_features.get(mood)
    if not features:
        print(f"Unknown mood '{mood}'. Available moods are: {', '.join(mood_to_features.keys())}")
        return []

    valence_range = features["valence"]
    energy_range = features["energy"]

    # Filter songs based on the mood's feature ranges
    matching_songs = [
        song for song in songs
        if valence_range[0] <= song["valence"] <= valence_range[1]
        and energy_range[0] <= song["energy"] <= energy_range[1]
    ]

    # Limit to the top 8 matches (or fewer if not enough songs match)
    return matching_songs[:8]

# Step 4: Main script to generate a playlist
if __name__ == "__main__":
    try:
        # Step 4a: Load the dataset
        dataset_path = "./playlist_data-2.csv"  # Ensure the file name and path are correct
        songs = load_dataset(dataset_path)

        # Step 4b: User selects a mood
        mood = "chill"  # Change this to "happy" or "sad" for testing

        # Step 4c: Get recommendations
        playlist = recommend_songs(mood, songs, mood_to_features)

        # Step 4d: Display the playlist
        if playlist:
            print(f"\nPlaylist for mood '{mood}':")
            for song in playlist:
                print(f"- {song['track_name']} by {song['artist_name']}")
        else:
            print(f"No songs found for mood '{mood}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
