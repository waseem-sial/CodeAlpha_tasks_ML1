import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('spotify_dataset.csv')

# Inspect the data
print("Dataset Head:")
print(data.head())

# Create a user-item matrix
user_song_matrix = data.pivot_table(index='user_id', columns='song_id', values='play_count', fill_value=0)

# Calculate the cosine similarity between users
user_similarity = cosine_similarity(user_song_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_song_matrix.index, columns=user_song_matrix.index)

# Function to get song recommendations for a user
def recommend_songs(user_id, user_song_matrix, user_similarity_df, num_recommendations=10):
    # Get the similarity scores for the user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # Get the songs played by the similar users
    similar_users_songs = user_song_matrix.loc[similar_users.index]

    # Sum the play counts for the songs and sort them
    song_recommendations = similar_users_songs.sum(axis=0).sort_values(ascending=False)

    # Filter out the songs already played by the user
    user_songs = user_song_matrix.loc[user_id]
    song_recommendations = song_recommendations[user_songs[user_songs > 0].index.to_list()]

    return song_recommendations.head(num_recommendations).index.tolist()

# Example recommendation for a user
user_id = 'user_1'
recommendations = recommend_songs(user_id, user_song_matrix, user_similarity_df, num_recommendations=3)
print(f"Recommendations for user {user_id}: {recommendations}")

