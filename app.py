import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Step 1: Dataset with real movie names and genres
data = {
    'User': ['User1', 'User1', 'User1', 'User2', 'User2', 'User2', 'User3', 'User3', 'User4', 'User4', 'User4', 'User4'],
    'Movie': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'The Shawshank Redemption', 'The Godfather', 'Pulp Fiction', 'The Godfather', 'Pulp Fiction', 'The Shawshank Redemption', 'The Dark Knight', 'Pulp Fiction', 'The Lord of the Rings'],
    'Genre': ['Drama', 'Crime', 'Action', 'Drama', 'Crime', 'Crime', 'Crime', 'Crime', 'Drama', 'Action', 'Crime', 'Fantasy'],
    'Rating': [5, 4, 5, 5, 4, 3, 4, 5, 5, 5, 3, 5]
}

df = pd.DataFrame(data)

# Create a dictionary to map movies to genres
movie_genre = df.set_index('Movie')['Genre'].to_dict()

# Step 2: Create the User-Item Matrix
user_item_matrix = df.pivot_table(index='User', columns='Movie', values='Rating')

# Step 3: Calculate Similarity
# Filling NaN values with 0 for the similarity calculation
user_item_matrix_filled = user_item_matrix.fillna(0)

# Calculating the cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix_filled)

# Creating a DataFrame for the similarity matrix
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Step 4: Function to Make Recommendations based on genre
def get_recommendations(user_likes, user_item_matrix, user_similarity_df, movie_genre):
    # Create a new row for the target user with their selected movies
    target_user_ratings = pd.Series(index=user_item_matrix.columns)
    for movie in user_likes:
        target_user_ratings[movie] = 5  # Assuming the user likes the movie with a high rating
    
    # Add the new user ratings to the user-item matrix
    user_item_matrix = pd.concat([user_item_matrix, target_user_ratings.to_frame().T], ignore_index=True)
    user_item_matrix = user_item_matrix.fillna(0)
    
    # Calculate the similarity with the new user
    new_user_similarity = cosine_similarity(user_item_matrix)
    
    # Creating a DataFrame for the new similarity matrix
    new_user_similarity_df = pd.DataFrame(new_user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # Calculate the weighted sum of ratings for each movie
    weighted_ratings = user_item_matrix.T.dot(new_user_similarity_df.iloc[-1])
    
    # Normalize by the sum of the similarity scores
    recommendation_scores = weighted_ratings / new_user_similarity_df.iloc[-1].sum()
    
    # Filter out movies that the target user has already rated
    recommendations = recommendation_scores[target_user_ratings.isnull()]
    
    # Sort the recommendations by score
    recommendations = recommendations.sort_values(ascending=False)
    
    # Get the genre of the liked movies
    liked_genres = [movie_genre[movie] for movie in user_likes]
    
    # Recommend movies that match the liked genres
    genre_recommendations = recommendations[recommendations.index.map(lambda x: movie_genre[x] in liked_genres)]
    
    # Recommend the movie with the highest score
    if not genre_recommendations.empty:
        top_recommendation = genre_recommendations.idxmax()
        top_recommendation_score = genre_recommendations.max()
        return top_recommendation, top_recommendation_score
    else:
        return None, None

# Step 5: Streamlit App
st.title('Movie Recommendation System')

# Movie selection
movies = df['Movie'].unique()
user_likes = st.multiselect('Select movies you like', movies)

if user_likes:
    top_recommendation, score = get_recommendations(user_likes, user_item_matrix, user_similarity_df, movie_genre)
    if top_recommendation:
        st.write(f"Top recommendation based on your selection: {top_recommendation} (Score: {score})")
    else:
        st.write("No recommendations available based on your selection.")