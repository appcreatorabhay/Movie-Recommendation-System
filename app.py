import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Define content-based recommender function
def content_based_recommender(movie_title, num_recommendations):
    # Check if the movie title exists in the dataset
    if movie_title not in movies_df['title'].values:
        return "Movie not found in the dataset."

    # Find the row index of the input movie title
    movie_index = movies_df.index[movies_df['title'] == movie_title].tolist()[0]

    # Extract genres of all movies
    genres = movies_df['genres']

    # Initialize CountVectorizer to convert text data into token counts
    count_vectorizer = CountVectorizer()
    genre_matrix = count_vectorizer.fit_transform(genres)

    # Calculate cosine similarity between the input movie and all other movies
    similarity_scores = cosine_similarity(genre_matrix, genre_matrix[movie_index])

    # Enumerate through similarity scores and keep track of movie indices
    movie_indices_scores = list(enumerate(similarity_scores))

    # Sort movie indices based on similarity scores
    sorted_movie_indices = sorted(movie_indices_scores, key=lambda x: x[1], reverse=True)

    # Exclude the input movie itself
    sorted_movie_indices = sorted_movie_indices[1:]

    # Recommend top N similar movies
    top_movie_indices = [index for index, _ in sorted_movie_indices[:num_recommendations]]
    recommended_movies = movies_df.iloc[top_movie_indices]

    return recommended_movies[['title', 'genres']]

# Define collaborative recommender function
def collaborative_recommender(user_id, num_recommendations, k):
    # Merge ratings and movies data
    merged_df = pd.merge(ratings_df, movies_df, on='movieId')

    # Create a user-item matrix
    user_movie_matrix = merged_df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    # Use Nearest Neighbors to find similar users
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_movie_matrix)

    # Find the index of the target user
    target_user_index = user_movie_matrix.index.get_loc(user_id)

    # Use kneighbors to find K similar users
    _, similar_users_indices = knn_model.kneighbors(user_movie_matrix.iloc[target_user_index].values.reshape(1, -1), n_neighbors=k+1)

    # Flatten the list of similar users indices
    similar_users_indices = similar_users_indices.flatten()

    # Get movies watched by the target user
    movies_watched = user_movie_matrix.iloc[target_user_index][user_movie_matrix.iloc[target_user_index] > 0].index.tolist()

    # Create a list of movies recommended by similar users, excluding movies already watched
    recommended_movies = []
    for similar_user_index in similar_users_indices:
        similar_user_movies = user_movie_matrix.iloc[similar_user_index][user_movie_matrix.iloc[similar_user_index] > 0].index.tolist()
        recommended_movies.extend([movie for movie in similar_user_movies if movie not in movies_watched])

    # Get top N recommended movies
    recommended_movies = list(set(recommended_movies))[:num_recommendations]

    return pd.DataFrame({'title': recommended_movies})

# Define popularity-based recommender function
def popularity_recommender(genre, min_reviews, num_recommendations):
    # Merge ratings and movies data
    merged_df = pd.merge(ratings_df, movies_df, on='movieId')

    # Filter by genre and minimum review threshold
    genre_movies = merged_df[(merged_df['genres'].str.contains(genre)) & (merged_df['rating'] >= min_reviews)]

    if genre_movies.empty:
        return "No movies found for the given genre and minimum review threshold."

    # Group by movie title and calculate mean rating
    movie_ratings = genre_movies.groupby('title')['rating'].mean()

    # Sort by ratings in descending order
    sorted_movies = movie_ratings.sort_values(ascending=False).reset_index()

    # Recommend top N movies
    top_movies = sorted_movies.head(num_recommendations)

    return top_movies

# Background image
image_file_path = "Movie_Recommendar.jpg"
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url("{image_file_path}") center center no-repeat;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Main content
st.title("Movie Recommender System")

# Sidebar with user input
st.sidebar.title("User Input")
model_choice = st.sidebar.selectbox("Choose a model", ("Popularity-based", "Content-based", "Collaborative-based"))

if model_choice == "Collaborative-based":
    user_id_input = st.sidebar.number_input("Enter User ID", min_value=1)
    num_recommendations_input = st.sidebar.number_input("Enter number of recommendations", min_value=1, max_value=10)
    k_input = st.sidebar.number_input("Enter value of K", min_value=1)

    # Get recommendations only if user inputs are provided
    if user_id_input and num_recommendations_input and k_input:
        recommendations = collaborative_recommender(user_id_input, num_recommendations_input, k_input)
        st.subheader("Recommendations")
        st.write(recommendations)

elif model_choice == "Content-based":
    movie_title = st.sidebar.selectbox("Select Movie Title", movies_df['title'].unique())
    num_recommendations_input = st.sidebar.number_input("Enter number of recommendations", min_value=1, max_value=10)

    # Get recommendations only if user inputs are provided
    if movie_title and num_recommendations_input:
        recommendations = content_based_recommender(movie_title, num_recommendations_input)
        st.subheader("Recommendations")
        st.write(recommendations)

else:  # Popularity-based
    genres_list = movies_df['genres'].str.split('|').explode().unique().tolist()
    genre = st.sidebar.selectbox("Select Genre", genres_list)
    min_reviews_threshold = st.sidebar.number_input("Enter Minimum Reviews Threshold", min_value=1)
    num_recommendations = st.sidebar.number_input("Enter number of recommendations", min_value=1, max_value=10)

    # Get recommendations only if user inputs are provided
    if genre and min_reviews_threshold and num_recommendations:
        recommendations = popularity_recommender(genre, min_reviews_threshold, num_recommendations)
        st.subheader("Recommendations")
        st.write(recommendations)
