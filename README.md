# codsoft_taskno.4
# Movie Recommendation System

## Overview
This project is a Movie Recommendation System built using Python and Streamlit. The system recommends movies based on user preferences using collaborative filtering with cosine similarity.

## Features
- Input: User can select movies they like from a list.
- Output: The system recommends a movie based on the selected movies and their genres.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required dependencies:
    ```bash
    pip install pandas scikit-learn streamlit
    ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to interact with the Movie Recommendation System.

## Code Explanation
### Step 1: Dataset Creation
A dataset with real movie names, genres, and ratings is created:
```python
data = {
    'User': ['User1', 'User1', 'User1', 'User2', 'User2', 'User2', 'User3', 'User3', 'User4', 'User4', 'User4', 'User4'],
    'Movie': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'The Shawshank Redemption', 'The Godfather', 'Pulp Fiction', 'The Godfather', 'Pulp Fiction', 'The Shawshank Redemption', 'The Dark Knight', 'Pulp Fiction', 'The Lord of the Rings'],
    'Genre': ['Drama', 'Crime', 'Action', 'Drama', 'Crime', 'Crime', 'Crime', 'Crime', 'Drama', 'Action', 'Crime', 'Fantasy'],
    'Rating': [5, 4, 5, 5, 4, 3, 4, 5, 5, 5, 3, 5]
}

df = pd.DataFrame(data)
```

### Step 2: User-Item Matrix Creation
A pivot table is created to represent the user-item matrix:
```python
user_item_matrix = df.pivot_table(index='User', columns='Movie', values='Rating')
```

### Step 3: Calculate User Similarity
Cosine similarity is calculated to find similarities between users:
```python
user_item_matrix_filled = user_item_matrix.fillna(0)
user_similarity = cosine_similarity(user_item_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
```

### Step 4: Recommendation Function
A function is defined to make recommendations based on user likes:
```python
def get_recommendations(user_likes, user_item_matrix, user_similarity_df, movie_genre):
    # Function implementation here
```

### Step 5: Streamlit App
A simple Streamlit app is created to interact with the user:
```python
st.title('Movie Recommendation System')
movies = df['Movie'].unique()
user_likes = st.multiselect('Select movies you like', movies)

if user_likes:
    top_recommendation, score = get_recommendations(user_likes, user_item_matrix, user_similarity_df, movie_genre)
    if top_recommendation:
        st.write(f"Top recommendation based on your selection: {top_recommendation} (Score: {score})")
    else:
        st.write("No recommendations available based on your selection.")
```

## Contributing
Feel free to fork this project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
