import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("C:/Users/Hp/OneDrive/Documents/GitHub/CS_2025_27/Personal Practice Projects/Global Hack Week/Recommendation Engine/Data/TMDB_movie_dataset_v11.csv")

vectorizer = CountVectorizer(stop_words='english')
genre_matrix = vectorizer.fit_transform(movies['genres'].fillna(""))

def recommend_movies(movie_title, n=5):
    if movie_title not in movies['title'].values:
        return f"Movie '{movie_title}' not found in dataset."
    
    idx = movies[movies['title'] == movie_title].index[0]

    sim_scores = cosine_similarity(genre_matrix[idx], genre_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1][1:n+1]

    recommendations = movies.iloc[sim_indices][['title', 'genres', 'vote_average']]
    recommendations = recommendations.sort_values(by="vote_average", ascending=False)

    return recommendations.head(n)

def top_movies_by_genre(genre, n=5, min_votes=50):
    if "vote_count" in movies.columns:
        filtered = movies[movies['vote_count'] >= min_votes]
    else:
        filtered = movies

    genre_filtered = filtered[filtered['genres'].str.contains(genre, case=False, na=False)]
    
    if genre_filtered.empty:
        return f"No movies found for genre '{genre}'"
    
    return genre_filtered.sort_values(by="vote_average", ascending=False).head(n)[['title', 'vote_average']]

if __name__ == "__main__":
    movie_choice = input("Enter a movie title you like: ")
    print(f"Movies similar to '{movie_choice}'")
    print(recommend_movies(movie_choice, 5))

    genre_choice = input("Enter the genre you want top movies from: ")
    print(f"Movies in genre '{genre_choice}'")
    print(top_movies_by_genre(genre_choice, 5))