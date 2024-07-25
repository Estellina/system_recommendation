import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
credits = pd.read_csv('dataset_fillm/credits.csv')
keywords = pd.read_csv('dataset_fillm/keywords.csv')
links = pd.read_csv('links.csv')
links_small = pd.read_csv('dataset_fillm/links_small.csv')
ratings_small = pd.read_csv('dataset_fillm/ratings_small.csv')
tmdb_5000_credits = pd.read_csv('dataset_fillm/tmdb_5000_credits.csv')
tmdb_5000_movies = pd.read_csv('dataset_fillm/tmdb_5000_movies.csv')
movies_metadata = pd.read_csv('dataset_fillm/movies_metadata.csv', low_memory=False)

# Ensure the 'id' columns are of the same type
movies_metadata['id'] = movies_metadata['id'].astype(str)
tmdb_5000_movies['id'] = tmdb_5000_movies['id'].astype(str)
tmdb_5000_credits['movie_id'] = tmdb_5000_credits['movie_id'].astype(str)

# Convert release_date to datetime
movies_metadata['release_date'] = pd.to_datetime(movies_metadata['release_date'], errors='coerce')
tmdb_5000_movies['release_date'] = pd.to_datetime(tmdb_5000_movies['release_date'], errors='coerce')

# Extract year from release_date
movies_metadata['year'] = movies_metadata['release_date'].dt.year
tmdb_5000_movies['year'] = tmdb_5000_movies['release_date'].dt.year

# Drop rows with missing years
movies_metadata = movies_metadata.dropna(subset=['year'])
tmdb_5000_movies = tmdb_5000_movies.dropna(subset=['year'])

# Distribution of movie ratings
plt.figure(figsize=(10, 6))
sns.histplot(ratings_small['rating'], bins=10, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Number of movies released per year
plt.figure(figsize=(14, 7))
movies_per_year = tmdb_5000_movies['year'].value_counts().sort_index()
sns.barplot(x=movies_per_year.index, y=movies_per_year.values, palette='viridis')
plt.xticks(rotation=90)
plt.title('Number of Movies Released per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# Average rating of movies per year
ratings_small['tmdbId'] = ratings_small['movieId'].map(links_small.set_index('movieId')['tmdbId'])
ratings_small['tmdbId'] = ratings_small['tmdbId'].astype(str)
avg_ratings_per_year = ratings_small.merge(movies_metadata[['id', 'year']], left_on='tmdbId', right_on='id')
avg_ratings_per_year = avg_ratings_per_year.groupby('year')['rating'].mean().reset_index()

plt.figure(figsize=(14, 7))
sns.lineplot(data=avg_ratings_per_year, x='year', y='rating', marker='o')
plt.title('Average Movie Rating per Year')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.show()

# Most frequent genres
tmdb_5000_movies['genres'] = tmdb_5000_movies['genres'].apply(lambda x: [i['name'] for i in eval(x)] if pd.notnull(x) else [])
all_genres = tmdb_5000_movies.explode('genres')['genres'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=all_genres.index, y=all_genres.values)
plt.title('Most Frequent Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Most frequent keywords
tmdb_5000_movies['keywords'] = tmdb_5000_movies['keywords'].apply(lambda x: [i['name'] for i in eval(x)] if pd.notnull(x) else [])
all_keywords = tmdb_5000_movies.explode('keywords')['keywords'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=all_keywords.index, y=all_keywords.values)
plt.title('Most Frequent Keywords')
plt.xlabel('Keyword')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Top cast members by number of movies
credits['cast'] = credits['cast'].apply(lambda x: [i['name'] for i in eval(x)])
all_cast = credits.explode('cast')['cast'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=all_cast.index, y=all_cast.values)
plt.title('Top Cast Members by Number of Movies')
plt.xlabel('Cast Member')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Top directors by number of movies
tmdb_5000_credits['directors'] = tmdb_5000_credits['crew'].apply(lambda x: [i['name'] for i in eval(x) if i['job'] == 'Director'])
all_directors = tmdb_5000_credits.explode('directors')['directors'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=all_directors.index, y=all_directors.values)
plt.title('Top Directors by Number of Movies')
plt.xlabel('Director')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
