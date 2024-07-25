import pandas as pd

# Load the datasets
credits = pd.read_csv('dataset_fillm/credits.csv')
keywords = pd.read_csv('dataset_fillm/keywords.csv')
links = pd.read_csv('links.csv')
links_small = pd.read_csv('dataset_fillm/links_small.csv')
ratings_small = pd.read_csv('dataset_fillm/ratings_small.csv')
tmdb_5000_credits = pd.read_csv('dataset_fillm/tmdb_5000_credits.csv')
tmdb_5000_movies = pd.read_csv('dataset_fillm/tmdb_5000_movies.csv')
movies_metadata = pd.read_csv('dataset_fillm/movies_metadata.csv', low_memory=False)

# Print columns of each DataFrame to identify discrepancies
print("credits columns:", credits.columns)
print("keywords columns:", keywords.columns)
print("links columns:", links.columns)
print("links_small columns:", links_small.columns)
print("ratings_small columns:", ratings_small.columns)
print("tmdb_5000_credits columns:", tmdb_5000_credits.columns)
print("tmdb_5000_movies columns:", tmdb_5000_movies.columns)
print("movies_metadata columns:", movies_metadata.columns)
