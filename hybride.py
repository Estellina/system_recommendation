import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

# Chemin d'accès aux fichiers de données
ratings_path = r'C:\Users\ethan\OneDrive\Bureau\The movies dataset\ratings_small.csv'
movies_path = r'C:\Users\ethan\OneDrive\Bureau\The movies dataset\movies_metadata.csv'

# charger les données dans dataframe pandas pour eviter les erreurs de memoire
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path, low_memory=False)

# Assurer que "id" dans les films est le bon type
movies = movies[['id', 'title', 'genres']]
movies.dropna(subset=['id', 'genres'], inplace=True)  # assure qu'il n'y a pas de données "NaN" dans 'id' ou 'genres'
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')  # Converti 'id' en nombre
movies.dropna(subset=['id'], inplace=True)  # Supprime les données "NaN" dans 'id'
movies['id'] = movies['id'].astype(int)  # Convertis les valeurs de 'id' en entiers

# Traitement des genres
movies['genres'] = movies['genres'].apply(eval).apply(lambda x: [d['name'] for d in x] if isinstance(x, list) else [])
movies['genres'] = movies['genres'].apply(lambda x: '|'.join(x) if isinstance(x, list) else x)

# Creation de table pivot pour combler les valeurs par zero
ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Converti la table pivot en matrice
R = ratings_pivot.values

# Normalisation de la matrice
R_mean = np.mean(R, axis=1)
R_demeaned = R - R_mean.reshape(-1, 1)

# Matrice de décomposition en valeurs singulières
U, sigma, Vt = svds(R_demeaned, k=50)

# Convertir sigma en une matrice diagonale
sigma = np.diag(sigma)

# Construction de la matrice
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + R_mean.reshape(-1, 1)

# Convertir la matrice reconstruite en DataFrame
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_pivot.columns)

# Fonction pour obtenir des recommandations de films recupere les films non notés par l'utilisateur
def get_movie_recommendations(user_id, genre, num_recommendations=10):
    if user_id not in ratings_pivot.index:
        raise ValueError(f"L'ID de l'utilisateur {user_id} n'existe pas.")

    user_row_number = ratings_pivot.index.get_loc(user_id)
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)

    user_data = ratings[ratings.userId == user_id]
    user_full = (user_data.merge(movies, how='left', left_on='movieId', right_on='id').
                 sort_values(['rating'], ascending=False))

    recommendations = (movies[~movies['id'].isin(user_full['movieId']) & movies['genres'].str.contains(genre, case=False, na=False)].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                             left_on='id',
                             right_on='movieId').
                       rename(columns={user_row_number: 'Predictions'}).
                       sort_values('Predictions', ascending=False))

    return user_full, recommendations.head(num_recommendations)

# Fonction pour obtenir les trois genres les plus regardés par l'utilisateur
def get_top_genres(user_id):
    user_data = ratings[ratings.userId == user_id]
    user_movies = user_data.merge(movies, how='left', left_on='movieId', right_on='id')

    # Separe les genres et compte le nombre de fois qu'ils occurrent
    genre_counts = user_movies['genres'].str.split('|').explode().value_counts()
    return genre_counts.head(3).index.tolist()

# Fonction pour créer un compte utilisateur
def create_account():
    new_user_id = ratings['userId'].max() + 1
    print(f"Votre nouveau ID utilisateur est : {new_user_id}")
    return new_user_id

# Fonction pour noter un film
def rate_movie(user_id):
    movie_title = input("Veuillez entrer le nom du film : ").strip()
    movie = movies[movies['title'].str.contains(movie_title, case=False, na=False)]

    if movie.empty:
        print("Film introuvable.")
        return

    print("Films trouvés :")
    print(movie[['id', 'title']])

    try:
        movie_id = int(input("Veuillez entrer l'ID du film que vous voulez noter : "))
        if movie_id not in movie['id'].values:
            print("ID du film invalide.")
            return
        rating = float(input("Veuillez entrer votre note (sur 5) : "))
        if rating < 0 or rating > 5:
            print("Note invalide. Veuillez entrer une note entre 0 et 5.")
            return

        new_rating = pd.DataFrame({'userId': [user_id], 'movieId': [movie_id], 'rating': [rating]})
        global ratings
        ratings = pd.concat([ratings, new_rating], ignore_index=True)
        print("Film noté avec succès.")
    except ValueError:
        print("Entrée invalide.")

# Fonction pour la Partie interactive
def interactive_recommender():
    while True:
        try:
            has_account = input("Avez-vous un compte ? (o/n) : ").strip().lower()
            if has_account not in ['o', 'n']:
                raise ValueError("Entrée invalide. Veuillez entrer 'o' pour oui ou 'n' pour non.")
            break
        except ValueError as e:
            print(e)

    if has_account == 'o':
        while True:
            try:
                user_id = int(input("Veuillez entrer l'ID de l'utilisateur : "))
                if user_id not in ratings['userId'].unique():
                    raise ValueError(f"L'ID de l'utilisateur {user_id} n'existe pas.")
                break
            except ValueError as e:
                print(e)
                print("Veuillez entrer un ID utilisateur valide.")
    else:
        user_id = create_account()
        rate_movie(user_id)

    while True:
        print("\nQue voulez-vous faire ?")
        print("1. Noter un film")
        print("2. Obtenir des recommandations")
        try:
            choice = int(input("Entrez le numéro de votre choix : "))
            if choice not in [1, 2]:
                raise ValueError("Choix invalide. Veuillez entrer 1 ou 2.")
        except ValueError as e:
            print(e)
            continue

        if choice == 1:
            rate_movie(user_id)
        elif choice == 2:
            top_genres = get_top_genres(user_id)
            print(f"\nVoici le Top 3 des genres que tu as le plus regardé: {', '.join(top_genres)}")

            genres = movies['genres'].str.split('|').explode().unique()
            print("\nGenres disponibles:", ", ".join(genres))
            genre = input("Choisis un genre: ")

            already_rated, predictions = get_movie_recommendations(user_id, genre)
            print("\nFilms déjà notés par l'utilisateur:")
            print(already_rated[['title', 'genres', 'rating']])
            print("\nTop recommandations de films:")
            print(predictions[['title', 'genres', 'Predictions']])

        more = input("Voulez-vous continuer ? (o/n) : ").strip().lower()
        if more == 'n':
            break

# lancement de la commande interactive
interactive_recommender()