import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Charger le dataset
manga_cleaned = pd.read_csv('manga_cleaned.csv')
manga_cleaned = manga_cleaned[['id', 'title', 'synopsis']].dropna().reset_index(drop=True)

# Fonction pour obtenir les embeddings avec sentence-transformers
def get_embeddings(text_list, model_name='all-MiniLM-L6-v2', batch_size=32):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_list, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings)

# Calculer ou charger les embeddings
embeddings_file = 'synopsis_embeddings.npy'

if os.path.exists(embeddings_file):
    try:
        synopsis_embeddings = np.load(embeddings_file)
        print("Loaded embeddings from file.")
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        print("Computing embeddings...")
        synopsis_embeddings = get_embeddings(manga_cleaned['synopsis'].tolist())
        np.save(embeddings_file, synopsis_embeddings)
        print("Saved embeddings to file.")
else:
    print("Computing embeddings...")
    synopsis_embeddings = get_embeddings(manga_cleaned['synopsis'].tolist())
    np.save(embeddings_file, synopsis_embeddings)
    print("Saved embeddings to file.")

# Construire ou charger l'index FAISS
index_file = 'faiss_index.bin'

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if os.path.exists(index_file):
    try:
        index = faiss.read_index(index_file)
        print("Loaded FAISS index from file.")
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        print("Building FAISS index...")
        index = build_faiss_index(synopsis_embeddings)
        faiss.write_index(index, index_file)
        print("Saved FAISS index to file.")
else:
    print("Building FAISS index...")
    index = build_faiss_index(synopsis_embeddings)
    faiss.write_index(index, index_file)
    print("Saved FAISS index to file.")

# Fonction pour obtenir des recommandations
def get_recommendations_from_prompt(prompt, index=index, n_recommendations=10):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    prompt_embedding = model.encode([prompt])
    distances, nearest_indices = index.search(np.array(prompt_embedding), n_recommendations)
    recommended_titles = manga_cleaned['title'].iloc[nearest_indices[0]]
    return recommended_titles
