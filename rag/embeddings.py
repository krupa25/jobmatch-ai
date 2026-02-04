# This file creates embeddings (vectors) for text chunks

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    embeddings = MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)