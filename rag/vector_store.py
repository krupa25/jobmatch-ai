# rag/vector_store.py
# ---------------------------------------------
# Mini Vector Store (FAISS replacement)
# ---------------------------------------------
# On Apple Silicon, faiss-cpu via pip can cause Rosetta/signing issues.
# This file implements the SAME idea as FAISS:
# - store embeddings
# - search by cosine similarity
#
# It is slower than FAISS, but PERFECT for:
# - learning
# - small datasets (like one resume)
# - reliable local development on Mac
# ---------------------------------------------

import os
import json
from typing import List, Dict, Tuple

import numpy as np


def save_embeddings_and_metadata(
    resume_id: str,
    embeddings: np.ndarray,
    chunks: List[Dict],
    base_dir: str = "storage/vectors",
) -> None:
    """
    Save embeddings + chunk metadata to disk.

    embeddings: numpy array shape (n_chunks, dim)
    chunks: list of dicts containing chunk_text + offsets
    """
    os.makedirs(base_dir, exist_ok=True)

    # Save embeddings as .npy (fast numpy format)
    emb_path = os.path.join(base_dir, f"{resume_id}.npy")
    np.save(emb_path, embeddings)

    # Save metadata as JSON
    meta_path = os.path.join(base_dir, f"{resume_id}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_embeddings_and_metadata(
    resume_id: str,
    base_dir: str = "storage/vectors",
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load embeddings + chunk metadata from disk.
    """
    emb_path = os.path.join(base_dir, f"{resume_id}.npy")
    meta_path = os.path.join(base_dir, f"{resume_id}.json")

    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Embeddings or metadata not found for this resume_id")

    embeddings = np.load(emb_path)  # shape (n_chunks, dim)

    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return embeddings, chunks


def cosine_search(
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform cosine similarity search.

    embeddings: shape (n_chunks, dim) (assumed normalized)
    query_vec: shape (1, dim) (assumed normalized)
    Returns:
      scores: shape (top_k,)
      indices: shape (top_k,)
    """

    # Cosine similarity between normalized vectors is just dot product
    # scores_all shape -> (n_chunks,)
    scores_all = embeddings @ query_vec[0]

    # Get top_k indices (largest scores)
    top_k = min(top_k, embeddings.shape[0])
    idxs = np.argsort(scores_all)[::-1][:top_k]

    scores = scores_all[idxs]
    return scores, idxs
