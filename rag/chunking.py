# This file contains logic for splitting the resume text into smaller chunks

from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
    """
    Split text into chunks with overlap
     Why overlap?
    - If a skill/idea is near the boundary, overlap keeps context.

    chunk_size: approx number of characters per chunk (simple for now)
    overlap: how many characters to repeat between chunks
    Returns: list of chunks with metadata
    """

    # If text is empty, return no chunks

    if not text or not text.strip():
        return []
    
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        chunks.append({
            "chunk_text": chunk,
            "start_char": start,
            "end_char": end
        })

        start = end - overlap

        if start < 0:
            start = 0

        if end == text_length:
            break

    return chunks