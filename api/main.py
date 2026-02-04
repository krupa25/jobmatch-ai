# FastAPI is a python framework to build APIS
from fastapi import FastAPI
from fastapi import UploadFile, File
import tempfile
import os
from api.services.resume_parser import parse_resume_pdf
import uuid
from pydantic import BaseModel
from rag.chunking import chunk_text
from rag.embeddings import embed_texts
from rag.vector_store import (
    save_embeddings_and_metadata,
    load_embeddings_and_metadata,
    cosine_search,
)

# Creating an app object
# This is the main Fast API application

app = FastAPI(title="JobMatch AI")

# Simple endpoint to check if the server is running
# /health is used commonly in production for monitoring the API

@app.get("/health")
def health_check():
    return {"status":"ok"}

# Dummy Endpoint for day 1
# Later will replace it with the actual code

@app.post("/api/analyze")

def analyze_dummy():
    return {
        "match_score" : 72,
        "strengths" : ["Python", "Machine Learning"],
        "gaps" : ["Docker", "CI/CD"],
        "message" : "Day 1 , Sample response for API Trial"
    }

# This endpoint accepts a PDF resume and returns extracted text + sections
@app.post("/api/parse_resume")
async def parse_resume_endpoint(resume: UploadFile = File(...)):
    """
    1) Receive uploaded PDF
    2) Save to temp file
    3) Parse text + sections
    4) Delete temp file
    5) Return JSON
    """

    # Only allow PDF files
    if not resume.filename.lower().endswith(".pdf"):
        return {"error": "Please upload a PDF file only."}

    # Create a temp file to store the uploaded PDF bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        # Read uploaded file bytes and write to temp file
        tmp.write(await resume.read())
        temp_path = tmp.name  # save path

    try:
        # Parse the resume using our service
        result = parse_resume_pdf(temp_path)
        return result

    finally:
        # Always delete temp file to keep system clean
        os.remove(temp_path)


class IndexRequest(BaseModel):
    raw_text: str  # resume text from Day 2 output

class RetrieveRequest(BaseModel):
    resume_id: str
    job_description: str
    top_k: int = 5

@app.post("/api/index_resume")
def index_resume(req: IndexRequest):
    """
    Takes resume text, chunks it, creates embeddings,
    builds a FAISS index, and saves it to disk.
    """
    # Create a unique ID for this resume
    resume_id = str(uuid.uuid4())

    # Chunk the resume text
    chunks = chunk_text(req.raw_text)

    # Convert chunk texts into embeddings
    texts = [c["chunk_text"] for c in chunks]
    embeddings = embed_texts(texts)

    # Build FAISS index
    #index = build_faiss_index(embeddings)

    # Save embeddings + metadata (instead of FAISS index)
    save_embeddings_and_metadata(resume_id, embeddings, chunks)

    return {"resume_id": resume_id, "num_chunks": len(chunks)}

@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest):
    """
    Takes a resume_id and job description,
    searches the resume chunks and returns top-k relevant ones.
    """
   # Load stored embeddings + metadata
    embeddings, chunks = load_embeddings_and_metadata(req.resume_id)

    # Embed job description
    query_vec = embed_texts([req.job_description])  # shape (1, dim)

    # Search using cosine similarity
    scores, idxs = cosine_search(embeddings, query_vec, top_k=req.top_k)


    results = []
    for score, i in zip(scores, idxs):
        if i == -1:
            continue
        results.append({
            "score": float(score),
            "chunk": chunks[i]["chunk_text"],
            "start_char": chunks[i]["start_char"],
            "end_char": chunks[i]["end_char"],
        })

    return {"resume_id": req.resume_id, "results": results}
