# FastAPI is a python framework to build APIS
from fastapi import FastAPI
from fastapi import UploadFile, File
import tempfile
import os
from api.services.resume_parser import parse_resume_pdf

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


