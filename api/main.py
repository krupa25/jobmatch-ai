# FastAPI is a python framework to build APIS
from fastapi import FastAPI

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