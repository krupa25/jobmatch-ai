import streamlit as st
import requests


st.set_page_config(
    page_title="JobMatch AI",
    layout="centered"
)

st.title("JobMatch AI")
st.write("Day 1: Connecting Streamlit UI to FastAPI backend")

API_BASE = "http://127.0.0.1:8000"

st.subheader("1) Test Backend Health")

#If user clicks this button, we GET the /health

if st.button("Check /health"):
    try:
        r = requests.get(f"{API_BASE}/health", timeout = 10)
        data = r.json()

        st.success("Backend is running")
        st.json(data)

    except Exception as e:

        st.error("Could not reach backend")
        st.write("Make sure FastAPI is running on port 8000")
        st.write(f"Error: {e}")


st.subheader("2) Test /api/analyze (Dummy)")

st.write("This will call POST /api/analyze and show the dummy JSON response")


if st.button("Run Dummy Analyze"):
    try:
        r = requests.post(f"{API_BASE}/api/analyze", timeout=30)

        data = r.json()

        st.success("Analyze endpoint worked")
        st.json(data)

    except Exception as e:
        st.error("Analyze endpoint failed")
        st.write("Make sure FastAPI is running, and /api/analyze is a POST endpoint")
        st.write(f"Error: {e}")