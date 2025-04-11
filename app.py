from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = FastAPI()

# Enable CORS so frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

# Load model and assessment data at startup
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_json("assessments.json")

@app.post("/recommend")
async def recommend(query: Query):
    job_text = query.query.strip()
    if not job_text or df.empty:
        return []

    descriptions = df["description"].astype(str).tolist()
    job_embedding = model.encode([job_text])
    assessment_embeddings = model.encode(descriptions)

    similarities = cosine_similarity(job_embedding, assessment_embeddings)[0]
    df_results = df.copy()
    df_results["similarity"] = similarities

    top_results = df_results.sort_values(by="similarity", ascending=False).head(5)
    return top_results.to_dict(orient="records")
