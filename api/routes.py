import pandas as pd
import pickle
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from schemas import ContentRequest, ContentResponse
from pathlib import Path

router = APIRouter()
MODEL_PATH = Path("/app/models")
DATA_PATH = MODEL_PATH / "processed_data.csv"
VECTORIZER_PATH = MODEL_PATH / "tfidf_vectorizer.pkl"

# Read the data and load the model
if not DATA_PATH.exists() or not VECTORIZER_PATH.exists():
    raise FileNotFoundError("Model artifacts are missing. Run the Airflow training pipeline first.")

df = pd.read_csv(f"{MODEL_PATH}/processed_data.csv")
with VECTORIZER_PATH.open("rb") as f:
    tfidf = pickle.load(f)

# Endpoint router
@router.post("/recommendations", response_model=List[ContentResponse])
async def get_recommendations(request: ContentRequest):
    """Get content recommendations based on title"""
    try:
        idx = df[df['title'] == request.title].index[0]
        from sklearn.metrics.pairwise import cosine_similarity
        tfidf_matrix = tfidf.transform(df['combined_features'])
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[-request.n_recommendation-1:-1][::-1]

        return [
            {
                'title': df.iloc[i]['title'],
                'type': df.iloc[i]['type'],
                'genre': df.iloc[i]['listed_in'],
                'similarity_score': float(sim_scores[i])
            } for i in top_indices
        ]

    except IndexError:
        raise HTTPException(status_code=404, detail="Title not found in the dataset.")