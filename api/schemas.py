from pydantic import BaseModel
from typing import Optional

class ContentRequest(BaseModel):
    title: str
    n_recommendations: Optional[int] = 5

class ContentResponse(BaseModel):
    title: str
    type: str
    genre: str
    similarity_score: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool