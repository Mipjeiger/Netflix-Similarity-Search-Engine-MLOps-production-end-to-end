import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
import logging
from kfp.dsl import component, Input, Output, Dataset, Model

"""Component: Train Model
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.26.2", "scikit-learn==1.5.2"]
)
def train_model(
    processed_data: Input[Dataset],
    model_output: Output[Model],
    n_features: int = 5000
) -> str:
    """Train TF-IDF model"""
    df = pd.read_csv(processed_data.path)

    # Define TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=n_features,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

    # Save model
    with open(f"{model_output.path}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    df.to_csv(f"{model_output.path}/processed_data.csv", index=False)
    logger.info(f"Model trained and saved to {model_output.path}")

    return json.dumps({
        'vocabulary_size': len(vectorizer.vocabulary_),
        'matrix_shape': tfidf_matrix.shape
    })