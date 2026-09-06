import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
import logging
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

"""
Component: Evaluate Model
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.26.2", "scikit-learn==1.5.2"]
)
def evaluate_model(
    processed_data: Input[Dataset],
    model_input: Input[Model],
    metrics_output: Output[Metrics]
) -> str:
    """Evaluate model performance"""
    df = pd.read_csv(processed_data.path)

    with open(f"{model_input.path}/tfidf_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)

    tfidf_matrix = vectorizer.transform(df["combined_features"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Calculate matrix
    sample_size = min(100, len(df))
    indices = np.random.choice(len(df), sample_size, replace=False)

    precision_scores = []
    for idx in indices:
        sim_scores = similarity_matrix[idx]
        top_indices = np.argsort(sim_scores)[-6:-1]  # Get top 5 similar items excluding itself
        actual_genre = df.iloc[idx]["listed_in"]
        recommended_genres = df.iloc[top_indices]["listed_in"].values
        precision = sum(1 for g in recommended_genres if g == actual_genre) / len(recommended_genres)
        precision_scores.append(precision)

    avg_precision = np.mean(precision_scores)

    # Save metrics
    with open(metrics_output.path, 'w') as f:
        json.dump({'precision': float(avg_precision)}, f)
    logger.info(f"Model evaluation completed with average precision: {avg_precision}")

    return json.dumps({"avg_precision": float(avg_precision)})