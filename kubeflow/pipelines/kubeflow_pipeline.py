import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from kfp.dsl import Artifact, HTML
from typing import NamedTuple
from dotenv import load_dotenv
from pathlib import Path
import os

"""
Kubeflow Pipeline for Netflix Content Recommendation
====================================================
This pipeline handles:
1. Load Dataset
2. Preprocess Data
3. Train Model
4. Evaluate Model
5. Deploy Model
"""

# Define envrionment variable for MinIO S3 access
ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Path Configuration for DATA PATH
DATA_LOCAL_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "netflix_titles.csv"

# ============================================
# Component 1: Load Dataset
# ============================================
@component(base_image="mipjeiger/netflix-kfp-base:v1")
def load_data(
    data_path: str,
    output_data: Output[Dataset]
) -> NamedTuple('LoadDataOutputs', [('dataset_shape', str), ('columns', str)]):
    """Load Netflix dataset from source"""
    import pandas as pd
    import json
    import logging
    import os
    import requests
    from collections import namedtuple
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from: {data_path}")
    
    # Handle GitHub blob URLs
    if "github.com" in data_path and "/blob/" in data_path:
        data_path = data_path.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        logger.info(f"Converted to raw URL: {data_path}")
    
    try:
        # Load data from Minio/S3
        if data_path.startswith("s3://") or data_path.startswith("minio://"):
            s3_path = data_path.replace("minio://", "s3://")
            storage_options = {
                "key": os.getenv("AWS_ACCESS_KEY_ID"),
                "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "client_kwargs": {
                    "endpoint_url": os.getenv("MINIO_ENDPOINT", "http://minio:9000")
                }
            }
            df = pd.read_csv(s3_path, storage_options=storage_options)
        
        # Handle HTTP/HTTPS URLs
        elif data_path.startswith("http://") or data_path.startswith("https://"):
            # Try to read directly first
            try:
                df = pd.read_csv(data_path)
            except Exception as e:
                logger.warning(f"Direct read failed: {e}. Trying with requests...")
                response = requests.get(data_path)
                response.raise_for_status()

                # Use StringIO to read CSV from response text
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
        
        # Local file
        else:
            df = pd.read_csv(data_path)
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        # Try alternative method - download via requests
        if data_path.startswith("http"):
            logger.info("Attempting to download via requests...")
            import requests
            from io import StringIO
            response = requests.get(data_path)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
        else:
            raise
    
    # Save to KFP output artifact path
    df.to_csv(output_data.path, index=False)
    
    # Return metadata
    shape = df.shape
    columns = df.columns.tolist()
    
    logger.info(f"Data loaded successfully! Shape: {shape}")

    LoadDataOutputs = namedtuple('LoadDataOutputs', ['dataset_shape', 'columns'])
    return LoadDataOutputs(
        dataset_shape=json.dumps({"rows": shape[0], "cols": shape[1]}),
        columns=json.dumps(columns)
    )

# ============================================
# Component 2: Preprocess Data
# ============================================
@component(base_image="mipjeiger/netflix-kfp-base:v1")
def preprocess_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    output_features: Output[Artifact]
) -> NamedTuple('PreprocessOutputs', [('processed_shape', str), ('missing_values', str)]):
    """Preprocess Netflix data"""
    import pandas as pd
    import numpy as np
    import json
    import logging
    from collections import namedtuple
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load data
    df = pd.read_csv(input_data.path)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Handle missing values
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df['date_added'] = df['date_added'].fillna(method='ffill')
    df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
    df['duration'] = df['duration'].fillna(df['duration'].mode()[0])
    
    # Convert date
    df['date_added'] = df['date_added'].fillna(method='ffill')
    df['date_added'] = df['date_added'].astype(str).str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], format='mixed', errors='coerce')

    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    
    # Create combined features for similarity
    text_cols = ['director', 'cast', 'listed_in', 'description']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].astype(str).str.strip()

    df['combined_features'] = df.apply(
        lambda r: " ".join(
            [str(r[c]) for c in text_cols if c in r and str(r[c]).strip() != "Unknown" \
             and str(r[c]).strip() != "nan" and str(r[c]).strip() != "None"]
        ), axis=1
    )
    
    # Save processed data
    df.to_csv(output_data.path, index=False)
    
    # Save feature info
    feature_info = {
        'features': df['combined_features'].tolist()[:5],
        'feature_names': df.columns.tolist(),
        'n_samples': len(df)
    }
    with open(output_features.path, 'w') as f:
        json.dump(feature_info, f)
    
    # Missing values summary
    missing = df.isnull().sum().to_dict()
    
    logger.info(f"Preprocessed data shape: {df.shape}")
    logger.info(f"Missing values: {missing}")
    
    PreprocessOutputs = namedtuple('PreprocessOutputs', ['processed_shape', 'missing_values'])
    return PreprocessOutputs(
        processed_shape=json.dumps({"rows": df.shape[0], "cols": df.shape[1]}),
        missing_values=json.dumps(missing)
    )

# ============================================
# Component 3: Train Model
# ============================================
@component(base_image="mipjeiger/netflix-kfp-base:v1")
def train_model(
    processed_data: Input[Dataset],
    model_output: Output[Model],
    vectorizer_output: Output[Artifact],
    n_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.8
) -> NamedTuple('TrainModelOutputs', [('model_info', str), ('train_metrics', str)]):
    """Train TF-IDF recommendation model"""
    import os
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import pickle
    import json
    import mlflow
    import logging
    from collections import namedtuple
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load processed data
    df = pd.read_csv(processed_data.path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=n_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Save model artifacts
    os.makedirs(model_output.path, exist_ok=True)
    with open(f"{model_output.path}/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save similarity matrix (if not too large)
    if similarity_matrix.shape[0] <= 10000:
        np.save(f"{model_output.path}/similarity_matrix.npy", similarity_matrix)
    
    # Save processed data for recommendations
    df.to_csv(f"{model_output.path}/processed_data.csv", index=False)
    
    # Save vectorizer info
    vectorizer_info = {
        'n_features': n_features,
        'min_df': min_df,
        'max_df': max_df,
        'vocabulary_size': len(vectorizer.vocabulary_),
        'matrix_shape': tfidf_matrix.shape
    }
    with open(vectorizer_output.path, 'w') as f:
        json.dump(vectorizer_info, f)
    
    # Calculate metrics
    sparsity = 1 - (similarity_matrix > 0).sum().sum() / (similarity_matrix.shape[0] ** 2)
    avg_similarity = similarity_matrix[similarity_matrix < 1].mean()
    
    metrics = {
        'n_features': n_features,
        'vocabulary_size': len(vectorizer.vocabulary_),
        'sparsity': float(sparsity),
        'avg_similarity': float(avg_similarity),
        'n_samples': len(df),
        'n_components': tfidf_matrix.shape[1]
    }
    
    logger.info(f"Training metrics: {metrics}")

    TrainModelOutputs = namedtuple('TrainModelOutputs', ['model_info', 'train_metrics'])
    return TrainModelOutputs(
        json.dumps({'model_type': 'tfidf_cosine', 'status': 'success'}),
        json.dumps(metrics)
    )

# ============================================
# Component 4: Evaluate Model
# ============================================
@component(base_image="mipjeiger/netflix-kfp-base:v1")
def evaluate_model(
    processed_data: Input[Dataset],
    model_input: Input[Model],
    metrics_output: Output[Metrics],
    eval_report: Output[Artifact]
) -> NamedTuple('EvaluateOutputs', [('eval_metrics', str), ('status', str)]):
    """Evaluate the trained model"""
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import pickle
    import json
    import logging
    from collections import namedtuple
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load data
    df = pd.read_csv(processed_data.path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Load vectorizer
    model_path = model_input.path
    with open(f"{model_path}/tfidf_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Transform data
    tfidf_matrix = vectorizer.transform(df['combined_features'])
    
    # Calculate similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Evaluate on sample
    sample_size = min(100, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    
    precision_scores = []
    recall_scores = []
    
    for idx in sample_indices:
        sim_scores = similarity_matrix[idx]
        top_indices = np.argsort(sim_scores)[-6:-1]  # Top 5 (excluding self)
        
        # Check if same genre appears in recommendations
        actual_genre = df.iloc[idx]['listed_in']
        recommended_genres = df.iloc[top_indices]['listed_in'].values
        
        # Precision: fraction of recommendations with same genre
        precision = sum(1 for g in recommended_genres if g == actual_genre) / len(recommended_genres)
        precision_scores.append(precision)
        
        # Recall: fraction of relevant content found
        recall = sum(1 for g in recommended_genres if g == actual_genre) / 1  # Only 1 relevant per query
        recall_scores.append(recall)
    
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-8)
    
    # Prepare evaluation metrics
    eval_metrics = {
        'precision': float(avg_precision),
        'recall': float(avg_recall),
        'f1_score': float(f1_score),
        'sample_size': sample_size,
        'avg_similarity': float(np.mean(similarity_matrix[similarity_matrix < 1]))
    }
    
    # Save metrics
    with open(metrics_output.path, 'w') as f:
        json.dump(eval_metrics, f)
    
    # Save evaluation report
    report = {
        'metrics': eval_metrics,
        'model_type': 'tfidf_cosine',
        'evaluation_timestamp': pd.Timestamp.now().isoformat()
    }
    with open(eval_report.path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation metrics: {eval_metrics}")   
    status = 'success' if avg_precision > 0.5 else 'needs_improvement'

    EvaluateModelOutputs = namedtuple('EvaluateOutputs', ['eval_metrics', 'status'])
    return EvaluateModelOutputs(
        json.dumps(eval_metrics),
        status
    )

# ============================================
# Component 5: Deploy Model
# ============================================
@component(base_image="mipjeiger/netflix-kfp-base:v1")
def deploy_model(
    model_input: Input[Model],
    model_name: str = "netflix_content_model",
    deploy_target: str = "mlflow"
) -> str:
    """Deploy model to production"""
    import pickle
    import mlflow
    import os
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    model_path = model_input.path
    
    # Load vectorizer
    with open(f"{model_path}/tfidf_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Register with MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow.kubeflow.svc.cluster.local:5000'))
    mlflow.set_experiment("kubeflow_netflix")
    
    with mlflow.start_run(run_name="kubeflow_pipeline"):
        mlflow.sklearn.log_model(
            sk_model=vectorizer,
            artifact_path="tfidf_model",
            registered_model_name=model_name
        )
        
        # Log model info
        mlflow.log_params({
            'model_type': 'tfidf_cosine',
            'deployed_by': 'kubeflow'
        })
    
    logger.info(f"✅ Model '{model_name}' deployed successfully!")
    return f"Model {model_name} deployed to MLflow"

# ============================================
# Define the Pipeline
# ============================================
@dsl.pipeline(
    name="Netflix Content Recommendation Pipeline",
    description="End-to-end ML pipeline for Netflix content recommendations",
    pipeline_root="minio://mlpipeline/v2/artifacts"
)
def netflix_pipeline(
    data_path: str = "https://github.com/Mipjeiger/Netflix-Similarity-Search-Engine-MLOps-production-end-to-end/blob/main/data/raw/netflix_titles.csv",
    n_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.8,
    do_deploy: bool = True,  
    model_name: str = "netflix_content_model"
):
    """Kubeflow pipeline for Netflix content recommendation""" 
    
    # Step 1: Load Data
    load_task = load_data(
        data_path=data_path
    )
    
    # Step 2: Preprocess Data
    preprocess_task = preprocess_data(
        input_data=load_task.outputs['output_data']
    )
    
    # Step 3: Train Model
    train_task = train_model(
        processed_data=preprocess_task.outputs['output_data'],
        n_features=n_features,
        min_df=min_df,
        max_df=max_df
    )
    
    # Step 4: Evaluate Model
    evaluate_task = evaluate_model(
        processed_data=preprocess_task.outputs['output_data'],
        model_input=train_task.outputs['model_output']
    )
    
    # Step 5: Deploy Model (conditional)
    with dsl.Condition(
        evaluate_task.outputs['status'] == 'success',
        name="deploy_condition"
    ):
        with dsl.Condition(
            do_deploy == True,
            name="deploy_enabled"
        ):
            deploy_task = deploy_model(
                model_input=train_task.outputs['model_output'],
                model_name=model_name
            )

# ============================================
# Compile the Pipeline
# ============================================
if __name__ == "__main__":
    import kfp.compiler as compiler
    
    # Compile to YAML
    compiler.Compiler().compile(
        pipeline_func=netflix_pipeline,
        package_path="pipeline.yaml"
    )
    
    print("✅ Pipeline compiled successfully to pipeline.yaml")