import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from kfp.dsl import Artifact, HTML
from typing import NamedTuple
import json

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

# ============================================
# Component 1: Load Dataset
# ============================================
@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "gcsfs==2023.9.2"
    ]
)
def load_data(
    data_path: str,
    output_data: Output[Dataset]
) -> NamedTuple('LoadDataOutputs', [('dataset_shape', str), ('columns', str)]):
    """Load Netflix dataset from source"""
    import pandas as pd
    import json
    import logging
    from collections import namedtuple
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from: {data_path}")
    
    # Load data (can be from GCS, local, or URL)
    if data_path.startswith('gs://'):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(data_path) as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(data_path)
    
    # Save to output
    df.to_csv(output_data.path, index=False)
    
    # Return metadata
    shape = df.shape
    columns = df.columns.tolist()
    
    logger.info(f"Data loaded successfully! Shape: {shape}")
    logger.info(f"Columns: {columns[:5]}...")

    LoadDataOutputs = namedtuple('LoadDataOutputs', ['dataset_shape', 'columns'])
    return LoadDataOutputs(
        dataset_shape=json.dumps({"rows": shape[0], "cols": shape[1]}),
        columns=json.dumps(columns)
    )

# ============================================
# Component 2: Preprocess Data
# ============================================
@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.5.2"
    ]
)
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
    df['date_added'] = pd.to_datetime(df['date_added'])
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    
    # Create combined features for similarity
    text_cols = ['director', 'cast', 'listed_in', 'description']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    df['combined_features'] = df.apply(
        lambda r: " ".join(
            [str(r[c]) for c in text_cols if c in r and str(r[c]).strip() != "Unknown"]
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
@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.5.2",
        "mlflow==2.6.0"
    ]
)
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
@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.5.2"
    ]
)
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
@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "mlflow==2.6.0",
        "scikit-learn==1.5.2",
    ]
)
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
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
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
    pipeline_root="gs://your-bucket/kubeflow-pipelines/netflix"
)
def netflix_pipeline(
    data_path: str = "gs://netflix-data/netflix_titles.csv",
    n_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.8,
    do_deploy: bool = True,  # 👈 Fixed: Changed name from deploy_model to do_deploy
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