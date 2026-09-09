from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle
import json
import os
import logging
import sys
from io import StringIO

sys.path.append('/opt/airflow')
from src.data.preprocess import preprocess_netflix_data

"""
Netflix Content Model Training Pipeline
Weekly retraining with MLflow tracking (Optional)
This Airflow DAG defines training and evaluate the task manually by airflow trigger CLI command
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email': ['cubudida@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1
}

# Configuration - robust Variable.get with env fallback (variables may not exist at parse time)
def _get_var(name, default):
    try:
        return Variable.get(name, default_var=default)
    except Exception:
        return os.getenv(name.upper(), default)

MLFLOW_TRACKING_URI = _get_var('mlflow_tracking_uri', os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
MODEL_NAME = _get_var('model_name', os.getenv('MODEL_NAME', 'netflix_content_model'))
DATA_PATH = _get_var('data_path', '/opt/airflow/data/raw/netflix_titles.csv')
PROCESSED_PATH = _get_var('processed_data_path', '/opt/airflow/data/processed')
MODEL_PATH = _get_var('model_path', '/opt/airflow/data/models')

def extract_data(**context):
    """Extract data from the source CSV file."""
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    logger.info(f"✅ Source data file found at {DATA_PATH}. Extracting...")
    return DATA_PATH

def preprocess_data(**context):
    """Preprocess the extracted data."""
    ti = context['task_instance']

    # Try pulling key-specific XCom or general return value
    raw_csv_path = ti.xcom_pull(task_ids='extract_data', key='return_value')
    if not raw_csv_path:
        raw_csv_path = ti.xcom_pull(task_ids='extract_data')

    # Fallback to default DATA_PATH if XCom is empty
    if not raw_csv_path or not os.path.exists(str(raw_csv_path)):
        logger.warning(f"XCom for raw data path is empty or file does not exist. Falling back to default DATA_PATH: {DATA_PATH}")
        raw_csv_path = DATA_PATH

    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Input dataset not found at path: {raw_csv_path}")

    # Read dataset from verified path
    df = pd.read_csv(raw_csv_path)
    df_processed = preprocess_netflix_data(df)

    # Save processed data output
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    output_filepath = os.path.join(PROCESSED_PATH, "netflix_processed.csv")
    df_processed.to_csv(output_filepath, index=False)

    ti.xcom_push(key='processed_data_path', value=output_filepath)
    logger.info(f"✅ Processed data saved successfully to {output_filepath}")
    return output_filepath

def train_model(**context):
    """Train the content-based recommendation model."""
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Ensure artifact mount exists and is writable
    for p in ["/mlflow", "/mlflow/artifacts", MODEL_PATH]:
        try:
            os.makedirs(p, exist_ok=True)
        except PermissionError as e:
            logger.warning(f"Could not mkdir {p}: {e}")

    mlflow.set_experiment("netflix_content_recommendation")
    ti = context['task_instance']
    processed_path = ti.xcom_pull(task_ids='preprocess_data', key='processed_data_path')

    if not processed_path or not os.path.exists(processed_path):
        processed_path = os.path.join(PROCESSED_PATH, "netflix_processed.csv")
         
    # Read json data into DataFrame
    df = pd.read_csv(processed_path)

    # Mlflow training
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(df['combined_features'].fillna(''))
        similarity = cosine_similarity(tfidf_matrix)

        # Mlflow logging
        mlflow.log_params({'max_features': 5000, 'train_size': len(df)})
        mlflow.log_metrics({'sparsity': 1 - (similarity > 0).sum().sum() / (similarity.shape[0]**2)})

        os.makedirs(MODEL_PATH, exist_ok=True)
        with open(f"{MODEL_PATH}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(tfidf, f)

        # Mlflow sklearn model logging
        mlflow.sklearn.log_model(tfidf, name="tfidf_model", registered_model_name=MODEL_NAME)
        run_id = run.info.run_id
        ti.xcom_push(key='mlflow_run_id', value=run_id)
        return run_id

def evaluate_model(**context):
    """Evaluate trained vectorizer precision."""
    ti = context['task_instance']
    run_id = ti.xcom_pull(task_ids='train_model', key='mlflow_run_id')

    if not run_id:
        logger.error("No MLflow run ID found. Skipping evaluation.")
        return

    processed_file = os.path.join(PROCESSED_PATH, "netflix_processed.csv")
    df = pd.read_csv(processed_file)
    
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', MLFLOW_TRACKING_URI))
    client = mlflow.tracking.MlflowClient()
    
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
    except Exception as e:
        logger.error(f"Failed to fetch model versions: {e}")
        return

    if not versions:
        logger.error("No registered model versions found. Skipping evaluation.")
        return
    
    latest_ver = versions[0].version
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest_ver}")
    
    sample = df.sample(min(100, len(df)), random_state=42)
    tfidf_matrix = model.transform(sample['combined_features'].fillna(''))
    sim = cosine_similarity(tfidf_matrix)
    
    precision = np.mean([
        1 if sample.iloc[i]['listed_in'] in sample.iloc[np.argsort(sim[i])[-6:-1]]['listed_in'].values else 0 
        for i in range(len(sample))
    ])

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric('evaluation_precision', float(precision))
        logger.info(f"✅ Model evaluation completed. Precision@5: {precision:.4f}")
        
    ti.xcom_push(key='avg_precision', value=precision)
    return precision

def promote_model(**context):
    """Promote top model version using MLflow Aliases."""
    ti = context['task_instance']
    precision = ti.xcom_pull(task_ids='evaluate_model', key='avg_precision')

    if precision is not None and precision >= 0.6:
        client = mlflow.tracking.MlflowClient()
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', MLFLOW_TRACKING_URI))
        
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

        if versions:
            latest_version = versions[0].version
            try:
                # MLflow 3.x standard alias promotion
                client.set_registered_model_alias(MODEL_NAME, "production", latest_version)
                logger.info(f"✅ Model version {latest_version} promoted to alias 'production'")
            except Exception:
                # Fallback for legacy MLflow stages
                client.transition_model_version_stage(MODEL_NAME, latest_version, stage='Production')
                logger.info(f"✅ Model version {latest_version} transitioned to Production stage")

def generate_report(**context):
    """Write run summary report to disk."""
    ti = context['task_instance']
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'run_id': ti.xcom_pull(task_ids='train_model', key='mlflow_run_id'),
        'precision': ti.xcom_pull(task_ids='evaluate_model', key='avg_precision'),
        'promoted': 'Yes' if ti.xcom_pull(task_ids='promote_model') else 'No'
    }
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    report_path = f"{MODEL_PATH}/metrics_report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"✅ Report saved to {report_path}")

# DAG Configuration
with DAG(
    'netflix_model_training_pipeline',
    default_args=default_args,
    description='A weekly training pipeline for Netflix content recommendation model',
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=['netflix', 'mlflow', 'training']
) as dag:

    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data
    )

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )

    promote = PythonOperator(
        task_id='promote_model',
        python_callable=promote_model
    )

    report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report
    )

    # Task Pipeline Pipeline execution order
    extract >> preprocess >> train >> evaluate >> promote >> report