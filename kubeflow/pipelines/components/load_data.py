import pandas as pd
import json
import logging
from kfp.dsl import component, Input, Output, Dataset

"""
Component: Load Data
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.26.2"]
)
def load_data(data_path: str, output_data: Output[Dataset]) -> str:
    """Load Netflix dataset"""
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        df.to_csv(output_data.path, index=False)
        logger.info(f"Data loaded and saved to {output_data.path}")
        return json.dumps({"rows": len(df), "columns": len(df.columns)})

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise