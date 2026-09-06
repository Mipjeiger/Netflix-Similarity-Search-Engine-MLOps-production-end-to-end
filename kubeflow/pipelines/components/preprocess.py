import pandas as pd
import json
import logging
from kfp.dsl import component, Input, Output, Dataset

"""
Component: Preprocess Data
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.26.2"]
)
def preprocess_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset]
) -> str:
    """Preprocess Netflix dataset"""
    try:
        df = pd.read_csv(input_data.path)

        # Handle missing values
        df['director'].fillna('Unknown', inplace=True)
        df['cast'].fillna('Unknown', inplace=True)
        df['country'].fillna('Unknown', inplace=True)
        df['rating'].fillna(df['rating'].mode()[0], inplace=True)

        # Create combined features
        df['combined_features'] = df.apply(
            lambda r: ' '.join([
                str(r['director']) if r['director'] != 'Unknown' else '',
                str(r['cast']) if r['cast'] != 'Unknown' else '',
                str(r['listed_in']) if r['listed_in'] != 'Unknown' else '',
                str(r['description']) if r['description'] != 'Unknown' else ''
            ]), axis=1
        )

        df.to_csv(output_data.path, index=False)
        logger.info(f"Preprocessed {len(df)} rows")

        return json.dumps({"rows": len(df), "columns": len(df.columns)})

    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise