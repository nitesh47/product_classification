import pandas as pd
import os
import logging
from typing import Any, Dict
from tensorflow.python.lib.io import file_io
import yaml


def read_csv_file(data_path: str) -> pd.DataFrame:
    """
    This function read the CSV files

    """
    for filename in os.listdir(data_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_path, filename)
            data = pd.read_csv(filepath)
    return data


def load_model_params(model_params_path: str) -> Dict[str, Any]:
    """
    The purpose of this function is to load the model parameters from the provided path.
    """
    try:
        logging.info(f'Read model params file from: "{model_params_path}".')
        with file_io.FileIO(model_params_path, "r") as f:
            model_params = yaml.safe_load(f)
        if not model_params:
            raise Exception(
                f"Failed to load the model parameters file in the"
                f'following path: "{model_params_path}", the file was not found.'
            )
        logging.info(f"The model params:\n{model_params}")
        logging.info("Model parameters loaded successfully.")
        return model_params
    except Exception as e:
        logging.error(f"Error loading model parameters: {e}")
        raise
