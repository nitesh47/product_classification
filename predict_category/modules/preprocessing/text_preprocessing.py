import logging
import re
import numpy as np
import pandas as pd
import os
import warnings
from predict_category.modules.common.constants import (
    FEATURE_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    TITLE,
    QUERY,
)

warnings.filterwarnings("ignore")


class DataProcessor:
    """
    DataProcessor handles the reading, cleaning, and saving of data from JSON log files
    for machine learning training and testing.
    """

    def __init__(self):
        self.df = None
        self.test_data = None
        self.training_data = None

    def read_json_file(self, data_path):
        try:
            logging.info("Reading JSON log file...")
            for filename in os.listdir(data_path):
                if filename.endswith(".log"):
                    filepath = os.path.join(data_path, filename)
                    self.df = pd.read_json(filepath, lines=True)
                    logging.info(f"Loaded data from {filename}")
            logging.info(f"Length of the dataframe: {len(self.df)}")
            if self.df is None or self.df.empty:
                raise ValueError("No log file found or the file is empty.")
        except Exception as e:
            logging.error(f"Error reading JSON file: {e}")
            raise

    def split_test_and_training_data(self):
        try:
            logging.info("Splitting data into test and training sets...")
            if LABEL_COLUMN_NAME not in self.df.columns:
                raise ValueError(f"{LABEL_COLUMN_NAME} not found in the DataFrame.")
            self.test_data = self.df.loc[pd.isna(self.df[LABEL_COLUMN_NAME])]
            logging.info(f"Length of the test dataframe: {len(self.test_data)}")
            self.training_data = self.df.loc[pd.notna(self.df[LABEL_COLUMN_NAME])]
            logging.info(f"Length of the training dataframe: {len(self.training_data)}")
            if self.test_data.empty:
                logging.warning("No test data available.")
            if self.training_data.empty:
                logging.warning("No training data available.")
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise

    def clean_labels(self):
        try:
            logging.info("Cleaning labels...")
            logging.info(
                f"Length of training dataframe before "
                f"cleaning labels: {len(self.training_data)}"
            )
            self.training_data[LABEL_COLUMN_NAME] = (
                self.training_data[LABEL_COLUMN_NAME]
                .replace("", np.nan)
                .dropna()
                .str.lower()
                .str.strip()
            )
            logging.info(
                f"Length of training dataframe after "
                f"cleaning labels: {len(self.training_data)}"
            )
        except Exception as e:
            logging.error(f"Error cleaning labels: {e}")
            raise

    def text_preprocessing(self, text: str) -> str:
        try:
            text = text.strip().casefold()
            text = re.sub(r"[^A-Za-z\s]", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        except Exception as e:
            logging.error(f"Error during text preprocessing: {e}")
            raise

    def apply_text_preprocessing(self, data_types: list):
        try:
            for data_type in data_types:
                if data_type == "training":
                    data = self.training_data
                    column_name = TITLE
                    logging.info(
                        f"Preprocessing text in column: "
                        f"{column_name} for training data..."
                    )
                elif data_type == "test":
                    data = self.test_data
                    column_name = QUERY
                    logging.info(
                        f"Preprocessing text in column: {column_name} for test data..."
                    )
                else:
                    raise ValueError("data_type must be either 'training' or 'test'")

                data[column_name] = data[column_name].replace("", np.nan)
                data = data.dropna(subset=[column_name])
                data[FEATURE_COLUMN_NAME] = data[column_name].apply(
                    self.text_preprocessing
                )
                data[FEATURE_COLUMN_NAME] = data[FEATURE_COLUMN_NAME].replace(
                    "", np.nan
                )
                data = data.dropna(subset=[FEATURE_COLUMN_NAME])
                if data_type == "training":
                    self.training_data = data
                else:
                    self.test_data = data

        except Exception as e:
            logging.error(f"Error applying text preprocessing to {data_type} data: {e}")
            raise

    def save_dataframes(self, output_dir):
        try:
            os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
            test_data_path = os.path.join(output_dir, "test", "search_data.csv")
            train_data_path = os.path.join(
                output_dir, "train", "productview_training_data.csv"
            )
            if not self.test_data.empty:
                self.test_data.to_csv(test_data_path, index=False)
                logging.info(f"Saved test data to {test_data_path}...")
            else:
                logging.warning("No test data to save.")

            if not self.training_data.empty:
                self.training_data.to_csv(train_data_path, index=False)
                logging.info(
                    f"Saved preprocessed training data to {train_data_path}..."
                )
            else:
                logging.warning("No training data to save.")
        except Exception as e:
            logging.error(f"Error saving dataframes: {e}")
            raise
