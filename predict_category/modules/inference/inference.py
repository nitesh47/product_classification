from typing import Any
import tensorflow as tf
import tensorflow_hub as hub
import logging
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.lib.io import file_io
import joblib
import numpy as np
import os
from predict_category.modules.common.functions import read_csv_file
from predict_category.modules.common.functions import load_model_params
from predict_category.modules.training.train import ModelTrainer
from predict_category.modules.common.constants import (
    ENCODING,
    MAX_LENGTH,
    INFERENCE,
    BATCH_SIZE,
    LABEL_ENCODER_PKL,
    FEATURE_COLUMN_NAME,
    PREDICTED_CATEGORY,
    PROB_SCORES,
    FILE_NAME_PREFIX,
)


class InferenceModel:
    """
    Class to handle the loading of the model, label encoder, and making predictions.

    """

    def __init__(self, model_path: str, model_params_path: str):
        try:
            self.model_params = load_model_params(model_params_path)
            self.model = self.load_model(model_path)
            self.label_encoder = self.load_label_encoder(
                f"{model_path}/{LABEL_ENCODER_PKL}"
            )
        except Exception as e:
            logging.error(f"Error initializing InferenceModel: {e}")
            raise

    def load_model(self, model_path: str) -> Any:
        """Loads the model from the given path and returns it."""
        logging.info(f'Loading the Model... The path: "{model_path}"')
        try:
            model = tf.saved_model.load(model_path)
            model = self.build_model_from_pb(model)
            if not model:
                raise Exception("Failed to load the model")
            logging.info("Model was loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {e}")
            raise

    def build_model_from_pb(self, pb_model: Any) -> Any:
        """Builds and returns a Keras model from a TensorFlow Hub protobuf model."""
        try:
            input_ids_layer = tf.keras.layers.Input(
                shape=(self.model_params[ENCODING][MAX_LENGTH],),
                name="input_word",
                dtype="int32",
            )
            keras_layer = hub.KerasLayer(pb_model, trainable=False)(input_ids_layer)
            model = tf.keras.Model([input_ids_layer], keras_layer)
            return model
        except Exception as e:
            logging.error(f"Error building model from protobuf: {e}")
            raise

    def load_label_encoder(self, label_encoder_path: str) -> LabelEncoder:
        """Loads the label encoder from the specified file path."""
        logging.info(f'Loading the Label Encoder... The path: "{label_encoder_path}"')
        try:
            with file_io.FileIO(label_encoder_path, mode="rb") as encoder_file:
                label_encoder = joblib.load(encoder_file)
            if not label_encoder:
                raise Exception("Failed to load the Label Encoder")
            logging.info("Label Encoder was loaded successfully.")
            return label_encoder
        except Exception as e:
            logging.error(f"Error loading label encoder from {label_encoder_path}: {e}")
            raise

    def predict(self, inputs: tf.data.Dataset):
        """Makes predictions on the provided dataset and returns transformed categories
        and probability scores."""
        try:
            prediction = self.model.predict(
                inputs, batch_size=self.model_params[INFERENCE][BATCH_SIZE]
            )
            prob_scores = np.max(prediction, axis=1)
            prediction_transform = tf.argmax(prediction, axis=1).numpy()
            transform_cat = self.label_encoder.inverse_transform(prediction_transform)
            return transform_cat, prob_scores
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise


class CSVProcessor:
    """
    Class to process CSV files for inference.
    """

    def __init__(self, inference_model: InferenceModel, trainer: ModelTrainer):
        self.inference_model = inference_model
        self.trainer = trainer

    def process_csv(self, file_path: str, prediction_dir: str) -> None:
        """Processes the given CSV file and saves predictions to the specified
        directory."""
        try:
            df = read_csv_file(file_path)
            x_train = self.trainer.tokenize_batch(list(df[FEATURE_COLUMN_NAME]))
            tf_dataset = tf.data.Dataset.from_tensor_slices(x_train)
            tf_dataset = tf_dataset.batch(
                self.inference_model.model_params[INFERENCE][BATCH_SIZE]
            )
            tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

            predictions, prob_scores = self.inference_model.predict(tf_dataset)

            df[PREDICTED_CATEGORY] = predictions
            df[PROB_SCORES] = prob_scores

            file_path = os.path.join(
                prediction_dir, "cleaned_data", "test", FILE_NAME_PREFIX + ".csv"
            )
            df.to_csv(file_path, index=False)
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error processing CSV: {e}")
            raise


class InferenceCLI:
    """
    Command-line interface for running inference on CSV files or raw strings.
    """

    def __init__(self, model_path: str, model_params_path: str):
        try:
            self.inference_model = InferenceModel(model_path, model_params_path)
            self.trainer = ModelTrainer(self.inference_model.model_params)
        except Exception as e:
            logging.error(f"Error initializing InferenceCLI: {e}")
            raise

    def run_csv_inference(self, csv_path: str, prediction_dir: str):
        """Runs inference on the specified CSV file and saves predictions
        to the given directory."""
        try:
            processor = CSVProcessor(self.inference_model, self.trainer)
            processor.process_csv(csv_path, prediction_dir)
        except Exception as e:
            logging.error(f"Error running CSV inference: {e}")
            raise

    def run_string_inference(self, input_string: str):
        """Runs inference on the provided string and returns the prediction
        and probability score."""
        try:
            tokens = self.trainer.tokenize_batch([input_string])
            tf_dataset = (
                tf.data.Dataset.from_tensor_slices(tokens)
                .batch(1)
                .prefetch(tf.data.AUTOTUNE)
            )
            prediction, prob_score = self.inference_model.predict(tf_dataset)
            return prediction[0], prob_score[0]
        except Exception as e:
            logging.error(f"Error during string inference: {e}")
            raise
