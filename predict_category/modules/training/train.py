import os
import json
import joblib
import logging
import warnings
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Any, Dict, Iterable, List, Tuple
from predict_category.modules.common.functions import read_csv_file, load_model_params
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from predict_category.modules.common.constants import (
    TRANSFORMERS,
    TOKENIZER,
    LABEL_COLUMN_NAME,
    ENCODED_LABEL,
    DATA,
    VAL_SIZE,
    RANDOM_STATE,
    ENCODING,
    BATCH_SIZE,
    MAX_LENGTH,
    MODEL,
    TRAIN,
    OPTIMIZATION,
    INIT_LR,
    EPOCHS,
    PREDICTED_CATEGORY,
    ACCURACY,
    COHEN,
    MATHEW,
    METRICS_FULL_REPORT,
    LABEL_ENCODER_PKL,
    METRICS_JSON,
    METRICS_DIR,
    EARLY_STOPPING,
    VERBOSE,
    VAL_LOSS,
    MIN_LR,
    MIN_DELTA,
    REDUCE_ON_PLATEAU,
    PATIENCE,
    FACTOR,
    FEATURE_COLUMN_NAME,
    CALLBACKS,
)

AUTO = tf.data.experimental.AUTOTUNE


warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    Manages the training of a text classification model using DistilBERT.
    """

    def __init__(self, model_params: Dict[str, Any]):
        self.model_params = model_params
        self.tokenizer = self.get_tokenizer()
        self.label_encoder = LabelEncoder()

    def get_tokenizer(self) -> Any:
        try:
            logging.info("Download the tokenizer from Huggingface.")
            return DistilBertTokenizer.from_pretrained(
                self.model_params[TRANSFORMERS][TOKENIZER]
            )
        except Exception as e:
            logging.error(f"Error downloading tokenizer: {e}")
            raise

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Perform label encoding.")
            df[ENCODED_LABEL] = self.label_encoder.fit_transform(
                list(df[LABEL_COLUMN_NAME].values)
            )
            logging.info(
                f"The total number of samples: "
                f"{len(df[ENCODED_LABEL])}; classes: {len(self.label_encoder.classes_)}"
            )
            return df
        except Exception as e:
            logging.error(f"Error in label encoding: {e}")
            raise

    def split_to_train_val(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logging.info("Perform train/validation split.")
            return train_test_split(
                df,
                test_size=self.model_params[DATA][VAL_SIZE],
                random_state=self.model_params[RANDOM_STATE],
                stratify=list(df[LABEL_COLUMN_NAME]),
            )
        except Exception as e:
            logging.error(f"Error during train/validation split: {e}")
            raise

    def tokenize_batch(self, texts: List[str]) -> Any:
        try:
            logging.info("Tokenizing batch of texts.")
            tokenized_data = []
            for i in range(0, len(texts), self.model_params[ENCODING][BATCH_SIZE]):
                batch_text = texts[i : i + self.model_params[ENCODING][BATCH_SIZE]]
                tokenized_batch = self.tokenizer(
                    batch_text,
                    max_length=self.model_params[ENCODING][MAX_LENGTH],
                    padding="max_length",
                    truncation=True,
                    return_tensors="tf",
                )["input_ids"]
                tokenized_data.append(tokenized_batch)
            return tf.concat(tokenized_data, axis=0)
        except Exception as e:
            logging.error(f"Error during tokenization: {e}")
            raise

    def build_and_train_model(
        self, train_dataset, val_dataset, callbacks: List[Any], n_steps: int
    ):
        """
        Build and train the model using TFDistilBertModel and TensorFlow's Keras API.
        """
        try:
            logging.info("Building the model.")
            bert_model = TFDistilBertModel.from_pretrained(
                self.model_params[TRANSFORMERS][MODEL]
            )

            input_ids = tf.keras.layers.Input(
                shape=(self.model_params[ENCODING][MAX_LENGTH],), dtype="int32"
            )
            bert_output = bert_model(input_ids)[0]
            cls_token = bert_output[:, 0, :]
            output = tf.keras.layers.Dense(
                len(self.label_encoder.classes_), activation="softmax"
            )(cls_token)

            model = tf.keras.models.Model(inputs=input_ids, outputs=output)

            logging.info("Compiling the model.")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.model_params[TRAIN][OPTIMIZATION][INIT_LR]
                ),
                loss=SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

            logging.info("Training the model.")
            train_start = time.time()
            model.fit(
                train_dataset,
                validation_data=val_dataset,
                steps_per_epoch=n_steps,
                epochs=self.model_params[TRAIN][EPOCHS],
                batch_size=self.model_params[TRAIN][BATCH_SIZE],
                callbacks=callbacks,
            )
            train_time_minutes = round((time.time() - train_start) / 60, 2)
            logging.info(
                f"The training has finished, took {train_time_minutes} minutes."
            )
            return model
        except Exception as e:
            logging.error(f"Error during model building or training: {e}")
            raise


class ModelEvaluator:
    """
    Evaluates a trained text classification model and calculates performance metrics.
    """

    def __init__(self, model: Any, label_encoder: LabelEncoder):
        self.model = model
        self.label_encoder = label_encoder

    def calculate_scores(
        self, x_val: Iterable[int], y_val_true: Iterable[int]
    ) -> Tuple[Any, Any, dict]:
        try:
            logging.info("Calculating the metrics.")
            prediction = self.model.predict(x_val)
            y_val_pred = tf.argmax(prediction, axis=1).numpy()
            y_val_pred = self.label_encoder.inverse_transform(y_val_pred)
            y_val_true = self.label_encoder.inverse_transform(y_val_true)
            metrics = self.get_metrics(y_val_true, y_val_pred)
            logging.info(f"Metrics: {metrics}")
            return y_val_pred, y_val_true, metrics
        except Exception as e:
            logging.error(f"Error during score calculation: {e}")
            raise

    def val_data(self, df_val: pd.DataFrame, y_pred: Iterable[str]):
        df_val[PREDICTED_CATEGORY] = y_pred
        return df_val

    def get_metrics(
        self,
        true_values: Iterable[str],
        predicted_values: Iterable[str],
        round_n: int = 3,
    ) -> Dict[str, float]:
        """
        Prints classification report with accuracy.
        :param true_values: Iterable with the actual values.
        :param predicted_values: Iterable with the predicted values.
        :param round_n: The number after the decimal point.
        :return: The dictionary with the model metrics.
        """
        metrics = dict()

        acc_value = accuracy_score(true_values, predicted_values)
        metrics[ACCURACY] = round(acc_value, round_n)
        cohen = cohen_kappa_score(true_values, predicted_values)
        metrics[COHEN] = round(cohen, round_n)
        matthew = matthews_corrcoef(true_values, predicted_values)
        metrics[MATHEW] = round(matthew, round_n)

        f1_macro = f1_score(true_values, predicted_values, average="macro")
        metrics["f1-score macro"] = round(f1_macro, round_n)
        precision_macro = precision_score(
            true_values, predicted_values, average="macro"
        )
        metrics["Precision macro"] = round(precision_macro, round_n)
        recall_macro = recall_score(true_values, predicted_values, average="macro")
        metrics["Recall macro"] = round(recall_macro, round_n)

        f1_micro = f1_score(true_values, predicted_values, average="micro")
        metrics["f1-score micro"] = round(f1_micro, round_n)
        precision_micro = precision_score(
            true_values, predicted_values, average="micro"
        )
        metrics["Precision micro"] = round(precision_micro, round_n)
        recall_micro = recall_score(true_values, predicted_values, average="micro")
        metrics["Recall micro"] = round(recall_micro, round_n)

        f1_weighted = f1_score(true_values, predicted_values, average="weighted")
        metrics["f1-score weighted"] = round(f1_weighted, round_n)
        precision_weighted = precision_score(
            true_values, predicted_values, average="weighted"
        )
        metrics["Precision weighted"] = round(precision_weighted, round_n)
        recall_weighted = recall_score(
            true_values, predicted_values, average="weighted"
        )
        metrics["Recall weighted"] = round(recall_weighted, round_n)

        metrics[METRICS_FULL_REPORT] = classification_report(
            true_values, predicted_values, output_dict=True
        )
        return metrics


class ArtifactManager:
    """
    Handles the saving of model artifacts, including the trained model,
    label encoder, and evaluation metrics.
    """

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir

    def save_artifacts(
        self,
        model: Any,
        label_encoder: LabelEncoder,
        metrics: Dict[str, float],
        df_val_df: pd.DataFrame,
    ) -> None:
        try:
            if not os.path.exists(self.artifacts_dir):
                os.makedirs(self.artifacts_dir)
            joblib.dump(
                label_encoder, os.path.join(self.artifacts_dir, LABEL_ENCODER_PKL)
            )
            metrics_dir_path = os.path.join(self.artifacts_dir, METRICS_DIR)
            os.makedirs(metrics_dir_path, exist_ok=True)
            with open(os.path.join(metrics_dir_path, METRICS_JSON), "w") as f:
                json.dump(metrics, f, ensure_ascii=False)
            model.save(self.artifacts_dir)
            df_val_df.to_csv(
                os.path.join(metrics_dir_path, "val_predicted.csv"), index=False
            )
            logging.info("The artifacts are successfully saved.")
        except Exception as e:
            logging.error(f"Error saving artifacts: {e}")
            raise


def get_callbacks(callback_params: Dict[str, Any]) -> List[Any]:

    """
    A custom function to provide the needed callbacks based on the parameters specified.
    """
    try:
        callbacks = []
        if EARLY_STOPPING in callback_params:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=VAL_LOSS,
                verbose=callback_params[VERBOSE],
                min_delta=callback_params[EARLY_STOPPING][MIN_DELTA],
                patience=callback_params[EARLY_STOPPING][PATIENCE],
                mode="auto",
            )
            callbacks.append(early_stopping)
        if REDUCE_ON_PLATEAU in callback_params:
            reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=VAL_LOSS,
                verbose=callback_params[VERBOSE],
                factor=callback_params[REDUCE_ON_PLATEAU][FACTOR],
                patience=callback_params[REDUCE_ON_PLATEAU][PATIENCE],
                min_lr=callback_params[REDUCE_ON_PLATEAU][MIN_LR],
            )
            callbacks.append(reduce_on_plateau)
        logging.info(f"Callbacks: {callbacks}")
        return callbacks
    except Exception as e:
        logging.error(f"Error get_callbacks: {e}")
        raise


def transform_data_for_training(
    x_train: Iterable[int],
    y_train: Iterable[int],
    x_val: Iterable[int],
    y_val: Iterable[int],
    model_params: Dict,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Transforms training and validation data into TensorFlow datasets.

    This function takes training and validation input data and labels,
    creates TensorFlow datasets, and applies necessary transformations
    such as shuffling and batching for training.
    """
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .repeat()
        .shuffle(2048)
        .batch(model_params[TRAIN][BATCH_SIZE])
        .prefetch(AUTO)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(model_params[TRAIN][BATCH_SIZE])
        .cache()
        .prefetch(AUTO)
    )
    return train_dataset, val_dataset


def train_main(data_path: str, model_params_path: str, artifacts_dir: str) -> None:
    """
    Main function to train the model.

    This function loads the model parameters, reads the training data,
    encodes labels, splits the data into training and validation sets,
    tokenizes the text data, builds and trains the model, evaluates
    its performance, and saves the artifacts.
    """
    model_params = load_model_params(model_params_path)
    df = read_csv_file(data_path)
    df = df.drop_duplicates(
        subset=[FEATURE_COLUMN_NAME, LABEL_COLUMN_NAME], keep="first"
    )

    trainer = ModelTrainer(model_params)
    df_encoded = trainer.encode_labels(df)
    df_train, df_val = trainer.split_to_train_val(df_encoded)

    x_train = trainer.tokenize_batch(list(df_train[FEATURE_COLUMN_NAME]))
    x_val = trainer.tokenize_batch(list(df_val[FEATURE_COLUMN_NAME]))
    y_train = list(df_train[ENCODED_LABEL])
    y_val = list(df_val[ENCODED_LABEL])

    train_dataset, val_dataset = transform_data_for_training(
        x_train, y_train, x_val, y_val, model_params
    )

    n_steps = x_train.shape[0] // model_params[TRAIN][BATCH_SIZE]
    callbacks = get_callbacks(model_params[TRAIN][CALLBACKS])

    model = trainer.build_and_train_model(
        train_dataset, val_dataset, callbacks, n_steps
    )

    evaluator = ModelEvaluator(model, trainer.label_encoder)
    y_pred, y_true, metrics = evaluator.calculate_scores(x_val, y_val)
    df_val_pred = evaluator.val_data(df_val, y_pred)

    artifact_manager = ArtifactManager(artifacts_dir)
    artifact_manager.save_artifacts(model, trainer.label_encoder, metrics, df_val_pred)
