import argparse
import logging
from predict_category.modules.training.train import train_main


def main():
    parser = argparse.ArgumentParser(
        description="Train a multi-class classification model."
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the CSV data file."
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default="config/model/dev.yml",
        help="Path to the model parameters YAML file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        required=True,
        help="Directory to save model artifacts.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        logging.info("Starting the training process...")
        train_main(args.data_path, args.model_params, args.artifacts_dir)
        logging.info("Training and evaluation completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
