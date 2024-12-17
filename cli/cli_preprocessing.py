import argparse
import logging
from predict_category.modules.preprocessing.text_preprocessing import DataProcessor


def main():
    parser = argparse.ArgumentParser(description="Process and save training/test data.")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the data directory."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save processed data.",
    )
    parser.add_argument(
        "--data-types",
        type=str,
        default="training,test",
        help="Comma-separated list of data types to "
        "process (default: 'training,test').",
    )

    args = parser.parse_args()
    data_types = args.data_types.split(",")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        processor = DataProcessor()
        processor.read_json_file(args.data_path)
        processor.split_test_and_training_data()
        processor.clean_labels()
        processor.apply_text_preprocessing(data_types)
        processor.save_dataframes(args.output_dir)
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
