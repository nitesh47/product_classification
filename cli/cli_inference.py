import argparse
import logging
from predict_category.modules.inference.inference import InferenceCLI


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run inference on CSV or raw string.")
    parser.add_argument("--csv", type=str, help="Path to the CSV file for inference.")
    parser.add_argument("--string", type=str, help="Input string for inference.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model."
    )
    parser.add_argument(
        "--model_params_path",
        type=str,
        default="config/model/dev.yml",
        help="Path to the model parameters.",
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        help="Directory to save predictions for CSV inference.",
    )
    try:
        args = parser.parse_args()

        cli = InferenceCLI(args.model_path, args.model_params_path)

        if args.csv:
            if not args.prediction_dir:
                logging.error(
                    "Prediction directory must be specified when using CSV inference."
                )
                return
            cli.run_csv_inference(args.csv, args.prediction_dir)
        elif args.string:
            prediction, prob_score = cli.run_string_inference(args.string)
            print(f"Prediction: {prediction}, Probability: {prob_score}")
        else:
            logging.error("No input provided. Please specify either --csv or --string.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
