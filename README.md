## Task Overview

To rank the search results better, we want to predict possible categories to boost for a given user search query.

For this purpose we collected `search` and `productview` events. As an applied scientist in the search team, we kindly ask you to develop a simple CLI to predict categories to boost for a given search term.

Please include the project with all the source code as a zip/tar.gz file.

You are free to use any library or programming language. We would like to run the project in our local machines easily.


`events.log` file is in JSONL format. Every row has an event serialized as json with the properties:
{
    "session": session id of the request,
    "name": name of the event,
    "query": (search event only) text the user searched,
    "title": (productview event only) title of the product,
    "product_id": (productview event only) id of the product,
    "category":(productview event only) category of the product,
}
# Vestiaire Collective Task

This package includes the code for category prediction that can be executed via the CLI.

## Setup

Make sure you have `pyenv` installed:

```bash
pyenv install 3.9.13
```

Create and activate the new virtual environment using pyenv:

```bash
pyenv virtualenv 3.9.13 vest_task
pyenv activate vest_task
```
## Note:
The code has been tested locally on `macOS`, which is why you'll find `tensorflow-macos` in the `pyproject.toml` file. If you’re not using `macOS`, you can remove this library before running the setup-dev file.

```bash
make setup-dev
```
## Notebook
In the notebook section, you will find the Exploratory Data Analysis (EDA), as well as the training, inference, and model evaluation processes.

## Run the Preprocessing pipeline:

#### Input Data Location:
The preprocessing pipeline will take source data from the `data/raw_data` directory, where you can find the `events.log` file.

`Output Data Locations`:
After processing, the final output will be stored in the following directories:

#### Training Data or Productview data:

`data/clean_data/train`: Contains product view data for training.
#### Testing Data or Search data:

`data/clean_data/test`: Contains search query data for testing.

```bash
python -m cli.cli_preprocessing --data-path data/raw_data --output-dir data/cleaned_data --data-types training,test
```

## Run the Training pipeline:

For training, I used the DistilBERT pre-trained language model and fine-tuned it on the provided data.

#### Model Storage:
The trained model is stored in the `model_artifacts` directory.

#### Model Metrics:
The model metrics are saved in the `model_artifacts/metrics` directory.

```bash
python -m cli.cli_train --data-path data/cleaned_data/train --artifacts-dir model_artifacts
```

## Run the Inference pipeline:
During inference, the search queries are taken from the `data/cleaned_data/test` directory. The process applies inference on this data and dumps the results into a CSV file located at:

`data/cleaned_data/test/search_data_prediction_file.csv`.
```bash
python -m cli.cli_inference --csv data/cleaned_data/test --model_path model_artifacts --prediction_dir data
```
You can also do the inference just passing the search query as a string:

```bash
python -m cli.cli_inference --string "coat" --model_path model_artifacts
```
