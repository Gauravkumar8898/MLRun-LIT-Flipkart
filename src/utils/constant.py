from pathlib import Path
import mlrun

import pandas

curr_path = Path(__file__).parents[1]
data_directory = curr_path / 'data'
model_download_directory = "/home/nashtech/PycharmProjects/LIT-flipkart/models"
flipkart_dataset_path = data_directory / "review_flipkart.csv"
transformed_dataset_path = data_directory / "transformed_data.csv"

project = mlrun.get_or_create_project('hugging-face-trainer', context="./", user_project=True)
