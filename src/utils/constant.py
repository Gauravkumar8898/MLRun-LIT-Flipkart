from pathlib import Path


import pandas

curr_path = Path(__file__).parents[1]
project_path = Path(__file__).parents[2]
data_directory = curr_path / 'data'
model_download_directory = "/home/nashtech/PycharmProjects/MLRun-transformer/models"
flipkart_dataset_path = data_directory / "review_flipkart.csv"
transformed_dataset_path = data_directory / "transformed_data.csv"
model_path = "/home/nashtech/PycharmProjects/MLRun-transformer/huggingface.pkl"
hugging_face = "/home/nashtech/PycharmProjects/MLRun-transformer/src/serving_model/serving_model.py"

true_model_path='/home/nashtech/PycharmProjects/MLRun-transformer/models'
file_path= curr_path / 'data_prep/data.py'

