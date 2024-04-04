from pathlib import Path


curr_path = Path(__file__).parents[1]
curr_path1 = Path(__file__).parents[2]
data_directory = curr_path / 'data'
model_path_ = curr_path1 / 'flipkart_model'
mlrun_model_path = curr_path1 / "models"
# print(mlrun_model_path)
data_points = 100

flipkart_dataset_path = data_directory / "review_flipkart.csv"


