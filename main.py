import time
import os
from src.model.training_component_transformer import ModelTraining
from src.data_prep.run_component import DataVisualization
from src.serving_model.model_serving import Serving_Model

if __name__ == '__main__':
    """For running visualizations """
    obj=DataVisualization()
    obj.runner()
    time.sleep(5)
    """For Model traning on ui"""
    obj2=ModelTraining()
    obj2.trainer_runner()
    time.sleep(5)
    """For model serving"""
    obj3=Serving_Model()
    obj3.serving_runner()


