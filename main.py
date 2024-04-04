import time
import os
from src.model.training_component_transformer import ModelTraining
from src.data_prep.final_runner import obj
from src.utils.constant import component_path

if __name__ == '__main__':
    """For running File """
    obj.runner()
    time.sleep(5)
    # obj2=ModelTraining()
    # obj2.trainer_runner()
