import mlrun
from src.data_prep.data_transformation import Eda
from src.utils.constant import file_path
import os
import logging
# mlrun.set_environment(artifact_path="./")
class DataVisualization():
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.project=mlrun.get_or_create_project("flipkart-review1", "./", user_project=True)

    def runner(self):
        obj = Eda()
        obj.eda_runner()
        data_gen_fn = self.project.set_function(func=f"{file_path}", name="flipkart-transformed", kind="job", image="mlrun/mlrun",
                                           handler="flipkart")
        self.project.save()
        gen_data_run = self.project.run_function("flipkart", params={"format": "csv"}, local=True)

        describe_func = mlrun.import_function("hub://describe")
        # describe_func.apply(mlrun.platforms.auto_mount())

        describe_run = describe_func.run(
            name="task-describe",
            handler='analyze',
            inputs={"table": os.path.abspath("artifacts/random_dataset.parquet")},
            params={"name": "flipkart dataset", "label_column": "Sentiment"},
            local=True
        )


