import mlrun
from src.data_prep.data_transformation import Eda
import os


project = mlrun.get_or_create_project("flipkart-review1", "./", user_project=True)

# obj = Eda()
# obj.eda_runner()
# data_gen_fn = project.set_function("data.py", name="flipkart-transformed", kind="job", image="mlrun/mlrun",
#                                    handler="flipkart")
# project.save()
# gen_data_run = project.run_function("flipkart", params={"format": "csv"}, local=True)
#
# describe_func = mlrun.import_function("hub://describe")
# # describe_func.apply(mlrun.platforms.auto_mount())
#
# describe_run = describe_func.run(
#     name="task-describe",
#     handler='analyze',
#     inputs={"table": os.path.abspath("artifacts/random_dataset.parquet")},
#     params={"name": "flipkart dataset", "label_column": "Sentiment"},
#     local=True
# )


