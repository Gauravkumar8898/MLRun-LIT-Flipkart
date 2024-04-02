import os
import mlrun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import logging

logging.set_verbosity("CRITICAL")

model_name = "akshatmehta98/roberta-base-fine-tuned-flipkart-reviews-am"
tokenizer = model_name
generation_config = GenerationConfig.from_pretrained(model_name)


# project = mlrun.get_or_create_project(
#     name="auto-trainer-test",
#     context="./",
#     user_project=True,
#     parameters={
#         "default_image": "yonishelach/mlrun-llm",
#     },
# )
project = mlrun.get_or_create_project("flipkart-review1", "./", user_project=True)

project.set_function(
    "auto-trainer.py",
    name="auto-trainer",
    kind="job",
    image="mlrun/mlrun",
    handler="finetune_llm",
)
project.save()

import transformers

training_arguments = {
    "learning_rate" : 2e-5,
    "per_device_train_batch_size" : 16,
    "per_device_eval_batch_size" : 16,
    "num_train_epochs" : 1,
    "weight_decay" : 0.01,
    "save_strategy" : "epoch",
    "push_to_hub": False,
    "optim" : "adafactor"
}

training_run = mlrun.run_function(
    function="auto-trainer",
    name="auto-trainer",
    local=True,
    params={
        "model": (model_name, "transformers.AutoModelForCausalLM"),
        "tokenizer": tokenizer,
        "train_dataset": "/home/nashtech/PycharmProjects/MLRun-transformer/src/data/transformed_data.csv",
        "training_config": training_arguments,
        "quantization_config": True,
        "lora_config": True,
        "dataset_columns_to_train": "quote",
        "lora_target_modules": ["query_key_value"],
        "model_pretrained_config": {"trust_remote_code": True, "use_cache": False},
    },
    handler="finetune_llm",
    outputs=["model"],
)

