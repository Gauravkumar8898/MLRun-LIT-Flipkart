import mlrun

project = mlrun.get_or_create_project('hugging-face-trainer', context="./", user_project=True)
hugging_face_classifier_trainer = mlrun.import_function("hub://hugging_face_classifier_trainer")
# accuracy_metric = load_metric('accuracy')
# precision_metric= load_metric('precision', average='weighted')
model_class = "transformers.AutoModelForSequenceClassification"
additional_parameters = {
    "TRAIN_output_dir": "finetuning-sentiment-model-3000-samples",
    "TRAIN_learning_rate": 2e-5,
    "TRAIN_per_device_train_batch_size": 16,
    "TRAIN_per_device_eval_batch_size": 16,
    "TRAIN_num_train_epochs": 3,
    "TRAIN_weight_decay": 0.01,
    "TRAIN_push_to_hub": False,
    "TRAIN_evaluation_strategy": "epoch",
    "TRAIN_eval_steps": 1,
    "TRAIN_logging_steps": 1,
    "CLASS_num_labels": 3,
    "ignore_mismatched_sizes": True,
    "optim": "adafactor"
}
train_run = hugging_face_classifier_trainer.run(params={
    "hf_dataset": "akshatmehta98/Flipkart-Dataset",
    "drop_columns": [
        "product_name",
        "product_price",
        "Rate",
        "Review",
        "labels"
    ],
    "pretrained_tokenizer": "akshatmehta98/roberta-base-fine-tuned-flipkart-reviews-am",
    "pretrained_model": "akshatmehta98/roberta-base-fine-tuned-flipkart-reviews-am",
    "model_class": "transformers.AutoModelForSequenceClassification",
    "label_name": "sentiment_code",
    "num_of_train_samples": 100,
    "metrics": ["accuracy"],
    "random_state": 42,

    **additional_parameters
},
    handler="train",
    local=True,
)
print(train_run.outputs)
print(train_run.artifact('loss_plot').show())
