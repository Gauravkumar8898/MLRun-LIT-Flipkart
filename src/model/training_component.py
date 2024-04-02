import logging
from src.utils.helpers import load_dataset
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, \
    TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np


class Flipkart_Pipeline():
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

    def drop_uneccasry(self, tokenised_dataset, col):
        tokenised_dataset = tokenised_dataset.remove_columns(col)
        tokenised_dataset = tokenised_dataset.rename_column('Summary', 'text')
        tokenised_dataset = tokenised_dataset.rename_column('sentiment_code', 'labels')
        return tokenised_dataset

    def tokenise(self, df):
        return self.tokenizer(df['Summary'], truncation=True)

    def compute_metrics(self, eval_pred):
        hub_model_id = 'akshatmehta98/roberta'
        load_recall = load_metric('recall')
        load_precision = load_metric('precision')

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        recall = load_recall.compute(predictions=predictions, references=labels, average="micro")["recall"]
        precision = load_precision.compute(predictions=predictions, references=labels, average="micro")["precision"]

        return {"recall": recall, "precision": precision}

    def runner(self):
        # obj = Flipkart_preprocessing()
        load_dataset()
        train_data, test_data = obj.runner_preprocesser()
        tokenized_train = train_data.map(self.tokenise, batched=True)
        tokenized_test = test_data.map(self.tokenise, batched=True)
        tokenized_train = self.drop_uneccasry(tokenized_train, '__index_level_0__')
        tokenized_test = self.drop_uneccasry(tokenized_test, '__index_level_0__')
        logging.info('Tokenised Dataset:', tokenized_train)
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment",
                                                                   num_labels=3)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        repo_name = 'roberta-fine-tuned-flipkart-reviews-an'

        training_args = TrainingArguments(
            output_dir=repo_name,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            save_strategy="epoch",
            push_to_hub=False,
            optim="adafactor"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        trainer.save_model()
        logging.info('Evaluation:', trainer.evaluate())
        trainer.push_to_hub()
