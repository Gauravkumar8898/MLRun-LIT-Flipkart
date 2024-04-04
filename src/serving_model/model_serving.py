import mlrun
import os
import pickle
from src.utils.constant import hugging_face, true_model_path,project_path,model_path
from transformers import AutoConfig,AutoModelForSequenceClassification
import logging

class Serving_Model():
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.project=mlrun.get_or_create_project("flipkart-review1", "./", user_project=True)


    def serving_runner(self):


        config=AutoConfig.from_pretrained(f'{true_model_path}/config.json')
        model=AutoModelForSequenceClassification.from_pretrained(f'{true_model_path}/pytorch_model.bin',config=config)
        with open(f'{project_path}/huggingface.pkl','wb') as f:
            pickle.dump(model,f)




        serving_fn = mlrun.code_to_function('serving_model', filename=hugging_face,
                                            kind='serving_model', image='mlrun/ml-models:1.5.0-rc9',
                                            requirements=['transformers==4.21.3', 'tensorflow==2.9.2', "torch==2.2.2", "Datasets==2.10.1"])

        serving_fn.add_model(
            'flipkart_review_model',
            class_name='HuggingFaceModelServer',
            model_path=model_path,  # This is not used, just for enabling the process.

            task="sentiment-analysis",
            model_class="AutoModelForSequenceClassification",
            tokenizer_class="AutoTokenizer",
            tokenizer_name="/home/nashtech/PycharmProjects/LIT-flipkart/models",
        )
        server = serving_fn.to_mock_server()
        result = server.test(
            '/v2/models/flipkart_review_model',
            body={"inputs": ["good product"]}
        )
        print(f"prediction: {result['outputs']}")
        # serving_fn.deploy()