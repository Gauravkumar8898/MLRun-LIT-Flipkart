import mlrun
from src.utils.constant import hugging_face, model_path

project = mlrun.get_or_create_project("hugging-tutorial-demo", "./", user_project=True)

serving_fn = mlrun.code_to_function('serving', filename=hugging_face,
                                    kind='serving', image='mlrun/ml-models:1.5.0-rc9',
                                    requirements=['transformers==4.21.3', 'tensorflow==2.9.2', "torch==2.2.2", "Datasets==2.10.1"])

serving_fn.add_model(
    'mymodel',
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