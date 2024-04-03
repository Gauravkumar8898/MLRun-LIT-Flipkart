import mlrun
serving_function = mlrun.import_function('function.yaml')
serving_function.add_model(
    'mymodel',
    class_name='HuggingFaceModelServer',
    model_path='123',  # This is not used, just for enabling the process.

    task="sentiment-analysis",
    model_class="AutoModelForSequenceClassification",
    model_name="akshatmehta98/roberta-base-fine-tuned-flipkart-reviews-am",
    tokenizer_class="AutoTokenizer",
    tokenizer_name="akshatmehta98/roberta-base-fine-tuned-flipkart-reviews-am",
)
server = serving_function.to_mock_server()
result = server.test(
    '/v2/models/mymodel',
    body={"inputs": ["Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."]}
)
print(f"prediction: {result['outputs']}")
serving_function.deploy()