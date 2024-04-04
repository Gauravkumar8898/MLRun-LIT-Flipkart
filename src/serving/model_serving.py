import os
import urllib.request
import mlrun
import pickle
from transformers import AutoModelForSequenceClassification,AutoConfig
model_path = os.path.abspath('/home/nashtech/mlrun-data/hugging-face-trainer-nashtech/hugging-face-classifier-trainer-train/0/model/')
config=AutoConfig.from_pretrained(f'{model_path}/config.json')
model=AutoModelForSequenceClassification.from_pretrained(f'{model_path}/pytorch_model.bin',config=config)
# with open('huggingface.pkl','wb') as f:
#     pickle.dump(model,f)
# print(model)
with open('/home/nashtech/mlrun-data/hugging-face-trainer-nashtech/hugging-face-classifier-trainer-train/0/model/huggingface.pkl','rb') as f:
    model=pickle.load(f)

# model_path=os.path.abspath('/home/nashtech/Desktop/Mlrun-LIT-Flipkart/src/serving/huggingface.pkl')
# model_path=os.path.join(model_path,'config.json')
# print(model_path)
# tokenizer=XLMRobertaTokenizer.from_pretrained('/home/nashtech/mlrun-data/hugging-face-trainer-nashtech/hugging-face-classifier-trainer-train/0/model/tokenizer.json')
project_name_base = 'serving-test'
project = mlrun.get_or_create_project(project_name_base, context="./", user_project=True)
serving_function_image = "mlrun/mlrun"
serving_model_class_name = "mlrun.frameworks.huggingface.HuggingFaceModelServer"
serving_fn = mlrun.new_function("serving", project=project.name, kind="serving", image=serving_function_image)

model_key = "hugging-face"
serving_fn.add_model(key=model_key,
                     class_name=serving_model_class_name,
                     model_path=model_path,
                     model=model,
                     )

mock_server = serving_fn.to_mock_server()
my_data = {"inputs":[["Hello there"],["No I dont know"]]}
mock_server.test(f"/v2/models/{model_key}/infer", body=my_data)
