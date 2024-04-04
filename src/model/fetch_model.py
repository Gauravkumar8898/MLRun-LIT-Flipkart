from mlrun.artifacts import get_model
import zipfile
import logging
from src.utils.constant import model_download_directory, project

logging.basicConfig(level=logging.INFO)


class LoadModel:

    @staticmethod
    def load_model_mlrun_artifact():
        models = project.list_models()
        model_path = None
        for model in models:
            model_path = model.uri
        model_obj, model_file, extra_data = get_model(model_path)
        token_object = extra_data['tokenizer']
        dest_dir = model_download_directory
        src_zip_file = str(token_object)
        obj = LoadModel()
        obj.extract_zip(src_zip_file, dest_dir)
        src_zip_file = model_obj
        obj.extract_zip(src_zip_file, dest_dir)

    @staticmethod
    def extract_zip(src_zip_file, dest_dir):
        try:
            with zipfile.ZipFile(src_zip_file, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            logging.info("Extraction successful!")
        except Exception as e:
            logging.info(f"Error extracting zip file: {e}")


obj1 = LoadModel()
obj1.load_model_mlrun_artifact()
