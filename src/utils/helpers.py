import pandas as pd


def load_dataset(data_path):
    """
       Use Pandas read_csv function to read the dataset from the specified file path.
       Return the loaded dataset.
       """
    data_set = pd.read_csv(data_path)
    return data_set
