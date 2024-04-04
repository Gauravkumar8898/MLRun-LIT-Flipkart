import pandas as pd
from src.utils.constant import flipkart_dataset_path, transformed_dataset_path
from src.utils.helpers import load_dataset
from datasets import Dataset
import logging


class Eda:
    @staticmethod
    def eda_runner():
        df = load_dataset(flipkart_dataset_path)
        new_df = df
        new_df = new_df.dropna()
        new_df['product_price'] = pd.to_numeric(new_df['product_price'], errors='coerce')
        new_df['Rate'] = pd.to_numeric(new_df['Rate'], errors='coerce')
        new_df['sentiment_code'] = pd.Categorical(new_df.Sentiment).codes
        new_df['sentiment_code'] = new_df['sentiment_code'].astype('Int64')
        pd.DataFrame.iteritems = pd.DataFrame.items
        new_df.to_csv(transformed_dataset_path, index=False)

    # @staticmethod
    # def splitting(self, final_data):
    #     train = final_data.shuffle(seed=42).select([i for i in list(range(30000))])
    #     test = final_data.shuffle(seed=42).select([i for i in list(range(30000, 40000))])
    #     return train, test
    #
    # @staticmethod
    # def punctuation_handling(self, df):
    #     df.dropna(inplace=True)
    #     final_dataset = Dataset.from_pandas(df[['Summary', 'sentiment_code']])
    #     train_dataset, test_dataset = self.splitting(final_dataset)
    #     logging.info('New train dataset:\n%s', train_dataset)
    #     logging.info('New test dataset:\n%s', test_dataset)
    #     return train_dataset, test_dataset

#
# obj = Eda()
# obj.eda_runner()
