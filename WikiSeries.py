import numpy as np
import pandas as pd
from datetime import timedelta
import re
from sklearn.preprocessing import OrdinalEncoder


class WikiSeries:
    def __init__(self, pickle_path) -> None:
        super().__init__()
        self.df = pd.read_pickle(pickle_path)
        self.data_start_date = None
        self.data_end_date = None

    def get_time_block_series(self, start_date, end_date):
        pass

    def training_series(self):
        pass

    def validation_series(self):
        pass

    # TODO should be able to get any sample, transformed correctly, not only validation
    def get_validation_sample(self, sample_ind):
        pass

    def denormalize_series(self, series, sample_ind):
        pass

    def prepare_dates(self, pred_steps):
        self.pred_length = timedelta(pred_steps)

        first_day = pd.to_datetime(self.data_start_date)
        last_day = pd.to_datetime(self.data_end_date)

        self.val_pred_start = last_day - self.pred_length + timedelta(1)
        self.val_pred_end = last_day

        self.train_pred_start = self.val_pred_start - self.pred_length
        self.train_pred_end = self.val_pred_start - timedelta(days=1)

        self.enc_length = self.train_pred_start - first_day

        self.train_enc_start = first_day
        self.train_enc_end = self.train_enc_start + self.enc_length - timedelta(1)

        self.val_enc_start = self.train_enc_start + self.pred_length
        self.val_enc_end = self.val_enc_start + self.enc_length - timedelta(1)

        print('Train encoding:', self.train_enc_start, '-', self.train_enc_end)
        print('Train prediction:', self.train_pred_start, '-', self.train_pred_end, '\n')
        print('Val encoding:', self.val_enc_start, '-', self.val_enc_end)
        print('Val prediction:', self.val_pred_start, '-', self.val_pred_end)

        print('\nEncoding interval:', self.enc_length.days)
        print('Prediction interval:', self.pred_length.days)
        self.prepare_series()

    def prepare_series(self):
        pass

    @staticmethod
    def preprocess_csv_to_pickle(input_path, output_path):
        def extract_meta(page):
            parts = page.split("_")
            [project, access, agent] = parts[-3:]
            name = " ".join(parts[0:-3])
            match_lang = re.search("([a-z][a-z])\.wikipedia\.org", project)
            lang = match_lang.group(1) if match_lang else 'na'
            return [name, lang, access, agent]

        original_dataset = pd.read_csv(input_path)
        print("Before cleaning:")
        original_dataset.info()
        original_dataset.dropna(axis='index', inplace=True)
        original_dataset.reset_index(inplace=True, drop=True)
        print("After cleaning:")
        original_dataset.info()
        meta_dataset = original_dataset.copy()
        meta_dataset['name'], meta_dataset['lang'], meta_dataset['access'], meta_dataset['agent'] = zip(*meta_dataset['Page'].apply(extract_meta))
        meta_dataset = meta_dataset[['name', 'lang', 'access', 'agent'] + [c for c in meta_dataset if c not in ['Page', 'name', 'lang', 'access', 'agent']]]
        meta_dataset.to_pickle(output_path)
