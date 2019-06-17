from WikiSeries import WikiSeries
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import pickle


class WikiSeriesClassic(WikiSeries):

    def __init__(self, pickle_path, extracted=False, ordinal_encoder_pickle=None) -> None:
        super().__init__(pickle_path)

        if extracted:
            with open(ordinal_encoder_pickle, 'rb') as f:
                self.ordinal_encoder = pickle.load(f)
            self.fixed_columns = 5
        else:
            self.fixed_columns = 4
            print("Encoding meta columns...")
            self.encode_meta()
            print("Extracting median...")
            self.extract_median()
            print("Melting data...")
            self.melt_data()
            print("Extracting date features for melted data...")
            self.extract_date_features()
            print("Extracting lag features...")
            self.extract_lag_features()

        # TODO do dates after lag which will cut two days, new dates should start on '2015-07-03'
        # self.data_start_date = self.df.columns[]
        # self.data_end_date = self.df.columns[-1]
        # print('Data ranges from %s to %s' % (self.data_start_date, self.data_end_date))

        # self.date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in self.df.columns[self.fixed_columns:]]),data=[i for i in range(len(self.df.columns[self.fixed_columns:]))])
        # self.date_to_index = self.date_to_index + self.fixed_columns

    def extract_lag_features(self):
        self.df['last_hits'] = self.df.groupby(['name', 'lang', 'access', 'agent'])['hits'].shift()
        self.df['last_diff'] = self.df.groupby(['name', 'lang', 'access', 'agent'])['last_hits'].diff()
        self.df.dropna(inplace=True,)

    def extract_date_features(self):
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['week_of_year'] = self.df['date'].dt.weekofyear

    def melt_data(self):
        self.df = self.df.melt(id_vars=['name', 'lang', 'access', 'agent', 'median'], var_name='date', value_name='hits')
        self.df['date'] = pd.to_datetime(self.df['date'])

    def extract_median(self):
        self.df.insert(self.fixed_columns, 'median', self.df.iloc[:, self.fixed_columns:].median(axis=1))
        self.fixed_columns = self.fixed_columns + 1

    def encode_meta(self):
        self.ordinal_encoder = OrdinalEncoder()
        self.df.iloc[:, :self.fixed_columns] = self.ordinal_encoder.fit_transform(self.df.iloc[:, :self.fixed_columns])

    def save_to_disk(self, path_df, path_encoder):
        self.df.to_pickle(path_df)

        with open(path_encoder, "wb") as f:
            pickle.dump(self.ordinal_encoder, f)

    def prepare_series(self):
        pass
        # self.training_encoder = self.df.iloc[:,np.concatenate([np.arange(self.fixed_columns), self.date_to_index[self.train_enc_start:self.train_enc_end].values])]
        # self.training_decoder = self.df.iloc[:,np.concatenate([np.arange(self.fixed_columns), self.date_to_index[self.train_pred_start:self.train_pred_end].values])]
        # self.validation_encoder = self.df.iloc[:,np.concatenate([np.arange(self.fixed_columns), self.date_to_index[self.val_enc_start:self.val_enc_end].values])]
        # self.validation_decoder = self.df.iloc[:,np.concatenate([np.arange(self.fixed_columns), self.date_to_index[self.val_pred_start:self.val_pred_end].values])]
