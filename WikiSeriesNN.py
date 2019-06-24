from datetime import timedelta

import numpy as np
import pandas as pd

from WikiSeries import WikiSeries


class WikiSeriesNN(WikiSeries):
    def __init__(self, pickle_path) -> None:
        super().__init__(pickle_path)

        self.separate_meta()
        self.data_start_date = self.df.columns[0]
        self.data_end_date = self.df.columns[-1]
        print('Data ranges from %s to %s' % (self.data_start_date, self.data_end_date))
        self.date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in self.df.columns[0:]]),
                                       data=[i for i in range(len(self.df.columns[0:]))])
        self.series_array = self.df.values

    def separate_meta(self):
        self.meta = self.df.iloc[:, :4]
        self.df = self.df.iloc[:, 4:]

    def get_time_block_series(self, start_date, end_date):
        inds = self.date_to_index[start_date:end_date]
        return self.series_array[:, inds]

    def transform_series_encode(self, series, compute_mean=True):
        series_array = np.log1p(series)
        if compute_mean:
            self.encode_series_mean = series_array.mean(axis=1).reshape(-1, 1)
        series_array = series_array - self.encode_series_mean
        series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))

        return series_array

    def transform_series_decode(self, series):
        series_array = np.log1p(series)
        series_array = series_array - self.encode_series_mean
        series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))

        return series_array

    def training_series(self):
        encoder_input_data = self.get_time_block_series(self.train_enc_start, self.train_enc_end)
        encoder_input_data = self.transform_series_encode(encoder_input_data)

        decoder_target_data = self.get_time_block_series(self.train_pred_start, self.train_pred_end)
        decoder_target_data = self.transform_series_decode(decoder_target_data)

        return encoder_input_data, decoder_target_data

    def validation_series(self):
        encoder_input_data = self.get_time_block_series(self.val_enc_start, self.val_enc_end)
        encoder_input_data = self.transform_series_encode(encoder_input_data)

        decoder_target_data = self.get_time_block_series(self.val_pred_start, self.val_pred_end)
        decoder_target_data = self.transform_series_decode(decoder_target_data)

        return encoder_input_data, decoder_target_data

    def get_validation_sample(self, sample_ind):
        return self.validation_encoder[sample_ind:sample_ind + 1, :, :], self.validation_decoder[sample_ind, :, :1].reshape(-1, 1)

    def get_sample(self, index=None, page_name=None, lang=None, access=None, agent=None):
        if index is None:
            if page_name is None or lang is None or access is None or agent is None:
                return None
            index = self.meta[self.meta['name'] == page_name][self.meta['lang'] == lang][self.meta['access'] == access][self.meta['agent'] == agent]
            print(index)
            if len(index.index) == 0:
                return None
            index = index.index[0]

        encode, decode = self.get_validation_sample(index)
        return index, self.meta.iloc[index, :], encode, decode

    def denormalize_series(self, series, sample_ind):
        new_series = series + self.encode_series_mean[sample_ind, 0]
        new_series = np.expm1(new_series)
        return new_series

    