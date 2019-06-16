import numpy as np
from keras.layers import Input, Conv1D, Dense, Dropout, Lambda
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam

from WikiModel import WikiModel
from WikiSeries import WikiSeries


class Seq2SeqConv(WikiModel):
    def __init__(self, series: WikiSeries, pred_steps) -> None:
        super().__init__(series, pred_steps)

    def build_model(self):
        # convolutional layer parameters
        n_filters = 32
        filter_width = 2
        dilation_rates = [2 ** i for i in range(8)]

        # define an input history series and pass it through a stack of dilated causal convolutions.
        history_seq = Input(shape=(None, 1))
        x = history_seq

        for dilation_rate in dilation_rates:
            x = Conv1D(filters=n_filters,
                       kernel_size=filter_width,
                       padding='causal',
                       dilation_rate=dilation_rate)(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(.2)(x)
        x = Dense(1)(x)

        # extract the last 14 time steps as the training target
        def slice(x, seq_length):
            return x[:, -seq_length:, :]

        pred_seq_train = Lambda(slice, arguments={'seq_length': self.pred_steps})(x)

        self.model = Model(history_seq, pred_seq_train)

    def fit(self, epochs=10):
        first_n_samples = 40000
        batch_size = 2 ** 11

        encoder_input_data = self.series.training_encoder
        decoder_target_data = self.series.training_decoder

        encoder_input_data = encoder_input_data[:first_n_samples]
        decoder_target_data = decoder_target_data[:first_n_samples]

        # we append a lagged history of the target series to the input data,
        # so that we can train with teacher forcing
        lagged_target_history = decoder_target_data[:, :-1, :1]
        encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)

        print(encoder_input_data.shape)
        print(decoder_target_data.shape)

        self.model.compile(Adam(), loss='mean_absolute_error')
        self.history = self.model.fit(encoder_input_data, decoder_target_data,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      validation_split=0.2)

    def predict(self, input_seq, target, feed_truth):
        history_sequence = input_seq.copy()
        pred_sequence = np.zeros((1, self.pred_steps, 1))  # initialize output (pred_steps time steps)

        for i in range(self.pred_steps):

            # record next time step prediction (last time step of model output)
            last_step_pred = self.model.predict(history_sequence)[0, -1, 0]
            pred_sequence[0, i, 0] = last_step_pred

            if feed_truth:
                last_step_pred = target[i][0]

            # add the next time step prediction to the history sequence
            history_sequence = np.concatenate([history_sequence,
                                               last_step_pred.reshape(-1, 1, 1)], axis=1)

        return pred_sequence

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)
