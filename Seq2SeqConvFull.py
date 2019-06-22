import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from keras.optimizers import Adam
from keras.models import load_model

from WikiModel import WikiModel
from WikiSeries import WikiSeries


class Seq2SeqConvFull(WikiModel):
    def __init__(self, series: WikiSeries, pred_steps) -> None:
        super().__init__(series, pred_steps)

    def build_model(self):
        # convolutional layer parameters
        n_filters = 32
        filter_width = 2
        power_2 = 11
        dilation_rates = [2 ** i for i in range(power_2)] * 2

        # define an input history series and pass it through a stack of dilated causal convolutions.
        history_seq = Input(shape=(None, 1))
        x = history_seq

        skips = []
        for dilation_rate in dilation_rates:
            # preprocessing - equivalent to time-distributed dense
            x = Conv1D(16, 1, padding='same', activation='relu')(x)

            # filter convolution
            x_f = Conv1D(filters=n_filters,
                         kernel_size=filter_width,
                         padding='causal',
                         dilation_rate=dilation_rate)(x)

            # gating convolution
            x_g = Conv1D(filters=n_filters,
                         kernel_size=filter_width,
                         padding='causal',
                         dilation_rate=dilation_rate)(x)

            # multiply filter and gating branches
            z = Multiply()([Activation('tanh')(x_f),
                            Activation('sigmoid')(x_g)])

            # postprocessing - equivalent to time-distributed dense
            z = Conv1D(16, 1, padding='same', activation='relu')(z)

            # residual connection
            x = Add()([x, z])

            # collect skip connections
            skips.append(z)

        # add all skip connection outputs
        out = Activation('relu')(Add()(skips))

        # final time-distributed dense layers
        out = Conv1D(2 ** (power_2 - 1), 1, padding='same')(out)
        out = Activation('relu')(out)
        out = Dropout(.2)(out)
        out = Conv1D(1, 1, padding='same')(out)

        # extract the last 60 time steps as the training target
        def slice(x, seq_length):
            return x[:, -seq_length:, :]

        pred_seq_train = Lambda(slice, arguments={'seq_length': 60})(out)

        self.model = Model(history_seq, pred_seq_train)
        self.model.compile(Adam(), loss='mean_absolute_error')

    def fit(self, epochs=10, batch_size=2 ** 11):
        first_n_samples = 40000

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
