import os

import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam

from WikiModel import WikiModel
from WikiSeries import WikiSeries


class Seq2Seq(WikiModel):
    ENCODER_PKL = "encoder.h5"
    DECODER_PKL = "decoder.h5"

    def __init__(self, series: WikiSeries, pred_steps) -> None:
        super().__init__(series, pred_steps)

    def build_model(self):
        self.latent_dim = 50  # LSTM hidden units
        dropout = .20

        # Define an input series and encode it with an LSTM.
        self.encoder_inputs = Input(shape=(None, 1))
        encoder = LSTM(self.latent_dim, dropout=dropout, return_state=True)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)

        # We discard `encoder_outputs` and only keep the final states. These represent the "context"
        # vector that we use as the basis for decoding.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        # This is where teacher forcing inputs are fed in.
        self.decoder_inputs = Input(shape=(None, 1))

        # We set up our decoder using `encoder_states` as initial state.
        # We return full output sequences and return internal states as well.
        # We don't use the return states in the training model, but we will use them in inference.
        self.decoder_lstm = LSTM(self.latent_dim, dropout=dropout, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                                  initial_state=self.encoder_states)

        self.decoder_dense = Dense(1)  # 1 continuous output at each timestep
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

    def fit(self, epochs=10, batch_size=2 ** 11):
        first_n_samples = 20000

        encoder_input_data = self.series.training_encoder
        decoder_target_data = self.series.training_decoder

        encoder_input_data = encoder_input_data[:first_n_samples]
        decoder_target_data = decoder_target_data[:first_n_samples]

        # lagged target series for teacher forcing
        decoder_input_data = np.zeros(decoder_target_data.shape)
        decoder_input_data[:, 1:, 0] = decoder_target_data[:, :-1, 0]
        decoder_input_data[:, 0, 0] = encoder_input_data[:, -1, 0]

        self.model.compile(Adam(), loss='mean_absolute_error')
        self.history = self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      validation_split=0.2)

        # from our previous model - mapping encoder sequence to state vectors
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        # A modified version of the decoding stage that takes in predicted target inputs
        # and encoded state vectors, returning predicted target outputs and decoder state vectors.
        # We need to hang onto these state vectors to run the next step of the inference loop.
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]

        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model = Model([self.decoder_inputs] + decoder_states_inputs,
                                   [decoder_outputs] + decoder_states)

    def predict(self, input_seq, target, feed_truth):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, 1))

        # Populate the first target sequence with end of encoding series pageviews
        target_seq[0, 0, 0] = input_seq[0, -1, 0]

        # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
        # (to simplify, here we assume a batch of size 1).

        decoded_seq = np.zeros((1, self.pred_steps, 1))

        for i in range(self.pred_steps):

            output, h, c = self.decoder_model.predict([target_seq] + states_value)

            decoded_seq[0, i, 0] = output[0, 0, 0]

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 1))
            if feed_truth:
                target_seq[0, 0, 0] = target[i][0]
            else:
                target_seq[0, 0, 0] = output[0, 0, 0]

            # Update states
            states_value = [h, c]

        return decoded_seq

    def save_model(self, path):
        try:
            if not os.path.isdir(path):
                os.mkdir(path)

            encoder_path = os.path.join(path, self.ENCODER_PKL)
            decoder_path = os.path.join(path, self.DECODER_PKL)

            self.encoder_model.save(encoder_path)
            self.decoder_model.save(decoder_path)
        except:
            print("Couldn't save your model")

    def load_model(self, path):
        try:
            encoder_path = os.path.join(path, self.ENCODER_PKL)
            decoder_path = os.path.join(path, self.DECODER_PKL)

            self.encoder_model = load_model(encoder_path)

            self.decoder_model = load_model(decoder_path)

        except:
            print("Couldn't load your model")
