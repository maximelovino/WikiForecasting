from WikiSeries import WikiSeries
from matplotlib import pyplot as plt
import numpy as np


class WikiModel:

    def __init__(self, series: WikiSeries, pred_steps, summary=True):
        self.series = series
        self.pred_steps = pred_steps
        series.prepare_dates(pred_steps)
        self.build_model()
        if summary:
            self.model.summary()

    def fit(self, epochs=10, batch_size=2 ** 11):
        pass

    def build_model(self):
        pass

    def history_plot(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])

        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error Loss')
        plt.title('Loss Over Time')
        plt.legend(['Train', 'Valid'])

    def normalise_reshape_prediction(self, encode, target, prediction, sample_ind):
        encode = encode.reshape(-1, 1)
        prediction = prediction.reshape(-1, 1)

        prediction = self.series.denormalize_series(prediction, sample_ind)
        encode = self.series.denormalize_series(encode, sample_ind)
        target = self.series.denormalize_series(target, sample_ind)

        return encode, target, prediction

    
    def normalise_reshape_prediction_batches(self, encode_all, target_all, prediction_all, samples_indexes):
        encode = encode_all[:,:,0].T
        prediction = prediction_all[:,:,0].T

        prediction = self.series.denormalize_series(prediction, samples_indexes)
        encode = self.series.denormalize_series(encode, samples_indexes)
        target = self.series.denormalize_series(target_all, samples_indexes)

        return encode, target, prediction
        
        
    def predict(self, input_seq, target, feed_truth):
        pass

    def predictBatch(self, input_seq, target, feed_truth, batch=2**10):
        history_sequence = input_seq.copy()
        pred_sequence = np.zeros((history_sequence.shape[0], self.pred_steps, 1))  # initialize output (pred_steps time steps)
        print(target.shape)
        print(input_seq.shape)
        

        for i in range(self.pred_steps):

            # record next time step prediction (last time step of model output)
            last_step_pred = self.model.predict(history_sequence,batch_size=batch)[:, -1, 0]
            pred_sequence[:, i, 0] = last_step_pred
            
            if feed_truth:
                last_step_pred = target[i,:]
                
            # add the next time step prediction to the history sequence
            history_sequence = np.concatenate([history_sequence,
                                               last_step_pred.reshape(-1, 1, 1)], axis=1)

        return pred_sequence
    
    
    def save_model(self, path):
        pass

    def load_model(self, path):
        pass
