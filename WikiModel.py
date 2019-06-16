from WikiSeries import WikiSeries
from matplotlib import pyplot as plt


class WikiModel:

    def __init__(self, series: WikiSeries, pred_steps, summary=True):
        self.series = series
        self.pred_steps = pred_steps
        series.prepare_dates(pred_steps)
        self.build_model()
        if summary:
            self.model.summary()

    def fit(self, epochs=10):
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

    def predict(self, input_seq, target, feed_truth):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass
