from WikiModel import WikiModel
from WikiSeries import WikiSeries
from lightgbm import LGBMRegressor
from sklearn.externals import joblib


class LGBM(WikiModel):
    def __init__(self, series: WikiSeries, pred_steps):
        super().__init__(series, pred_steps, summary=False)
        self.series.encode_meta()
        print("Extracting series features")
        #TODO how to do with median? Store elsewhere and then copy again?
        self.series.prepare_series_classic()
        print("Melting dataset")
        self.series.melt()
        print("Extracting temporal features")
        # TODO here again we have to extract everywhere?
        print("Extracting lag?")
        # TODO how will this work?

    def build_model(self):
        self.model = LGBMRegressor(n_estimators=1000, learning_rate=0.01, categorical_feature=[0, 1, 2, 3])
