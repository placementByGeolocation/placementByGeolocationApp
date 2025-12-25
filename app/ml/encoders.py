from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FixedMeanTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Mean Target Encoding только для строго фиксированных колонок.
    Никаких параметров cols снаружи нет.
    """

    FIXED_COLS = [
        'name', 'brand', 'nearest_museum_name',
        'nearest_theatre_name', 'nearest_gallery_name',
        'nearest_park_name'
    ]

    # если хотим - фиксируем сглаживание (можно 0.0)
    SMOOTHING = 0.0

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y)

        self.global_mean_ = float(np.mean(y))
        self.mapping_ = {}

        cols_present = [c for c in self.FIXED_COLS if c in X.columns]

        for col in cols_present:
            tmp = pd.DataFrame({"x": X[col], "y": y})
            stats = tmp.groupby("x")["y"].agg(["mean", "count"])

            if self.SMOOTHING and self.SMOOTHING > 0:
                enc = (stats["count"] * stats["mean"] + self.SMOOTHING * self.global_mean_) / (stats["count"] + self.SMOOTHING)
            else:
                enc = stats["mean"]

            self.mapping_[col] = enc

        self.cols_ = cols_present
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in self.cols_:
            X[col] = X[col].map(self.mapping_[col]).fillna(self.global_mean_).astype(float)

        return X
