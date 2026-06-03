from sklearn.base import BaseEstimator, RegressorMixin


class CurrentWeightBaselineRegressor(BaseEstimator, RegressorMixin):
    """Predict next-week smoothed weight from the current 7-day average."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X["Weight_7Day_Avg"].to_numpy()
