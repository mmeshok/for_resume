from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    """Gradient boosting regressor."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
        subsample_size=0.5,
        replace=False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.base_pred_: float = None
        self.trees_: list = None
        self.subsample_size = subsample_size
        self.replace = replace

    def _subsample(self, X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """Return subsample of data."""
        size = int(len(X) * self.subsample_size)
        idx = [np.random.choice(len(X), size=size, replace=self.replace)]
        sub_X = X[tuple(idx)]
        sub_y = y[tuple(idx)]
        return sub_X, sub_y

    def _mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> (float, np.ndarray):
        """Calculate MSE loss function and gradient."""
        loss = ((y_pred - y_true) ** 2).mean()
        grad = y_pred - y_true
        if self.verbose:
            print(loss, grad)
        return loss, grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        """Fit the model to the data.

        Args:
            X (np.ndarray of shape (n_samples, n_features)): train data
            y (np.ndarray): target values

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        if not self.trees_:
            _, sub_y = self._subsample(X, y)
            self.base_pred_ = sub_y.mean()
            self.trees_ = []

        for i in range(self.n_estimators):
            if self.loss == "mse":
                loss, grad = self._mse(y, self.predict(X))
            else:
                loss, grad = self.loss(y, self.predict(X))

            sub_X, sub_grad = self._subsample(X, grad)

            self.trees_.append(
                DecisionTreeRegressor(
                    max_depth=self.max_depth, min_samples_split=self.min_samples_split
                )
            )
            self.trees_[-1].fit(sub_X, -sub_grad)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target values.

        Args:
            X (np.ndarray of shape (n_samples, n_features)): input data

        Returns:
            predictions (np.ndarray): Predict values.
        """
        predictions = np.array([self.base_pred_] * X.shape[0])
        for tree in self.trees_:
            predictions += tree.predict(X) * self.learning_rate

        return predictions
