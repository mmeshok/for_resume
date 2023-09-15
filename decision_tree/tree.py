from __future__ import annotations


from dataclasses import dataclass
import numpy as np


@dataclass
class Node:
    """Decision tree node."""

    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None


@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""

    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y).

        Args:
            X (np.ndarray): train data
            y (np.ndarray): train target values

        Returns:
            DecisionTreeRegressor: fitted tree
        """

        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        mse = ((y - y.mean()) ** 2).mean()
        return mse

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weithed mse criterion for a two given sets of target values"""
        mse_left = self._mse(y_left)
        mse_right = self._mse(y_right)
        weighted_mse = (mse_left * len(y_left) + mse_right * len(y_right)) / (
            len(y_left) + len(y_right)
        )
        return weighted_mse

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        feature_errors = []
        for feature in range(X.shape[1]):
            errors = []
            thresholds = sorted([x for x in set(X[:, feature])])[:-1]
            for threshold in thresholds:
                y_left = y[X[:, feature] <= threshold]
                y_right = y[X[:, feature] > threshold]
                errors.append([self._weighted_mse(y_left, y_right), threshold])
            errors.sort()
            feature_errors.append([*errors[0], feature])
        feature_errors.sort()
        best_idx, best_thr = feature_errors[0][2], feature_errors[0][1]
        return best_idx, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        if len(y) < self.min_samples_split or depth > self.max_depth:
            return None

        node = Node()
        try:
            node.feature, node.threshold = self._best_split(X, y)
            node.n_samples = X.shape[0]
            node.value = int(np.round(y.mean()))
            node.mse = self._mse(y)
            node.left = self._split_node(
                X[X[:, node.feature] <= node.threshold],
                y[X[:, node.feature] <= node.threshold],
                depth + 1,
            )

            node.right = self._split_node(
                X[X[:, node.feature] > node.threshold],
                y[X[:, node.feature] > node.threshold],
                depth + 1,
            )

        except:
            return None

        return node

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        return self._as_json(self.tree_)

    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        if (not node.left) and (not node.right):
            return (
                f'{{"value": {node.value}, "n_samples": {node.n_samples},'
                f' "mse": {node.mse:.2f}}}'
            )
        else:
            json = (
                f'{{"feature": {node.feature}, "threshold": {node.threshold},'
                f' "n_samples": {node.n_samples}, "mse": {node.mse:.2f},'
                f' "left": {self._as_json(node.left)},'
                f' "right": {self._as_json(node.right)}}}'
            )
            return json

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X.

        Args:
            X (np.ndarray): shape (n_samples, n_features)
                            The input samples.

        Returns:
            np.ndarray: array of shape (n_samples,)
                        The predicted values.
        """

        return np.array([self._predict_one_sample(x) for x in X])

    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""

        def search(node: Node, ftrs: np.ndarray) -> int:
            if (not node.left) and (not node.right):
                return node.value
            if ftrs[node.feature] <= node.threshold:
                return search(node.left, ftrs)
            else:
                return search(node.right, ftrs)

        return search(self.tree_, features)
