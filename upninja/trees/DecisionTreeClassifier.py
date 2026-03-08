# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from typing import Optional

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class Node:
    """
    Класс узла решающешо дерева.
    """

    def __init__(
        self, feature=None, threshold=None, left=None, right=None, gain=None, value=None
    ):
        """
        Инициализирует новый экземпляр класса Node.

        Аргументы:
            feature: признак, используемый для разделения в этом узле. Значение по умолчанию равно None.
            threshold: порог, используемый для разделения на этом узле. Значение по умолчанию равно None.
            left: левый дочерний узел. По умолчанию равно None.
            right: правый дочерний узел. По умолчанию равно None.
            gain: прирост информации при разделении. По умолчанию равно None.
            value: если этот узел является конечным, этот атрибут представляет прогнозируемое значение
                для целевой переменной. По умолчанию установлено значение "Нет".
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_samples: int = 2,
        max_depth: int = 2,
        criterion: str = "entropy",
        bins: Optional[int] = None
    ):
        """
        Конструктор для класса DecisionTree.

        Параметры:
            min_samples (int): минимальное количество выборок, необходимое для разделения внутреннего узла.
            max_depth (int): максимальная глубина дерева решений.
            criterion (srt): критерий построения разбиения в узлах дерева.
            bins (int): бининг - ускоряет построение модели на больших данных.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.criterion = criterion if criterion in ("entropy", "gini") else "entropy"
        self.bins = bins

    def _entropy(self, y: np.ndarray) -> float:
        """
        Вычисляет энтропийный критерий разбиения по формуле:
            entropy = sum_{i=0}^{N} p_i * log_2(p_i)
        """
        entropy = 0.0
        n = len(y)
        for label in np.unique(y):
            p = np.sum(y == label) / n
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _gini(self, y: np.ndarray) -> float:
        """
        Вычисляет gini:
            gini = 1 - sum_{i=0}^{N}p_i**2
        """
        gini = 1.0
        n = len(y)
        for label in np.unique(y):
            p = np.sum(y == label) / n
            gini -= p ** 2
        return gini

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == "entropy":
            return self._entropy(y)
        else:
            return self._gini(y)

    def _information_gain(
        self, parent: np.ndarray, left: np.ndarray, right: np.ndarray
    ) -> float:
        n = len(parent)
        wl = len(left) / n
        wr = len(right) / n
        gain = self._impurity(parent) - (wl * self._impurity(left) + wr * self._impurity(right))
        return gain

    def _get_thresholds(self, values: np.ndarray) -> np.ndarray:
        unique_vals = np.sort(np.unique(values))
        if len(unique_vals) < 2:
            return np.array([])
        if self.bins is not None and len(unique_vals) - 1 >= self.bins:
            hist_bins = np.histogram(values, bins=self.bins)[1]
            thresholds = hist_bins[1:-1]
        else:
            thresholds = np.convolve(unique_vals, [0.5, 0.5], mode="valid")
        return thresholds

    def _is_leaf(self, y: np.ndarray, n_samples: int, depth: int) -> bool:
        if len(np.unique(y)) == 1:
            return True
        if depth >= self.max_depth:
            return True
        if n_samples < self.min_samples:
            return True
        return False

    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        values, counts = np.unique(y, return_counts=True)
        return float(values[np.argmax(counts)])

    def _calculate_leaf_proba(self, y: np.ndarray) -> float:
        # Возвращает вероятность класса 1.
        return np.mean(y == self.classes_[1]) if len(self.classes_) == 2 else np.mean(y)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> dict:
        best_split = {"gain": -1, "feature": None, "threshold": None}
        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            thresholds = self._split_values.get(feature_idx, None)
            if thresholds is None:
                thresholds = self._get_thresholds(X[:, feature_idx])
                self._split_values[feature_idx] = thresholds

            for thresh in thresholds:
                left_mask = X[:, feature_idx] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                gain = self._information_gain(y, y_left, y_right)

                if gain > best_split["gain"]:
                    best_split.update({
                        "gain": gain,
                        "feature": feature_idx,
                        "threshold": thresh,
                        "left_X": X[left_mask],
                        "right_X": X[right_mask],
                        "left_y": y_left,
                        "right_y": y_right
                    })

        return best_split

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples = X.shape[0]

        if self._is_leaf(y, n_samples, depth):
            proba = self._calculate_leaf_proba(y)
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value, gain=proba)

        best_split = self._best_split(X, y)
        if best_split["gain"] <= 0 or best_split["feature"] is None:
            proba = self._calculate_leaf_proba(y)
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value, gain=proba)

        # Update feature importance
        weight = n_samples / self.n_samples_
        self.feature_importances_[best_split["feature"]] += weight * best_split["gain"]

        left_child = self._build_tree(
            best_split["left_X"], best_split["left_y"], depth + 1
        )
        right_child = self._build_tree(
            best_split["right_X"], best_split["right_y"], depth + 1
        )

        return Node(
            feature=best_split["feature"],
            threshold=best_split["threshold"],
            left=left_child,
            right=right_child,
            gain=best_split["gain"]
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Only binary classification is supported.")

        self.n_samples_, self.n_features_in_ = X.shape
        self.feature_importances_ = np.zeros(self.n_features_in_)
        self._split_values = {}

        self.root = self._build_tree(X, y)
        self._fitted = True
        return self

    def _predict_row_proba(self, x: np.ndarray) -> float:
        node = self.root
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.gain  # stores class probability

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_fitted") or not self._fitted:
            raise ValueError("This DecisionTreeClassifier instance is not fitted yet.")

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        probas = np.array([self._predict_row_proba(row) for row in X])
        # Return shape (n_samples, 2)
        return np.vstack((1 - probas, probas)).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # искусственные данные
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 5], [5, 4], [6, 6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    y = y.reshape((X.shape[0], 1))

    clf = DecisionTreeClassifier(min_samples=2, max_depth=3)
    clf.fit(X, y)

    predictions = clf.predict(X)
    print(predictions)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
