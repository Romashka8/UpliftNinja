# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class Node:
    """
    Класс узла решающего дерева.
    """

    def __init__(
        self, feature=None, threshold=None, left=None, right=None, gain=None, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value


class UpliftTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_samples: int = 200,
        max_depth: int = 3,
        criterion: str = "linear",
        bins: Optional[int] = None,
        min_samples_treatment: int = 10,
        control_name: int = 0,
        treatment_name: int = 1,
    ):
        """
        Конструктор для uplift-дерева.

        Параметры:
            min_samples (int): минимальное общее число наблюдений в узле для разбиения.
            max_depth (int): максимальная глубина дерева.
            criterion (str): критерий разбиения — 'uplift' (по умолчанию), 'kl', 'chi2', 'delta'.
            bins (int): опциональное количество бинов для ускорения поиска порогов.
            min_samples_treatment (int): минимальное число наблюдений в группе treatment и control в каждом дочернем узле.
            control_name (int): метка контрольной группы в векторе treatment.
            treatment_name (int): метка экспериментальной группы в векторе treatment.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.criterion = criterion
        self.criterion_f = self._weighted_uplift_sq if self.criterion == "squared" else self._linear_uplift
        self.bins = bins
        self.min_samples_treatment = min_samples_treatment
        self.control_name = control_name
        self.treatment_name = treatment_name

    @staticmethod
    def _weighted_uplift_sq(stats):
            uplift = stats["uplift"]
            n_t, n_c = stats["n_t"], stats["n_c"]
            weight = (n_t * n_c) / (n_t + n_c + 1e-8)
            return uplift ** 2 * weight

    @staticmethod
    def _linear_uplift(stats):
        uplift = stats["uplift"]
        n_t, n_c = stats["n_t"], stats["n_c"]
        weight = np.sqrt(n_t * n_c + 1e-8)
        return uplift * weight

    def _compute_uplift_stats(self, y: np.ndarray, w: np.ndarray):
        """
        Вычисляет статистики по группам: конверсии, размеры.
        y — целевая переменная (0/1), w — treatment indicator (control/treatment).
        """
        mask_t = w == self.treatment_name
        mask_c = w == self.control_name

        n_t = np.sum(mask_t)
        n_c = np.sum(mask_c)
        if n_t == 0 or n_c == 0:
            return None

        p_t = np.mean(y[mask_t]) if n_t > 0 else 0.0
        p_c = np.mean(y[mask_c]) if n_c > 0 else 0.0

        return {
            "n_t": n_t,
            "n_c": n_c,
            "p_t": p_t,
            "p_c": p_c,
            "uplift": p_t - p_c,
        }

    def _uplift_criterion(self, stats_left, stats_right):
        """Критерий на основе взвешенного квадрата uplift."""
        if stats_left is None or stats_right is None:
            return -np.inf

        # Веса подузлов
        nL = stats_left["n_t"] + stats_left["n_c"]
        nR = stats_right["n_t"] + stats_right["n_c"]
        N = nL + nR + 1e-8

        gain = (nL / N) * self.criterion_f(stats_left) + (nR / N) * self.criterion_f(stats_right)
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

    def _is_leaf(self, y: np.ndarray, w: np.ndarray, n_samples: int, depth: int) -> bool:
        if depth >= self.max_depth:
            return True
        if n_samples < self.min_samples:
            return True

        # Проверка минимального размера treatment/control
        n_t = np.sum(w == self.treatment_name)
        n_c = np.sum(w == self.control_name)
        if n_t < self.min_samples_treatment or n_c < self.min_samples_treatment:
            return True

        # Если uplift статистически неотличим от нуля
        stats = self._compute_uplift_stats(y, w)
        if stats is not None and abs(stats["uplift"]) < 1e-8:
            return True

        return False

    def _calculate_leaf_value(self, y: np.ndarray, w: np.ndarray) -> float:
        stats = self._compute_uplift_stats(y, w)
        return stats["uplift"] if stats is not None else 0.0

    def _best_split(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
        best_split = {"gain": -np.inf, "feature": None, "threshold": None}
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

                # Проверка минимального размера treatment/control в обоих потомках
                w_left, w_right = w[left_mask], w[right_mask]
                n_t_l = np.sum(w_left == self.treatment_name)
                n_c_l = np.sum(w_left == self.control_name)
                n_t_r = np.sum(w_right == self.treatment_name)
                n_c_r = np.sum(w_right == self.control_name)

                if (n_t_l < self.min_samples_treatment or n_c_l < self.min_samples_treatment or
                    n_t_r < self.min_samples_treatment or n_c_r < self.min_samples_treatment):
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                stats_left = self._compute_uplift_stats(y_left, w_left)
                stats_right = self._compute_uplift_stats(y_right, w_right)

                gain = self._uplift_criterion(stats_left, stats_right)

                if gain > best_split["gain"]:
                    best_split.update({
                        "gain": gain,
                        "feature": feature_idx,
                        "threshold": thresh,
                        "left_X": X[left_mask],
                        "right_X": X[right_mask],
                        "left_y": y_left,
                        "right_y": y_right,
                        "left_w": w_left,
                        "right_w": w_right,
                    })

        return best_split

    def _build_tree(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int = 0) -> Node:
        n_samples = X.shape[0]

        if self._is_leaf(y, w, n_samples, depth):
            leaf_value = self._calculate_leaf_value(y, w)
            return Node(value=leaf_value, gain=leaf_value)

        best_split = self._best_split(X, y, w)
        if best_split["gain"] == -np.inf or best_split["feature"] is None:
            leaf_value = self._calculate_leaf_value(y, w)
            return Node(value=leaf_value, gain=leaf_value)

        left_child = self._build_tree(
            best_split["left_X"], best_split["left_y"], best_split["left_w"], depth + 1
        )
        right_child = self._build_tree(
            best_split["right_X"], best_split["right_y"], best_split["right_w"], depth + 1
        )

        return Node(
            feature=best_split["feature"],
            threshold=best_split["threshold"],
            left=left_child,
            right=right_child,
            gain=best_split["gain"]
        )

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(w, np.ndarray):
            w = np.array(w)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if w.ndim == 2 and w.shape[1] == 1:
            w = w.ravel()

        # Проверка уникальных значений treatment
        unique_w = np.unique(w)
        if not ({self.control_name, self.treatment_name} <= set(unique_w)):
            raise ValueError(f"Treatment vector must contain both {self.control_name} (control) and {self.treatment_name} (treatment).")

        self.n_samples_, self.n_features_in_ = X.shape
        self._split_values = {}

        self.root = self._build_tree(X, y, w)
        self._fitted = True
        return self

    def _predict_row(self, x: np.ndarray) -> float:
        node = self.root
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_fitted") or not self._fitted:
            raise ValueError("This UpliftTreeClassifier instance is not fitted yet.")

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        predictions = np.array([self._predict_row(row) for row in X])
        return predictions

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
