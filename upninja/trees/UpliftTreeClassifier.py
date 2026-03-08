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
            criterion (str): критерий разбиения — 'squared', 'linear', или 'dml'.
            bins (int): опциональное количество бинов для ускорения поиска порогов.
            min_samples_treatment (int): минимальное число наблюдений в группе treatment и control в каждом дочернем узле.
            control_name (int): метка контрольной группы в векторе treatment.
            treatment_name (int): метка экспериментальной группы в векторе treatment.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.criterion = criterion
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

    def _fit_propensity_model(self, X: np.ndarray, w: np.ndarray):
        """Обучает простую модель propensity score (логистическая регрессия)."""
        from sklearn.linear_model import LogisticRegression
        w_binary = (w == self.treatment_name).astype(int)
        model = LogisticRegression(max_iter=1000, solver='liblinear')
        model.fit(X, w_binary)
        return model

    def _compute_dml_pseudo_outcome(self, y: np.ndarray, w: np.ndarray, X: np.ndarray):
        """Вычисляет дебиазированный псевдо-outcome для DML."""
        prop_model = self._fit_propensity_model(X, w)
        e_hat = prop_model.predict_proba(X)[:, 1]
        e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)
        w_binary = (w == self.treatment_name).astype(float)
        # Используем y - E[y] как приближение; для бинарного y можно взять y - y.mean()
        mu_hat = np.full_like(y, y.mean())
        pseudo_y = (w_binary - e_hat) / (e_hat * (1 - e_hat)) * (y - mu_hat)
        return pseudo_y

    def _mse_criterion(self, pseudo_y_left: np.ndarray, pseudo_y_right: np.ndarray):
        """Критерий MSE для DML (аналог econml)."""
        nL, nR = len(pseudo_y_left), len(pseudo_y_right)
        if nL == 0 or nR == 0:
            return -np.inf

        total = np.concatenate([pseudo_y_left, pseudo_y_right])
        mse_parent = np.var(total, ddof=0)
        mse_left = np.var(pseudo_y_left, ddof=0) if nL > 1 else 0.0
        mse_right = np.var(pseudo_y_right, ddof=0) if nR > 1 else 0.0

        gain = mse_parent - (nL * mse_left + nR * mse_right) / (nL + nR)
        return gain

    def _uplift_criterion(self, stats_left, stats_right, pseudo_y_left=None, pseudo_y_right=None):
        if self.criterion == "dml":
            if pseudo_y_left is None or pseudo_y_right is None:
                return -np.inf
            return self._mse_criterion(pseudo_y_left, pseudo_y_right)
        else:
            if stats_left is None or stats_right is None:
                return -np.inf
            nL = stats_left["n_t"] + stats_left["n_c"]
            nR = stats_right["n_t"] + stats_right["n_c"]
            N = nL + nR + 1e-8
            if self.criterion == "squared":
                f = self._weighted_uplift_sq
            else:  # linear
                f = self._linear_uplift
            gain = (nL / N) * f(stats_left) + (nR / N) * f(stats_right)
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

        n_t = np.sum(w == self.treatment_name)
        n_c = np.sum(w == self.control_name)
        if n_t < self.min_samples_treatment or n_c < self.min_samples_treatment:
            return True

        if self.criterion != "dml":
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

        pseudo_y_full = None
        if self.criterion == "dml":
            pseudo_y_full = self._compute_dml_pseudo_outcome(y, w, X)

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

                w_left, w_right = w[left_mask], w[right_mask]
                n_t_l = np.sum(w_left == self.treatment_name)
                n_c_l = np.sum(w_left == self.control_name)
                n_t_r = np.sum(w_right == self.treatment_name)
                n_c_r = np.sum(w_right == self.control_name)

                if (n_t_l < self.min_samples_treatment or n_c_l < self.min_samples_treatment or
                    n_t_r < self.min_samples_treatment or n_c_r < self.min_samples_treatment):
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                if self.criterion == "dml":
                    pseudo_y_left = pseudo_y_full[left_mask]
                    pseudo_y_right = pseudo_y_full[right_mask]
                    gain = self._uplift_criterion(None, None, pseudo_y_left, pseudo_y_right)
                else:
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
