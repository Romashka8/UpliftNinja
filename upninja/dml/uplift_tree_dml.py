# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from typing import Optional

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class UpliftTreeRegressorDML(BaseEstimator, RegressorMixin):
    """
    Uplift-tree, using sklearn.tree.DecisionTreeRegressor.
    """

    def __init__(
        self,
        min_samples: int = 200,
        max_depth: int = 3,
        control_name: int = 0,
        treatment_name: int = 1,
        random_state: Optional[int] = None,
    ):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.control_name = control_name
        self.treatment_name = treatment_name
        self.random_state = random_state

    def _compute_pseudo_outcome(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """Transform (X, y, w) -> pseudo_y for regressor."""

        # DML: fit propensity and build pseudo-outcome
        w_binary = (w == self.treatment_name).astype(int)
        prop_model = LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=self.random_state
        )
        prop_model.fit(X, w_binary)
        e_hat = prop_model.predict_proba(X)[:, 1]
        e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)
        mu_hat = y.mean()  # simple eval E[Y]
        pseudo_y = (w_binary - e_hat) / (e_hat * (1 - e_hat)) * (y - mu_hat)
        return pseudo_y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> "UpliftTreeRegressorDML":
        X, y, w = np.asarray(X), np.asarray(y), np.asarray(w)
        if len(y) != X.shape[0] or len(w) != X.shape[0]:
            raise ValueError("X, y, and w must contain the same number of samples.")

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if w.ndim == 2 and w.shape[1] == 1:
            w = w.ravel()

        unique_w = np.unique(w)
        if not ({self.control_name, self.treatment_name} <= set(unique_w)):
            raise ValueError(
                f"Treatment must contain both {self.control_name} and {self.treatment_name}."
            )

        self.n_samples_, self.n_features_in_ = X.shape

        pseudo_y = self._compute_pseudo_outcome(X, y, w)
        self._tree_regressor = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples,
            random_state=self.random_state,
        )
        self._tree_regressor.fit(X, pseudo_y)
        self._is_dml = True

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict uplift values for input samples."""
        check_is_fitted(self, "_tree_regressor")

        X = np.asarray(X)
        return self._tree_regressor.predict(X)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
