# ----------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import random
from typing import Optional
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from .UpliftTreeClassifier import UpliftTreeClassifier

# ----------------------------------------------------------------------------------------------------------------------------------------


class UpliftRandomForest:
    def __init__(
        self,
        n_estimators: int = 10,
        max_features: float = 0.5,
        max_samples: float = 0.5,
        max_depth: int = 3,
        min_samples: int = 200,
        bins: Optional[int] = None,
        forest_criterion: str = "none",
        tree_criterion: str = "linear",
        random_state: int = 42,
        min_samples_treatment: int = 10,
        control_name: int = 0,
        treatment_name: int = 1,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        # Forest parameters
        self.n_estimators = n_estimators
        self.max_features = max_features if 0.0 <= max_features <= 1.0 else 0.5
        self.max_samples = max_samples if 0.0 <= max_samples <= 1.0 else 0.5
        
        # Tree parameters
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.bins = bins
        self.forest_criterion = forest_criterion
        self.tree_criterion = tree_criterion
        self.min_samples_treatment = min_samples_treatment
        self.control_name = control_name
        self.treatment_name = treatment_name
        
        # Execution
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Storage
        self.forest = []
        self._fitted = False
        self._using_dml = False

    def __str__(self):
        atr = self.__dict__
        res = "".join([i + "=" + str(atr[i]) + "," + " " for i in atr])[:-2]
        return "UpliftRandomForest class: " + res

    def _preprocess_dml(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray):
        """Global DML preprocessing using YOUR models if available."""
        if self.forest_criterion != "dml":
            return None

        # Propensity model: P(T=1 | X)
        w_binary = (w == self.treatment_name).astype(int)
        prop_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=self.random_state)
        prop_model.fit(X, w_binary)
        e_hat = prop_model.predict_proba(X)[:, 1]
        e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)

        # Outcome model: E[Y | X, T]
        X_w = np.hstack([X.values, w.reshape(-1, 1)])
        

        mu_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=self.random_state)

        mu_model.fit(X_w, y)
        mu_hat = mu_model.predict(X_w)

        # DML pseudo-outcome
        pseudo_y = (w_binary - e_hat) / (e_hat * (1 - e_hat)) * (y - mu_hat)
        self._using_dml = True
        return pseudo_y

    def _fit_single_tree(self, X, y, w, cols_idx, rows_idx, tree_idx, pseudo_y=None):
        """Fit a single tree with specified tree_criterion."""
        X_sample = X.iloc[rows_idx][cols_idx].values
        w_sample = w.iloc[rows_idx].values

        if self._using_dml:
            y_sample = pseudo_y[rows_idx]
            # В DML-режиме дерево работает как регрессор на pseudo_y
            tree = UpliftTreeClassifier(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                bins=self.bins,
                criterion=self.tree_criterion,
                min_samples_treatment=self.min_samples_treatment,
                control_name=self.control_name,
                treatment_name=self.treatment_name,
            )
        else:
            y_sample = y.iloc[rows_idx].values
            tree = UpliftTreeClassifier(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                bins=self.bins,
                criterion=self.tree_criterion,
                min_samples_treatment=self.min_samples_treatment,
                control_name=self.control_name,
                treatment_name=self.treatment_name,
            )

        tree.fit(X_sample, y_sample, w_sample)
        return (tree, cols_idx)

    def fit(self, X: pd.DataFrame, y: pd.Series, w: pd.Series) -> None:
        n_samples, n_features = X.shape
        random.seed(self.random_state)
        init_cols = list(X.columns)
        init_rows = list(range(n_samples))
        cols_sample_cnt = max(1, int(np.round(self.max_features * n_features)))
        rows_sample_cnt = max(1, int(np.round(self.max_samples * n_samples)))

        # DML preprocessing (only if forest_criterion == 'dml')
        pseudo_y_global = None
        if self.forest_criterion == "dml":
            if self.verbose > 0:
                print("[UpliftRandomForest] Fitting global DML models...")
            pseudo_y_global = self._preprocess_dml(X, y.values, w.values)
            if self.verbose > 0:
                print("[UpliftRandomForest] DML preprocessing done.")

        # Bootstrap samples
        bootstrap_samples = []
        for i in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_sample_cnt)
            rows_idx = random.choices(init_rows, k=rows_sample_cnt)
            bootstrap_samples.append((cols_idx, rows_idx, i))

        # Fit trees
        if self.n_jobs is not None and self.n_jobs != 1:
            if self.verbose > 0:
                print(f"[UpliftRandomForest] Fitting {self.n_estimators} trees in parallel (n_jobs={self.n_jobs})...")

            self.forest = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self._fit_single_tree)(
                    X, y, w, cols_idx, rows_idx, tree_idx, pseudo_y_global
                )
                for cols_idx, rows_idx, tree_idx in bootstrap_samples
            )
        else:
            if self.verbose > 0:
                print(f"[UpliftRandomForest] Fitting {self.n_estimators} trees sequentially...")

            self.forest = []
            for i, (cols_idx, rows_idx, tree_idx) in enumerate(bootstrap_samples):
                if self.verbose > 0 and (i + 1) % self.verbose == 0:
                    print(f"[UpliftRandomForest] Tree {i + 1}/{self.n_estimators}")

                tree_tuple = self._fit_single_tree(
                    X, y, w, cols_idx, rows_idx, tree_idx, pseudo_y_global
                )
                self.forest.append(tree_tuple)

        self._fitted = True
        if self.verbose > 0:
            print("[UpliftRandomForest] Fitting completed.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError("This UpliftRandomForest instance is not fitted yet.")

        predictions = np.zeros(X.shape[0])
        for tree, cols_idx in self.forest:
            X_sub = X[cols_idx].values
            pred = tree.predict(X_sub)
            predictions += pred

        return predictions / self.n_estimators

# ----------------------------------------------------------------------------------------------------------------------------------------
