# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

from typing import Any, Dict, Callable

from causalml.inference.tree.uplift import UpliftTreeClassifier as UpliftTreeClassifierCM

from econml.dml import CausalForestDML

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold


# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class UpliftTune:
    def __init__(
        self,
        uplift_model_class: Any,
        data: np.ndarray,
        target: np.ndarray,
        treatment: np.ndarray,
        space: dict,
        base_model_class: Any = None,
        rnd_seed: int = 42,
        max_evals: int = 50,
        cv: int = 3,
        verbose: bool = False,
    ):
        self.base_model_class = base_model_class
        self.uplift_model_class = uplift_model_class
        self.data = data
        self.target = target
        self.treatment = treatment
        self.space = space
        self.rnd_seed = rnd_seed
        self.max_evals = max_evals
        self.cv = cv
        self.verbose = verbose

        self.trials = Trials()
        self.best_params = None
        self.best_score = None

    def tune(self) -> Dict:
        objective = self._create_objective()

        best = fmin(
            fn=objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials,
            rstate=np.random.default_rng(self.rnd_seed),
            show_progressbar=self.verbose,
        )

        self.best_params = space_eval(self.space, best)
        self.best_score = -min(self.trials.losses())

        if self.verbose:
            print(f"Optimization completed. Best score: {self.best_score:.4f}")
            print(f"Best parameters: {self.best_params}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trials": self.trials,
        }

    def _create_objective(self) -> Callable:
        def objective(params: Dict) -> Dict:
            try:
                if self.base_model_class:
                    model = self.base_model_class(**params)
                    uplift_model = self.uplift_model_class(model)
                else:
                    uplift_model = self.uplift_model_class(**params)

                score = self._cross_val_score(
                    model=uplift_model,
                    X=self.data,
                    y=self.target,
                    treatment=self.treatment,
                    cv=self.cv,
                )

                return {"loss": -score, "status": STATUS_OK, "params": params}

            except Exception as e:
                if self.verbose:
                    print(f"Error with params {params}: {e}")
                return {"loss": 0.0, "status": STATUS_OK, "params": params}

        return objective

    def _cross_val_score(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
        cv: int = 3,
    ) -> float:
        scores = []

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.rnd_seed)

        for train_idx, val_idx in skf.split(X, treatment):
            if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                treatment_train, treatment_val = treatment[train_idx], treatment[val_idx]                
            else:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                treatment_train, treatment_val = treatment.iloc[train_idx], treatment.iloc[val_idx]

            try:
                if isinstance(model, UpliftTreeClassifierCM):
                    model.fit(X_train, treatment_train, y_train)
                    predictions = model.predict(X_val)[:, 0]
                if isinstance(model, CausalForestDML):
                    model.fit(Y=y_train, T=treatment_train, X=X_train)
                    predictions = model.effect(X_val)
                else:
                    model.fit(X_train, y_train, treatment_train)
                    predictions = model.predict(X_val)

                score = self._calculate_uplift_score(
                    predictions=predictions, treatment=treatment_val, target=y_val
                )

                scores.append(score)

            except Exception as e:
                if self.verbose:
                    print(f"Error in fold: {e}")
                scores.append(0.0)

        return np.mean(scores) if scores else 0.0

    def _calculate_uplift_score(
        self,
        predictions: np.ndarray,
        treatment: np.ndarray,
        target: np.ndarray,
        rate: float = 0.3
    ) -> float:

        return self._uplift_at_k_score(predictions, treatment, target, rate)

    def _uplift_at_k_score(
        self,
        predictions: np.ndarray,
        treatment: np.ndarray,
        target: np.ndarray,
        rate: float = 0.3,
    ) -> float:
        order = np.argsort(-predictions)

        n_total = len(predictions)
        n_top = int(n_total * rate)

        top_indices = order[:n_top]

        if isinstance(treatment, np.ndarray):
            treatment = pd.Series(treatment).map({"treatment": 1, "control": 0})
            target = pd.Series(target)

        treatment_mask = treatment.iloc[top_indices] == 1
        control_mask = treatment.iloc[top_indices] == 0

        treatment_conv = (
            target.iloc[top_indices][treatment_mask].mean() if treatment_mask.any() else 0
        )
        control_conv = (
            target.iloc[top_indices][control_mask].mean() if control_mask.any() else 0
        )

        uplift = treatment_conv - control_conv

        return uplift

    def plot_optimization_history(self):
        import matplotlib.pyplot as plt

        losses = [trial["result"]["loss"] for trial in self.trials.trials]
        iterations = range(1, len(losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, losses, "b-o", linewidth=2, markersize=6)
        plt.xlabel("Iteration")
        plt.ylabel("Loss (negative uplift)")
        plt.title("Hyperparameter Optimization History")
        plt.grid(True, alpha=0.3)

        best_idx = np.argmin(losses)
        plt.scatter(
            best_idx + 1,
            losses[best_idx],
            color="red",
            s=100,
            zorder=5,
            label=f"Best: {-losses[best_idx]:.4f}",
        )

        plt.legend()
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
