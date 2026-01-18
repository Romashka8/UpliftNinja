# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np

from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

from typing import Any, Dict, Callable

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold

import warnings

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class UpliftTune:
    def __init__(
        self,
        base_model_class: Any,
        uplift_model_class: Any,
        data: np.ndarray,
        target: np.ndarray,
        treatment: np.ndarray,
        space: dict,
        rnd_seed: int = 42,
        max_evals: int = 50,
        cv: int = 3,
        scoring_metric: str = "qini",
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
        self.scoring_metric = scoring_metric
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
                model = self.base_model_class(**params)
                uplift_model = self.uplift_model_class(model)

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
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            treatment_train, treatment_val = treatment[train_idx], treatment[val_idx]

            try:
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
        rate: float = 0.3,
        method: str = None,
    ) -> float:
        method = method or self.scoring_metric

        if method == "uplift":
            return self._uplift_at_k_score(predictions, treatment, target, rate)
        elif method == "qini":
            return self._qini_score(predictions, treatment, target)
        elif method == "auuc":
            return self._auuc_score(predictions, treatment, target)
        else:
            warnings.warn(f"Unknown scoring method: {method}. Using uplift@k")
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

        treatment_mask = treatment[top_indices] == 1
        control_mask = treatment[top_indices] == 0

        treatment_conv = (
            target[top_indices][treatment_mask].mean() if treatment_mask.any() else 0
        )
        control_conv = (
            target[top_indices][control_mask].mean() if control_mask.any() else 0
        )

        uplift = treatment_conv - control_conv

        return uplift

    def _qini_score(
        self,
        predictions: np.ndarray,
        treatment: np.ndarray,
        target: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        order = np.argsort(-predictions)
        predictions_sorted = predictions[order]
        treatment_sorted = treatment[order]
        target_sorted = target[order]

        n = len(predictions)
        bin_size = n // n_bins
        qini_values = []

        cumulative_treatment = 0
        cumulative_control = 0
        cumulative_n_treatment = 0
        cumulative_n_control = 0

        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else n

            bin_treatment = treatment_sorted[start:end] == 1
            bin_control = treatment_sorted[start:end] == 0

            treatment_conv = (
                target_sorted[start:end][bin_treatment].sum()
                if bin_treatment.any()
                else 0
            )
            control_conv = (
                target_sorted[start:end][bin_control].sum() if bin_control.any() else 0
            )

            n_treatment = bin_treatment.sum()
            n_control = bin_control.sum()

            cumulative_treatment += treatment_conv
            cumulative_control += control_conv
            cumulative_n_treatment += n_treatment
            cumulative_n_control += n_control

            if cumulative_n_treatment > 0 and cumulative_n_control > 0:
                uplift_rate = (
                    cumulative_treatment / cumulative_n_treatment
                    - cumulative_control / cumulative_n_control
                )
                qini_values.append(uplift_rate)

        qini_score = np.trapezoid(qini_values) if qini_values else 0

        return qini_score

    def _auuc_score(
        self, predictions: np.ndarray, treatment: np.ndarray, target: np.ndarray
    ) -> float:
        return self._qini_score(predictions, treatment, target)

    def get_best_model(self) -> BaseEstimator:
        if self.best_params is None:
            raise ValueError("Model has not been tuned yet. Call tune() first.")

        model = self.base_model_class(**self.best_params)
        uplift_model = self.uplift_model_class(model)
        uplift_model.fit(self.data, self.target, self.treatment)
        return model

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


def findBestParams(
    model_class, data, target, space, cv=3, scoring="roc_auc", max_evals=5
):
    def objective(space):
        clf = model_class(**space)
        scores = cross_val_score(clf, data, target, cv=cv, scoring=scoring, n_jobs=-1)
        loss = -scores.mean()
        return {"loss": loss, "status": STATUS_OK}

    trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(0),
    )

    best_params = {k: v for k, v in best.items()}
    best_score = -min(trials.losses())

    return {"best_params": best_params, "best_score": best_score, "trials": trials}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
