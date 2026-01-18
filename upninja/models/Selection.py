# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from typing import Any, Optional, Union

from hyperopt import fmin, tpe, STATUS_OK, Trials

from sklearn.model_selection import cross_val_score

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class UpliftTune:
    def __init__(
        self,
        model_class: Any,
        data: np.array,
        target: np.array,
        treatment: np.array,
        space: dict,
        rnd_seed: int,
        max_evals: int = 5,
    ):
        self.model_class = model_class
        self.space = space
        self.objective = self.get_objective_(
            data=data, target=target, treatment=treatment
        )
        self.rnd_seed = rnd_seed
        self.max_evals = max_evals
        self.trials = Trials()

    def tune(self):
        best = fmin(
            fh=self.objective,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials,
            rstate=np.random.default_rng(self.rnd_seed),
        )

        best_params = {k: v for k, v in best.items()}
        best_score = -min(self.trials.losses())

        return {
            "best_params": best_params,
            "best_score": best_score,
            "trials": self.trials,
        }

    def scoring_(
        self,
        prediction: np.array,
        treatment: np.array,
        target: np.array,
        rate: float = 0.3,
    ) -> float:
        """
        Calculate Uplift Score:
            - calc top uplift@k score;
            - try to reach maximum uplift with hyperopt
        """
        order = np.argsort(-prediction)
        treatment_n = int((treatment == 1).sum() * rate)
        treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
        control_n = int((treatment == 0).sum() * rate)
        control_p = target[order][treatment[order] == 0][:control_n].mean()
        score = treatment_p - control_p
        return score

    def get_objective_(self, data: np.array, target: np.array, treatment: np.array):
        def uplift_objective(self):
            model = self.model_class(**self.space)
            predictions = model.predict(data)
            scores = self.scoring_(predictions, treatment, target)
            loss = -scores.mean()
            return {"loss": loss, "status": STATUS_OK}

        return uplift_objective


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
