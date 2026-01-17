# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from hyperopt import fmin, tpe, STATUS_OK, Trials

from sklearn.model_selection import cross_val_score

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


def findBestParams(
        model_class,
        data,
        target,
        space,
        cv=3,
        scoring="roc_auc",
        max_evals=5):
    def objective(space):
        clf = model_class(**space)
        scores = cross_val_score(
            clf,
            data,
            target,
            cv=cv,
            scoring=scoring,
            n_jobs=-1)
        loss = -scores.mean()
        return {"loss": loss, "status": STATUS_OK}

    trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(0)
    )

    best_params = {k: v for k, v in best.items()}
    best_score = -min(trials.losses())

    return {
        "best_params": best_params,
        "best_score": best_score,
        "trials": trials}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
