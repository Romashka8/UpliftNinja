# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from hyperopt import fmin, tpe, STATUS_OK, Trials

from sklearn.model_selection import cross_validate, cross_val_score

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


def baseModelSelection(models, data, target):
    result = []
    scorring = ["accuracy", "roc_auc", "f1"]
    for model_name in models:
        scores = cross_validate(
            models[model_name],
            data,
            target,
            scoring=scorring)
        for score in scores:
            scores[score] = np.mean(scores[score])
        scores["model_name"] = model_name
        result.append(scores)
    result = pd.DataFrame(result)
    result = pd.concat(
        [result["model_name"], result.drop(["model_name"], axis=1)], axis=1)
    return result.sort_values(by=["fit_time",
                                  "score_time",
                                  "test_roc_auc",
                                  "test_accuracy",
                                  "test_f1"],
                              ascending=[True,
                                         True,
                                         False,
                                         False,
                                         False])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
