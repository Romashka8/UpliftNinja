# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import time

import numpy as np
import pandas as pd

from sklift.metrics import uplift_at_k
from sklift.metrics import weighted_average_uplift
from sklift.metrics import qini_auc_score
from sklift.metrics import uplift_auc_score


import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


def upliftComparingHist(model_name_1,
                        model_predictions_1,
                        model_name_2,
                        model_predictions_2
                        ):
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.despine(left=True, right=True, top=True)

    sns.histplot(
        model_predictions_1,
        bins=100,
        alpha=0.4,
        label=model_name_1,
        color='purple',
        edgecolor='k',
        linewidth=1,
        ax=ax)
    sns.histplot(
        model_predictions_2,
        bins=100,
        alpha=0.4,
        label=model_name_2,
        color='deeppink',
        edgecolor='k',
        linewidth=1,
        ax=ax)
    ax.set_xlabel('uplift')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Comparison of {model_name_1} & {model_name_2} uplift')
    ax.legend()

    return fig

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


def scoreUpliftAtK(model_w_predictions,
                   target_test,
                   treatment_test,
                   k=0.2
                   ):
    results = {}
    for model_name, predictions in model_w_predictions:
        up_k_best = uplift_at_k(y_true=target_test,
                                uplift=predictions,
                                treatment=treatment_test,
                                strategy='overall',
                                k=k)

        results[model_name] = round(up_k_best, 4)

    return results

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


def scorePipelines(
        pipelines_unfited,
        data_train,
        target_train,
        data_test,
        target_test,
        treatment_train_in,
        treatment_test_in):
    results = []
    for pipeline in pipelines_unfited:
        scores = {}
        scores["model_name"] = pipeline
        steps = pipeline.split("-")
        if "treeCausal" in steps:
            treatment_train = treatment_train_in.copy()
            treatment_train = treatment_train.map({0: 'no-control', 1: 'control'})
            treatment_train = treatment_train
            treatment_test = treatment_test_in.copy()
            treatment_test = treatment_test.map({0: 'no-control', 1: 'control'})
            treatment_test = treatment_test
        else:
            treatment_train = treatment_train_in.copy()
            treatment_test = treatment_test_in.copy()
        start_fit_time = time.time()
        if "treeCausal" in steps:
            pipelines_unfited[pipeline] = pipelines_unfited[pipeline]\
                .fit(data_train,
                     treatment_train,
                     target_train)
        else:
            pipelines_unfited[pipeline] = pipelines_unfited[pipeline]\
                .fit(data_train,
                     target_train,
                     treatment_train)
        scores["fit_time"] = time.time() - start_fit_time
        start_score_time = time.time()
        score = pipelines_unfited[pipeline].predict(data_test)
        scores["score_time"] = time.time() - start_score_time
        if "treeCausal" in steps:
            treatment_test = treatment_test_in.copy()
            score = score[:, 1] - score[:, 0]
        scores["weighted_average_uplift_test"] = weighted_average_uplift(
            target_test, score, treatment_test, bins=10)
        scores["auqc_test"] = qini_auc_score(
            target_test.values, score, treatment_test.values)
        scores["auuq_test"] = uplift_auc_score(
            target_test.values, score, treatment_test.values)
        results.append(scores)
    result = pd.DataFrame(results)
    result = pd.concat(
        [result["model_name"], result.drop(["model_name"], axis=1)], axis=1)
    return result.sort_values(by=["fit_time",
                                  "score_time",
                                  "weighted_average_uplift_test",
                                  "auqc_test",
                                  "auuq_test"],
                              ascending=[True,
                                         True,
                                         False,
                                         False,
                                         False])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
