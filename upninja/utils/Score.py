# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from sklift.metrics import uplift_at_k

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
