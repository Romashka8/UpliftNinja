# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np

from hyperopt import hp


# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# DecisionTreeClassifier
dt_base_params = {
    "max_depth": 5,
    "min_samples_split": 0.05,
    "min_samples_leaf": 0.05,
    "criterion": "gini",
    "random_state": 0,
}

dt_hp_space = {
    "max_depth": hp.uniformint("max_depth", 2, 20),
    "min_samples_split": hp.uniform("min_samples_split", 0.01, 0.2),
    "min_samples_leaf": hp.uniform("min_samples_leaf", 0.01, 0.2),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "random_state": 0,
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# RandomForestClassifier
rf_base_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 0.05,
    "min_samples_leaf": 0.05,
    "criterion": "gini",
    "random_state": 0,
}

rf_hp_space = {
    "n_estimators": hp.uniformint("n_estimators", 50, 200),
    "max_depth": hp.uniformint("max_depth", 2, 20),
    "min_samples_split": hp.uniform("min_samples_split", 0.01, 0.2),
    "min_samples_leaf": hp.uniform("min_samples_leaf", 0.01, 0.2),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "random_state": 0,
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# CatBoostClassifier
cb_base_params = {
    "iterations": 100,
    "depth": 6,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3,
    "random_seed": 0,
    "verbose": False,
}

cb_hp_space = {
    "iterations": hp.uniformint("iterations", 100, 1000),
    "depth": hp.uniformint("depth", 2, 10),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
    "l2_leaf_reg": hp.uniform("l2_leaf_reg", 1, 10),
    "random_seed": 0,
    "verbose": False,
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
