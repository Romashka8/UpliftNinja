# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np

from hyperopt import hp

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# LogisticRegression
log_reg_base_params = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "random_state": 0
}

log_reg_hp_space = {
    "C": hp.uniform("C", 0.01, 10),
    "solver": hp.choice("solver", ["newton-cg", "lbfgs", "liblinear", "saga"]),
    "max_iter": hp.uniformint("max_iter", 100, 500),
    "penalty": "l2",
    "solver": "lbfgs",
    "random_state": 0
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# KNeighborsClassifier
knn_base_params = {
    "n_neighbors": 5,
    "weights": "uniform",
    "metric": "euclidean"
}

knn_hp_space = {
    "n_neighbors": hp.choice("n_neighbors", range(1, 21)),
    "weights": hp.choice("weights", ["uniform", "distance"]),
    "metric": hp.choice("metric", ["euclidean", "manhattan", "minkowski"]),
    "p": hp.choice("p", [1, 2])
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# DecisionTreeClassifier
dt_base_params = {
    "max_depth": 5,
    "min_samples_split": 0.05,
    "min_samples_leaf": 0.05,
    "criterion": "gini",
    "random_state": 0
}

dt_hp_space = {
    "max_depth": hp.uniformint("max_depth", 2, 20),
    "min_samples_split": hp.uniform("min_samples_split", 0.01, 0.2),
    "min_samples_leaf": hp.uniform("min_samples_leaf", 0.01, 0.2),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "random_state": 0
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# RandomForestClassifier
rf_base_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 0.05,
    "min_samples_leaf": 0.05,
    "criterion": "gini",
    "random_state": 0
}

rf_hp_space = {
    "n_estimators": hp.uniformint("n_estimators", 50, 200),
    "max_depth": hp.uniformint("max_depth", 2, 20),
    "min_samples_split": hp.uniform("min_samples_split", 0.01, 0.2),
    "min_samples_leaf": hp.uniform("min_samples_leaf", 0.01, 0.2),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "random_state": 0
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# XGBClassifier
xgb_base_params = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 0
}

xgb_hp_space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300]),
    "max_depth": hp.choice("max_depth", [3, 5, 7]),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
    "subsample": hp.uniform("subsample", 0.5, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "random_state": 0
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# CatBoostClassifier
cb_base_params = {
    "iterations": 100,
    "depth": 6,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3,
    "random_seed": 0,
    "verbose": False
}

cb_hp_space = {
    "iterations": hp.uniformint("iterations", 100, 1000),
    "depth": hp.uniformint("depth", 2, 10),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
    "l2_leaf_reg": hp.uniform("l2_leaf_reg", 1, 10),
    "random_seed": 0,
    "verbose": False
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# LGBMClassifier
lgbm_base_params = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 0
}

lgbm_hp_space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300]),
    "max_depth": hp.choice("max_depth", [3, 5, 7]),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
    "subsample": hp.uniform("subsample", 0.5, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "random_state": 0
}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
