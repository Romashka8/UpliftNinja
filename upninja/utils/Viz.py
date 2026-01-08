# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np

from sklift.metrics import (
    qini_curve,
    perfect_qini_curve,
    uplift_by_percentile
)

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_qini_curve(y_true,
                    uplift,
                    treatment,
                    perfect=True,
                    ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    x_curve, y_curve = qini_curve(y_true, uplift, treatment)
    ax.plot(x_curve, y_curve, label="Model", color="steelblue")

    if perfect:
        x_perfect, y_perfect = perfect_qini_curve(y_true, treatment)
        ax.plot(
            x_perfect,
            y_perfect,
            label="Perfect",
            color="red",
            linestyle="--")

    ax.plot([0, x_curve[-1]], [0, 0], label="Random",
            color="black", linestyle="--")

    ax.set_xlabel("Number targeted")
    ax.set_ylabel("Uplift")
    ax.set_title("Qini Curve")
    ax.legend()
    ax.grid(True)
    return ax


def plot_uplift_by_percentile(y_true, uplift, treatment, bins=10, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    df = uplift_by_percentile(y_true, uplift, treatment, bins=bins, std=False)
    
    percentiles = df.index
    uplift_vals = df["uplift"]
    
    ax.bar(
        percentiles,
        uplift_vals,
        color="lightblue",
        edgecolor="black"
    )
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Uplift")
    ax.set_title("Uplift by Percentile")
    ax.grid(axis="y")
    return ax

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
