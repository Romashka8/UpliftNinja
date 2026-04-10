# ----------------------------------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklift.metrics import uplift_by_percentile
from sklift.metrics.metrics import (
    uplift_auc_score,
    uplift_curve,
    perfect_uplift_curve,
    qini_curve,
    perfect_qini_curve,
    qini_auc_score,
)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class UpliftCurveDisplay:
    def __init__(
        self,
        x_actual,
        y_actual,
        x_baseline=None,
        y_baseline=None,
        x_perfect=None,
        y_perfect=None,
        random=None,
        perfect=None,
        estimator_name=None,
    ):
        self.x_actual = x_actual
        self.y_actual = y_actual
        self.x_baseline = x_baseline
        self.y_baseline = y_baseline
        self.x_perfect = x_perfect
        self.y_perfect = y_perfect
        self.random = random
        self.perfect = perfect
        self.estimator_name = estimator_name

    def plot(self, auc_score, ax=None, name=None, title=None, **kwargs):
        name = self.estimator_name if name is None else name

        line_kwargs = {}
        if auc_score is not None and name is not None:
            line_kwargs["label"] = f"{name} ({title} = {auc_score:0.2f})"
        elif auc_score is not None:
            line_kwargs["label"] = f"{title} = {auc_score:0.2f}"
        elif name is not None:
            line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        if ax is None:
            fig, ax = plt.subplots()

        (self.line_,) = ax.plot(self.x_actual, self.y_actual, **line_kwargs)

        if self.random:
            ax.plot(self.x_baseline, self.y_baseline, label="Random")
            ax.fill_between(self.x_actual, self.y_actual, self.y_baseline, alpha=0.2)

        if self.perfect:
            ax.plot(self.x_perfect, self.y_perfect, label="Perfect")

        ax.set_xlabel("Number targeted")
        ax.set_ylabel("Number of incremental outcome")

        if self.random == self.perfect:
            variance = False
        else:
            variance = True

        if len(ax.lines) > 4:
            ax.lines.pop(len(ax.lines) - 1)
            if variance == False:
                ax.lines.pop(len(ax.lines) - 1)

        if "label" in line_kwargs:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        self.ax_ = ax
        self.figure_ = ax.figure

        return self


# ----------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_uplift_by_percentile(
    y_true,
    uplift,
    treatment,
    strategy="overall",
    kind="line",
    bins=10,
    string_percentiles=True,
):
    strategy_methods = ["overall", "by_group"]
    kind_methods = ["line", "bar"]

    n_samples = len(y_true)

    df = uplift_by_percentile(
        y_true,
        uplift,
        treatment,
        strategy=strategy,
        std=True,
        bins=bins,
        string_percentiles=False,
    )

    percentiles = df.index[:bins].values.astype(float)

    response_rate_trmnt = df.loc[percentiles, "response_rate_treatment"].values
    std_trmnt = df.loc[percentiles, "std_treatment"].values

    response_rate_ctrl = df.loc[percentiles, "response_rate_control"].values
    std_ctrl = df.loc[percentiles, "std_control"].values

    uplift_score = df.loc[percentiles, "uplift"].values
    std_uplift = df.loc[percentiles, "std_uplift"].values

    if kind == "line":
        _, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        axes.errorbar(
            percentiles,
            response_rate_trmnt,
            yerr=std_trmnt,
            linewidth=2,
            color="forestgreen",
            label="treatment\nresponse rate",
        )
        axes.errorbar(
            percentiles,
            response_rate_ctrl,
            yerr=std_ctrl,
            linewidth=2,
            color="orange",
            label="control\nresponse rate",
        )
        axes.errorbar(
            percentiles,
            uplift_score,
            yerr=std_uplift,
            linewidth=2,
            color="red",
            label="uplift",
        )
        axes.fill_between(
            percentiles, response_rate_trmnt, response_rate_ctrl, alpha=0.1, color="red"
        )

        if np.amin(uplift_score) < 0:
            axes.axhline(y=0, color="black", linewidth=1)

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + [
                f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}"
                for i in range(len(percentiles) - 1)
            ]
            axes.set_xticks(percentiles)
            axes.set_xticklabels(percentiles_str, rotation=45)
        else:
            axes.set_xticks(percentiles)

        axes.legend(loc="upper right")
        axes.set_title(f"Uplift by percentile")
        axes.set_xlabel("Percentile")
        axes.set_ylabel("Uplift = treatment response rate - control response rate")

    else:  # kind == 'bar'
        delta = percentiles[0]
        fig, axes = plt.subplots(
            ncols=1, nrows=2, figsize=(8, 6), sharex=True, sharey=True
        )
        fig.text(
            0.04,
            0.5,
            "Uplift = treatment response rate - control response rate",
            va="center",
            ha="center",
            rotation="vertical",
        )

        axes[1].bar(
            np.array(percentiles) - delta / 6,
            response_rate_trmnt,
            delta / 3,
            yerr=std_trmnt,
            color="forestgreen",
            label="treatment\nresponse rate",
        )
        axes[1].bar(
            np.array(percentiles) + delta / 6,
            response_rate_ctrl,
            delta / 3,
            yerr=std_ctrl,
            color="orange",
            label="control\nresponse rate",
        )
        axes[0].bar(
            np.array(percentiles),
            uplift_score,
            delta / 1.5,
            yerr=std_uplift,
            color="red",
            label="uplift",
        )

        axes[0].legend(loc="upper right")
        axes[0].tick_params(axis="x", bottom=False)
        axes[0].axhline(y=0, color="black", linewidth=1)
        axes[0].set_title(f"Uplift by percentile")

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + [
                f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}"
                for i in range(len(percentiles) - 1)
            ]
            axes[1].set_xticks(percentiles)
            axes[1].set_xticklabels(percentiles_str, rotation=45)

        else:
            axes[1].set_xticks(percentiles)

        axes[1].legend(loc="upper right")
        axes[1].axhline(y=0, color="black", linewidth=1)
        axes[1].set_xlabel("Percentile")
        axes[1].set_title("Response rate by percentile")

    plt.show()

    return axes


# ----------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_uplift_curve(
    y_true, uplift, treatment, random=True, perfect=True, ax=None, name=None, **kwargs
):
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    x_actual, y_actual = uplift_curve(y_true, uplift, treatment)

    if random:
        x_baseline, y_baseline = x_actual, x_actual * y_actual[-1] / len(y_true)
    else:
        x_baseline, y_baseline = None, None

    if perfect:
        x_perfect, y_perfect = perfect_uplift_curve(y_true, treatment)
    else:
        x_perfect, y_perfect = None, None

    viz = UpliftCurveDisplay(
        x_actual=x_actual,
        y_actual=y_actual,
        x_baseline=x_baseline,
        y_baseline=y_baseline,
        x_perfect=x_perfect,
        y_perfect=y_perfect,
        random=random,
        perfect=perfect,
        estimator_name=name,
    )

    auc = uplift_auc_score(y_true, uplift, treatment)

    return viz.plot(auc, ax=ax, title="AUC", **kwargs)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_qini_curve(
    y_true,
    uplift,
    treatment,
    random=True,
    perfect=True,
    negative_effect=True,
    ax=None,
    name=None,
    **kwargs,
):
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    x_actual, y_actual = qini_curve(y_true, uplift, treatment)

    if random:
        x_baseline, y_baseline = x_actual, x_actual * y_actual[-1] / len(y_true)
    else:
        x_baseline, y_baseline = None, None

    if perfect:
        x_perfect, y_perfect = perfect_qini_curve(y_true, treatment, negative_effect)
    else:
        x_perfect, y_perfect = None, None

    viz = UpliftCurveDisplay(
        x_actual=x_actual,
        y_actual=y_actual,
        x_baseline=x_baseline,
        y_baseline=y_baseline,
        x_perfect=x_perfect,
        y_perfect=y_perfect,
        random=random,
        perfect=perfect,
        estimator_name=name,
    )

    auc = qini_auc_score(y_true, uplift, treatment, negative_effect)

    return viz.plot(auc, ax=ax, title="AUC", **kwargs)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
