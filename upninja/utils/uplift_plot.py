# ----------------------------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Any
from numpy.typing import ArrayLike, NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
    """Display object for uplift and qini curves.

    Stores curve coordinates and provides a ``plot`` method similar to
    scikit-learn display objects.
    """

    def __init__(
        self,
        x_actual: NDArray,
        y_actual: NDArray,
        x_baseline: NDArray | None = None,
        y_baseline: NDArray | None = None,
        x_perfect: NDArray | None = None,
        y_perfect: NDArray | None = None,
        random: bool = False,
        perfect: bool = False,
        estimator_name: str | None = None,
    ) -> None:
        self.x_actual = x_actual
        self.y_actual = y_actual
        self.x_baseline = x_baseline
        self.y_baseline = y_baseline
        self.x_perfect = x_perfect
        self.y_perfect = y_perfect
        self.random = random
        self.perfect = perfect
        self.estimator_name = estimator_name

    def plot(
        self,
        auc_score: float | None,
        ax: Axes | None = None,
        name: str | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> "UpliftCurveDisplay":
        """Plot the stored curve on a matplotlib axis.

        Parameters
        ----------
        auc_score : float | None
            Curve AUC value shown in the legend label.
        ax : Axes | None, default=None
            Existing matplotlib axis. If None, a new figure and axis are created.
        name : str | None, default=None
            Display name for the curve.
        title : str | None, default=None
            Metric name used in the legend label.
        **kwargs : Any
            Additional keyword arguments passed to ``ax.plot``.

        Returns
        -------
        UpliftCurveDisplay
            The current display instance.
        """
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
    y_true: ArrayLike,
    uplift: ArrayLike,
    treatment: ArrayLike,
    strategy: str = "overall",
    kind: str = "line",
    bins: int = 10,
    string_percentiles: bool = True,
) -> Axes | NDArray:
    """Plot treatment rate, control rate, and uplift by percentile.

    Parameters
    ----------
    y_true : ArrayLike
        Binary outcome values.
    uplift : ArrayLike
        Predicted uplift scores.
    treatment : ArrayLike
        Treatment group indicator.
    strategy : str, default="overall"
        Strategy passed to ``sklift.metrics.uplift_by_percentile``.
    kind : str, default="line"
        Plot type: ``"line"`` or ``"bar"``.
    bins : int, default=10
        Number of percentile bins.
    string_percentiles : bool, default=True
        Whether to show percentile intervals as string labels on the x-axis.

    Returns
    -------
    Axes | NDArray
        Matplotlib axes object for ``kind="line"`` or axes array for
        ``kind="bar"``.
    """
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
    y_true: ArrayLike,
    uplift: ArrayLike,
    treatment: ArrayLike,
    random: bool = True,
    perfect: bool = True,
    ax: Axes | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> UpliftCurveDisplay:
    """Compute and plot the uplift curve.

    Parameters
    ----------
    y_true : ArrayLike
        Binary outcome values.
    uplift : ArrayLike
        Predicted uplift scores.
    treatment : ArrayLike
        Treatment group indicator.
    random : bool, default=True
        Whether to plot the random baseline.
    perfect : bool, default=True
        Whether to plot the perfect uplift curve.
    ax : Axes | None, default=None
        Existing matplotlib axis. If None, a new one is created.
    name : str | None, default=None
        Curve label shown in the legend.
    **kwargs : Any
        Additional keyword arguments passed to the main curve plot.

    Returns
    -------
    UpliftCurveDisplay
        Display object containing the computed curve.
    """
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
    y_true: ArrayLike,
    uplift: ArrayLike,
    treatment: ArrayLike,
    random: bool = True,
    perfect: bool = True,
    negative_effect: bool = True,
    ax: Axes | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> UpliftCurveDisplay:
    """Compute and plot the qini curve.

    Parameters
    ----------
    y_true : ArrayLike
        Binary outcome values.
    uplift : ArrayLike
        Predicted uplift scores.
    treatment : ArrayLike
        Treatment group indicator.
    random : bool, default=True
        Whether to plot the random baseline.
    perfect : bool, default=True
        Whether to plot the perfect qini curve.
    negative_effect : bool, default=True
        Whether negative treatment effects are allowed in the perfect curve.
    ax : Axes | None, default=None
        Existing matplotlib axis. If None, a new one is created.
    name : str | None, default=None
        Curve label shown in the legend.
    **kwargs : Any
        Additional keyword arguments passed to the main curve plot.

    Returns
    -------
    UpliftCurveDisplay
        Display object containing the computed curve.
    """
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
