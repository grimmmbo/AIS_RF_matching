"""
Shared figure-producing functions for the run_phase4*.py scripts

Every function here is a pure view over already-computed data (a
pandas Series/DataFrame or precomputed curve arrays) and draws exactly
one figure. Nothing in this module reads files or performs the
statistics/data-wrangling that produces its inputs -- that lives in
scripts.evaluation.metrics, .bias_experiments, .baseline_comparison,
and .open_set.

Every function returns the matplotlib Figure it drew, and accepts an
optional save_path to also write it to disk (parent directories
created as needed). This lets the same functions be used both by the
run_phase4*.py scripts (batch, always saving) and interactively in the
notebooks (call the function, then plt.show()).
"""
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

PathLike = Union[str, Path]

GROUP_LABELS = ["Correct", "Chosen", "Should-be"]

# Shared, easily-tunable text size for every figure in this module
# (title/axis labels/tick labels) -- change this one value to resize
# text across all of them at once.
FONT_SIZE = 18

# Mean/median annotation text is deliberately smaller than FONT_SIZE --
# at FONT_SIZE the mean (right of box) and median (left of the next
# box) labels of adjacent groups run into each other.
ANNOTATION_FONT_SIZE = 11


def _mean_median_legend_handles() -> list:
    """Proxy artists for the mean/median annotations _draw_group_boxplot draws as text, not plot elements"""
    return [
        Line2D([0], [0], color="tab:blue", label="Median"),
        Line2D([0], [0], color="tab:orange", marker="^", linestyle="None", markersize=8, label="Mean"),
    ]


def _maybe_save(fig: plt.Figure, save_path: Optional[PathLike]) -> None:
    """Save fig to disk and close it (batch mode only -- interactive callers never pass save_path)"""
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_group_boxplot(
    ax: plt.Axes,
    data: Sequence[pd.Series],
    labels: Sequence[str],
    mean_fmt: str = "{:.2f}",
    median_fmt: str = "{:.2f}",
) -> None:
    """Draw a Correct/Chosen/Should-be style boxplot with mean/median labels on ax"""
    meanpointprops = dict(marker="^", markerfacecolor="tab:orange", markeredgecolor="tab:orange")
    medianprops = dict(color="tab:blue")
    # Narrower boxes (default width is 0.5) leave more clearance before
    # the offset mean/median labels of the neighboring group, which is
    # what was causing adjacent groups' labels to run into each other.
    ax.boxplot(
        data, tick_labels=labels, widths=0.35, showfliers=False, showmeans=True,
        medianprops=medianprops, meanprops=meanpointprops,
    )
    means = [x.mean() for x in data]
    medians = [x.median() for x in data]
    for i, (mean, median) in enumerate(zip(means, medians), start=1):
        ax.text(
            i + 0.2, mean, mean_fmt.format(mean), color="tab:orange", ha="left", va="center",
            fontsize=ANNOTATION_FONT_SIZE,
        )
        ax.text(
            i - 0.2, median, median_fmt.format(median), color="tab:blue", ha="right", va="center",
            fontsize=ANNOTATION_FONT_SIZE,
        )
    ax.grid(True, linewidth=0.15)


def boxplot_correct_chosen_shouldbe(
    df_correct: pd.DataFrame,
    df_incorrect: pd.DataFrame,
    df_shouldbe: pd.DataFrame,
    feature: str,
    ylabel: str,
    title: str,
    mean_fmt: str = "{:.2f}",
    median_fmt: str = "{:.2f}",
    figsize: tuple = (12, 6),
    dpi: int = 100,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Single-panel Correct/Chosen/Should-be boxplot of one feature"""
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        data = [df_correct[feature], df_incorrect[feature], df_shouldbe[feature]]
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _draw_group_boxplot(ax, data, GROUP_LABELS, mean_fmt, median_fmt)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(handles=_mean_median_legend_handles(), loc="best")
        fig.tight_layout()
        _maybe_save(fig, save_path)
    return fig


def boxplot_correct_chosen_shouldbe_compare(
    panels: Sequence[tuple],
    feature: str,
    ylabel: str,
    title: str,
    fmt: str = "{:.2f}",
    figsize: Optional[tuple] = None,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    N-panel Correct/Chosen/Should-be boxplot comparing multiple methods

    Args:
        panels: [(name, (df_correct, df_incorrect, df_shouldbe)), ...],
            one panel per method being compared
    """
    if figsize is None:
        figsize = (7 * len(panels), 6)
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (name, splits) in zip(axes, panels):
            data = [df[feature] for df in splits]
            _draw_group_boxplot(ax, data, GROUP_LABELS, fmt, fmt)
            ax.set_title(name)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(handles=_mean_median_legend_handles(), loc="best")
        fig.suptitle(title)
        fig.tight_layout()
        _maybe_save(fig, save_path)
    return fig


def diff_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str = "Frequency",
    bins=50,
    kde: bool = True,
    color: str = "tab:blue",
    figsize: tuple = (8, 6),
    dpi: int = 100,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Histogram of a (chosen - should-be) style difference series"""
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.histplot(series, bins=bins, kde=kde, color=color, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.15)
        fig.tight_layout()
        _maybe_save(fig, save_path)
    return fig


def count_histogram_comparison(
    series_a: pd.Series,
    series_b: pd.Series,
    label_a: str = "Chosen",
    label_b: str = "Should-be",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Number of trajectories",
    bins=range(0, 9100, 50),
    xlim: Optional[tuple] = (0, 4500),
    color_a: str = "tab:blue",
    color_b: str = "tab:orange",
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Overlaid histograms comparing two groups (e.g. chosen vs should-be track length)"""
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(series_a, bins=bins, alpha=0.6, label=label_a, color=color_a, ax=ax)
        sns.histplot(series_b, bins=bins, alpha=0.6, label=label_b, color=color_b, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.set_rasterized(True)
        ax.grid(True, linewidth=0.15)
        if xlim is not None:
            ax.set_xlim(*xlim)
        fig.tight_layout()
        _maybe_save(fig, save_path)
    return fig


def candidate_count_histogram(
    counts: pd.Series, binwidth: int = 1, save_path: Optional[PathLike] = None
) -> plt.Figure:
    """Histogram of #AIS candidates per RF signal, first bin (single-match) highlighted"""
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(counts, binwidth=binwidth, edgecolor="black", color="#78add2", ax=ax)

        first_patch = ax.patches[0]
        xtick_positions = [p.get_x() + p.get_width() / 2 for p in ax.patches]
        xtick_labels = [int(p.get_x() + p.get_width() / 2) for p in ax.patches]
        ax.add_patch(
            patches.Rectangle(
                (first_patch.get_x(), 0),
                first_patch.get_width(),
                first_patch.get_height(),
                linewidth=1.5,
                edgecolor="black",
                linestyle="--",
                facecolor="#ffb26e",
            )
        )
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)

        ax.set_xlabel("Number of AIS candidates per RF signal")
        ax.set_ylabel("Frequency")
        ax.grid(True, linewidth=0.15)
        fig.tight_layout()
        _maybe_save(fig, save_path)
    return fig


def score_margin_histogram(
    correct_margin: pd.Series,
    incorrect_margin: pd.Series,
    title: str = "",
    xlabel: str = "Score margin (best − second-best)",
    bins=50,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Overlaid histogram of the top1-vs-runner-up score margin, split by top-1 correctness"""
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(correct_margin, bins=bins, alpha=0.6, label="Correct top-1", color="tab:blue", ax=ax)
        sns.histplot(incorrect_margin, bins=bins, alpha=0.6, label="Incorrect top-1", color="tab:orange", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if own_fig:
            ax.set_ylabel("Number of RF points")
        if legend:
            ax.legend()
        ax.grid(True, linewidth=0.15)
        fig = ax.figure
        if own_fig:
            fig.tight_layout()
            _maybe_save(fig, save_path)
    return fig


def score_margin_grid(
    margin_by_model: Mapping[str, tuple],
    suptitle: str = "",
    ncols: int = 3,
    figsize: tuple = (16, 9),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Grid of score_margin_histogram panels, one per model

    Args:
        margin_by_model: model name -> (correct_margin, incorrect_margin)
    """
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        n = len(margin_by_model)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        for ax, (name, (correct_margin, incorrect_margin)) in zip(axes, margin_by_model.items()):
            score_margin_histogram(correct_margin, incorrect_margin, title=name, ax=ax, legend=False)
        axes[0].legend()
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(suptitle)
        fig.tight_layout()
        _maybe_save(fig, save_path)
    return fig


def metric_comparison_bars(
    comparisons: Mapping[str, pd.DataFrame],
    model_names: Sequence[str],
    metric_cols: Sequence[str] = ("precision", "recall", "f1_score", "accuracy"),
    suptitle: str = "",
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Grouped bar chart of precision/recall/f1/accuracy per model, one panel per view

    Args:
        comparisons: panel title -> metrics DataFrame (index = model name,
            columns include metric_cols), e.g. from baseline_comparison.evaluate_models
    """
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        x = np.arange(len(metric_cols))
        n_models = len(model_names)
        width = 0.8 / n_models

        fig, axes = plt.subplots(1, len(comparisons), figsize=(7 * len(comparisons), 5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (title, comparison) in zip(axes, comparisons.items()):
            for i, name in enumerate(model_names):
                offset = (i - (n_models - 1) / 2) * width
                ax.bar(x + offset, comparison.loc[name, list(metric_cols)], width, label=name)
            ax.set_xticks(x)
            ax.set_xticklabels(metric_cols)
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.grid(True, axis="y", linewidth=0.15)

        axes[0].set_ylabel("Score")
        axes[0].legend()
        fig.suptitle(suptitle)
        fig.tight_layout()
        _maybe_save(fig, save_path)
    return fig


def bootstrap_ci_bars(
    bootstrap_table: pd.DataFrame, suptitle: str = "", save_path: Optional[PathLike] = None
) -> plt.Figure:
    """Point-range plot of Recall@1 per model with 95% bootstrap CI whiskers

    Uses dots + horizontal CI whiskers rather than bars: the CIs are narrow
    (large n -> tight estimates), and a bar chart's zero baseline would make
    them invisible. A zoomed axis is only honest without that baseline.

    MRR is omitted: with ~2.6 candidates per query its only reachable
    reciprocal-rank values are ~{1, 0.5, 0.33}, so it adds little over
    Recall@1 here.
    """
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        fig, ax = plt.subplots(figsize=(7, 5))
        models = bootstrap_table["model"]
        y_pos = np.arange(len(models))
        x = bootstrap_table["recall@1"]
        xerr = [x - bootstrap_table["recall@1_ci_lo"], bootstrap_table["recall@1_ci_hi"] - x]
        ax.errorbar(
            x, y_pos, xerr=xerr, fmt="o", color="#78add2", ecolor="#78add2",
            elinewidth=2, capsize=4, markersize=7,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.invert_yaxis()
        span = max(
            float(bootstrap_table["recall@1_ci_hi"].max() - bootstrap_table["recall@1_ci_lo"].min()), 1e-3
        )
        pad = max(span * 0.5, 0.01)
        ax.set_xlim(
            max(0.0, float(bootstrap_table["recall@1_ci_lo"].min()) - pad),
            min(1.0, float(bootstrap_table["recall@1_ci_hi"].max()) + pad),
        )
        ax.grid(True, axis="x", linewidth=0.15)
        fig.suptitle(suptitle)
        fig.tight_layout()
        _maybe_save(fig, save_path)
    return fig


def roc_pr_overall(
    curves: Mapping[str, dict],
    suptitle: str = "",
    negatives_label: str = "all negatives",
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Two-panel overall ROC + Precision-Recall chart, one curve per model

    Args:
        curves: model name -> {"fpr", "tpr", "roc_auc", "precision", "recall", "ap"},
            e.g. from open_set.compute_roc / open_set.compute_pr
        negatives_label: Short description of the negative class shown
            in the ROC panel's own title, e.g. "all leave-one-out
            negatives" or "all dark-vessel negatives"
    """
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for name, c in curves.items():
            axes[0].plot(c["fpr"], c["tpr"], label=f"{name} (AUC={c['roc_auc']:.3f})")
            axes[1].plot(c["recall"], c["precision"], label=f"{name} (AP={c['ap']:.3f})")

        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axes[0].set_xlabel("False accept rate")
        axes[0].set_ylabel("True accept rate")
        axes[0].set_title(f"Overall ROC ({negatives_label})")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, linewidth=0.15)

        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Overall Precision-Recall")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linewidth=0.15)

        if suptitle:
            fig.suptitle(suptitle)
        fig.tight_layout()
        _maybe_save(fig, save_path)
    return fig


def roc_single_panel(
    curves: Mapping[str, dict], title: str = "", save_path: Optional[PathLike] = None
) -> plt.Figure:
    """Single ROC panel, one curve per model (e.g. the scored-rejections-only ROC)"""
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE}):
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, c in curves.items():
            ax.plot(c["fpr"], c["tpr"], label=f"{name} (AUC={c['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False accept rate")
        ax.set_ylabel("True accept rate")
        ax.set_title(title)
        ax.legend(fontsize=13)
        ax.grid(True, linewidth=0.15)
        fig.tight_layout()
        _maybe_save(fig, save_path)
    return fig
