"""
Draws every phase 4 figure from its shaped input data. This is the one
place that knows each figure's title/axis labels and which data it's
drawn from.

Each run_phase4*.py script calls the matching function here once, at
the end of main(), with the exact DataFrames it just computed (and
also stored with save_plot_data). scripts/evaluation/regenerate_plots.py
calls the SAME functions with the DataFrames reloaded from
reports/plot_data. Either way the title/label text is written exactly
once, here -- neither caller repeats it.

Each phaseN_* function takes `data: dict[str, pd.DataFrame]` (the
figure inputs a script stores with save_plot_data) and `fig_dir: Path`,
draws whatever keys are present (silently skipping the rest) and
returns how many figures it drew.
"""
import re
from pathlib import Path

import pandas as pd
from matplotlib.cbook import boxplot_stats

from scripts.evaluation import plots
from scripts.evaluation.bias_experiments import points_by_group, points_diff

def plot_data_path(table_dir: Path) -> Path:
    """Pickle holding a phase's figure inputs.

    reports/tables/<phase>/<error_model> maps to
    reports/plot_data/<phase>/<error_model>.pkl, so the result tables
    folder only holds results, not point-level figure inputs.
    """
    reports = table_dir.parents[2]
    return (reports / "plot_data" / table_dir.parent.name
            / f"{table_dir.name}.pkl")


def save_plot_data(data: dict[str, pd.DataFrame], table_dir: Path) -> None:
    """Save a phase's figure inputs, keyed as the phaseN_* functions expect."""
    path = plot_data_path(table_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(data, path)


def load_plot_data(table_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Load what save_plot_data stored, or None if it was not saved yet."""
    path = plot_data_path(table_dir)
    return pd.read_pickle(path) if path.exists() else None


# (name, ylabel, title, xlabel, mean_fmt, median_fmt) for the boxplot +
# diff-histogram figures of run_phase4_evaluation.py's experiments 2-4
# (track-length bias experiment 1 is handled separately below, since it
# also draws a count_histogram_comparison and uses fixed formats).
DIFF_EXPERIMENTS = [
    ("experiment2a_intra_time", "Seconds", "Mean intra-time difference: Correct vs Incorrect",
     "Mean intra-time", "mean {:.2f}", "median {:.2f}"),
    ("experiment2b_intra_distance", "Kilometers", "Mean intra-distance difference: Correct vs Incorrect",
     "Mean intra-distance", "mean {:.3f}", "median {:.3f}"),
    ("experiment3_stationarity", "Proportion of Time Stationary (< 5 km/h)",
     "Stationary Behavior (stationary_ratio)", "Stationary ratio", "mean {:.2f}", "median {:.2f}"),
    ("experiment4a_time", "Seconds", "Time difference (without outliers)",
     "Time", "mean {:.1f}", "median {:.1f}"),
    ("experiment4b_distance", "Kilometers", "Distance difference (without outliers)",
     "Distance", "mean {:.1f}", "median {:.1f}"),
]

# ROC figures draw only these curves (in this order), not every model in
# the CSVs: both PHMM Forward scores, both log-odds variants, the one
# geometric NN baseline reported in the paper (the best of Euclidean /
# Haversine / Segment, chosen by AUC) and NN time-weighted. A model
# missing from the data (e.g. a CSV written before it was added) is
# skipped, not an error. Keys are the CSV model names; values are the
# legend labels.
ROC_PLOT_MODELS = {
    "PHMM Forward": "PHMM Forward (alpha=1)",
    "PHMM Forward (corrected)": "PHMM Forward (corrected)",
    "Log-odds PHMM (raw)": "PHMM log-odds (raw)",
    "Log-odds PHMM (n^alpha)": "PHMM log-odds (n^alpha)",
    "NN Segment": "NN point-to-segment",
    "NN Time-weighted": "NN time-weighted",
}


def _roc_models(names) -> list:
    """The ROC_PLOT_MODELS entries present in names, in ROC_PLOT_MODELS order"""
    present = set(names)
    return [name for name in ROC_PLOT_MODELS if name in present]


# (slug, title_suffix) for run_phase4c_open_set_evaluation.py's two
# negative-class constructions
OPEN_SET_SLUGS = [
    ("darkvessel", "dark-vessel test, held-out vessels"),
    ("leaveoneout", "leave-one-out negatives, own-vessel exclusion only"),
]


def phase4_evaluation(data: dict, fig_dir: Path) -> int:
    n = 0

    counts = data.get("candidate_counts")
    if counts is not None:
        plots.candidate_count_histogram(
            counts.set_index(["RF_track_id", "RF_signal_id"])["count"],
            save_path=fig_dir / "candidate_count_histogram.png",
        )
        n += 1

    margins = data.get("score_margins")
    if margins is not None:
        top1 = margins["rank_of_true"] == 1
        plots.score_margin_histogram(
            margins.loc[top1, "margin"], margins.loc[~top1, "margin"],
            title="Top-1 score margin: correct vs incorrect picks (PHMM Forward)",
            save_path=fig_dir / "score_margin_histogram.png",
        )
        n += 1

    for suffix in ("before_correction", "after_correction"):
        points = data.get(f"experiment1_{suffix}_points")
        if points is None:
            continue
        df_correct, df_incorrect, df_shouldbe = points_by_group(points)
        plots.boxplot_correct_chosen_shouldbe(
            df_correct, df_incorrect, df_shouldbe, "value", "Number of AIS messages per trajectory",
            "Number of AIS points per track: Correct vs Chosen vs Should-be",
            mean_fmt="mean {:.1f}", median_fmt="median {:.1f}",
            save_path=fig_dir / f"experiment1_{suffix}_boxplot.png",
        )
        plots.count_histogram_comparison(
            df_incorrect["value"], df_shouldbe["value"],
            title=f"AIS message counts: Chosen vs. Should-be ({suffix})",
            xlabel="Number of AIS messages per trajectory",
            save_path=fig_dir / f"experiment1_{suffix}_count_histogram.png",
        )
        plots.diff_histogram(
            points_diff(df_incorrect, df_shouldbe),
            "Distribution of track length difference (chosen − should-be)",
            "Difference in number of AIS messages (chosen − should-be)",
            save_path=fig_dir / f"experiment1_{suffix}_diff_histogram.png",
        )
        n += 3

    for name, ylabel, title, xlabel, mean_fmt, median_fmt in DIFF_EXPERIMENTS:
        points = data.get(f"{name}_points")
        if points is None:
            continue
        df_correct, df_incorrect, df_shouldbe = points_by_group(points)
        plots.boxplot_correct_chosen_shouldbe(
            df_correct, df_incorrect, df_shouldbe, "value", ylabel, title,
            mean_fmt=mean_fmt, median_fmt=median_fmt, dpi=150,
            save_path=fig_dir / f"{name}_boxplot.png",
        )
        plots.diff_histogram(
            points_diff(df_incorrect, df_shouldbe),
            f"{xlabel} error (chosen − should-be)", f"Difference in {xlabel.lower()} (chosen - should-be)",
            figsize=(12, 6), dpi=150, color="#78add2",
            save_path=fig_dir / f"{name}_diff_histogram.png",
        )
        n += 2

    return n


def phase4b_baseline_comparison(data: dict, fig_dir: Path, model_names: list, random_baseline_name: str) -> int:
    n = 0

    comparison_all = data.get("metrics_all_candidates")
    comparison_multi = data.get("metrics_multimatch_candidates")
    if comparison_all is not None and comparison_multi is not None:
        plots.metric_comparison_bars(
            {"All candidate pairs": comparison_all, "Multimatch candidate pairs": comparison_multi},
            model_names + [random_baseline_name],
            suptitle="PHMM Forward vs. Nearest-Neighbor baselines vs. random-choice floor",
            save_path=fig_dir / "metrics_comparison_bars.png",
        )
        n += 1

    margins = data.get("score_margins_multimatch")
    if margins is not None:
        margin_by_model = {}
        for name in model_names:
            summary = margins[margins["model"] == name]
            top1 = summary["rank_of_true"] == 1
            margin_by_model[name] = (summary.loc[top1, "margin"], summary.loc[~top1, "margin"])
        plots.score_margin_grid(
            margin_by_model,
            suptitle="Top-1 score margin: correct vs incorrect picks, per method",
            save_path=fig_dir / "score_margin_grid.png",
        )
        n += 1

    bootstrap_table = data.get("bootstrap_ci")
    if bootstrap_table is not None:
        plots.bootstrap_ci_bars(
            bootstrap_table,
            suptitle="Recall@1 per method with 95% CI",
            save_path=fig_dir / "bootstrap_ci_bars.png",
        )
        n += 1

    return n


# Experiments (by number) that are also drawn as one title-less boxplot per
# method, for placing side by side in a paper figure
PER_METHOD_BOXPLOT_EXPERIMENTS = ("1",)
# Shorter value-label format for these plots (message counts, whole numbers)
PER_METHOD_FMT = {"1": "{:.0f}"}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def phase4c_bias_experiments(data: dict, fig_dir: Path, experiments: list, method_names: list) -> int:
    n = 0
    for slug, feature, ylabel, fmt, alternative, hypothesis, paired in experiments:
        key = f"experiment_{slug.split('.')[0]}_{feature}_points"
        points = data.get(key)
        if points is None:
            continue
        present = points["method"].unique()
        panels = [
            (name, points_by_group(points[points["method"] == name].drop(columns="method")))
            for name in method_names if name in present
        ]
        plots.boxplot_correct_chosen_shouldbe_compare(
            panels, "value", ylabel,
            f"Experiment {slug}: Correct vs Chosen vs Should-be", fmt=fmt,
            save_path=fig_dir / f"{key.removesuffix('_points')}_boxplot.png",
        )
        n += 1
        if slug.split(".")[0] in PER_METHOD_BOXPLOT_EXPERIMENTS:
            whiskers = [
                stats for _, splits in panels for df in splits
                for stats in boxplot_stats(df["value"], whis=1.5)
            ]
            span = max(w["whishi"] for w in whiskers) - min(w["whislo"] for w in whiskers)
            ylim = (min(w["whislo"] for w in whiskers) - 0.04 * span, max(w["whishi"] for w in whiskers) + 0.16 * span)
            for i, (name, splits) in enumerate(panels):
                plots.boxplot_single_method(
                    splits, "value", ylabel, fmt=PER_METHOD_FMT.get(slug.split(".")[0], fmt), ylim=ylim, legend=(i == len(panels) - 1),
                    save_path=fig_dir / f"{key.replace('#', '').removesuffix('_points')}_boxplot_{_slug(name)}.png",
                )
                n += 1
    return n


def phase4c_open_set_evaluation(data: dict, fig_dir: Path) -> int:
    n = 0
    for slug, title_suffix in OPEN_SET_SLUGS:
        summary = data.get(f"open_set_summary_{slug}")

        roc_overall = data.get(f"roc_curve_overall_{slug}")
        pr_overall = data.get(f"pr_curve_overall_{slug}")
        if summary is not None and roc_overall is not None and pr_overall is not None:
            auc = summary.set_index("model")["auc_overall_raw"]
            ap = summary.set_index("model")["ap_overall_raw"]
            overall_curves = {
                ROC_PLOT_MODELS[name]: {
                    "fpr": roc_overall.loc[roc_overall["model"] == name, "fpr"].to_numpy(),
                    "tpr": roc_overall.loc[roc_overall["model"] == name, "tpr"].to_numpy(),
                    "roc_auc": auc[name],
                    "precision": pr_overall.loc[pr_overall["model"] == name, "precision"].to_numpy(),
                    "recall": pr_overall.loc[pr_overall["model"] == name, "recall"].to_numpy(),
                    "ap": ap[name],
                }
                for name in _roc_models(auc.index)
            }
            plots.roc_pr_overall(
                overall_curves,
                suptitle=f"Overall ROC/PR ({title_suffix})",
                negatives_label=f"all {title_suffix}",
                save_path=fig_dir / f"roc_pr_overall_{slug}.png",
            )
            n += 1

        roc_scored = data.get(f"roc_curve_scored_only_{slug}")
        if summary is not None and roc_scored is not None:
            auc_scored = summary.set_index("model")["auc_scored_only_raw"]
            scored_only_curves = {
                ROC_PLOT_MODELS[name]: {
                    "fpr": roc_scored.loc[roc_scored["model"] == name, "fpr"].to_numpy(),
                    "tpr": roc_scored.loc[roc_scored["model"] == name, "tpr"].to_numpy(),
                    "roc_auc": auc_scored[name],
                }
                for name in _roc_models(auc_scored.index)
            }
            plots.roc_single_panel(
                scored_only_curves,
                save_path=fig_dir / f"roc_scored_only_{slug}.png",
            )
            n += 1

    return n
