"""
Draws every phase 4 figure from its shaped input data. This is the one
place that knows each figure's title/axis labels and which data it's
drawn from.

Each run_phase4*.py script calls the matching function here once, at
the end of main(), with the exact DataFrames it just computed (and
also saved to CSV). scripts/evaluation/regenerate_plots.py calls the
SAME functions with the equivalent DataFrames reloaded from those
CSVs. Either way the title/label text is written exactly once, here --
neither caller repeats it.

Each phaseN_* function takes `data: dict[str, pd.DataFrame]` keyed by
that phase's reports/tables/*.csv stem and `fig_dir: Path`, draws
whatever keys are present (silently skipping the rest -- the caller,
not this module, is responsible for explaining a missing key, since
only it knows whether that means "not computed yet" or "no CSV on
disk"), and returns how many figures it drew.
"""
from pathlib import Path

from scripts.evaluation import plots
from scripts.evaluation.bias_experiments import points_by_group, points_diff

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

# CSV stems run_phase4_evaluation.py writes and phase4_evaluation() reads,
# exposed so regenerate_plots.py doesn't have to restate this list
PHASE4_EVALUATION_KEYS = (
    ["candidate_counts", "score_margins", "experiment1_before_correction_points", "experiment1_after_correction_points"]
    + [f"{name}_points" for name, *_ in DIFF_EXPERIMENTS]
)

# (slug, title_suffix) for run_phase4c_open_set_evaluation.py's two
# negative-class constructions
OPEN_SET_SLUGS = [
    ("darkvessel", "dark-vessel negatives, held-out registry split"),
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


def phase4c_bias_experiments(data: dict, fig_dir: Path, experiments: list, phmm_name: str, nn_name: str) -> int:
    n = 0
    for slug, feature, ylabel, fmt, alternative, hypothesis, paired in experiments:
        key = f"experiment_{slug.split('.')[0]}_{feature}_points"
        points = data.get(key)
        if points is None:
            continue
        splits_phmm = points_by_group(points[points["method"] == phmm_name].drop(columns="method"))
        splits_nn = points_by_group(points[points["method"] == nn_name].drop(columns="method"))
        plots.boxplot_correct_chosen_shouldbe_compare(
            splits_phmm, splits_nn, "value", ylabel,
            f"Experiment {slug}: Correct vs Chosen vs Should-be",
            names=(phmm_name, nn_name), fmt=fmt,
            save_path=fig_dir / f"{key.removesuffix('_points')}_boxplot.png",
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
                name: {
                    "fpr": roc_overall.loc[roc_overall["model"] == name, "fpr"].to_numpy(),
                    "tpr": roc_overall.loc[roc_overall["model"] == name, "tpr"].to_numpy(),
                    "roc_auc": auc[name],
                    "precision": pr_overall.loc[pr_overall["model"] == name, "precision"].to_numpy(),
                    "recall": pr_overall.loc[pr_overall["model"] == name, "recall"].to_numpy(),
                    "ap": ap[name],
                }
                for name in auc.index
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
                name: {
                    "fpr": roc_scored.loc[roc_scored["model"] == name, "fpr"].to_numpy(),
                    "tpr": roc_scored.loc[roc_scored["model"] == name, "tpr"].to_numpy(),
                    "roc_auc": auc_scored[name],
                }
                for name in auc_scored.index
            }
            plots.roc_single_panel(
                scored_only_curves,
                title=f"ROC, scored rejections only ({title_suffix})",
                save_path=fig_dir / f"roc_scored_only_{slug}.png",
            )
            n += 1

    return n
