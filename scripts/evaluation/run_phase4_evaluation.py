"""
PHMM Forward closed-set evaluation

Evaluates the PHMM Forward alignment on the closed-set preselection
results, then runs the misclassification-bias experiments (track
length, intra-track AIS sampling density, vessel stationarity,
point-level time/distance proximity to the nearest AIS fix) that
motivate the length correction in
scripts/evaluation/phmm_length_correction.py. Each experiment is shown
once uncorrected (raw normalized_forward_score) and once corrected
(alpha tuned on a held-out validation split) -- the same experiments
run_phase4c_bias_experiments.py runs for the NN Time-weighted
baseline.

See notebooks/04_evaluation.ipynb for the interactive, plot-only
version of this script.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

from scripts.evaluation import generate_plots
from scripts.evaluation.bias_experiments import compute_time_dist_diff, compute_track_behavior, points_frame
from scripts.evaluation.metrics import (
    calculate_metrics,
    deterministic_best_alignments,
    get_best_alignments,
    stratified_ranking_metrics,
    summarize_ranking,
)
from scripts.evaluation.phmm_length_correction import correct_forward_score, tune_alpha
from scripts.modeling.transition_probabilities.AIS_RF_distribution import AIS_RF_probability

logger = logging.getLogger(__name__)

# (section, item, value) rows collected into diagnostics.csv
Diagnostics = list[tuple[str, str, float]]

MERGE_COLS = ["RF_track_id", "RF_signal_id", "AIS_track_id", "is_true_match"]
GROUP_COLS = ["RF_signal_id", "RF_track_id"]


def load_data(data_dir: str, base_dir: str = "./data/processed") -> dict[str, pd.DataFrame]:
    """
    Args:
        data_dir: Error-model-specific folder (uniform: same as
            base_dir; other models: a subfolder of it)
        base_dir: Folder holding files shared across error models.
            Default assumes cwd = repo root; pass "../data/processed"
            when calling from a notebook (cwd = notebooks/).
    """
    return {
        "AIS": pd.read_pickle(f"{base_dir}/AIS_sample_no_RF_5000.pkl"),
        "AIS_stats": pd.read_pickle(f"{base_dir}/statistics_sample_5000.pkl").reset_index(drop=True),
        "train_set": pd.read_pickle(f"{data_dir}/train_data_sample_5000.pkl"),
        "preselection": pd.read_pickle(f"{data_dir}/AIS_RF_preselection_data.pkl"),
        "forward_results_raw": pd.read_pickle(f"{data_dir}/AIS_RF_forward_scores_data.pkl"),
    }


def run_ttest(label: str, group1: pd.Series, group2: pd.Series, alternative: str, paired: bool = False) -> dict:
    """One t-test (Welch's if independent, paired otherwise), returned as a result row"""
    if paired:
        res = ttest_rel(group1, group2, alternative=alternative)
        t_stat, p_value = res.statistic, res.pvalue
    else:
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False, alternative=alternative)
    rejected = p_value < 0.05

    return {
        "label": label, "alternative": alternative, "paired": paired,
        "t_stat": t_stat, "p_value": p_value, "h0_rejected": rejected,
    }


def split_correct_incorrect_shouldbe(df_chosen: pd.DataFrame, df_candidates: pd.DataFrame) -> tuple:
    df_best = df_chosen[df_chosen["chosen"]].reset_index(drop=True)
    df_correct = df_best[df_best["is_true_match"]].sort_values(GROUP_COLS).reset_index(drop=True)
    df_incorrect = df_best[~df_best["is_true_match"]].sort_values(GROUP_COLS).reset_index(drop=True)
    df_shouldbe = (
        df_candidates[df_candidates["is_true_match"]]
        .merge(df_incorrect[GROUP_COLS], on=GROUP_COLS, how="inner")
        .sort_values(GROUP_COLS).reset_index(drop=True)
    )
    return df_correct, df_incorrect, df_shouldbe


def add_trajectory_stats(df: pd.DataFrame, df_AIS_stats: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        df_AIS_stats, left_on="AIS_track_id", right_on="ID", how="left"
    ).sort_values(GROUP_COLS).reset_index(drop=True)


def preselection_overview(data: dict, diagnostics: Diagnostics) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_preselection = data["preselection"]
    df_preselection_multimatch = df_preselection.groupby(
        ["RF_track_id", "RF_signal_id"]
    ).filter(lambda x: x["AIS_track_id"].count() > 1).reset_index(drop=True)

    counts_df = df_preselection.groupby(["RF_track_id", "RF_signal_id"]).size().rename("count").reset_index()

    n_total_rf = df_preselection[GROUP_COLS].drop_duplicates().shape[0]
    n_multimatch_rf = df_preselection_multimatch[GROUP_COLS].drop_duplicates().shape[0]
    n_singlematch_rf = n_total_rf - n_multimatch_rf
    pct_filtered_out = round((n_singlematch_rf / n_total_rf) * 100, 2)

    avg_candidates = df_preselection_multimatch.groupby(GROUP_COLS)["AIS_track_id"].nunique().mean()
    diagnostics += [
        ("preselection", "RF signals", n_total_rf),
        ("preselection", "RF signals with several candidates", n_multimatch_rf),
        ("preselection", "RF signals with only their own track", n_singlematch_rf),
        ("preselection", "percent of RF signals filtered out", pct_filtered_out),
        ("preselection", "average candidates per multi-candidate RF signal", avg_candidates),
    ]

    return df_preselection_multimatch, counts_df


def closed_set_metrics(data: dict, df_preselection_multimatch: pd.DataFrame) -> tuple:
    df_forward_results_prefilter = data["preselection"].merge(
        data["forward_results_raw"], on=MERGE_COLS, how="inner", suffixes=("", "_y")
    )
    df_forward_results = df_preselection_multimatch.merge(
        data["forward_results_raw"], on=MERGE_COLS, how="inner", suffixes=("", "_y")
    )
    df_with_chosen = get_best_alignments(df_forward_results, "normalized_forward_score")


    return df_forward_results_prefilter, df_forward_results, df_with_chosen


def ranking_metrics_section(
    df_forward_results_prefilter: pd.DataFrame, df_forward_results: pd.DataFrame, table_dir: Path,
    diagnostics: Diagnostics,
) -> pd.DataFrame:
    ranking_summary_all = summarize_ranking(df_forward_results_prefilter, "normalized_forward_score", mode="max")
    ranking_summary_multi = summarize_ranking(df_forward_results, "normalized_forward_score", mode="max")

    pd.concat([
        stratified_ranking_metrics(ranking_summary_all).assign(population="all"),
        stratified_ranking_metrics(ranking_summary_multi).assign(population="multimatch"),
    ]).pipe(lambda t: t[["population", *[c for c in t.columns if c != "population"]]]).to_csv(
        table_dir / "stratified_ranking.csv", index=False
    )

    margin_data = ranking_summary_multi.dropna(subset=["margin"]).copy()
    top1_correct = margin_data["rank_of_true"] == 1
    diagnostics += [
        ("score margin", "median, correct top-1", margin_data.loc[top1_correct, "margin"].median()),
        ("score margin", "median, incorrect top-1", margin_data.loc[~top1_correct, "margin"].median()),
    ]
    return margin_data[["RF_signal_id", "RF_track_id", "margin", "rank_of_true"]]


def experiment1_boxplots_and_diffs(
    splits: tuple, suffix: str, with_two_sided: bool, ttest_rows: list,
) -> pd.DataFrame:
    df_correct, df_incorrect, df_shouldbe = splits

    points = points_frame(df_correct, df_incorrect, df_shouldbe, "#points")

    ttest_rows.append(run_ttest(
        f"experiment1_{suffix}: incorrect > shouldbe (#points)",
        df_incorrect["#points"], df_shouldbe["#points"], alternative="greater",
    ))
    if with_two_sided:
        ttest_rows.append(run_ttest(
            f"experiment1_{suffix}: incorrect vs shouldbe (#points, two-sided)",
            df_incorrect["#points"], df_shouldbe["#points"], alternative="two-sided",
        ))
    return points


def correction_section(
    data: dict, df_forward_results: pd.DataFrame, df_preselection_multimatch: pd.DataFrame, table_dir: Path,
    diagnostics: Diagnostics,
) -> dict:
    best_alpha, alpha_results, df_tuning, df_experiments = tune_alpha(
        df_forward_results, df_preselection_multimatch, data["AIS_stats"]
    )
    # Log-odds score (score_lo = log P_full - log P_null, see
    # AIS_RF_forward_alignment.py / states.NullMState): same tune_alpha
    # mechanism, grid and seed as the PHMM score above -- the
    # train_test_split itself doesn't depend on score values, so this
    # reproduces the identical tuning/experiments split (df_experiments).
    best_alpha_lo, alpha_results_lo, _, _ = tune_alpha(
        df_forward_results, df_preselection_multimatch, data["AIS_stats"],
        score_col="log_odds_score", exp_col="log_odds_score_exp",
    )

    alpha_search = pd.Series(alpha_results, name="precision").rename_axis("alpha").to_frame()
    alpha_search["precision_log_odds"] = pd.Series(alpha_results_lo)
    alpha_search.to_csv(table_dir / "alpha_search.csv")

    metrics_exp, chosen_exp, all_tracks_exp = correct_forward_score(
        df_experiments, df_preselection_multimatch, data["AIS_stats"], best_alpha
    )
    df_correct_exp, df_incorrect_exp, df_shouldbe_exp = split_correct_incorrect_shouldbe(chosen_exp, all_tracks_exp)

    metrics_before, chosen_before, all_tracks_before = correct_forward_score(
        df_experiments, df_preselection_multimatch, data["AIS_stats"], 1
    )

    # Log-odds primary variant: score_lo used directly, with NO length
    # normalization at all (not even n^1, unlike metrics_before above) --
    # deterministic_best_alignments gives a reproducible lowest-
    # candidate-index tie-break and reports how many RF points were tied.
    chosen_lo_primary, n_tied_lo_primary = deterministic_best_alignments(df_experiments, "log_odds_score")
    metrics_lo_primary = calculate_metrics(chosen_lo_primary)

    # Log-odds secondary variant: score_lo / n^alpha, alpha tuned above.
    metrics_lo_secondary, chosen_lo_secondary, all_tracks_lo_secondary = correct_forward_score(
        df_experiments, df_preselection_multimatch, data["AIS_stats"], best_alpha_lo,
        score_col="log_odds_score", exp_col="log_odds_score_exp",
    )

    def confusion_matrix(metrics, label=None):
        df = pd.DataFrame(
            [[metrics["TP"], metrics["FP"]], [metrics["FN"], metrics["TN"]]],
            index=["Actual Positive", "Actual Negative"],
            columns=["Predicted Positive", "Predicted Negative"],
        )
        if label is None:
            return df
        df.index.name = "actual"
        return df.assign(score=label).reset_index().set_index(["score", "actual"])

    diagnostics += [
        ("length correction", "RF signals in the evaluation split", len(df_experiments)),
        ("length correction", "log-odds (raw) RF signals with tied top candidates", n_tied_lo_primary),
    ]

    pd.concat([
        confusion_matrix(metrics_before, "PHMM Forward (alpha=1)"),
        confusion_matrix(metrics_lo_primary, "Log-odds PHMM (raw)"),
        confusion_matrix(metrics_exp, "PHMM Forward (n^alpha)"),
        confusion_matrix(metrics_lo_secondary, "Log-odds PHMM (n^alpha)"),
    ]).to_csv(table_dir / "confusion_matrices.csv")

    return {
        "best_alpha": best_alpha,
        "df_experiments": df_experiments,
        "df_correct_exp": df_correct_exp,
        "df_incorrect_exp": df_incorrect_exp,
        "df_shouldbe_exp": df_shouldbe_exp,
    }


def experiment_diff_section(
    df_correct: pd.DataFrame, df_incorrect: pd.DataFrame, df_shouldbe: pd.DataFrame,
    feature: str, alternative: str, hypothesis: str, with_two_sided: bool, ttest_rows: list,
) -> pd.DataFrame:
    """Shared points + t-test(s) block for experiments 2-4 (plotting is generate_plots.phase4_evaluation's job)"""
    points = points_frame(df_correct, df_incorrect, df_shouldbe, feature)

    ttest_rows.append(run_ttest(hypothesis, df_incorrect[feature], df_shouldbe[feature], alternative=alternative))
    if with_two_sided:
        ttest_rows.append(run_ttest(
            f"{hypothesis} (two-sided)", df_incorrect[feature], df_shouldbe[feature], alternative="two-sided",
        ))
    return points


def reference_comparison_section(
    data: dict, correction: dict, df_preselection_multimatch: pd.DataFrame, diagnostics: Diagnostics
) -> None:
    """
    Diagnostic-only comparison against a reference labeling built from
    the fixed AIS_RF_probability score instead of the PHMM Forward score
    """
    df_incorrect_exp = correction["df_incorrect_exp"]
    df_shouldbe_exp = correction["df_shouldbe_exp"]

    dist_diff = df_incorrect_exp["distance_diff"] - df_shouldbe_exp["distance_diff"]
    time_diff = df_incorrect_exp["time_diff"] - df_shouldbe_exp["time_diff"]
    for label, diff in (("distance", dist_diff), ("time", time_diff)):
        diagnostics += [
            ("reference labeling", f"{label} diff > 0", np.sum(diff > 0)),
            ("reference labeling", f"{label} diff < 0", np.sum(diff < 0)),
            ("reference labeling", f"{label} diff == 0", np.sum(diff == 0)),
        ]

    time_neg, dist_neg = time_diff < 0, dist_diff < 0
    total = len(time_diff)
    quadrant_pct = {
        "both": 100 * np.sum(time_neg & dist_neg) / total,
        "only_time": 100 * np.sum(time_neg & ~dist_neg) / total,
        "only_distance": 100 * np.sum(~time_neg & dist_neg) / total,
        "rest": 100 * np.sum(~time_neg & ~dist_neg) / total,
    }
    diagnostics += [("incorrect chosen quadrants (percent of pairs)", k, v) for k, v in quadrant_pct.items()]

    df_all_scores_preselected = df_preselection_multimatch.merge(data["forward_results_raw"], how="inner")
    df_all_scores_preselected = compute_time_dist_diff(df_all_scores_preselected, data["train_set"])
    df_all_scores_preselected["AIS_RF_score"] = df_all_scores_preselected.apply(
        lambda row: AIS_RF_probability(row["time_diff"], row["distance_diff"]), axis=1
    )
    df_all_scores_preselected["is_match_ref"] = df_all_scores_preselected["AIS_RF_score"].eq(
        df_all_scores_preselected.groupby(GROUP_COLS)["AIS_RF_score"].transform("max")
    )
    df_all_scores_preselected = df_all_scores_preselected.merge(correction["df_experiments"])

    metrics_ref, chosen_ref, all_tracks_ref = correct_forward_score(
        df_all_scores_preselected, df_preselection_multimatch, data["AIS_stats"],
        correction["best_alpha"], match_col="is_match_ref",
    )
    diagnostics += [("reference labeling metrics", k, v) for k, v in metrics_ref.items()]

    best_chosen_ref = chosen_ref[chosen_ref["chosen"]].reset_index(drop=True)
    df_correct_ref = best_chosen_ref[best_chosen_ref["is_match_ref"]].sort_values(GROUP_COLS).reset_index(drop=True)
    df_incorrect_ref = best_chosen_ref[~best_chosen_ref["is_match_ref"]].sort_values(GROUP_COLS).reset_index(drop=True)
    df_shouldbe_ref = (
        all_tracks_ref[all_tracks_ref["is_match_ref"]]
        .merge(df_incorrect_ref[GROUP_COLS], on=GROUP_COLS, how="inner")
        .sort_values(GROUP_COLS).reset_index(drop=True)
    )

    original = pd.concat([
        correction["df_correct_exp"][GROUP_COLS].assign(label_original="correct"),
        df_incorrect_exp[GROUP_COLS].assign(label_original="incorrect"),
        df_shouldbe_exp[GROUP_COLS].assign(label_original="should_be"),
    ], ignore_index=True).drop_duplicates(GROUP_COLS)

    reference = pd.concat([
        df_correct_ref[GROUP_COLS].assign(label_reference="correct"),
        df_incorrect_ref[GROUP_COLS].assign(label_reference="incorrect"),
        df_shouldbe_ref[GROUP_COLS].assign(label_reference="should_be"),
    ], ignore_index=True).drop_duplicates(GROUP_COLS)

    original_reference_df = original.merge(reference, on=GROUP_COLS, how="outer", indicator=True).fillna(
        {"label_original": "<missing>", "label_reference": "<missing>"}
    )
    transition = pd.crosstab(
        original_reference_df["label_original"], original_reference_df["label_reference"], dropna=False
    )
    diagnostics += [
        ("reference labeling transitions (generated -> nearest)", f"{old} -> {new}", count)
        for old, row in transition.iterrows() for new, count in row.items()
    ]


def main(data_dir: str, fig_dir: Path, table_dir: Path) -> None:
    data = load_data(data_dir)
    plot_data = {}
    diagnostics: Diagnostics = []

    df_preselection_multimatch, plot_data["candidate_counts"] = preselection_overview(data, diagnostics)

    df_forward_results_prefilter, df_forward_results, df_with_chosen = closed_set_metrics(
        data, df_preselection_multimatch
    )

    plot_data["score_margins"] = ranking_metrics_section(
        df_forward_results_prefilter, df_forward_results, table_dir, diagnostics
    )

    ttest_rows = []

    uncorrected_splits = split_correct_incorrect_shouldbe(df_with_chosen, df_forward_results)
    uncorrected_splits = tuple(add_trajectory_stats(df, data["AIS_stats"]) for df in uncorrected_splits)
    plot_data["experiment1_before_correction_points"] = experiment1_boxplots_and_diffs(
        uncorrected_splits, suffix="before_correction", with_two_sided=False, ttest_rows=ttest_rows,
    )

    correction = correction_section(data, df_forward_results, df_preselection_multimatch, table_dir, diagnostics)
    df_correct_exp, df_incorrect_exp, df_shouldbe_exp = (
        correction["df_correct_exp"], correction["df_incorrect_exp"], correction["df_shouldbe_exp"]
    )

    plot_data["experiment1_after_correction_points"] = experiment1_boxplots_and_diffs(
        (df_correct_exp, df_incorrect_exp, df_shouldbe_exp), suffix="after_correction",
        with_two_sided=True, ttest_rows=ttest_rows,
    )

    plot_data["experiment2a_intra_time_points"] = experiment_diff_section(
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp, "mean_intra_time_diff",
        alternative="less", hypothesis="experiment2a: incorrect < shouldbe (mean intra-time gap)",
        with_two_sided=True, ttest_rows=ttest_rows,
    )
    plot_data["experiment2b_intra_distance_points"] = experiment_diff_section(
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp, "mean_intra_dist_diff",
        alternative="less", hypothesis="experiment2b: incorrect < shouldbe (mean intra-distance gap)",
        with_two_sided=False, ttest_rows=ttest_rows,
    )

    df_correct_exp = compute_track_behavior(df_correct_exp, data["AIS"])
    df_incorrect_exp = compute_track_behavior(df_incorrect_exp, data["AIS"])
    df_shouldbe_exp = compute_track_behavior(df_shouldbe_exp, data["AIS"])
    plot_data["experiment3_stationarity_points"] = experiment_diff_section(
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp, "stationary_ratio",
        alternative="greater", hypothesis="experiment3: incorrect > shouldbe (stationary ratio)",
        with_two_sided=False, ttest_rows=ttest_rows,
    )

    df_correct_exp = compute_time_dist_diff(df_correct_exp, data["train_set"])
    df_incorrect_exp = compute_time_dist_diff(df_incorrect_exp, data["train_set"])
    df_shouldbe_exp = compute_time_dist_diff(df_shouldbe_exp, data["train_set"])
    plot_data["experiment4a_time_points"] = experiment_diff_section(
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp, "time_diff",
        alternative="less", hypothesis="experiment4a: incorrect < shouldbe (time to nearest AIS fix)",
        with_two_sided=True, ttest_rows=ttest_rows,
    )
    plot_data["experiment4b_distance_points"] = experiment_diff_section(
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp, "distance_diff",
        alternative="less", hypothesis="experiment4b: incorrect < shouldbe (distance to nearest AIS fix)",
        with_two_sided=True, ttest_rows=ttest_rows,
    )

    pd.DataFrame(ttest_rows).to_csv(table_dir / "ttest_results.csv", index=False)

    correction["df_correct_exp"], correction["df_incorrect_exp"], correction["df_shouldbe_exp"] = (
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp
    )
    reference_comparison_section(data, correction, df_preselection_multimatch, diagnostics)

    pd.DataFrame(diagnostics, columns=["section", "item", "value"]).to_csv(table_dir / "diagnostics.csv", index=False)
    generate_plots.save_plot_data(plot_data, table_dir)
    generate_plots.phase4_evaluation(plot_data, fig_dir)

    n_tied = next(v for sec, item, v in diagnostics if item.endswith("tied top candidates"))
    logger.info(
        "Phase 4: PHMM Forward alpha=%s (tuning split), %d tests written to ttest_results.csv, "
        "%d log-odds ties on the evaluation split.", correction["best_alpha"], len(ttest_rows), n_tied,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHMM Forward evaluation and bias experiments")
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="gaussian",
        help="RF bearing-error model whose data folder to read from (default: gaussian)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    DATA_DIR = f"./data/processed/{args.error_model}"
    FIG_DIR = Path(f"reports/figures/phase4_evaluation/{args.error_model}")
    TABLE_DIR = Path(f"reports/tables/phase4_evaluation/{args.error_model}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    main(DATA_DIR, FIG_DIR, TABLE_DIR)
