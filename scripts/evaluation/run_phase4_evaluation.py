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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

from scripts.evaluation import generate_plots
from scripts.evaluation.bias_experiments import compute_time_dist_diff, compute_track_behavior, points_frame
from scripts.evaluation.metrics import (
    calculate_metrics,
    get_best_alignments,
    mean_reciprocal_rank,
    recall_at_k,
    stratified_ranking_metrics,
    summarize_ranking,
)
from scripts.evaluation.phmm_length_correction import correct_forward_score, tune_alpha
from scripts.modeling.transition_probabilities.AIS_RF_distribution import AIS_RF_probability

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
    """One t-test (Welch's if independent, paired otherwise), printed and returned as a result row"""
    if paired:
        res = ttest_rel(group1, group2, alternative=alternative)
        t_stat, p_value = res.statistic, res.pvalue
    else:
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False, alternative=alternative)
    rejected = p_value < 0.05

    print(f"\033[1mH1: {label}\033[0m")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.10f} or {p_value:.2e}")
    print("H0 can be rejected" if rejected else "H0 cannot be rejected")

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


def preselection_overview(data: dict, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_preselection = data["preselection"]
    df_preselection_multimatch = df_preselection.groupby(
        ["RF_track_id", "RF_signal_id"]
    ).filter(lambda x: x["AIS_track_id"].count() > 1).reset_index(drop=True)

    counts_df = df_preselection.groupby(["RF_track_id", "RF_signal_id"]).size().rename("count").reset_index()
    counts_df.to_csv(table_dir / "candidate_counts.csv", index=False)

    n_total_rf = df_preselection[GROUP_COLS].drop_duplicates().shape[0]
    n_multimatch_rf = df_preselection_multimatch[GROUP_COLS].drop_duplicates().shape[0]
    n_singlematch_rf = n_total_rf - n_multimatch_rf
    pct_filtered_out = round((n_singlematch_rf / n_total_rf) * 100, 2)

    print("Number of unique RF signals in preselection:", n_total_rf)
    print("Number of unique RF signals in preselection with multiple AIS matches:", n_multimatch_rf)
    print("Number of unique RF signals in preselection with only their own AIS match:", n_singlematch_rf)
    print("Percentage of RF signals that are filtered out:", pct_filtered_out, "%")
    print("Percentage of RF signals kept for the rest of the results:", 100 - pct_filtered_out, "%")

    return df_preselection_multimatch, counts_df


def closed_set_metrics(data: dict, df_preselection_multimatch: pd.DataFrame) -> tuple:
    df_forward_results_prefilter = data["preselection"].merge(
        data["forward_results_raw"], on=MERGE_COLS, how="inner", suffixes=("", "_y")
    )
    df_with_chosen_all = get_best_alignments(df_forward_results_prefilter, "normalized_forward_score")
    print("All results (before removing single matches):")
    for key, value in calculate_metrics(df_with_chosen_all).items():
        print(f"  {key}: {value}")

    df_forward_results = df_preselection_multimatch.merge(
        data["forward_results_raw"], on=MERGE_COLS, how="inner", suffixes=("", "_y")
    )
    df_with_chosen = get_best_alignments(df_forward_results, "normalized_forward_score")
    print("\nResults after removing single matches (multimatch only):")
    for key, value in calculate_metrics(df_with_chosen).items():
        print(f"  {key}: {value}")

    avg_matches = df_forward_results.groupby(GROUP_COLS)["AIS_track_id"].nunique().mean()
    print(f"\nAverage number of AIS tracks per RF track: {avg_matches:.2f}")

    return df_forward_results_prefilter, df_forward_results, df_with_chosen


def ranking_metrics_section(
    df_forward_results_prefilter: pd.DataFrame, df_forward_results: pd.DataFrame, table_dir: Path,
) -> pd.DataFrame:
    ranking_summary_all = summarize_ranking(df_forward_results_prefilter, "normalized_forward_score", mode="max")
    ranking_summary_multi = summarize_ranking(df_forward_results, "normalized_forward_score", mode="max")

    print("Recall@k / MRR -- all candidate pairs")
    print("recall@k:", recall_at_k(ranking_summary_all))
    print("MRR:", mean_reciprocal_rank(ranking_summary_all))
    print("\nRecall@k / MRR -- multimatch RF signals only")
    print("recall@k:", recall_at_k(ranking_summary_multi))
    print("MRR:", mean_reciprocal_rank(ranking_summary_multi))

    stratified_all = stratified_ranking_metrics(ranking_summary_all)
    stratified_multi = stratified_ranking_metrics(ranking_summary_multi)
    stratified_all.to_csv(table_dir / "stratified_ranking_all_candidates.csv", index=False)
    stratified_multi.to_csv(table_dir / "stratified_ranking_multimatch_candidates.csv", index=False)
    print("\nStratified ranking metrics (all candidate pairs):")
    print(stratified_all)
    print("\nStratified ranking metrics (multimatch candidate pairs):")
    print(stratified_multi)

    margin_data = ranking_summary_multi.dropna(subset=["margin"]).copy()
    top1_correct = margin_data["rank_of_true"] == 1
    margin_cols = margin_data[["RF_signal_id", "RF_track_id", "margin", "rank_of_true"]]
    margin_cols.to_csv(table_dir / "score_margins.csv", index=False)
    print("\nMedian margin (correct top-1):", margin_data.loc[top1_correct, "margin"].median())
    print("Median margin (incorrect top-1):", margin_data.loc[~top1_correct, "margin"].median())

    return margin_cols


def experiment1_boxplots_and_diffs(
    splits: tuple, table_dir: Path, suffix: str, with_two_sided: bool, ttest_rows: list,
) -> pd.DataFrame:
    df_correct, df_incorrect, df_shouldbe = splits

    points = points_frame(df_correct, df_incorrect, df_shouldbe, "#points")
    points.to_csv(table_dir / f"experiment1_{suffix}_points.csv", index=False)

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
) -> dict:
    best_alpha, alpha_results, df_tuning, df_experiments = tune_alpha(
        df_forward_results, df_preselection_multimatch, data["AIS_stats"]
    )
    pd.Series(alpha_results, name="precision").rename_axis("alpha").to_csv(table_dir / "alpha_search.csv")
    print(f"Best alpha is {best_alpha}, with a precision score of {alpha_results[best_alpha]}")

    metrics_exp, chosen_exp, all_tracks_exp = correct_forward_score(
        df_experiments, df_preselection_multimatch, data["AIS_stats"], best_alpha
    )
    df_correct_exp, df_incorrect_exp, df_shouldbe_exp = split_correct_incorrect_shouldbe(chosen_exp, all_tracks_exp)

    metrics_before, chosen_before, all_tracks_before = correct_forward_score(
        df_experiments, df_preselection_multimatch, data["AIS_stats"], 1
    )

    def confusion_matrix(metrics):
        return pd.DataFrame(
            [[metrics["TP"], metrics["FP"]], [metrics["FN"], metrics["TN"]]],
            index=["Actual Positive", "Actual Negative"],
            columns=["Predicted Positive", "Predicted Negative"],
        )

    print(f"\nNumber of rows: {len(df_experiments)}")
    print("Confusion Matrix before correction:\n", confusion_matrix(metrics_before))
    print("Performance Metrics before correction:")
    for key in ["precision", "recall", "f1_score", "accuracy"]:
        print(f"  {key}: {metrics_before[key]:.4f}")
    print("\nConfusion Matrix after correction:\n", confusion_matrix(metrics_exp))
    print("Performance Metrics after correction:")
    for key in ["precision", "recall", "f1_score", "accuracy"]:
        print(f"  {key}: {metrics_exp[key]:.4f}")

    confusion_matrix(metrics_before).to_csv(table_dir / "confusion_matrix_before_correction.csv")
    confusion_matrix(metrics_exp).to_csv(table_dir / "confusion_matrix_after_correction.csv")

    return {
        "best_alpha": best_alpha,
        "df_experiments": df_experiments,
        "df_correct_exp": df_correct_exp,
        "df_incorrect_exp": df_incorrect_exp,
        "df_shouldbe_exp": df_shouldbe_exp,
    }


def experiment_diff_section(
    df_correct: pd.DataFrame, df_incorrect: pd.DataFrame, df_shouldbe: pd.DataFrame,
    feature: str, alternative: str, hypothesis: str, with_two_sided: bool, table_dir: Path, name: str,
    ttest_rows: list,
) -> pd.DataFrame:
    """Shared points-CSV + t-test(s) block for experiments 2-4 (plotting is generate_plots.phase4_evaluation's job)"""
    points = points_frame(df_correct, df_incorrect, df_shouldbe, feature)
    points.to_csv(table_dir / f"{name}_points.csv", index=False)

    ttest_rows.append(run_ttest(hypothesis, df_incorrect[feature], df_shouldbe[feature], alternative=alternative))
    if with_two_sided:
        ttest_rows.append(run_ttest(
            f"{hypothesis} (two-sided)", df_incorrect[feature], df_shouldbe[feature], alternative="two-sided",
        ))
    return points


def reference_comparison_section(
    data: dict, correction: dict, df_preselection_multimatch: pd.DataFrame, table_dir: Path
) -> None:
    """
    Diagnostic-only comparison against a reference labeling built from
    the fixed AIS_RF_probability score instead of the PHMM Forward score
    """
    df_incorrect_exp = correction["df_incorrect_exp"]
    df_shouldbe_exp = correction["df_shouldbe_exp"]

    dist_diff = df_incorrect_exp["distance_diff"] - df_shouldbe_exp["distance_diff"]
    time_diff = df_incorrect_exp["time_diff"] - df_shouldbe_exp["time_diff"]
    print("Distance diffs(>0, <0, ==0):",
          np.sum(dist_diff > 0), np.sum(dist_diff < 0), np.sum(dist_diff == 0))
    print("Time diffs(>0, <0, ==0):",
          np.sum(time_diff > 0), np.sum(time_diff < 0), np.sum(time_diff == 0))

    time_neg, dist_neg = time_diff < 0, dist_diff < 0
    total = len(time_diff)
    quadrant_pct = {
        "both": 100 * np.sum(time_neg & dist_neg) / total,
        "only_time": 100 * np.sum(time_neg & ~dist_neg) / total,
        "only_distance": 100 * np.sum(~time_neg & dist_neg) / total,
        "rest": 100 * np.sum(~time_neg & ~dist_neg) / total,
    }
    print("Percentages (of all pairs):", quadrant_pct)
    pd.Series(quadrant_pct).to_csv(table_dir / "incorrect_chosen_quadrants.csv")

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
    print("\nReference labeling metrics:")
    for key, value in metrics_ref.items():
        print(f"  {key}: {value}")

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
    print("\nTransition matrix (generated -> nearest):")
    print(transition)
    transition.to_csv(table_dir / "reference_transition_matrix.csv")

    changed = original_reference_df[
        original_reference_df["label_original"] != original_reference_df["label_reference"]
    ].copy()
    changed["transition"] = changed["label_original"] + " -> " + changed["label_reference"]
    print("\nTop transitions:")
    print(changed["transition"].value_counts())


def main(data_dir: str, fig_dir: Path, table_dir: Path) -> None:
    data = load_data(data_dir)
    plot_data = {}

    print("=== Preselection overview ===")
    df_preselection_multimatch, plot_data["candidate_counts"] = preselection_overview(data, table_dir)

    print("\n=== Closed-set metrics ===")
    df_forward_results_prefilter, df_forward_results, df_with_chosen = closed_set_metrics(
        data, df_preselection_multimatch
    )

    print("\n=== Ranking metrics ===")
    plot_data["score_margins"] = ranking_metrics_section(df_forward_results_prefilter, df_forward_results, table_dir)

    ttest_rows = []

    print("\n=== Experiment 1 (before correction) ===")
    uncorrected_splits = split_correct_incorrect_shouldbe(df_with_chosen, df_forward_results)
    uncorrected_splits = tuple(add_trajectory_stats(df, data["AIS_stats"]) for df in uncorrected_splits)
    plot_data["experiment1_before_correction_points"] = experiment1_boxplots_and_diffs(
        uncorrected_splits, table_dir, suffix="before_correction", with_two_sided=False, ttest_rows=ttest_rows,
    )

    print("\n=== PHMM length correction ===")
    correction = correction_section(data, df_forward_results, df_preselection_multimatch, table_dir)
    df_correct_exp, df_incorrect_exp, df_shouldbe_exp = (
        correction["df_correct_exp"], correction["df_incorrect_exp"], correction["df_shouldbe_exp"]
    )

    print("\n=== Experiment 1 (after correction) ===")
    plot_data["experiment1_after_correction_points"] = experiment1_boxplots_and_diffs(
        (df_correct_exp, df_incorrect_exp, df_shouldbe_exp), table_dir, suffix="after_correction",
        with_two_sided=True, ttest_rows=ttest_rows,
    )

    print("\n=== Experiment 2: intra-track AIS sampling density ===")
    plot_data["experiment2a_intra_time_points"] = experiment_diff_section(
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp, "mean_intra_time_diff",
        alternative="less", hypothesis="experiment2a: incorrect < shouldbe (mean intra-time gap)",
        with_two_sided=True, table_dir=table_dir, name="experiment2a_intra_time", ttest_rows=ttest_rows,
    )
    plot_data["experiment2b_intra_distance_points"] = experiment_diff_section(
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp, "mean_intra_dist_diff",
        alternative="less", hypothesis="experiment2b: incorrect < shouldbe (mean intra-distance gap)",
        with_two_sided=False, table_dir=table_dir, name="experiment2b_intra_distance", ttest_rows=ttest_rows,
    )

    print("\n=== Experiment 3: vessel stationarity ===")
    df_correct_exp = compute_track_behavior(df_correct_exp, data["AIS"])
    df_incorrect_exp = compute_track_behavior(df_incorrect_exp, data["AIS"])
    df_shouldbe_exp = compute_track_behavior(df_shouldbe_exp, data["AIS"])
    plot_data["experiment3_stationarity_points"] = experiment_diff_section(
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp, "stationary_ratio",
        alternative="greater", hypothesis="experiment3: incorrect > shouldbe (stationary ratio)",
        with_two_sided=False, table_dir=table_dir, name="experiment3_stationarity", ttest_rows=ttest_rows,
    )

    print("\n=== Experiment 4: point-level time/distance proximity to nearest AIS fix ===")
    df_correct_exp = compute_time_dist_diff(df_correct_exp, data["train_set"])
    df_incorrect_exp = compute_time_dist_diff(df_incorrect_exp, data["train_set"])
    df_shouldbe_exp = compute_time_dist_diff(df_shouldbe_exp, data["train_set"])
    plot_data["experiment4a_time_points"] = experiment_diff_section(
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp, "time_diff",
        alternative="less", hypothesis="experiment4a: incorrect < shouldbe (time to nearest AIS fix)",
        with_two_sided=True, table_dir=table_dir, name="experiment4a_time", ttest_rows=ttest_rows,
    )
    plot_data["experiment4b_distance_points"] = experiment_diff_section(
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp, "distance_diff",
        alternative="less", hypothesis="experiment4b: incorrect < shouldbe (distance to nearest AIS fix)",
        with_two_sided=True, table_dir=table_dir, name="experiment4b_distance", ttest_rows=ttest_rows,
    )

    pd.DataFrame(ttest_rows).to_csv(table_dir / "ttest_results.csv", index=False)

    print("\n=== Incorrect chosen matches: reference-labeling comparison ===")
    correction["df_correct_exp"], correction["df_incorrect_exp"], correction["df_shouldbe_exp"] = (
        df_correct_exp, df_incorrect_exp, df_shouldbe_exp
    )
    reference_comparison_section(data, correction, df_preselection_multimatch, table_dir)

    generate_plots.phase4_evaluation(plot_data, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHMM Forward evaluation and bias experiments")
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="gaussian",
        help="RF bearing-error model whose data folder to read from (default: gaussian)",
    )
    args = parser.parse_args()

    DATA_DIR = "./data/processed" if args.error_model == "uniform" else f"./data/processed/{args.error_model}"
    FIG_DIR = Path(f"reports/figures/phase4_evaluation/{args.error_model}")
    TABLE_DIR = Path(f"reports/tables/phase4_evaluation/{args.error_model}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    main(DATA_DIR, FIG_DIR, TABLE_DIR)
