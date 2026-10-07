"""
PHMM vs. NN Time-weighted misclassification-bias experiments

Re-runs the four misclassification-bias experiments from
run_phase4_evaluation.py (track length, intra-track AIS sampling
density, vessel stationarity, point-level time/distance proximity to
the nearest AIS fix) for the NN Time-weighted baseline, alongside a
freshly recomputed PHMM reference (length-corrected, same
alpha-tuning procedure) on the same held-out validation split
(train_test_split(..., random_state=100)), so the two methods are
compared on identical RF points. Question: are PHMM's biases still
present with the far simpler time+distance nearest-neighbor rule?

See notebooks/07_baseline_bias_experiments.ipynb for the interactive,
plot-only version of this script.
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
from sklearn.model_selection import train_test_split

from scripts.evaluation import generate_plots
from scripts.evaluation.bias_experiments import (
    add_trajectory_stats,
    compute_time_dist_diff,
    compute_track_behavior,
    points_frame,
    split_correct_incorrect_shouldbe,
)
from scripts.evaluation.metrics import calculate_metrics, get_best_alignments
from scripts.evaluation.phmm_length_correction import correct_forward_score, tune_alpha

PHMM_NAME = "PHMM Forward (corrected)"
NN_NAME = "NN Time-weighted"
LOGODDS_NAME = "Log-odds PHMM (n^alpha)"

# (experiment label, feature column, ylabel, fmt, alternative, hypothesis, paired)
EXPERIMENTS = [
    ("1. Track length", "#points", "Number of AIS messages per trajectory", "{:.1f}",
     "greater", "incorrect (chosen) > should-be", False),
    ("2a. Intra-time gap", "mean_intra_time_diff", "Seconds", "{:.2f}",
     "less", "incorrect (chosen) < should-be (mean intra-time gap)", False),
    ("2b. Intra-distance gap", "mean_intra_dist_diff", "Kilometers", "{:.3f}",
     "less", "incorrect (chosen) < should-be (mean intra-distance gap)", False),
    ("3. Stationarity", "stationary_ratio", "Proportion of time stationary (< 5 km/h)", "{:.2f}",
     "greater", "incorrect (chosen) > should-be (stationary ratio)", False),
    ("4a. Time to nearest AIS fix", "time_diff", "Seconds", "{:.2f}",
     "less", "incorrect (chosen) < should-be (time to nearest AIS fix)", True),
    ("4b. Distance to nearest AIS fix", "distance_diff", "Kilometers", "{:.3f}",
     "less", "incorrect (chosen) < should-be (distance to nearest AIS fix)", True),
]


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
        "forward_results": pd.read_pickle(f"{data_dir}/AIS_RF_forward_scores_data.pkl"),
        "nn_time_weighted": pd.read_pickle(f"{data_dir}/AIS_RF_nn_baseline_time_weighted_scores_data.pkl"),
    }


def build_validation_split(data: dict[str, pd.DataFrame]) -> dict:
    """Same multimatch filter + train_test_split(random_state=100) as run_phase4_evaluation.py"""
    df_preselection_multimatch = data["preselection"].groupby(
        ["RF_track_id", "RF_signal_id"]
    ).filter(lambda x: x["AIS_track_id"].count() > 1).reset_index(drop=True)

    df_forward_results = df_preselection_multimatch.merge(
        data["forward_results"],
        on=["RF_track_id", "RF_signal_id", "AIS_track_id", "is_true_match"],
        how="inner", suffixes=("", "_y"),
    )

    unique_rf_pairs = df_forward_results[["RF_signal_id", "RF_track_id"]].drop_duplicates()
    train_pairs, val_pairs = train_test_split(unique_rf_pairs, test_size=0.2, random_state=100)

    return {
        "preselection_multimatch": df_preselection_multimatch,
        "forward_results": df_forward_results,
        "tuning": df_forward_results.merge(val_pairs, on=["RF_signal_id", "RF_track_id"]),
        "experiments": df_forward_results.merge(train_pairs, on=["RF_signal_id", "RF_track_id"]),
        "n_val": len(train_pairs),
    }


def phmm_reference_splits(data, split) -> tuple:
    """Alpha-tune on the tuning split, then correct/incorrect/should-be on the validation split"""
    best_alpha, alpha_results, _, df_experiments = tune_alpha(
        split["forward_results"], split["preselection_multimatch"], data["AIS_stats"]
    )
    print(f"Best alpha: {best_alpha}, tuning-set precision: {alpha_results[best_alpha]}")

    metrics_phmm, chosen_phmm, all_tracks_phmm = correct_forward_score(
        df_experiments, split["preselection_multimatch"], data["AIS_stats"], best_alpha
    )
    print("PHMM Forward (corrected) metrics on the validation split:")
    for key, value in metrics_phmm.items():
        print(f"  {key}: {value}")

    return split_correct_incorrect_shouldbe(all_tracks_phmm, chosen_phmm)


def logodds_reference_splits(data, split) -> tuple:
    """Same alpha-tuning procedure as phmm_reference_splits, for the log-odds score"""
    best_alpha, alpha_results, _, df_experiments = tune_alpha(
        split["forward_results"], split["preselection_multimatch"], data["AIS_stats"],
        score_col="log_odds_score", exp_col="log_odds_score_exp",
    )
    print(f"Best log-odds alpha: {best_alpha}, tuning-set precision: {alpha_results[best_alpha]}")

    metrics_lo, chosen_lo, all_tracks_lo = correct_forward_score(
        df_experiments, split["preselection_multimatch"], data["AIS_stats"], best_alpha,
        score_col="log_odds_score", exp_col="log_odds_score_exp",
    )
    print(f"{LOGODDS_NAME} metrics on the validation split:")
    for key, value in metrics_lo.items():
        print(f"  {key}: {value}")

    return split_correct_incorrect_shouldbe(all_tracks_lo, chosen_lo)


def nn_time_weighted_splits(data, split) -> tuple:
    df_nn_candidates = split["experiments"][
        ["RF_signal_id", "RF_track_id", "AIS_track_id", "is_true_match"]
    ].merge(
        data["nn_time_weighted"],
        on=["RF_signal_id", "RF_track_id", "AIS_track_id", "is_true_match"],
        how="inner",
    )
    chosen_nn = get_best_alignments(df_nn_candidates, "nn_distance", mode="min")
    metrics_nn = calculate_metrics(chosen_nn)
    print("NN Time-weighted metrics on the validation split:")
    for key, value in metrics_nn.items():
        print(f"  {key}: {value}")

    return split_correct_incorrect_shouldbe(df_nn_candidates, chosen_nn)


def enrich(df_correct, df_incorrect, df_shouldbe, data, add_trajectory: bool):
    """
    Attach trajectory-level, behavior, and point-level features, and
    sort each split by RF point so incorrect/should-be rows are paired
    for the paired t-tests in experiments 4a/4b
    """
    if add_trajectory:
        df_correct = add_trajectory_stats(df_correct, data["AIS_stats"])
        df_incorrect = add_trajectory_stats(df_incorrect, data["AIS_stats"])
        df_shouldbe = add_trajectory_stats(df_shouldbe, data["AIS_stats"])

    df_correct = compute_track_behavior(df_correct, data["AIS"])
    df_incorrect = compute_track_behavior(df_incorrect, data["AIS"])
    df_shouldbe = compute_track_behavior(df_shouldbe, data["AIS"])

    df_correct = compute_time_dist_diff(df_correct, data["train_set"])
    df_incorrect = compute_time_dist_diff(df_incorrect, data["train_set"])
    df_shouldbe = compute_time_dist_diff(df_shouldbe, data["train_set"])

    sort_cols = ["RF_signal_id", "RF_track_id"]
    df_correct = df_correct.sort_values(sort_cols).reset_index(drop=True)
    df_incorrect = df_incorrect.sort_values(sort_cols).reset_index(drop=True)
    df_shouldbe = df_shouldbe.sort_values(sort_cols).reset_index(drop=True)
    return df_correct, df_incorrect, df_shouldbe


def report_ttest(group1_phmm, group2_phmm, group1_nn, group2_nn, alternative, hypothesis, paired=False):
    """
    Run the same (one-sided + two-sided) t-test for PHMM and NN Time-weighted

    group1/group2 follow the incorrect/should-be convention: group1 =
    incorrect (chosen), group2 = should-be.
    """
    test = ttest_rel if paired else lambda a, b, alternative: ttest_ind(
        a, b, equal_var=False, alternative=alternative
    )

    print(f"\033[1mH1: {hypothesis}\033[0m")
    for name, group1, group2 in [(PHMM_NAME, group1_phmm, group2_phmm), (NN_NAME, group1_nn, group2_nn)]:
        res_one = test(group1, group2, alternative=alternative)
        res_two = test(group1, group2, alternative="two-sided")
        t_one, p_one = (res_one.statistic, res_one.pvalue) if paired else res_one
        t_two, p_two = (res_two.statistic, res_two.pvalue) if paired else res_two
        verdict = "H0 rejected (H1 direction)" if p_one < 0.05 else (
            "H0 rejected (opposite direction)" if p_two < 0.05 else "H0 not rejected"
        )
        print(f"  {name:<18} t={t_one:8.4f}  p(one-sided)={p_one:.2e}  p(two-sided)={p_two:.2e}  -> {verdict}")
    print()


# Column-label aliases for the two original models, kept exactly as
# before; any other model name in effect_row()'s models list gets
# "{name} p"/"{name} effect" columns instead.
_COLUMN_LABELS = {"PHMM": "PHMM", "NN Time-weighted": "NN-TW"}


def effect_row(experiment, feature, models, paired=False):
    """
    One bias_experiments_summary.csv row: p-value + effect direction
    for each (name, dfi, dfs) in models

    models: list of (name, df_incorrect, df_shouldbe) tuples. Passing
    additional models beyond the original ("PHMM", "NN Time-weighted")
    adds columns, never changes the existing two.
    """
    test = ttest_rel if paired else lambda a, b, alternative: ttest_ind(
        a, b, equal_var=False, alternative=alternative
    )
    row = {"experiment": experiment}
    for name, dfi, dfs in models:
        res = test(dfi[feature], dfs[feature], alternative="two-sided")
        t_stat, p_value = (res.statistic, res.pvalue) if paired else res
        direction = "chosen > should-be" if t_stat > 0 else "chosen < should-be"
        label = _COLUMN_LABELS.get(name, name)
        row[f"{label} p"] = p_value
        row[f"{label} effect"] = direction if p_value < 0.05 else "n.s."
    return row


def misclassification_overlap(df_incorrect_phmm: pd.DataFrame, df_incorrect_nn: pd.DataFrame) -> dict:
    """
    How much do PHMM's and NN Time-weighted's misclassified RF signals
    overlap

    Both methods run on the same validation split, but may be fooled
    by different RF signals. A low Jaccard overlap means each method
    has its own distinct failure mode rather than a shared blind spot
    (plausible, since which competing AIS tracks land in a candidate
    set is an accident of which real vessels are nearby --
    generate_synthethic_RF_data.py's error model has no notion of
    other trajectories). A high overlap, especially many shared
    mistakes landing on the same wrong track, would point to a real
    common blind spot.
    """
    phmm_ids = set(zip(df_incorrect_phmm["RF_signal_id"], df_incorrect_phmm["RF_track_id"]))
    nn_ids = set(zip(df_incorrect_nn["RF_signal_id"], df_incorrect_nn["RF_track_id"]))
    both = phmm_ids & nn_ids
    union = phmm_ids | nn_ids

    phmm_pick = df_incorrect_phmm.set_index(["RF_signal_id", "RF_track_id"])["AIS_track_id"]
    nn_pick = df_incorrect_nn.set_index(["RF_signal_id", "RF_track_id"])["AIS_track_id"]
    both_idx = pd.MultiIndex.from_tuples(sorted(both), names=["RF_signal_id", "RF_track_id"])
    same_wrong_track = int((phmm_pick.loc[both_idx] == nn_pick.loc[both_idx]).sum()) if len(both_idx) else 0

    result = {
        "n_phmm_incorrect": len(phmm_ids),
        "n_nn_incorrect": len(nn_ids),
        "n_both_incorrect": len(both),
        "n_phmm_only": len(phmm_ids - nn_ids),
        "n_nn_only": len(nn_ids - phmm_ids),
        "jaccard_overlap": len(both) / len(union) if union else float("nan"),
        "pct_of_phmm_mistakes_shared": len(both) / len(phmm_ids) if phmm_ids else float("nan"),
        "pct_of_nn_mistakes_shared": len(both) / len(nn_ids) if nn_ids else float("nan"),
        "of_shared_mistakes_same_wrong_track": same_wrong_track,
        "of_shared_mistakes_same_wrong_track_pct": same_wrong_track / len(both) if both else float("nan"),
    }
    print("Misclassification overlap, PHMM vs NN Time-weighted:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return result


def main(data_dir: str, fig_dir: Path, table_dir: Path) -> None:
    data = load_data(data_dir)
    split = build_validation_split(data)
    print("RF points in the shared validation split:", split["n_val"])

    df_correct_phmm, df_incorrect_phmm, df_shouldbe_phmm = phmm_reference_splits(data, split)
    print("PHMM correct / incorrect (chosen) / should-be:",
          len(df_correct_phmm), len(df_incorrect_phmm), len(df_shouldbe_phmm))

    df_correct_nn, df_incorrect_nn, df_shouldbe_nn = nn_time_weighted_splits(data, split)
    print("NN Time-weighted correct / incorrect (chosen) / should-be:",
          len(df_correct_nn), len(df_incorrect_nn), len(df_shouldbe_nn))

    df_correct_lo, df_incorrect_lo, df_shouldbe_lo = logodds_reference_splits(data, split)
    print(f"{LOGODDS_NAME} correct / incorrect (chosen) / should-be:",
          len(df_correct_lo), len(df_incorrect_lo), len(df_shouldbe_lo))

    overlap = misclassification_overlap(df_incorrect_phmm, df_incorrect_nn)
    pd.Series(overlap).to_csv(table_dir / "misclassification_overlap.csv", header=["value"])

    df_correct_phmm, df_incorrect_phmm, df_shouldbe_phmm = enrich(
        df_correct_phmm, df_incorrect_phmm, df_shouldbe_phmm, data, add_trajectory=False
    )
    df_correct_nn, df_incorrect_nn, df_shouldbe_nn = enrich(
        df_correct_nn, df_incorrect_nn, df_shouldbe_nn, data, add_trajectory=True
    )
    # Matches phmm_reference_splits' own enrich() call: #points already
    # comes along via correct_forward_score's AIS_stats merge.
    df_correct_lo, df_incorrect_lo, df_shouldbe_lo = enrich(
        df_correct_lo, df_incorrect_lo, df_shouldbe_lo, data, add_trajectory=False
    )
    splits_phmm = (df_correct_phmm, df_incorrect_phmm, df_shouldbe_phmm)
    splits_nn = (df_correct_nn, df_incorrect_nn, df_shouldbe_nn)
    splits_lo = (df_correct_lo, df_incorrect_lo, df_shouldbe_lo)

    plot_data = {}
    summary_rows = []
    for slug, feature, ylabel, fmt, alternative, hypothesis, paired in EXPERIMENTS:
        print(f"\n=== Experiment {slug} ===")
        key = f"experiment_{slug.split('.')[0]}_{feature}_points"
        points = pd.concat([
            points_frame(*splits_phmm, feature).assign(method=PHMM_NAME),
            points_frame(*splits_nn, feature).assign(method=NN_NAME),
            points_frame(*splits_lo, feature).assign(method=LOGODDS_NAME),
        ], ignore_index=True)
        points.to_csv(table_dir / f"{key}.csv", index=False)
        plot_data[key] = points
        report_ttest(
            df_incorrect_phmm[feature], df_shouldbe_phmm[feature],
            df_incorrect_nn[feature], df_shouldbe_nn[feature],
            alternative=alternative, hypothesis=hypothesis, paired=paired,
        )
        summary_rows.append(effect_row(
            slug, feature,
            [("PHMM", df_incorrect_phmm, df_shouldbe_phmm),
             ("NN Time-weighted", df_incorrect_nn, df_shouldbe_nn),
             (LOGODDS_NAME, df_incorrect_lo, df_shouldbe_lo)],
            paired=paired,
        ))

    summary_table = pd.DataFrame(summary_rows).set_index("experiment")
    print("\n=== Summary: is PHMM's bias gone in NN Time-weighted / the log-odds score? ===")
    print(summary_table)
    summary_table.to_csv(table_dir / "bias_experiments_summary.csv")

    generate_plots.phase4c_bias_experiments(plot_data, fig_dir, EXPERIMENTS, [PHMM_NAME, LOGODDS_NAME, NN_NAME])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PHMM vs. NN Time-weighted misclassification-bias experiments"
    )
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="gaussian",
        help="RF bearing-error model whose data folder to read from (default: gaussian)",
    )
    args = parser.parse_args()

    DATA_DIR = f"./data/processed/{args.error_model}"
    FIG_DIR = Path(f"reports/figures/phase4c_bias_experiments/{args.error_model}")
    TABLE_DIR = Path(f"reports/tables/phase4c_bias_experiments/{args.error_model}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    main(DATA_DIR, FIG_DIR, TABLE_DIR)
