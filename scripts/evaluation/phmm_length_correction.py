"""
PHMM Forward score correction for AIS track-length bias

The PHMM Forward score is a product of per-step probabilities, so
longer AIS tracks accumulate more terms and their raw forward_score is
not directly comparable to a shorter track's. correct_forward_score
divides by #points ** alpha to counteract this before picking each RF
signal's best alignment. Used by 04_evaluation.ipynb (to tune and
apply the correction) and 07_baseline_bias_experiments.ipynb (to
reproduce the same corrected PHMM reference on the shared validation
split, for comparison against the NN Time-weighted baseline).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.evaluation.metrics import get_best_alignments, calculate_metrics

DEFAULT_ALPHAS = np.arange(0.7, 1.3, 0.01)


def tune_alpha(
    df_forward_results_multimatch: pd.DataFrame,
    df_preselection_multimatch: pd.DataFrame,
    df_AIS_stats: pd.DataFrame,
    alphas=DEFAULT_ALPHAS,
    test_size: float = 0.2,
    random_state: int = 100,
    score_col: str = "forward_score",
    exp_col: str = "normalized_forward_score_exp",
) -> tuple[float, dict, pd.DataFrame, pd.DataFrame]:
    """
    Split multimatch RF pairs into a tuning/experiments split and search
    alpha on the tuning split only

    Same split used throughout run_phase4_evaluation.py,
    run_phase4b_baseline_comparison.py, and
    run_phase4c_bias_experiments.py, so every PHMM-corrected result
    they report is tuned and evaluated identically: alpha is chosen on
    the 20% tuning split, then applied out-of-sample to the 80%
    experiments split -- reporting corrected-score metrics on the
    tuning split itself would be optimistic, since alpha was chosen to
    maximize precision there.

    score_col/exp_col let this same tuning procedure be reused for a
    different score column (e.g. "log_odds_score") without duplicating
    it -- the train_test_split itself doesn't depend on score values,
    so calling this with a different score_col reproduces the
    identical split (same seed, same pairs).

    Args:
        df_forward_results_multimatch: Preselection-multimatch merged
            with raw forward scores (one row per RF point, AIS
            candidate pair, restricted to multimatch RF signals)
        df_preselection_multimatch: Preselected AIS-RF candidate pairs,
            restricted to RF signals with multiple AIS candidates
        df_AIS_stats: Statistics per AIS track (e.g. #points)
        alphas: Candidate alpha values to search
        test_size, random_state: train_test_split() parameters for the
            tuning/experiments split
        score_col: Raw score column to correct (default "forward_score")
        exp_col: Name for the length-corrected output column

    Returns:
        (best_alpha, alpha -> tuning-set precision, df_tuning, df_experiments)
    """
    unique_rf_pairs = df_forward_results_multimatch[["RF_signal_id", "RF_track_id"]].drop_duplicates()
    train_pairs, val_pairs = train_test_split(unique_rf_pairs, test_size=test_size, random_state=random_state)
    df_tuning = df_forward_results_multimatch.merge(val_pairs, on=["RF_signal_id", "RF_track_id"])
    df_experiments = df_forward_results_multimatch.merge(train_pairs, on=["RF_signal_id", "RF_track_id"])

    alpha_results = {}
    for alpha in alphas:
        alpha = round(float(alpha), 2)
        precision, _, _ = correct_forward_score(
            df_tuning, df_preselection_multimatch, df_AIS_stats, alpha,
            score_col=score_col, exp_col=exp_col,
        )
        alpha_results[alpha] = precision["precision"]
    best_alpha = sorted(alpha_results.items(), key=lambda x: x[1], reverse=True)[0][0]

    return best_alpha, alpha_results, df_tuning, df_experiments


def correct_forward_score(
    df_all_forward_results: pd.DataFrame,
    df_preselection_multimatch: pd.DataFrame,
    df_AIS_stats: pd.DataFrame,
    alpha: float,
    match_col: str = "is_true_match",
    score_col: str = "forward_score",
    exp_col: str = "normalized_forward_score_exp",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Apply a correction to forward scores based on AIS track statistics.

    Args:
        df_all_forward_results (pd.DataFrame): Forward alignment results.
        df_preselection_multimatch (pd.DataFrame): Preselected AIS-RF
            candidate pairs, restricted to RF signals with multiple
            AIS candidates.
        df_AIS_stats (pd.DataFrame): Statistics per AIS track (e.g., number of points).
        alpha (float): Exponent factor used for normalization.
        match_col (str): Column marking the true-match candidate, passed
            through to calculate_metrics (e.g. "is_true_match", or the
            AIS_RF_probability-based reference label "is_match_ref").
        score_col (str): Raw score column to correct (default
            "forward_score"; pass "log_odds_score" to correct the
            log-odds score with this same procedure).
        exp_col (str): Name for the length-corrected output column.

    Returns:
        tuple: (metrics, chosen alignments, all track stats)
    """
    forward_results = df_preselection_multimatch.merge(
        df_all_forward_results,
        on=['RF_track_id', 'RF_signal_id', 'AIS_track_id', 'is_true_match'],
        how='inner',
        suffixes=('', '_y')
    )

    all_tracks_stats = forward_results.merge(df_AIS_stats, left_on='AIS_track_id', right_on='ID', how='left')
    all_tracks_stats.drop(all_tracks_stats.filter(regex='_y$').columns, axis=1, inplace=True)

    all_tracks_stats[exp_col] = all_tracks_stats[score_col] / (all_tracks_stats['#points'] ** alpha)

    chosen = get_best_alignments(all_tracks_stats, exp_col)
    return calculate_metrics(chosen, match_col=match_col), chosen, all_tracks_stats
