"""
Shared evaluation metrics for AIS-RF candidate scoring methods

Used by both the pHMM evaluation notebook (04_evaluation.ipynb) and the
baseline comparison notebook (05_baseline_comparison.ipynb) so that
top-1 confusion-matrix metrics, ranking-quality metrics, and the
significance tests comparing methods are computed identically in both
places rather than duplicated inline.

All functions operate on a "candidate dataframe": one row per
(RF point, AIS candidate) pair, as produced by the preselection step
and merged with a scoring method's output (e.g. normalized_forward_score
or nn_distance). An RF point is identified by group_cols, which
defaults to ("RF_signal_id", "RF_track_id").
"""
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

DEFAULT_GROUP_COLS = ("RF_signal_id", "RF_track_id")

CANDIDATE_COUNT_BINS = [0, 1, 4, 9, np.inf]
CANDIDATE_COUNT_LABELS = ["1", "2-4", "5-9", "10+"]


# === Top-1 / confusion-matrix metrics ===

def get_best_alignments(
    df: pd.DataFrame,
    score_col: str,
    mode: str = "max",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> pd.DataFrame:
    """
    Select the best-scoring candidate for each RF point

    Args:
        df: Candidate rows, one row per (RF point, AIS candidate) pair
        score_col: Column holding each candidate's score
        mode: "max" if a higher score is better (e.g. forward score),
            "min" if a lower score is better (e.g. nearest-neighbor
            distance)
        group_cols: Columns identifying a single RF point

    Returns:
        Copy of df with an added boolean "chosen" column, True for the
        top-ranked candidate of each RF point
    """
    group_cols = list(group_cols)
    scores_by_group = df.groupby(group_cols)[score_col]
    idx = scores_by_group.idxmax() if mode == "max" else scores_by_group.idxmin()
    idx = idx.reset_index(drop=True)

    df = df.copy()
    df["chosen"] = False
    df.loc[idx, "chosen"] = True
    return df


def deterministic_best_alignments(
    df: pd.DataFrame,
    score_col: str,
    mode: str = "max",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    tie_break_col: str = "AIS_track_id",
) -> tuple[pd.DataFrame, int]:
    """
    get_best_alignments(), with an explicit, deterministic tie-break
    and a count of how many RF points were actually tied

    get_best_alignments()'s groupby(...).idxmax()/idxmin() already
    resolves ties to the first row of the group in the frame's current
    order; sorting by tie_break_col first (ascending) before delegating
    to it, unmodified, makes that deterministically "lowest candidate
    index" with no change to get_best_alignments() or its other callers.

    Args:
        df, score_col, mode, group_cols: see get_best_alignments()
        tie_break_col: Column whose lowest value wins a tie

    Returns:
        (result of get_best_alignments() on the sorted frame, number of
        RF points -- groups -- where more than one candidate attained
        the winning score)
    """
    group_cols = list(group_cols)
    sorted_df = df.sort_values(group_cols + [tie_break_col]).reset_index(drop=True)

    extreme = sorted_df.groupby(group_cols)[score_col].transform("max" if mode == "max" else "min")
    tie_counts = sorted_df[sorted_df[score_col] == extreme].groupby(group_cols).size()
    n_tied = int((tie_counts > 1).sum())

    return get_best_alignments(sorted_df, score_col, mode, group_cols), n_tied


def compute_confusion_counts(
    df_with_chosen: pd.DataFrame,
    match_col: str = "is_true_match",
    chosen_col: str = "chosen",
) -> tuple[int, int, int, int]:
    """
    Compute confusion matrix counts: TP, FP, FN, TN

    Args:
        df_with_chosen: Candidate rows with a boolean chosen_col (e.g.
            from get_best_alignments) and a boolean match_col
        match_col: Column marking the true-match candidate
        chosen_col: Column marking the candidate a method selected

    Returns:
        (TP, FP, FN, TN) counts
    """
    is_match = df_with_chosen[match_col]
    is_chosen = df_with_chosen[chosen_col]
    TP = int((is_match & is_chosen).sum())
    FP = int((~is_match & is_chosen).sum())
    FN = int((is_match & ~is_chosen).sum())
    TN = int((~is_match & ~is_chosen).sum())
    return TP, FP, FN, TN


def calculate_metrics(
    df_with_chosen: pd.DataFrame,
    match_col: str = "is_true_match",
    chosen_col: str = "chosen",
) -> dict:
    """
    Compute precision, recall, F1 score, accuracy and confusion counts

    Args:
        df_with_chosen: Candidate rows with a boolean chosen_col and a
            boolean match_col
        match_col: Column marking the true-match candidate
        chosen_col: Column marking the candidate a method selected

    Returns:
        Dict with precision, recall, f1_score, accuracy, TP, FP, FN, TN
    """
    TP, FP, FN, TN = compute_confusion_counts(
        df_with_chosen, match_col=match_col, chosen_col=chosen_col
    )
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
    }


# === Ranking-quality metrics ===

def summarize_ranking(
    df: pd.DataFrame,
    score_col: str,
    mode: str = "max",
    ks: Sequence[int] = (1, 3, 5),
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    match_col: str = "is_true_match",
) -> pd.DataFrame:
    """
    Reduce a per-candidate scored dataframe to one row per RF point

    This is the shared building block for recall@k, MRR, score-margin,
    and candidate-set-size stratification: rank candidates within each
    RF point by score, locate the true match's rank (if present), and
    derive the top-1/runner-up score margin, so downstream metric
    functions only need to aggregate over this per-point summary
    instead of re-ranking candidates themselves.

    Args:
        df: Candidate rows, one row per (RF point, AIS candidate) pair
        score_col: Column holding each candidate's score
        mode: "max" if a higher score is better, "min" if a lower
            score is better
        ks: Recall@k cutoffs to compute hit indicators for
        group_cols: Columns identifying a single RF point
        match_col: Boolean column marking the true-match candidate.
            RF points with no True row (e.g. open-set/leave-one-out
            evaluation) get rank_of_true = NaN, reciprocal_rank = 0,
            and hit@k = False for all k

    Returns:
        DataFrame with one row per RF point (group_cols as columns)
        and: candidate_count, rank_of_true, reciprocal_rank, margin,
        and a "hit@<k>" boolean column for each k in ks
    """
    group_cols = list(group_cols)
    ascending = mode == "min"

    working = df[group_cols + [score_col, match_col]].reset_index(drop=True)
    working = working.copy()
    working["rank"] = (
        working.groupby(group_cols)[score_col]
        .rank(method="first", ascending=ascending)
        .astype(int)
    )

    candidate_count = working.groupby(group_cols).size()
    candidate_count = candidate_count.rename("candidate_count")

    true_ranks = (
        working[working[match_col]]
        .drop_duplicates(group_cols, keep="first")
        .set_index(group_cols)["rank"]
        .rename("rank_of_true")
    )

    # Top-1 vs runner-up score margin, oriented so a larger value always
    # means a more confident top-1 pick regardless of mode. Also keep the
    # raw top-1 score itself (whichever candidate ranked best), used e.g.
    # for open-set accept/reject thresholding.
    top2 = working[working["rank"] <= 2]
    pivoted = top2.pivot_table(index=group_cols, columns="rank", values=score_col)
    top1_score = pivoted[1].rename("top1_score")
    if 2 in pivoted.columns:
        margin = pivoted[1] - pivoted[2] if mode == "max" else pivoted[2] - pivoted[1]
    else:
        margin = pd.Series(np.nan, index=pivoted.index)
    margin = margin.rename("margin")

    summary = candidate_count.to_frame().join([true_ranks, top1_score, margin])
    summary["reciprocal_rank"] = np.where(
        summary["rank_of_true"].notna(), 1.0 / summary["rank_of_true"], 0.0
    )
    for k in ks:
        summary[f"hit@{k}"] = (
            summary["rank_of_true"].notna() & (summary["rank_of_true"] <= k)
        )

    return summary.reset_index()


def recall_at_k(
    summary: pd.DataFrame, ks: Sequence[int] = (1, 3, 5)
) -> dict:
    """
    Aggregate recall@k over a summarize_ranking() output

    Args:
        summary: Per-RF-point ranking summary from summarize_ranking()
        ks: Cutoffs to report recall@k for; each needs a matching
            "hit@<k>" column in summary

    Returns:
        Dict mapping k to the fraction of RF points whose true match
        ranked within the top k candidates
    """
    return {k: float(summary[f"hit@{k}"].mean()) for k in ks}


def mean_reciprocal_rank(summary: pd.DataFrame) -> float:
    """
    Mean reciprocal rank over a summarize_ranking() output

    Args:
        summary: Per-RF-point ranking summary from summarize_ranking()

    Returns:
        Mean of 1 / rank_of_true across all RF points (0 for points
        where the true match is absent from the candidate set)
    """
    return float(summary["reciprocal_rank"].mean())


def add_candidate_count_bucket(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket RF points by how many candidates survived prefiltering

    Args:
        summary: Per-RF-point ranking summary from summarize_ranking(),
            must contain a "candidate_count" column

    Returns:
        Copy of summary with an added "candidate_count_bucket" column
        (categories: "1", "2-4", "5-9", "10+")
    """
    summary = summary.copy()
    summary["candidate_count_bucket"] = pd.cut(
        summary["candidate_count"],
        bins=CANDIDATE_COUNT_BINS,
        labels=CANDIDATE_COUNT_LABELS,
    )
    return summary


def stratified_ranking_metrics(
    summary: pd.DataFrame, ks: Sequence[int] = (1, 3, 5)
) -> pd.DataFrame:
    """
    Recall@k and MRR overall and per candidate-set-size bucket

    Args:
        summary: Per-RF-point ranking summary from summarize_ranking()
        ks: Recall@k cutoffs to report

    Returns:
        DataFrame with one row for "all" plus one row per
        candidate-count bucket, columns: bucket, n_points,
        recall@<k> (for each k), mrr
    """
    summary = add_candidate_count_bucket(summary)

    def _row(label: str, subset: pd.DataFrame) -> dict:
        row = {"bucket": label, "n_points": len(subset)}
        row.update(
            {f"recall@{k}": subset[f"hit@{k}"].mean() for k in ks}
        )
        row["mrr"] = subset["reciprocal_rank"].mean()
        return row

    rows = [_row("all", summary)]
    for label in CANDIDATE_COUNT_LABELS:
        subset = summary[summary["candidate_count_bucket"] == label]
        if len(subset) > 0:
            rows.append(_row(label, subset))

    return pd.DataFrame(rows)


# === Statistical comparison between methods ===

def mcnemar_test(correct_a, correct_b) -> dict:
    """
    Paired McNemar's test on binary top-1 correctness between methods

    Compares two scoring methods evaluated on the same RF points,
    using each point's correct/incorrect top-1 outcome. Uses the exact
    binomial form for small numbers of discordant pairs (< 25, as
    commonly recommended, e.g. by statsmodels' mcnemar implementation)
    and the continuity-corrected chi-square form otherwise.

    Args:
        correct_a: Boolean array-like, per-point top-1 correctness for
            method A, aligned by RF point with correct_b
        correct_b: Boolean array-like, per-point top-1 correctness for
            method B

    Returns:
        Dict with n10 (A correct, B wrong), n01 (A wrong, B correct),
        statistic, p_value, and method ("exact" or "chi2")
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)

    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n_discordant = n10 + n01

    if n_discordant == 0:
        return {
            "n10": n10, "n01": n01,
            "statistic": 0.0, "p_value": 1.0, "method": "exact",
        }

    if n_discordant < 25:
        result = binomtest(min(n10, n01), n_discordant, 0.5)
        return {
            "n10": n10, "n01": n01,
            "statistic": float(min(n10, n01)),
            "p_value": float(result.pvalue),
            "method": "exact",
        }

    statistic = (abs(n10 - n01) - 1) ** 2 / n_discordant
    p_value = float(1 - chi2.cdf(statistic, df=1))
    return {
        "n10": n10, "n01": n01,
        "statistic": float(statistic), "p_value": p_value, "method": "chi2",
    }


def bootstrap_ci(
    summary: pd.DataFrame,
    value_col: str,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: Optional[int] = None,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean of a per-RF-point column

    Resamples RF points (rows of summary) with replacement n_boot
    times and takes the empirical percentile interval of the
    resampled means. Used for recall@k (value_col="hit@<k>") and MRR
    (value_col="reciprocal_rank").

    Args:
        summary: Per-RF-point ranking summary from summarize_ranking()
        value_col: Column to average, e.g. "hit@1" or "reciprocal_rank"
        n_boot: Number of bootstrap resamples
        ci: Confidence level, e.g. 0.95 for a 95% interval
        random_state: Seed for reproducibility

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    values = summary[value_col].to_numpy(dtype=float)
    n = len(values)
    point_estimate = float(values.mean())

    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[i] = values[sample_idx].mean()

    alpha = (1 - ci) / 2
    lower, upper = np.quantile(boot_means, [alpha, 1 - alpha])
    return point_estimate, float(lower), float(upper)
