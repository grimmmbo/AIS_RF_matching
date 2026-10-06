"""
Shared building blocks for misclassification-bias experiments

Used by 04_evaluation.ipynb (PHMM Forward) and
07_baseline_bias_experiments.ipynb (NN Time-weighted baseline) so that
the correct/incorrect/should-be split and the trajectory/point-level
feature computations used to test for misclassification biases are
defined once instead of duplicated per notebook.

All functions operate on the same "candidate dataframe" convention as
scripts.evaluation.metrics: one row per (RF point, AIS candidate)
pair, an RF point identified by group_cols.
"""
from typing import Sequence

import pandas as pd
from haversine import haversine

from scripts.evaluation.metrics import DEFAULT_GROUP_COLS


def split_correct_incorrect_shouldbe(
    df_candidates: pd.DataFrame,
    df_chosen: pd.DataFrame,
    match_col: str = "is_true_match",
    chosen_col: str = "chosen",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split top-1 picks into correct, incorrect (chosen), and should-be matches

    Mirrors the "Split Forward results into three dataframes" step of
    04_evaluation.ipynb, generalized to any scoring method's chosen
    alignments: (1) correct matches, where the top-1 pick is the true
    match; (2) incorrect (chosen) matches, where the top-1 pick is
    wrong; and (3) should-be matches, the true-match candidate for
    each RF point in (2) (i.e. the correct candidate the method should
    have picked instead).

    Args:
        df_candidates: Full candidate rows (one per RF point, AIS
            candidate pair), used to look up each should-be candidate
        df_chosen: Candidate rows with a boolean chosen_col, e.g. from
            metrics.get_best_alignments
        match_col: Column marking the true-match candidate
        chosen_col: Column marking the candidate a method selected
        group_cols: Columns identifying a single RF point

    Returns:
        (df_correct, df_incorrect, df_shouldbe)
    """
    group_cols = list(group_cols)
    best = df_chosen[df_chosen[chosen_col]]
    df_correct = best[best[match_col]]
    df_incorrect = best[~best[match_col]]
    df_shouldbe = (
        df_candidates[df_candidates[match_col]]
        .merge(df_incorrect[group_cols], on=group_cols, how="inner")
    )
    return df_correct, df_incorrect, df_shouldbe


POINTS_COLS = ["RF_signal_id", "RF_track_id", "group", "value"]


def points_frame(
    df_correct: pd.DataFrame, df_incorrect: pd.DataFrame, df_shouldbe: pd.DataFrame,
    feature: str, group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> pd.DataFrame:
    """
    Long-format (RF_signal_id, RF_track_id, group, value) table of one
    feature across the three splits -- saved as a plain CSV result
    file so the boxplot/histogram figures built from it can be redrawn
    later without recomputing the split (see
    scripts/evaluation/regenerate_plots.py)
    """
    group_cols = list(group_cols)
    frames = []
    for group, df in [("correct", df_correct), ("incorrect", df_incorrect), ("shouldbe", df_shouldbe)]:
        frame = df[group_cols].copy()
        frame["group"] = group
        frame["value"] = df[feature].to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[group_cols + ["group", "value"]]


def points_by_group(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inverse of points_frame: (df_correct, df_incorrect, df_shouldbe), each with a "value" column"""
    return tuple(
        points[points["group"] == group].drop(columns="group").reset_index(drop=True)
        for group in ("correct", "incorrect", "shouldbe")
    )


def points_diff(
    df_incorrect: pd.DataFrame, df_shouldbe: pd.DataFrame, group_cols: Sequence[str] = DEFAULT_GROUP_COLS
) -> pd.Series:
    """(chosen - should-be) difference of a "value" column, aligned by RF point"""
    merged = df_incorrect.merge(df_shouldbe, on=list(group_cols), suffixes=("_chosen", "_shouldbe"), how="inner")
    return merged["value_chosen"] - merged["value_shouldbe"]


def add_trajectory_stats(df: pd.DataFrame, df_AIS_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Attach per-AIS-track trajectory statistics to a candidate/chosen dataframe

    Args:
        df: Candidate rows with an AIS_track_id column
        df_AIS_stats: Per-AIS-track statistics (e.g. #points,
            mean_intra_time_diff, mean_intra_dist_diff), indexed by ID

    Returns:
        Copy of df merged with df_AIS_stats on AIS_track_id == ID,
        sorted by (RF_signal_id, RF_track_id)
    """
    return df.merge(
        df_AIS_stats, left_on="AIS_track_id", right_on="ID", how="left"
    ).sort_values(["RF_signal_id", "RF_track_id"]).reset_index(drop=True)


def compute_track_behavior(
    df: pd.DataFrame, df_AIS: pd.DataFrame, stationary_threshold: float = 5
) -> pd.DataFrame:
    """
    Adds per-track behavior statistics to the given dataframe

    Args:
        df: Candidate rows with an AIS_track_id column
        df_AIS: Raw AIS position dataframe with ID and speed_kph columns
        stationary_threshold: Speed (km/h) below which a position is
            considered stationary

    Returns:
        Copy of df merged with each AIS track's stationary_ratio (the
        fraction of its positions below stationary_threshold)
    """
    df_AIS = df_AIS.copy()
    df_AIS["is_stationary"] = df_AIS["speed_kph"] < stationary_threshold
    track_behavior_stats = (
        df_AIS
        .groupby("ID")
        .agg(stationary_ratio=("is_stationary", "mean"))
        .reset_index()
        .set_index("ID")
    )
    return df.merge(track_behavior_stats, left_on="AIS_track_id", right_on="ID", how="left")


def compute_time_dist_diff(df_best: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    """
    For each best RF-AIS alignment, compute:
      - The closest AIS point in time
      - The time difference (in seconds)
      - The geographic distance difference (using haversine)

    Args:
        df_best: DataFrame with best RF-AIS alignments
        df_train: Original training dataset containing RF and AIS data

    Returns:
        pd.DataFrame: Merged DataFrame with extra columns for closest
        AIS timestamp, AIS coordinates, time difference, and distance
        difference
    """
    df_RF = df_train[df_train["RF"].notna()].copy()
    df_RF = df_RF[["ID", "RF_Timestamp", "RF"]]
    df_RF["RF_signal_id"] = df_RF.groupby("ID").cumcount() + 1

    df_merged = df_best.merge(
        df_RF.rename({"ID": "RF_track_id"}, axis=1),
        on=["RF_signal_id", "RF_track_id"],
        how="left"
    )

    df_AIS = df_train[df_train["AIS"].notna()].copy()
    df_AIS_grouped = df_AIS.groupby("ID")

    def compute_diffs(row):
        seq = df_AIS_grouped.get_group(row["AIS_track_id"])
        AIS_time, AIS_coords = min(
            zip(seq["AIS_Timestamp"], seq["AIS"]),
            key=lambda x: abs(x[0] - row["RF_Timestamp"]).total_seconds()
        )
        time_diff = abs((row["RF_Timestamp"] - AIS_time).total_seconds())
        distance_diff = haversine(row["RF"], AIS_coords)

        return pd.Series({
            "AIS_Timestamp": AIS_time,
            "AIS": AIS_coords,
            "time_diff": time_diff,
            "distance_diff": distance_diff
        })

    cols = ["AIS_Timestamp", "AIS", "time_diff", "distance_diff"]
    df_merged[cols] = df_merged.apply(compute_diffs, axis=1)

    return df_merged
