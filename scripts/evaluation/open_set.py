"""
Shared building blocks for open-set (accept/reject) evaluation

Used by run_phase4c_open_set_evaluation.py and
06_open_set_evaluation.ipynb.

Every accept/reject frame built here has a "rejection_type" column:
- "positive": the true track is a candidate (genuine match)
- "automatic_rejection": no candidate survived the prefilter, so the
  RF point is rejected before any scoring model runs
- "scored_rejection": 1+ candidates passed the prefilter and the
  scoring model had to judge none of them confident enough
"""
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from scripts.evaluation.metrics import summarize_ranking

KEY_COLS = ["RF_track_id", "RF_signal_id", "AIS_track_id", "is_true_match"]

# (display name, closed-set results file, leave-one-out results file, score column, "max"/"min")
MODELS = [
    ("PHMM Forward",
     "AIS_RF_forward_scores_data.pkl", "AIS_RF_forward_scores_leaveoneout_data.pkl",
     "normalized_forward_score", "max"),
    ("NN Euclidean",
     "AIS_RF_nn_baseline_euclidean_scores_data.pkl", "AIS_RF_nn_baseline_euclidean_scores_leaveoneout_data.pkl",
     "nn_distance", "min"),
    ("NN Haversine",
     "AIS_RF_nn_baseline_haversine_scores_data.pkl", "AIS_RF_nn_baseline_haversine_scores_leaveoneout_data.pkl",
     "nn_distance", "min"),
    ("NN Segment",
     "AIS_RF_nn_baseline_segment_scores_data.pkl", "AIS_RF_nn_baseline_segment_scores_leaveoneout_data.pkl",
     "nn_distance", "min"),
    ("NN Time-weighted",
     "AIS_RF_nn_baseline_time_weighted_scores_data.pkl", "AIS_RF_nn_baseline_time_weighted_scores_leaveoneout_data.pkl",
     "nn_distance", "min"),
]


LOG_ODDS_MODELS = [
    ("Log-odds PHMM (raw)",
     "AIS_RF_forward_scores_data.pkl", "AIS_RF_forward_scores_leaveoneout_data.pkl",
     "log_odds_score", "max"),
    ("Log-odds PHMM (n^alpha)",
     "AIS_RF_forward_scores_data.pkl", "AIS_RF_forward_scores_leaveoneout_data.pkl",
     "log_odds_score_corrected", "max"),
]


def apply_alpha_correction(
    df: pd.DataFrame, df_AIS_stats: pd.DataFrame, score_col: str, alpha: float, output_col: str,
) -> pd.DataFrame:
    """
    Add a length-corrected score column (score_col / #points ** alpha)
    to every row of df

    Unlike phmm_length_correction.correct_forward_score(), this has no
    train/tuning split -- open-set evaluation reuses an alpha already
    chosen elsewhere (on the closed-set data), it doesn't re-tune one,
    and applies it uniformly to every row (single- and multi-candidate
    alike; for a single-candidate RF signal this rescales its one score
    but can't change which candidate "wins").
    """
    df = df.merge(df_AIS_stats[["ID", "#points"]], left_on="AIS_track_id", right_on="ID", how="left")
    df[output_col] = df[score_col] / (df["#points"] ** alpha)
    return df.drop(columns=["ID", "#points"])


def load_closed_results(data_dir: str, models: Sequence = MODELS) -> dict[str, pd.DataFrame]:
    """Load each model's closed-set scored candidates"""
    return {name: pd.read_pickle(f"{data_dir}/{closed_file}") for name, closed_file, _, _, _ in models}


def load_loo_results(data_dir: str, models: Sequence = MODELS) -> dict[str, pd.DataFrame]:
    """Load each model's leave-one-out scored candidates"""
    return {name: pd.read_pickle(f"{data_dir}/{loo_file}") for name, _, loo_file, _, _ in models}


def _fill_inf_with_sentinel(series: pd.Series) -> pd.Series:
    """
    Replace +-inf placeholders with finite values just beyond the real
    range of the column, so downstream sklearn ROC/PR functions (which
    reject inf/NaN) still treat them as more extreme than every genuine
    score, without changing the ROC/PR/AUC result
    """
    finite = series[np.isfinite(series)]
    return series.replace({
        -np.inf: finite.min() - 1 if len(finite) else -1.0,
        np.inf: finite.max() + 1 if len(finite) else 1.0,
    })


def _automatic_rejection_rows(
    universe: pd.DataFrame, scored_rejections: pd.DataFrame, group_cols=("RF_signal_id", "RF_track_id"),
):
    """RF points in universe that never appear in scored_rejections -- zero surviving candidates"""
    group_cols = list(group_cols)
    scored_points = scored_rejections[group_cols]
    missing = universe.merge(scored_points, on=group_cols, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"][group_cols].copy()
    missing["candidate_count"] = 0
    missing["top1_score"] = np.nan
    missing["margin"] = np.nan
    return missing


def _finalize_accept_frame(
    pos_summary: pd.DataFrame, scored_rejections: pd.DataFrame, automatic_rejections: pd.DataFrame, mode: str,
) -> pd.DataFrame:
    """
    Assemble accept_score from a positive frame and a (scored +
    automatic rejection) negative frame -- shared by
    build_open_set_frame and build_dark_vessel_frame

    accept_score orients each method's top-1 score so higher always
    means more likely a genuine match (raw score for max-is-better
    methods, negated distance for min-is-better ones).
    """
    combined = pd.concat([pos_summary, scored_rejections, automatic_rejections], ignore_index=True)

    sign = 1 if mode == "max" else -1
    combined["accept_score"] = np.where(
        combined["top1_score"].notna(), sign * combined["top1_score"], -np.inf
    )
    # ROC/PR helpers (sklearn) reject +-inf; fold the placeholder into a
    # finite sentinel just beyond the column's real range instead
    combined["accept_score"] = _fill_inf_with_sentinel(combined["accept_score"])
    return combined


def build_open_set_frame(
    model_name: str,
    df_preselection: pd.DataFrame,
    df_preselection_loo: pd.DataFrame,
    closed_results: dict[str, pd.DataFrame],
    loo_results: dict[str, pd.DataFrame],
    score_col: str,
    mode: str,
) -> pd.DataFrame:
    """
    Combined positive (closed-set) / negative (leave-one-out) frame for
    one scoring method's accept/reject analysis, one row per RF point

    Positives: closed-set RF points, true track present as a candidate.
    Negatives: leave-one-out RF points -- zero surviving candidates is
    an automatic rejection, 1+ candidates needs the scoring model to
    judge none of them confident enough.

    This negative class only hides a vessel's own track from its own
    query; every other vessel stays searchable. See
    build_dark_vessel_frame for a negative class that is never a
    candidate for anyone.

    Returns:
        pd.DataFrame: one row per RF point, columns candidate_count,
        top1_score, margin, label (1 = should-accept, 0 = should-reject),
        rejection_type ("positive", "automatic_rejection",
        "scored_rejection"), accept_score
    """
    rf_universe = df_preselection[["RF_signal_id", "RF_track_id"]].drop_duplicates()

    closed_merged = df_preselection.merge(closed_results[model_name], on=KEY_COLS, how="inner")
    pos_summary = summarize_ranking(closed_merged, score_col, mode=mode)
    pos_summary["label"] = 1
    pos_summary["rejection_type"] = "positive"

    loo_merged = df_preselection_loo.merge(loo_results[model_name], on=KEY_COLS, how="inner")
    scored_rejections = summarize_ranking(loo_merged, score_col, mode=mode)
    scored_rejections["label"] = 0
    scored_rejections["rejection_type"] = "scored_rejection"

    automatic_rejections = _automatic_rejection_rows(rf_universe, scored_rejections)
    automatic_rejections["label"] = 0
    automatic_rejections["rejection_type"] = "automatic_rejection"

    return _finalize_accept_frame(pos_summary, scored_rejections, automatic_rejections, mode)


def split_registry_dark_vessels(
    vessel_ids: Sequence, dark_fraction: float = 0.2, random_state: int = 100,
) -> tuple[set, set]:
    """
    Partition the full vessel universe into a registry pool (the only
    vessels ever offered as a search candidate) and a dark-vessel pool
    (queried, but never searchable by anyone), for a held-out open-set
    negative

    Leave-one-out only excludes a query's own track, leaving every
    other vessel in the candidate pool. This fixes the split once over
    the whole universe, so a dark vessel's RF signal is scored only
    against vessels in a registry that has never seen it.

    Args:
        vessel_ids: The full universe of AIS track IDs
        dark_fraction: Share of vessels held out as "dark"
        random_state: Seed, for a reproducible split

    Returns:
        (registry_ids, dark_ids), disjoint sets partitioning vessel_ids
    """
    # Vessel IDs are (MMSI, track_id) tuples here, not scalars -- shuffle
    # index positions rather than the values themselves, so np.random
    # never has to treat a tuple as an array-like and flatten it
    ids = sorted(set(vessel_ids))
    rng = np.random.default_rng(random_state)
    shuffled_idx = rng.permutation(len(ids))
    n_dark = int(round(len(ids) * dark_fraction))
    dark_ids = {ids[i] for i in shuffled_idx[:n_dark]}
    registry_ids = {ids[i] for i in shuffled_idx[n_dark:]}
    return registry_ids, dark_ids


def build_dark_vessel_frame(
    model_name: str,
    df_preselection: pd.DataFrame,
    closed_results: dict[str, pd.DataFrame],
    registry_ids: set,
    dark_ids: set,
    score_col: str,
    mode: str,
) -> pd.DataFrame:
    """
    Registry-only positive (closed-set) / dark-vessel negative
    accept/reject frame for one scoring method, built by filtering the
    existing closed-set data -- no new alignment/scoring run needed

    The prefilter and every scoring method score one RF signal against
    one candidate track at a time, independent of the rest of the
    pool. So restricting the closed-set candidates to registry_ids
    reproduces exactly what a fresh, registry-only run would produce.

    Positives: RF signals whose true track is in the registry, scored
    only against registry candidates. Negatives: RF signals whose true
    track is a dark vessel, scored only against the registry -- no
    candidate is ever correct for these by construction. Split into
    automatic vs. scored rejections exactly as in build_open_set_frame.

    Returns:
        pd.DataFrame, same shape as build_open_set_frame's output
    """
    merged = df_preselection.merge(closed_results[model_name], on=KEY_COLS, how="inner")
    registry_candidates = merged[merged["AIS_track_id"].isin(registry_ids)]

    positive_rows = registry_candidates[registry_candidates["RF_track_id"].isin(registry_ids)]
    pos_summary = summarize_ranking(positive_rows, score_col, mode=mode)
    pos_summary["label"] = 1
    pos_summary["rejection_type"] = "positive"

    dark_rows = registry_candidates[registry_candidates["RF_track_id"].isin(dark_ids)]
    scored_rejections = summarize_ranking(dark_rows, score_col, mode=mode)
    scored_rejections["label"] = 0
    scored_rejections["rejection_type"] = "scored_rejection"

    dark_universe = df_preselection.loc[
        df_preselection["RF_track_id"].isin(dark_ids), ["RF_signal_id", "RF_track_id"]
    ].drop_duplicates()
    automatic_rejections = _automatic_rejection_rows(dark_universe, scored_rejections)
    automatic_rejections["label"] = 0
    automatic_rejections["rejection_type"] = "automatic_rejection"

    return _finalize_accept_frame(pos_summary, scored_rejections, automatic_rejections, mode)


def compute_roc(frame: pd.DataFrame, score_col: str, label_col: str = "label") -> dict:
    """ROC curve + AUC for one model's accept/reject frame, as a plots.roc_* input"""
    fpr, tpr, _ = roc_curve(frame[label_col], frame[score_col])
    return {"fpr": fpr, "tpr": tpr, "roc_auc": auc(fpr, tpr)}


def compute_pr(frame: pd.DataFrame, score_col: str, label_col: str = "label") -> dict:
    """Precision-recall curve + AP for one model's accept/reject frame"""
    precision, recall, _ = precision_recall_curve(frame[label_col], frame[score_col])
    ap = average_precision_score(frame[label_col], frame[score_col])
    return {"precision": precision, "recall": recall, "ap": ap}
