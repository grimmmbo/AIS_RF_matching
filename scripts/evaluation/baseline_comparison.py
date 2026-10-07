"""
Shared building blocks for the PHMM Forward vs. Nearest-Neighbor
baseline comparison

Used by run_phase4b_baseline_comparison.py and, for interactive
re-use, 05_baseline_comparison.ipynb.
"""
import pickle
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.evaluation.metrics import (
    DEFAULT_GROUP_COLS,
    bootstrap_ci,
    calculate_metrics,
    get_best_alignments,
    mcnemar_test,
    stratified_ranking_metrics,
    summarize_ranking,
)
from scripts.evaluation.phmm_length_correction import correct_forward_score, tune_alpha

# (display name, results file, score column, "max"/"min", checkpoint file)
MODELS = [
    ("PHMM Forward", "AIS_RF_forward_scores_data.pkl",
     "normalized_forward_score", "max", "AIS_RF_forward_scores_checkpoint.pkl"),
    ("NN Euclidean", "AIS_RF_nn_baseline_euclidean_scores_data.pkl",
     "nn_distance", "min", "AIS_RF_nn_baseline_euclidean_scores_checkpoint.pkl"),
    ("NN Haversine", "AIS_RF_nn_baseline_haversine_scores_data.pkl",
     "nn_distance", "min", "AIS_RF_nn_baseline_haversine_scores_checkpoint.pkl"),
    ("NN Segment", "AIS_RF_nn_baseline_segment_scores_data.pkl",
     "nn_distance", "min", "AIS_RF_nn_baseline_segment_scores_checkpoint.pkl"),
    ("NN Time-weighted", "AIS_RF_nn_baseline_time_weighted_scores_data.pkl",
     "nn_distance", "min", "AIS_RF_nn_baseline_time_weighted_scores_checkpoint.pkl"),
]

# Log-odds PHMM variants (score_lo = log P_full - log P_null, see
# AIS_RF_forward_alignment.py / states.NullMState). Kept out of MODELS
# itself so every existing call site using MODELS' default stays
# provably unaffected; pass models=MODELS + LOG_ODDS_MODELS explicitly
# wherever the new rows are wanted. Both point at the same results/
# checkpoint files as "PHMM Forward" -- the new columns ride along in
# the same per-(RF, candidate) cache.
LOG_ODDS_MODELS = [
    ("Log-odds PHMM (raw)", "AIS_RF_forward_scores_data.pkl",
     "log_odds_score", "max", "AIS_RF_forward_scores_checkpoint.pkl"),
    ("Log-odds PHMM (n^alpha)", "AIS_RF_forward_scores_data.pkl",
     "log_odds_score_corrected", "max", "AIS_RF_forward_scores_checkpoint.pkl"),
]


def load_model_results(data_dir: str, models: Sequence = MODELS) -> dict[str, pd.DataFrame]:
    """Load each model's scored-candidates pickle from data_dir"""
    return {name: pd.read_pickle(f"{data_dir}/{results_file}") for name, results_file, _, _, _ in models}


def random_choice_metrics(
    preselection_subset: pd.DataFrame,
    n_boot: int = 200,
    random_state: int = 42,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> dict:
    """
    Expected precision/recall/F1/accuracy of picking a candidate
    uniformly at random for each RF signal

    The floor "does matching at all beat chance" baseline for the
    comparison: for a given candidate set, a uniform random guess's
    closed-form expected top-1 accuracy is 1/candidate_count. Simulated
    (not just averaged in closed form) over n_boot repetitions so the
    result has the same TP/FP/FN/TN shape as calculate_metrics()'s
    output for every other model, and so a std is available.

    Args:
        preselection_subset: Candidate rows (one per RF point, AIS
            candidate pair), e.g. df_preselection or its multimatch subset
        n_boot: Number of random-draw repetitions to average over
        random_state: Seed, for a reproducible result
        group_cols: Columns identifying a single RF point

    Returns:
        Dict shaped like calculate_metrics()'s output (mean over
        n_boot repetitions), plus precision_std
    """
    group_cols = list(group_cols)
    rng = np.random.default_rng(random_state)
    df = preselection_subset[group_cols + ["is_true_match"]]

    metric_rows = []
    for _ in range(n_boot):
        seed = int(rng.integers(0, 2**31 - 1))
        chosen = df.groupby(group_cols, group_keys=False).sample(n=1, random_state=seed)
        df_chosen = df.copy()
        df_chosen["chosen"] = False
        df_chosen.loc[chosen.index, "chosen"] = True
        metric_rows.append(calculate_metrics(df_chosen))

    metrics_df = pd.DataFrame(metric_rows)
    means = metrics_df.mean()
    return {
        "precision": round(means["precision"], 4),
        "recall": round(means["recall"], 4),
        "f1_score": round(means["f1_score"], 4),
        "accuracy": round(means["accuracy"], 4),
        "precision_std": round(metrics_df["precision"].std(), 4),
        "TP": means["TP"], "FP": means["FP"], "FN": means["FN"], "TN": means["TN"],
    }


def build_corrected_phmm_results(
    df_preselection_multimatch: pd.DataFrame,
    df_forward_scores_raw: pd.DataFrame,
    df_AIS_stats: pd.DataFrame,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    score_col: str = "forward_score",
    exp_col: str = "normalized_forward_score_exp",
    raw_score_col: str = "normalized_forward_score",
    output_col: str = "normalized_forward_score",
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Length-corrected PHMM Forward results, restricted to alpha's
    held-out split

    Tunes alpha on a 20% held-out split of multimatch RF signals (same
    split as run_phase4_evaluation.py / run_phase4c_bias_experiments.py,
    via phmm_length_correction.tune_alpha), then applies it to the
    other 80% ("experiments" split) only -- alpha was chosen on the
    tuning split's own precision, so scoring corrected PHMM there too
    would be optimistic.

    correct_forward_score() only scores multimatch RF signals, so it
    would silently drop single-match signals that every other model
    keeps; those are added back from raw_score_col, which is already
    exact for a single candidate.

    score_col/exp_col/raw_score_col/output_col let this same procedure
    build a corrected results frame for a different score (e.g.
    score_col="log_odds_score") without duplicating it; the defaults
    reproduce the original PHMM Forward behavior exactly.

    Returns:
        (corrected results, with column output_col -- same shape as
        every other model's results DataFrame, a drop-in replacement
        for results["PHMM Forward"] when output_col is left at its
        default; best_alpha; df_preselection_multimatch restricted to
        the experiments split)
    """
    group_cols = list(group_cols)
    merge_cols = group_cols + ["AIS_track_id", "is_true_match"]

    df_forward_results_multimatch = df_preselection_multimatch.merge(
        df_forward_scores_raw, on=merge_cols, how="inner"
    )
    best_alpha, _, _, df_experiments = tune_alpha(
        df_forward_results_multimatch, df_preselection_multimatch, df_AIS_stats,
        score_col=score_col, exp_col=exp_col,
    )
    _, _, all_tracks_exp = correct_forward_score(
        df_experiments, df_preselection_multimatch, df_AIS_stats, best_alpha,
        score_col=score_col, exp_col=exp_col,
    )
    corrected_multimatch = all_tracks_exp[merge_cols + [exp_col]].rename(
        columns={exp_col: output_col}
    )

    is_singlematch = df_forward_scores_raw.merge(
        df_preselection_multimatch[group_cols].drop_duplicates(), on=group_cols, how="left", indicator=True
    )["_merge"] == "left_only"
    singlematch_results = df_forward_scores_raw.loc[is_singlematch, merge_cols + [raw_score_col]].rename(
        columns={raw_score_col: output_col}
    )

    corrected_results = pd.concat([corrected_multimatch, singlematch_results], ignore_index=True)
    df_preselection_multimatch_eval = df_preselection_multimatch.merge(
        df_experiments[group_cols].drop_duplicates(), on=group_cols, how="inner"
    )
    return corrected_results, best_alpha, df_preselection_multimatch_eval


def restrict_to_eval_population(
    df_preselection: pd.DataFrame,
    df_preselection_multimatch_eval: pd.DataFrame,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> pd.DataFrame:
    """
    All candidate pairs, with the alpha-tuning 20% of multimatch RF
    signals removed

    Single-match RF signals are unaffected by the length correction
    (only one candidate to choose, regardless of score) and kept in
    full; multimatch RF signals are restricted to the same experiments
    split used for the corrected PHMM results, so every model in the
    comparison is evaluated on one identical population.
    """
    group_cols = list(group_cols)
    singlematch_pairs = df_preselection.groupby(group_cols).filter(
        lambda x: len(x) == 1
    )[group_cols].drop_duplicates()
    eval_pairs = pd.concat(
        [singlematch_pairs, df_preselection_multimatch_eval[group_cols].drop_duplicates()], ignore_index=True
    )
    return df_preselection.merge(eval_pairs, on=group_cols, how="inner")


def evaluate_models(
    preselection_subset: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    models: Sequence = MODELS,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> pd.DataFrame:
    """
    Precision/recall/F1/accuracy for every model, on a subset of preselected candidate pairs

    Returns:
        pd.DataFrame: one row of metrics per model
    """
    merge_cols = list(group_cols) + ["AIS_track_id", "is_true_match"]
    metrics = {}
    for name, _, score_col, mode, _ in models:
        merged = preselection_subset.merge(results[name], on=merge_cols, how="inner")
        chosen = get_best_alignments(merged, score_col, mode=mode)
        metrics[name] = calculate_metrics(chosen)
    return pd.DataFrame(metrics).T


def avg_time_per_pair(checkpoint_path: str) -> float:
    """
    Average processing time per AIS-RF candidate pair from a checkpoint file

    Args:
        checkpoint_path: Path to a checkpoint file written by either
            compute_forward_score_parallel or compute_nn_score_parallel
    """
    total_time, total_pairs = 0.0, 0
    with open(checkpoint_path, "rb") as file:
        while True:
            try:
                _, result, time_iter = pickle.load(file)
            except EOFError:
                break
            total_time += time_iter
            total_pairs += len(result)
    return total_time / total_pairs


def runtime_comparison(data_dir: str, models: Sequence = MODELS) -> pd.DataFrame:
    """Average seconds per candidate pair for every model, from its checkpoint file"""
    return pd.DataFrame({
        "avg_seconds_per_pair": {
            name: avg_time_per_pair(f"{data_dir}/{checkpoint_file}")
            for name, _, _, _, checkpoint_file in models
        }
    })


def evaluate_ranking_models(
    preselection_subset: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    models: Sequence = MODELS,
    ks: tuple = (1, 3, 5),
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Recall@k, MRR and candidate-set-size stratification for every model

    Returns:
        (long-form table with one row per (model, candidate-count bucket),
        dict mapping model name to its per-RF-point ranking summary)
    """
    merge_cols = list(group_cols) + ["AIS_track_id", "is_true_match"]
    stratified_tables = []
    summaries = {}
    for name, _, score_col, mode, _ in models:
        merged = preselection_subset.merge(results[name], on=merge_cols, how="inner")
        summary = summarize_ranking(merged, score_col, mode=mode, ks=ks)
        summaries[name] = summary

        stratified = stratified_ranking_metrics(summary, ks=ks)
        stratified.insert(0, "model", name)
        stratified_tables.append(stratified)

    return pd.concat(stratified_tables, ignore_index=True), summaries


def compute_mcnemar_table(
    ranking_summaries: dict[str, pd.DataFrame],
    models: Sequence = MODELS,
    reference: str = "PHMM Forward",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> pd.DataFrame:
    """McNemar's test of top-1 hit/miss, reference model vs. every other model"""
    group_cols = list(group_cols)
    ref_summary = ranking_summaries[reference].set_index(group_cols)

    rows = []
    for name, *_ in models:
        if name == reference:
            continue
        other_summary = ranking_summaries[name].set_index(group_cols)
        aligned = ref_summary[["hit@1"]].join(
            other_summary[["hit@1"]], how="inner", lsuffix="_ref", rsuffix="_other"
        )
        result = mcnemar_test(aligned["hit@1_ref"], aligned["hit@1_other"])
        result["comparison"] = f"{reference} vs {name}"
        result["n_points"] = len(aligned)
        rows.append(result)

    return pd.DataFrame(rows)[["comparison", "n_points", "n10", "n01", "statistic", "p_value", "method"]]


def compute_bootstrap_table(
    ranking_summaries: dict[str, pd.DataFrame],
    models: Sequence = MODELS,
    n_boot: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap CIs (Recall@1, MRR) per model, 1000 resamples of RF points by default"""
    rows = []
    for name, *_ in models:
        summary = ranking_summaries[name]
        r1_est, r1_lo, r1_hi = bootstrap_ci(summary, "hit@1", n_boot=n_boot, random_state=random_state)
        mrr_est, mrr_lo, mrr_hi = bootstrap_ci(
            summary, "reciprocal_rank", n_boot=n_boot, random_state=random_state
        )
        rows.append({
            "model": name,
            "recall@1": r1_est, "recall@1_ci_lo": r1_lo, "recall@1_ci_hi": r1_hi,
            "mrr": mrr_est, "mrr_ci_lo": mrr_lo, "mrr_ci_hi": mrr_hi,
        })
    return pd.DataFrame(rows)
