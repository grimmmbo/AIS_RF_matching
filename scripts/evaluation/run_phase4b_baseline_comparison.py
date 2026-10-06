"""
Baseline comparison: PHMM Forward vs. Nearest-Neighbor

Compares the PHMM Forward alignment against a naive nearest-neighbor
baseline, run with four distance metrics as an ablation: plain
Euclidean, Haversine, point-to-segment, and time-weighted (see
scripts/modeling/AIS_RF_alignment/AIS_RF_nn_baseline.py). All five
variants are evaluated on the same preselected candidate pairs, both
on all candidates and restricted to RF signals with more than one AIS
candidate.

PHMM Forward is scored with its length-corrected score (see
scripts/evaluation/phmm_length_correction.py), tuned exactly as in
run_phase4_evaluation.py / run_phase4c_bias_experiments.py: alpha is
chosen on a 20% held-out split of multimatch RF signals, applied to
the other 80%. Every model here -- not just PHMM Forward -- is
restricted to that same 80% (plus all single-match signals, which the
correction can't affect), so no model is compared on data PHMM
Forward's alpha was tuned on.

See notebooks/05_baseline_comparison.ipynb for the interactive,
plot-only version of this script.
"""
import argparse
from pathlib import Path

import pandas as pd

from scripts.evaluation import generate_plots
from scripts.evaluation.baseline_comparison import (
    MODELS,
    build_corrected_phmm_results,
    compute_bootstrap_table,
    compute_mcnemar_table,
    evaluate_models,
    evaluate_ranking_models,
    load_model_results,
    random_choice_metrics,
    restrict_to_eval_population,
    runtime_comparison,
)

MODEL_NAMES = [name for name, *_ in MODELS]
RANDOM_BASELINE_NAME = "Random baseline"


def main(data_dir: str, fig_dir: Path, table_dir: Path, base_dir: str = "./data/processed") -> None:
    df_preselection = pd.read_pickle(f"{data_dir}/AIS_RF_preselection_data.pkl")
    df_AIS_stats = pd.read_pickle(f"{base_dir}/statistics_sample_5000.pkl").reset_index(drop=True)
    results = load_model_results(data_dir)

    df_preselection_multimatch = df_preselection.groupby(
        ["RF_track_id", "RF_signal_id"]
    ).filter(lambda x: x["AIS_track_id"].count() > 1).reset_index(drop=True)

    corrected_phmm, best_alpha, df_preselection_multimatch_eval = build_corrected_phmm_results(
        df_preselection_multimatch, results["PHMM Forward"], df_AIS_stats
    )
    print(f"PHMM length-correction alpha (tuned on a 20% held-out split): {best_alpha}")
    results["PHMM Forward"] = corrected_phmm

    df_preselection_eval = restrict_to_eval_population(df_preselection, df_preselection_multimatch_eval)
    print(
        "Multimatch RF signals held out for alpha tuning, excluded from every model's metrics below: "
        f"{df_preselection_multimatch[['RF_signal_id', 'RF_track_id']].drop_duplicates().shape[0] - df_preselection_multimatch_eval[['RF_signal_id', 'RF_track_id']].drop_duplicates().shape[0]}"
    )
    df_preselection, df_preselection_multimatch = df_preselection_eval, df_preselection_multimatch_eval

    print("Candidate pairs evaluated (all RF signals):", len(df_preselection))
    comparison_all = evaluate_models(df_preselection, results)
    print(comparison_all)

    print("\nCandidate pairs evaluated (multimatch RF signals):", len(df_preselection_multimatch))
    comparison_multimatch = evaluate_models(df_preselection_multimatch, results)
    print(comparison_multimatch)

    random_all = random_choice_metrics(df_preselection)
    random_multi = random_choice_metrics(df_preselection_multimatch)
    print(
        f"\nRandom-choice baseline (mean over 200 repetitions): "
        f"all candidates precision={random_all['precision']:.4f} (std={random_all['precision_std']:.4f}), "
        f"multimatch precision={random_multi['precision']:.4f} (std={random_multi['precision_std']:.4f})"
    )
    comparison_all.loc[RANDOM_BASELINE_NAME] = {k: random_all[k] for k in comparison_all.columns}
    comparison_multimatch.loc[RANDOM_BASELINE_NAME] = {k: random_multi[k] for k in comparison_multimatch.columns}

    comparison_all.to_csv(table_dir / "metrics_all_candidates.csv")
    comparison_multimatch.to_csv(table_dir / "metrics_multimatch_candidates.csv")
    plot_data = {"metrics_all_candidates": comparison_all, "metrics_multimatch_candidates": comparison_multimatch}

    runtimes = runtime_comparison(data_dir)
    print("\nRuntime comparison (avg seconds per candidate pair):")
    print(runtimes)
    runtimes.to_csv(table_dir / "runtime_comparison.csv")

    ranking_comparison_all, ranking_summaries_all = evaluate_ranking_models(df_preselection, results)
    print("\nRanking metrics, all RF signals:")
    print(ranking_comparison_all)
    ranking_comparison_all.to_csv(table_dir / "ranking_metrics_all_candidates.csv", index=False)

    ranking_comparison_multi, ranking_summaries_multi = evaluate_ranking_models(
        df_preselection_multimatch, results
    )
    print("\nRanking metrics, multimatch RF signals:")
    print(ranking_comparison_multi)
    ranking_comparison_multi.to_csv(table_dir / "ranking_metrics_multimatch_candidates.csv", index=False)

    margin_rows = []
    for name in MODEL_NAMES:
        summary = ranking_summaries_multi[name].dropna(subset=["margin"]).copy()
        margin_rows.append(summary[["RF_signal_id", "RF_track_id", "margin", "rank_of_true"]].assign(model=name))
    margins_multimatch = pd.concat(margin_rows, ignore_index=True)
    margins_multimatch.to_csv(table_dir / "score_margins_multimatch.csv", index=False)
    plot_data["score_margins_multimatch"] = margins_multimatch

    mcnemar_table = compute_mcnemar_table(ranking_summaries_all)
    print("\nMcNemar's test, PHMM Forward vs. each baseline (all RF signals):")
    print(mcnemar_table)
    mcnemar_table.to_csv(table_dir / "mcnemar_test.csv", index=False)

    bootstrap_table = compute_bootstrap_table(ranking_summaries_all)
    print("\nBootstrap CIs (Recall@1, MRR), all RF signals:")
    print(bootstrap_table)
    bootstrap_table.to_csv(table_dir / "bootstrap_ci.csv", index=False)
    plot_data["bootstrap_ci"] = bootstrap_table

    generate_plots.phase4b_baseline_comparison(plot_data, fig_dir, MODEL_NAMES, RANDOM_BASELINE_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PHMM Forward vs. Nearest-Neighbor baseline comparison"
    )
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="gaussian",
        help="RF bearing-error model whose data folder to read from (default: gaussian)",
    )
    args = parser.parse_args()

    DATA_DIR = "./data/processed" if args.error_model == "uniform" else f"./data/processed/{args.error_model}"
    FIG_DIR = Path(f"reports/figures/phase4b_baseline_comparison/{args.error_model}")
    TABLE_DIR = Path(f"reports/tables/phase4b_baseline_comparison/{args.error_model}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    main(DATA_DIR, FIG_DIR, TABLE_DIR)
