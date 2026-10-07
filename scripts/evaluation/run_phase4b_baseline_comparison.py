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
import logging
from pathlib import Path

import pandas as pd

from scripts.evaluation import generate_plots
from scripts.evaluation.baseline_comparison import (
    LOG_ODDS_MODELS,
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
ALL_MODELS = MODELS + LOG_ODDS_MODELS
NN_TIME_WEIGHTED = next(m for m in MODELS if m[0] == "NN Time-weighted")
RANDOM_BASELINE_NAME = "Random baseline"

logger = logging.getLogger(__name__)


def main(data_dir: str, fig_dir: Path, table_dir: Path, base_dir: str = "./data/processed") -> None:
    df_preselection = pd.read_pickle(f"{data_dir}/AIS_RF_preselection_data.pkl")
    df_AIS_stats = pd.read_pickle(f"{base_dir}/statistics_sample_5000.pkl").reset_index(drop=True)
    results = load_model_results(data_dir)

    df_preselection_multimatch = df_preselection.groupby(
        ["RF_track_id", "RF_signal_id"]
    ).filter(lambda x: x["AIS_track_id"].count() > 1).reset_index(drop=True)

    raw_forward_results = results["PHMM Forward"]
    corrected_phmm, best_alpha, df_preselection_multimatch_eval = build_corrected_phmm_results(
        df_preselection_multimatch, raw_forward_results, df_AIS_stats
    )
    results["PHMM Forward"] = corrected_phmm

    corrected_logodds, best_alpha_lo, _ = build_corrected_phmm_results(
        df_preselection_multimatch, raw_forward_results, df_AIS_stats,
        score_col="log_odds_score", exp_col="log_odds_score_exp",
        raw_score_col="log_odds_score", output_col="log_odds_score_corrected",
    )
    results["Log-odds PHMM (raw)"] = raw_forward_results
    results["Log-odds PHMM (n^alpha)"] = corrected_logodds

    n_signals = ["RF_signal_id", "RF_track_id"]
    n_held_out = (
        df_preselection_multimatch[n_signals].drop_duplicates().shape[0]
        - df_preselection_multimatch_eval[n_signals].drop_duplicates().shape[0]
    )
    df_preselection_eval = restrict_to_eval_population(df_preselection, df_preselection_multimatch_eval)
    df_preselection, df_preselection_multimatch = df_preselection_eval, df_preselection_multimatch_eval
    populations = {"all": df_preselection, "multimatch": df_preselection_multimatch}

    metrics = {name: evaluate_models(df, results, models=ALL_MODELS) for name, df in populations.items()}
    for name, df in populations.items():
        random = random_choice_metrics(df)
        metrics[name].loc[RANDOM_BASELINE_NAME] = {k: random[k] for k in metrics[name].columns}

    runtimes = runtime_comparison(data_dir, models=ALL_MODELS)
    metrics_table = pd.concat(
        [m.join(runtimes).rename_axis("model").reset_index().assign(population=name) for name, m in metrics.items()],
        ignore_index=True,
    )
    metrics_table.insert(0, "population", metrics_table.pop("population"))
    metrics_table.to_csv(table_dir / "metrics.csv", index=False)

    ranking = {}
    ranking_summaries = {}
    for name, df in populations.items():
        ranking[name], ranking_summaries[name] = evaluate_ranking_models(df, results, models=ALL_MODELS)
    pd.concat(
        [r.assign(population=name) for name, r in ranking.items()], ignore_index=True
    ).pipe(lambda t: t[["population", *[c for c in t.columns if c != "population"]]]).to_csv(
        table_dir / "ranking_metrics.csv", index=False
    )

    margin_rows = []
    for name in MODEL_NAMES + [m[0] for m in LOG_ODDS_MODELS]:
        summary = ranking_summaries["multimatch"][name].dropna(subset=["margin"]).copy()
        margin_rows.append(summary[["RF_signal_id", "RF_track_id", "margin", "rank_of_true"]].assign(model=name))
    margins_multimatch = pd.concat(margin_rows, ignore_index=True)

    mcnemar_table = pd.concat([
        compute_mcnemar_table(ranking_summaries["all"]),
        compute_mcnemar_table(
            ranking_summaries["all"], models=[NN_TIME_WEIGHTED] + LOG_ODDS_MODELS, reference="NN Time-weighted"
        ),
    ], ignore_index=True)
    mcnemar_table.to_csv(table_dir / "mcnemar_test.csv", index=False)

    bootstrap_table = compute_bootstrap_table(ranking_summaries["all"], models=ALL_MODELS)
    bootstrap_table.to_csv(table_dir / "bootstrap_ci.csv", index=False)

    plot_data = {
        "metrics_all_candidates": metrics["all"],
        "metrics_multimatch_candidates": metrics["multimatch"],
        "score_margins_multimatch": margins_multimatch,
        "bootstrap_ci": bootstrap_table,
    }
    generate_plots.save_plot_data(plot_data, table_dir)
    generate_plots.phase4b_baseline_comparison(plot_data, fig_dir, MODEL_NAMES, RANDOM_BASELINE_NAME)

    mn = metrics["multimatch"]["precision"]
    logger.info(
        "Baseline comparison (multimatch precision): Forward %.4f, log-odds %.4f, NN time-weighted %.4f, "
        "random %.4f. Tuned alpha: Forward %s, log-odds %s (%d signals held out for tuning).",
        mn["PHMM Forward"], mn["Log-odds PHMM (raw)"], mn["NN Time-weighted"], mn[RANDOM_BASELINE_NAME],
        best_alpha, best_alpha_lo, n_held_out,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PHMM Forward vs. Nearest-Neighbor baseline comparison"
    )
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="gaussian",
        help="RF bearing-error model whose data folder to read from (default: gaussian)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    DATA_DIR = f"./data/processed/{args.error_model}"
    FIG_DIR = Path(f"reports/figures/phase4b_baseline_comparison/{args.error_model}")
    TABLE_DIR = Path(f"reports/tables/phase4b_baseline_comparison/{args.error_model}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    main(DATA_DIR, FIG_DIR, TABLE_DIR)
