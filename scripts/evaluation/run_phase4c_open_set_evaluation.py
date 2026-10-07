"""
Open-set (accept/reject) evaluation: can the pipeline recognize an RF
point with no true AIS match, instead of always force-picking a
candidate? run_phase4_evaluation.py / run_phase4b_baseline_comparison.py
stay closed-set only (true track always included); this script reuses
their prefilter/scoring functions without that bypass.

Steps:
  2a. Diagnostic: would the true AIS track survive prefiltering alone?
  2b. Dark-vessel negatives (primary test): vessels are split into a
      registry pool (searchable) and a dark pool (queried, but never
      searchable by anyone). Built by filtering the closed-set data --
      see open_set.build_dark_vessel_frame for why that's equivalent
      to a fresh registry-only alignment run.
  2c. Leave-one-out negatives (secondary): each RF point's own true
      track is excluded from its own candidate pool and the pipeline
      rerun (scripts/modeling/AIS_RF_open_set/AIS_RF_leave_one_out.py).
      Skipped with a warning if that hasn't been run for this
      --error-model.
  2d. ROC/PR/AUC for both negative constructions, split into
      "automatic rejection" (no candidate survived the prefilter) vs.
      "scored rejection" (1+ candidates survived, scoring model judged
      none confident enough). Reported as full ROC/AUC, no tuned
      threshold -- see open_set.py.

See notebooks/06_open_set_evaluation.ipynb for the interactive,
plot-only version of this script.
"""
import argparse
from pathlib import Path

import pandas as pd

from scripts.evaluation import generate_plots
from scripts.evaluation.open_set import (
    FORWARD_CORRECTED_MODEL,
    LOG_ODDS_MODELS,
    MODELS,
    apply_alpha_correction,
    build_dark_vessel_frame,
    build_open_set_frame,
    compute_pr,
    compute_roc,
    load_closed_results,
    load_loo_results,
    split_registry_dark_vessels,
)
from scripts.evaluation.phmm_length_correction import tune_alpha


def diagnostic_prefilter_recall(data_dir: str) -> pd.DataFrame:
    df_true_match_diagnostic = pd.read_pickle(f"{data_dir}/true_match_prefilter_diagnostic.pkl")
    true_prefilter_recall = df_true_match_diagnostic["passed_prefilter"].mean()
    print(f"True prefilter recall: {true_prefilter_recall:.4f}")
    print(
        f"({df_true_match_diagnostic['passed_prefilter'].sum()} / "
        f"{len(df_true_match_diagnostic)} true tracks would pass stages 1-3 unaided)"
    )
    print(df_true_match_diagnostic["passed_prefilter"].value_counts())
    return df_true_match_diagnostic


def rejection_type_summary(n_universe: int, n_scored_rejections: int, label: str) -> dict:
    n_automatic_rejections = n_universe - n_scored_rejections
    print(f"Total RF points (universe): {n_universe}")
    print(f"Automatic rejections -- zero candidates survived the {label} prefilter, "
          f"no scoring model needed: {n_automatic_rejections} ({n_automatic_rejections / n_universe:.2%})")
    print(f"Scored rejections -- 1+ candidates survived, the scoring model had to judge none "
          f"confident enough: {n_scored_rejections} ({n_scored_rejections / n_universe:.2%})")
    return {
        "n_universe": n_universe,
        "n_automatic_rejections": n_automatic_rejections,
        "n_scored_rejections": n_scored_rejections,
    }


def open_set_report(
    frames: dict[str, pd.DataFrame], models, rej_summary: dict, table_dir: Path, slug: str, plot_data: dict,
) -> pd.DataFrame:
    """
    Compute + save the ROC/PR/AUC tables for one negative-class
    construction (dark-vessel or leave-one-out), and stash the curves
    into plot_data for generate_plots.phase4c_open_set_evaluation to
    draw later (titles/labels live there, not here)

    Reports AUC (the full ROC curve, i.e. every possible accept/reject
    threshold at once) rather than picking one operating threshold --
    no threshold is tuned anywhere in this script. AUC is reported both
    over every negative (including the automatic rejections, which
    inflate it since they're trivial) and restricted to scored
    rejections only, which is the harder, more informative number.
    """
    overall_curves = {
        name: {**compute_roc(frames[name], "accept_score"), **compute_pr(frames[name], "accept_score")}
        for name, *_ in models
    }
    roc_curve_overall = pd.concat([
        pd.DataFrame({"model": name, "fpr": c["fpr"], "tpr": c["tpr"]}) for name, c in overall_curves.items()
    ], ignore_index=True)
    roc_curve_overall.to_csv(table_dir / f"roc_curve_overall_{slug}.csv", index=False)
    plot_data[f"roc_curve_overall_{slug}"] = roc_curve_overall

    pr_curve_overall = pd.concat([
        pd.DataFrame({"model": name, "recall": c["recall"], "precision": c["precision"]})
        for name, c in overall_curves.items()
    ], ignore_index=True)
    pr_curve_overall.to_csv(table_dir / f"pr_curve_overall_{slug}.csv", index=False)
    plot_data[f"pr_curve_overall_{slug}"] = pr_curve_overall

    auc_rows = [
        {"model": name, "auc_overall_raw": c["roc_auc"], "ap_overall_raw": c["ap"]}
        for name, c in overall_curves.items()
    ]

    scored_only_curves = {
        name: compute_roc(frames[name][frames[name]["rejection_type"] != "automatic_rejection"], "accept_score")
        for name, *_ in models
    }
    roc_curve_scored_only = pd.concat([
        pd.DataFrame({"model": name, "fpr": c["fpr"], "tpr": c["tpr"]}) for name, c in scored_only_curves.items()
    ], ignore_index=True)
    roc_curve_scored_only.to_csv(table_dir / f"roc_curve_scored_only_{slug}.csv", index=False)
    plot_data[f"roc_curve_scored_only_{slug}"] = roc_curve_scored_only

    scored_only_auc_rows = [{"model": name, "auc_scored_only_raw": c["roc_auc"]} for name, c in scored_only_curves.items()]

    summary = pd.DataFrame(auc_rows).merge(pd.DataFrame(scored_only_auc_rows), on="model")
    summary["automatic_rejection_rate"] = rej_summary["n_automatic_rejections"] / rej_summary["n_universe"]
    summary = summary[[
        "model", "automatic_rejection_rate",
        "auc_overall_raw", "auc_scored_only_raw",
        "ap_overall_raw",
    ]]
    print(summary)
    summary.to_csv(table_dir / f"open_set_summary_{slug}.csv", index=False)
    plot_data[f"open_set_summary_{slug}"] = summary
    return summary


def combined_summary_from_frames(frames: dict[str, pd.DataFrame], open_set_summary: pd.DataFrame) -> pd.DataFrame:
    """Recall@1/MRR (from each frame's own positive rows) alongside its open-set AUC summary"""
    rows = []
    for name, frame in frames.items():
        positives = frame[frame["rejection_type"] == "positive"]
        rows.append({
            "model": name,
            "recall@1": positives["hit@1"].mean(),
            "mrr": positives["reciprocal_rank"].mean(),
        })
    return pd.DataFrame(rows).merge(open_set_summary, on="model")


def main(data_dir: str, fig_dir: Path, table_dir: Path, base_dir: str = "./data/processed") -> None:
    plot_data = {}
    df_preselection = pd.read_pickle(f"{data_dir}/AIS_RF_preselection_data.pkl")
    df_AIS_stats = pd.read_pickle(f"{base_dir}/statistics_sample_5000.pkl").reset_index(drop=True)
    closed_results = load_closed_results(data_dir)

    # Log-odds PHMM: "raw" reuses PHMM Forward's own cached frame (same
    # file, log_odds_score column added alongside forward_score by
    # AIS_RF_forward_alignment.py); "n^alpha" applies the length
    # correction's alpha -- re-tuned here the same way run_phase4_
    # evaluation.py / run_phase4b_baseline_comparison.py each
    # independently re-tune it (deterministic given random_state=100,
    # never persisted to disk) -- uniformly to every row, since open-set
    # evaluation reuses an alpha already chosen on the closed-set data
    # rather than re-tuning one here.
    closed_results["Log-odds PHMM (raw)"] = closed_results["PHMM Forward"]
    df_preselection_multimatch = df_preselection.groupby(
        ["RF_track_id", "RF_signal_id"]
    ).filter(lambda x: x["AIS_track_id"].count() > 1).reset_index(drop=True)
    df_forward_results_multimatch = df_preselection_multimatch.merge(
        closed_results["PHMM Forward"],
        on=["RF_track_id", "RF_signal_id", "AIS_track_id", "is_true_match"], how="inner",
    )
    best_alpha, _, _, _ = tune_alpha(
        df_forward_results_multimatch, df_preselection_multimatch, df_AIS_stats,
    )
    print(f"PHMM Forward length-correction alpha (tuned on the usual 20% held-out split): {best_alpha}")
    closed_results[FORWARD_CORRECTED_MODEL[0]] = apply_alpha_correction(
        closed_results["PHMM Forward"], df_AIS_stats, "forward_score", best_alpha, FORWARD_CORRECTED_MODEL[3],
    )
    best_alpha_lo, _, _, _ = tune_alpha(
        df_forward_results_multimatch, df_preselection_multimatch, df_AIS_stats,
        score_col="log_odds_score", exp_col="log_odds_score_exp",
    )
    print(f"Log-odds PHMM length-correction alpha (tuned on the usual 20% held-out split): {best_alpha_lo}")
    closed_results["Log-odds PHMM (n^alpha)"] = apply_alpha_correction(
        closed_results["PHMM Forward"], df_AIS_stats, "log_odds_score", best_alpha_lo, "log_odds_score_corrected",
    )

    print("=== 2a: true prefilter recall (diagnostic) ===")
    diagnostic_prefilter_recall(data_dir)

    print("\n=== 2b: dark-vessel negatives (primary open-set test) ===")
    vessel_ids = pd.concat([df_preselection["RF_track_id"], df_preselection["AIS_track_id"]]).unique()
    # vessel_ids are (MMSI, track_id) segment tuples (step03_make_
    # continuous_tracks.py splits one vessel's AIS messages into several
    # continuous-track segments at time gaps). Splitting on the segment
    # tuple directly could put two segments of the SAME vessel on
    # opposite sides of the registry/dark split; split unique MMSIs
    # instead (same function, same seed/fraction) and expand back, so
    # every segment of a vessel lands on the same side.
    unique_mmsis = {vid[0] for vid in vessel_ids}
    registry_mmsis, dark_mmsis = split_registry_dark_vessels(unique_mmsis)
    registry_ids = {vid for vid in vessel_ids if vid[0] in registry_mmsis}
    dark_ids = {vid for vid in vessel_ids if vid[0] in dark_mmsis}
    print(f"Vessel universe: {len(vessel_ids)} segments ({len(unique_mmsis)} unique MMSIs); "
          f"registry: {len(registry_ids)} segments ({len(registry_mmsis)} MMSIs); "
          f"dark (held out): {len(dark_ids)} segments ({len(dark_mmsis)} MMSIs)")

    dark_models = MODELS + [FORWARD_CORRECTED_MODEL] + LOG_ODDS_MODELS
    dark_vessel_frames = {
        name: build_dark_vessel_frame(name, df_preselection, closed_results, registry_ids, dark_ids, score_col, mode)
        for name, _, _, score_col, mode in dark_models
    }
    n_dark_universe = df_preselection[df_preselection["RF_track_id"].isin(dark_ids)][
        ["RF_signal_id", "RF_track_id"]
    ].drop_duplicates().shape[0]
    # The prefilter runs once in AIS_RF_preselection.py, before any
    # scoring, so automatic-vs-scored is identical across models --
    # compute it once from PHMM Forward's frame, not from all models.
    reference_frame = dark_vessel_frames[MODELS[0][0]]
    n_dark_scored_rejections = (reference_frame["rejection_type"] == "scored_rejection").sum()
    dark_rej_summary = rejection_type_summary(n_dark_universe, n_dark_scored_rejections, "dark-vessel")

    print("\n--- Dark-vessel rejection metrics ---")
    dark_open_set_summary = open_set_report(
        dark_vessel_frames, dark_models, dark_rej_summary, table_dir, slug="darkvessel", plot_data=plot_data,
    )

    print("\n=== Combined closed-set + open-set summary (dark-vessel, primary) ===")
    combined_dark = combined_summary_from_frames(dark_vessel_frames, dark_open_set_summary)
    print(combined_dark)
    combined_dark.to_csv(table_dir / "combined_phase1_phase2_summary_darkvessel.csv", index=False)

    print("\n=== 2c: leave-one-out negatives (secondary) ===")
    have_loo_data = True
    try:
        df_preselection_loo = pd.read_pickle(f"{data_dir}/AIS_RF_preselection_leaveoneout_data.pkl")
        loo_results = load_loo_results(data_dir)
    except FileNotFoundError as exc:
        print(f"Skipped -- leave-one-out data not found for this error model ({exc.filename}).")
        print("Run scripts/modeling/AIS_RF_open_set/AIS_RF_leave_one_out.py for this --error-model first.")
        have_loo_data = False

    if have_loo_data:
        rf_universe = df_preselection[["RF_signal_id", "RF_track_id"]].drop_duplicates()
        loo_surviving = df_preselection_loo[["RF_signal_id", "RF_track_id"]].drop_duplicates()
        loo_rej_summary = rejection_type_summary(len(rf_universe), len(loo_surviving), "leave-one-out")

        # Log-odds columns are only added to the leave-one-out cache by
        # a separate, deferred rerun of AIS_RF_leave_one_out.py -- only
        # include them here once that's actually been done, so a
        # missing column doesn't crash this secondary/supplementary test.
        has_log_odds_loo = "log_odds_score" in next(iter(loo_results.values())).columns
        if has_log_odds_loo:
            loo_models = MODELS + [FORWARD_CORRECTED_MODEL] + LOG_ODDS_MODELS
            loo_results[FORWARD_CORRECTED_MODEL[0]] = apply_alpha_correction(
                loo_results["PHMM Forward"], df_AIS_stats, "forward_score", best_alpha, FORWARD_CORRECTED_MODEL[3],
            )
            loo_results["Log-odds PHMM (raw)"] = loo_results["PHMM Forward"]
            loo_results["Log-odds PHMM (n^alpha)"] = apply_alpha_correction(
                loo_results["PHMM Forward"], df_AIS_stats, "log_odds_score", best_alpha_lo, "log_odds_score_corrected",
            )
        else:
            loo_models = MODELS
            print("Log-odds PHMM skipped for leave-one-out -- rerun AIS_RF_leave_one_out.py to include it.")

        print("\n--- Leave-one-out rejection metrics ---")
        loo_frames = {
            name: build_open_set_frame(
                name, df_preselection, df_preselection_loo, closed_results, loo_results, score_col, mode
            )
            for name, _, _, score_col, mode in loo_models
        }
        loo_open_set_summary = open_set_report(
            loo_frames, loo_models, loo_rej_summary, table_dir, slug="leaveoneout", plot_data=plot_data,
        )

        print("\n=== Combined closed-set + open-set summary (leave-one-out, secondary) ===")
        combined_loo = combined_summary_from_frames(loo_frames, loo_open_set_summary)
        print(combined_loo)
        combined_loo.to_csv(table_dir / "combined_phase1_phase2_summary_leaveoneout.csv", index=False)

    generate_plots.phase4c_open_set_evaluation(plot_data, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-set (accept/reject) evaluation")
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="gaussian",
        help="RF bearing-error model whose data folder to read from (default: gaussian)",
    )
    args = parser.parse_args()

    DATA_DIR = f"./data/processed/{args.error_model}"
    FIG_DIR = Path(f"reports/figures/phase4c_open_set_evaluation/{args.error_model}")
    TABLE_DIR = Path(f"reports/tables/phase4c_open_set_evaluation/{args.error_model}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    main(DATA_DIR, FIG_DIR, TABLE_DIR)
