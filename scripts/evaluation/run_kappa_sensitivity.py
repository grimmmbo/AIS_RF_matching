"""
Kappa sensitivity comparison: PHMM Forward / log-odds PHMM, alpha=1 vs.
alpha-tuned, under two kappa settings (kappa_ais, kappa_rf) -- the
exponents AIS_RF_forward_alignment.py applies to AISState's and
RFState's/MState's row-scaled transition probabilities. Default kappa
(0.5, 2) is the value hardcoded before --kappa existed; (1, 1) removes
that smoothing/sharpening entirely.

For each kappa this re-runs the existing alpha-tuning procedure
(phmm_length_correction.tune_alpha, same grid/seed/tie-break) on the
20% tuning split, then reports four variants' closed-set precision
(on the 80% validation split, exactly as run_phase4_evaluation.py's
correction_section does) and scored-only open-set AUC under the
MMSI-level registry/dark-vessel split (open_set.py,
build_dark_vessel_frame + compute_roc, exactly as
run_phase4c_open_set_evaluation.py's open_set_report does):
  - PHMM Forward alpha=1
  - PHMM Forward alpha-tuned
  - PHMM log-odds raw (no length normalization at all)
  - PHMM log-odds alpha-tuned

The default-kappa (0.5, 2) row is checked against the already-published
phase4_evaluation / phase4c_open_set_evaluation numbers before anything
is written; a mismatch stops the script rather than silently reporting
a diverged number.

Requires AIS_RF_forward_alignment.py to have already been run once per
kappa (its --kappa option caches scores to a kappa-suffixed file; the
default kappa reuses the existing, already-computed cache):
  uv run python -m scripts.modeling.AIS_RF_alignment.AIS_RF_forward_alignment --kappa 0.5,2
  uv run python -m scripts.modeling.AIS_RF_alignment.AIS_RF_forward_alignment --kappa 1,1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation.metrics import calculate_metrics, deterministic_best_alignments
from scripts.evaluation.open_set import apply_alpha_correction, build_dark_vessel_frame, compute_roc, split_registry_dark_vessels
from scripts.evaluation.phmm_length_correction import correct_forward_score, tune_alpha
from scripts.modeling.AIS_RF_alignment.AIS_RF_forward_alignment import DEFAULT_KAPPA, kappa_cache_suffix, parse_kappa

MERGE_COLS = ["RF_track_id", "RF_signal_id", "AIS_track_id", "is_true_match"]

VARIANT_ALPHA1 = "PHMM Forward alpha=1"
VARIANT_ALPHA_TUNED = "PHMM Forward alpha-tuned"
VARIANT_LO_RAW = "PHMM log-odds raw"
VARIANT_LO_TUNED = "PHMM log-odds alpha-tuned"


def parse_kappa_list(kappas_str: str) -> list[tuple[float, float]]:
    return [parse_kappa(part) for part in kappas_str.split(";")]


def load_forward_results(data_dir: str, kappa_ais: float, kappa_rf: float) -> pd.DataFrame:
    suffix = kappa_cache_suffix(kappa_ais, kappa_rf)
    path = Path(f"{data_dir}/AIS_RF_forward_scores_data{suffix}.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:\n"
            f"  uv run python -m scripts.modeling.AIS_RF_alignment.AIS_RF_forward_alignment "
            f"--kappa {kappa_ais},{kappa_rf}\nfirst."
        )
    return pd.read_pickle(path)


def closed_set_variants(
    df_forward_results_multimatch: pd.DataFrame,
    df_preselection_multimatch: pd.DataFrame,
    df_AIS_stats: pd.DataFrame,
) -> dict:
    """
    Tune alpha on the 20% tuning split (both score columns, same grid/
    seed/tie-break as run_phase4_evaluation.py), then compute all four
    variants' closed-set precision on the 80% validation split
    (df_experiments)

    Returns:
        dict with "best_alpha", "best_alpha_lo", and "precision" (dict
        variant name -> precision)
    """
    best_alpha, _, _, df_experiments = tune_alpha(
        df_forward_results_multimatch, df_preselection_multimatch, df_AIS_stats,
    )
    best_alpha_lo, _, _, _ = tune_alpha(
        df_forward_results_multimatch, df_preselection_multimatch, df_AIS_stats,
        score_col="log_odds_score", exp_col="log_odds_score_exp",
    )

    metrics_alpha1, _, _ = correct_forward_score(
        df_experiments, df_preselection_multimatch, df_AIS_stats, 1.0,
    )
    metrics_alpha_tuned, _, _ = correct_forward_score(
        df_experiments, df_preselection_multimatch, df_AIS_stats, best_alpha,
    )
    chosen_lo_raw, _ = deterministic_best_alignments(df_experiments, "log_odds_score")
    metrics_lo_raw = calculate_metrics(chosen_lo_raw)
    metrics_lo_tuned, _, _ = correct_forward_score(
        df_experiments, df_preselection_multimatch, df_AIS_stats, best_alpha_lo,
        score_col="log_odds_score", exp_col="log_odds_score_exp",
    )

    return {
        "best_alpha": best_alpha,
        "best_alpha_lo": best_alpha_lo,
        "precision": {
            VARIANT_ALPHA1: metrics_alpha1["precision"],
            VARIANT_ALPHA_TUNED: metrics_alpha_tuned["precision"],
            VARIANT_LO_RAW: metrics_lo_raw["precision"],
            VARIANT_LO_TUNED: metrics_lo_tuned["precision"],
        },
    }


def open_set_variants(
    df_closed_full: pd.DataFrame,
    df_preselection: pd.DataFrame,
    df_AIS_stats: pd.DataFrame,
    registry_ids: set,
    dark_ids: set,
    best_alpha: float,
    best_alpha_lo: float,
) -> dict:
    """
    Scored-only open-set AUC (dark-vessel/MMSI split) for all four
    variants, reusing open_set.py's own frame-building and ROC code
    """
    df_alpha_tuned = apply_alpha_correction(
        df_closed_full, df_AIS_stats, "forward_score", best_alpha, "forward_score_alpha_tuned",
    )
    df_lo_tuned = apply_alpha_correction(
        df_closed_full, df_AIS_stats, "log_odds_score", best_alpha_lo, "log_odds_score_corrected",
    )

    variant_frames = {
        VARIANT_ALPHA1: (df_closed_full, "normalized_forward_score"),
        VARIANT_ALPHA_TUNED: (df_alpha_tuned, "forward_score_alpha_tuned"),
        VARIANT_LO_RAW: (df_closed_full, "log_odds_score"),
        VARIANT_LO_TUNED: (df_lo_tuned, "log_odds_score_corrected"),
    }

    auc_scored_only = {}
    for name, (df_scored, score_col) in variant_frames.items():
        frame = build_dark_vessel_frame(
            name, df_preselection, {name: df_scored}, registry_ids, dark_ids, score_col, "max",
        )
        scored_only = frame[frame["rejection_type"] != "automatic_rejection"]
        auc_scored_only[name] = compute_roc(scored_only, "accept_score")["roc_auc"]

    return auc_scored_only


def _confusion_precision(confusion_csv: Path, label: str) -> float:
    df = pd.read_csv(confusion_csv)
    block = df[df["score"] == label]
    tp = block.loc[block["actual"] == "Actual Positive", "Predicted Positive"].iloc[0]
    fp = block.loc[block["actual"] == "Actual Negative", "Predicted Positive"].iloc[0]
    return tp / (tp + fp)


def load_expected_baseline(error_model: str) -> dict:
    """
    Expected closed-set precision / scored-only open-set AUC for the
    default kappa (0.5, 2), parsed from the already-published
    phase4_evaluation / phase4c_open_set_evaluation tables -- the
    ground truth this script's default-kappa row must reproduce
    """
    eval_dir = Path(f"reports/tables/phase4_evaluation/{error_model}")
    open_set_dir = Path(f"reports/tables/phase4c_open_set_evaluation/{error_model}")

    before_csv = eval_dir / "confusion_matrix_before_correction.csv"
    after_csv = eval_dir / "confusion_matrix_after_correction.csv"
    open_set_csv = open_set_dir / "open_set_summary_darkvessel.csv"

    precision = {
        VARIANT_ALPHA1: _confusion_precision(before_csv, "PHMM Forward (alpha=1)"),
        VARIANT_ALPHA_TUNED: _confusion_precision(after_csv, "PHMM Forward (n^alpha)"),
        VARIANT_LO_RAW: _confusion_precision(before_csv, "Log-odds PHMM (raw)"),
        VARIANT_LO_TUNED: _confusion_precision(after_csv, "Log-odds PHMM (n^alpha)"),
    }

    open_set_df = pd.read_csv(open_set_csv).set_index("model")["auc_scored_only_raw"]
    auc_scored_only = {
        VARIANT_ALPHA1: open_set_df["PHMM Forward"],
        # No existing published row uses an alpha-tuned (rather than
        # alpha=1) PHMM Forward score for open-set evaluation -- nothing
        # to check this variant against.
        VARIANT_ALPHA_TUNED: None,
        VARIANT_LO_RAW: open_set_df["Log-odds PHMM (raw)"],
        VARIANT_LO_TUNED: open_set_df["Log-odds PHMM (n^alpha)"],
    }

    return {"precision": precision, "auc_scored_only": auc_scored_only}


def validate_against_baseline(rows: list[dict], error_model: str, tol: float = 1e-6) -> None:
    """
    closed_set_precision is compared after rounding both sides to 4
    decimals, matching calculate_metrics()'s own round(precision, 4) --
    the "expected" value is parsed from a CSV that already went through
    that rounding, while the freshly computed one may not have (e.g.
    the log-odds-raw variant here calls calculate_metrics() directly,
    same as everywhere else, but comparing at full float precision
    would flag that existing rounding as a false mismatch)
    """
    try:
        expected = load_expected_baseline(error_model)
    except (FileNotFoundError, KeyError, IndexError) as exc:
        print(f"Could not load existing baseline results to validate against ({exc}); skipping check.")
        return

    mismatches = []
    for row in rows:
        variant = row["variant"]
        exp_precision = expected["precision"].get(variant)
        if exp_precision is not None and round(row["closed_set_precision"], 4) != round(exp_precision, 4):
            mismatches.append(
                f"{variant}: closed_set_precision {row['closed_set_precision']} != expected {exp_precision}"
            )
        exp_auc = expected["auc_scored_only"].get(variant)
        if exp_auc is not None and abs(row["auc_scored_only"] - exp_auc) > tol:
            mismatches.append(
                f"{variant}: auc_scored_only {row['auc_scored_only']} != expected {exp_auc}"
            )

    if mismatches:
        print("\nSTOP: kappa=0.5,2 does not reproduce the existing published results:")
        for line in mismatches:
            print(f"  - {line}")
        sys.exit(1)

    print("\nkappa=0.5,2 reproduces the existing published results exactly (within tolerance).")


def main(error_model: str, kappas: list[tuple[float, float]], base_dir: str = "./data/processed") -> pd.DataFrame:
    data_dir = f"./data/processed/{error_model}"

    df_preselection = pd.read_pickle(f"{data_dir}/AIS_RF_preselection_data.pkl")
    df_AIS_stats = pd.read_pickle(f"{base_dir}/statistics_sample_5000.pkl").reset_index(drop=True)
    df_preselection_multimatch = df_preselection.groupby(
        ["RF_track_id", "RF_signal_id"]
    ).filter(lambda x: x["AIS_track_id"].count() > 1).reset_index(drop=True)

    # Vessel universe / MMSI-level registry-dark split: independent of
    # kappa (it only depends on which vessels/segments exist in
    # df_preselection), so computed once and reused for every kappa --
    # same fix and same code as run_phase4c_open_set_evaluation.py.
    vessel_ids = pd.concat([df_preselection["RF_track_id"], df_preselection["AIS_track_id"]]).unique()
    unique_mmsis = {vid[0] for vid in vessel_ids}
    registry_mmsis, dark_mmsis = split_registry_dark_vessels(unique_mmsis)
    registry_ids = {vid for vid in vessel_ids if vid[0] in registry_mmsis}
    dark_ids = {vid for vid in vessel_ids if vid[0] in dark_mmsis}

    all_rows = []
    for kappa_ais, kappa_rf in kappas:
        print(f"\n=== kappa = {kappa_ais},{kappa_rf} ===")
        try:
            df_forward_results_raw = load_forward_results(data_dir, kappa_ais, kappa_rf)
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)

        df_forward_results_multimatch = df_preselection_multimatch.merge(
            df_forward_results_raw, on=MERGE_COLS, how="inner",
        )
        closed = closed_set_variants(df_forward_results_multimatch, df_preselection_multimatch, df_AIS_stats)

        auc_scored_only = open_set_variants(
            df_forward_results_raw, df_preselection, df_AIS_stats,
            registry_ids, dark_ids, closed["best_alpha"], closed["best_alpha_lo"],
        )

        variant_alpha = {
            VARIANT_ALPHA1: 1.0,
            VARIANT_ALPHA_TUNED: closed["best_alpha"],
            VARIANT_LO_RAW: np.nan,
            VARIANT_LO_TUNED: closed["best_alpha_lo"],
        }

        kappa_rows = [
            {
                "kappa_ais": kappa_ais,
                "kappa_rf": kappa_rf,
                "variant": variant,
                "alpha": variant_alpha[variant],
                "closed_set_precision": closed["precision"][variant],
                "auc_scored_only": auc_scored_only[variant],
            }
            for variant in [VARIANT_ALPHA1, VARIANT_ALPHA_TUNED, VARIANT_LO_RAW, VARIANT_LO_TUNED]
        ]
        for row in kappa_rows:
            print(f"  {row['variant']}: alpha={row['alpha']}, "
                  f"closed_set_precision={row['closed_set_precision']}, "
                  f"auc_scored_only={row['auc_scored_only']}")

        if (kappa_ais, kappa_rf) == DEFAULT_KAPPA:
            validate_against_baseline(kappa_rows, error_model)

        all_rows.extend(kappa_rows)

    result = pd.DataFrame(all_rows)
    print("\n=== Kappa sensitivity summary ===")
    print(result)

    table_dir = Path(f"reports/tables/phase4_evaluation/{error_model}")
    table_dir.mkdir(parents=True, exist_ok=True)
    out_path = table_dir / "kappa_sensitivity.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHMM kappa sensitivity comparison")
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="gaussian",
        help="RF bearing-error model whose data folder to read from (default: gaussian)",
    )
    parser.add_argument(
        "--kappas", default="0.5,2;1,1",
        help="Semicolon-separated list of kappa_ais,kappa_rf pairs to compare "
             "(default: 0.5,2;1,1)",
    )
    args = parser.parse_args()

    main(args.error_model, parse_kappa_list(args.kappas))
