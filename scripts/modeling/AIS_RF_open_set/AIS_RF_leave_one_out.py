"""
Build leave-one-out open-set negatives: rerun the unbypassed prefilter
and every scoring method with each RF signal's own true AIS track
excluded from its candidate pool
(compute_AIS_RF_alignments_parallel(include_true_match=False) drops it
before the prefilter runs, not just from a forced include).

RF signals with zero surviving candidates never appear in any output
here -- scripts/evaluation/open_set.py treats them as
"automatic_rejection" rows. RF signals with >=1 surviving candidate
are scored here and become that module's "scored_rejection" rows.

All outputs are suffixed "_leaveoneout" and never overwrite the
closed-set files (AIS_RF_preselection_data.pkl,
AIS_RF_forward_scores_data.pkl, AIS_RF_nn_baseline_*_scores_data.pkl),
which this script does not read or modify.
"""
import argparse
import os
from datetime import datetime

import pandas as pd

from scripts.modeling.AIS_RF_preselection.AIS_RF_preselection import (
    compute_AIS_RF_alignments_parallel,
)
from scripts.modeling.AIS_RF_alignment.AIS_RF_forward_alignment import (
    compute_forward_score_parallel,
)
from scripts.modeling.AIS_RF_alignment.AIS_RF_nn_baseline import (
    MODELS as NN_MODELS,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build leave-one-out open-set negatives: unbypassed "
            "prefilter + scoring with each RF signal's true AIS track "
            "excluded from its own candidate pool"
        )
    )
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="gaussian",
        help="RF bearing-error model whose data folder to read from and write to (default: gaussian)",
    )
    args = parser.parse_args()

    DATA_DIR = (
        "./data/processed" if args.error_model == "uniform"
        else f"./data/processed/{args.error_model}"
    )
    os.makedirs(DATA_DIR, exist_ok=True)

    df_train = pd.read_pickle(f"{DATA_DIR}/train_data_sample_5000.pkl")

    print("Building leave-one-out candidate pairs (true AIS track excluded)...")
    start_time = datetime.now()

    preselection_df, avg_time_per_iter = compute_AIS_RF_alignments_parallel(
        df_train,
        checkpoint_path=f"{DATA_DIR}/AIS_RF_preselection_leaveoneout_checkpoint.pkl",
        include_true_match=False,
    )
    preselection_df.to_pickle(f"{DATA_DIR}/AIS_RF_preselection_leaveoneout_data.pkl")

    print(f"Leave-one-out candidate pairs: {len(preselection_df)}")
    print(f"Processing took {datetime.now() - start_time} (hh:mm:ss.ms)")
    print(f"Average processing time per RF iteration took {avg_time_per_iter} seconds")

    print("\nScoring leave-one-out candidates with PHMM Forward...")
    start_time = datetime.now()
    forward_df, avg_time_per_iter = compute_forward_score_parallel(
        df_train, preselection_df,
        checkpoint_path=f"{DATA_DIR}/AIS_RF_forward_scores_leaveoneout_checkpoint.pkl",
    )
    forward_df.to_pickle(f"{DATA_DIR}/AIS_RF_forward_scores_leaveoneout_data.pkl")
    print(f"Processing took {datetime.now() - start_time} (hh:mm:ss.ms)")

    for model_name, compute_fn in NN_MODELS.items():
        print(f"\nScoring leave-one-out candidates with NN ({model_name})...")
        start_time = datetime.now()
        score_df, avg_time_per_iter = compute_fn(
            df_train, preselection_df,
            checkpoint_path=f"{DATA_DIR}/AIS_RF_nn_baseline_{model_name}_scores_leaveoneout_checkpoint.pkl",
        )
        score_df.to_pickle(f"{DATA_DIR}/AIS_RF_nn_baseline_{model_name}_scores_leaveoneout_data.pkl")
        print(f"Processing took {datetime.now() - start_time} (hh:mm:ss.ms)")

    print("\nDone. Leave-one-out open-set data written to", DATA_DIR)
