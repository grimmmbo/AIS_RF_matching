import pandas as pd
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from scripts.PHMM.viterbi_algorithm.states import *
from scripts.PHMM.viterbi_algorithm.forward import *


def compute_alignments(AIS_seq, RF_seq, RF_signal_id, RF_track_id, AIS_track_id, alpha=0.9):
    """
    Compute the forward alignment score between a single RF signal and an AIS sequence
    """
    states = [BeginState(), AISState(), RFState(), MState(), EndState()]
    forward_model = PHMM_forward(states, EndState())

    forward_score = forward_model.forward(AIS_seq, RF_seq)
    adjusted_score = forward_score / len(AIS_seq)
    adjusted_score_alpha = forward_score / (len(AIS_seq) ** alpha)

    is_true_match = RF_track_id == AIS_track_id

    return {
        "RF_signal_id": RF_signal_id,
        "RF_track_id": RF_track_id,
        "AIS_track_id": AIS_track_id,
        "forward_score": forward_score,
        "normalized_forward_score": adjusted_score,
        "normalized_forward_score_alpha": adjusted_score_alpha,
        "is_true_match": is_true_match
    }


def compute_forward_score_parallel(df_train, df_preselection):
    """
    Matches RF signals to AIS tracks using the forward algorithm, in parallel.
    Returns a DataFrame with alignment results.
    """
    results = []

    # Filter alleen RF-rijen
    df_RF = df_train[df_train["RF"].notna()].copy()
    df_RF["RF_signal_id"] = df_RF.groupby("ID").cumcount() + 1

    print("Preparing alignment tasks...")
    start_prep = time.time()

    # ✨ Maak snelle lookup dictionaries
    ais_dict = {track_id: group for track_id, group in df_train.groupby("ID")}
    rf_dict = {(row["ID"], row["RF_signal_id"]): row for _, row in df_RF.iterrows()}

    tasks = []
    for row in tqdm(df_preselection.itertuples(index=False), total=len(df_preselection), desc="Building task list"):
        try:
            rf_row = rf_dict.get((row.RF_track_id, row.RF_signal_id))
            ais_data = ais_dict.get(row.AIS_track_id)

            if rf_row is None or ais_data is None:
                continue

            RF_datetime = rf_row["RF_Timestamp"]
            RF_coordinates = rf_row["RF"]

            AIS_seq = [
                (dt, coord) for dt, coord in zip(ais_data["AIS_Timestamp"], ais_data["AIS"])
                if pd.notna(dt) and coord is not None
            ]

            RF_seq = [(RF_datetime, RF_coordinates)]
            tasks.append((AIS_seq, RF_seq, row.RF_signal_id, row.RF_track_id, row.AIS_track_id))

        except Exception as e:
            print(f"Error building task: {e}")

    end_prep = time.time()
    print(f"Task preparation completed in {end_prep - start_prep:.2f} seconds.\n")

    print("Starting parallel alignment...")
    start_align = time.time()

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(compute_alignments, AIS_seq, RF_seq, RF_signal_id, RF_track_id, AIS_track_id)
            for (AIS_seq, RF_seq, RF_signal_id, RF_track_id, AIS_track_id) in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Aligning RF to AIS", unit="task"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error during processing: {e}")

    end_align = time.time()
    print(f"\nParallel alignment completed in {end_align - start_align:.2f} seconds.")

    return pd.DataFrame(results)


# === MAIN SCRIPT ===
if __name__ == "__main__":
    print("Aligning AIS and RF data...")

    SOURCE_PATH1 = "../test_scripts/train_data_sample_5000.pkl"
    SOURCE_PATH2 = "../test_scripts/AIS_RF_preselection_df.pkl"
    DESTINATION_PATH = "../test_scripts/forward_scores_origineel_df.pkl"

    df_train = pd.read_pickle(SOURCE_PATH1)
    df_preselection = pd.read_pickle(SOURCE_PATH2)

    forward_score_df = compute_forward_score_parallel(df_train, df_preselection)
    forward_score_df.to_pickle(DESTINATION_PATH)

    print("Data saved to", DESTINATION_PATH)
