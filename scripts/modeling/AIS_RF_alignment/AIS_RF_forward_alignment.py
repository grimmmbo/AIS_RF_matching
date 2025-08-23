
import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

from scripts.modeling.hidden_states.states import *
from scripts.modeling.AIS_RF_alignment.forward import *

### RUNNING TIME ###
    # Total processing time 0:21:36.276858 (hh:mm:ss.ms)
    # Average processing time per forward alignment took 0.547953668199831 seconds

def compute_forward_scores(AIS_seq, RF_seq, RF_signal_id, RF_track_id, AIS_track_id):
    """
    Compute the forward alignment score between a single RF signal and an AIS sequence
    
    Args: 
        AIS_seq (list[tuple]): AIS sequence as (timestamp, coordinate) pairs
        RF_seq (list[tuple]): RF sequence as (timestamp, coordinate) 
        RF_signal_id (int): Unique identifier for the RF signals within its track
        RF_track_id (int): Identifier of the RF track
        AIS_track_id (int): Identifier of the AIS track
    
    Returns: 
        dict: Forward scores for a given RF signal and AIS sequence
    """
    start = time.time()
    # Define the PHMM states used for alignment
    states = [BeginState(), AISState(), RFState(), MState(), EndState()]
    forward_model = PHMM_forward(states, EndState())

    # Run the forward algorithm to compute the raw alignment score
    forward_score = forward_model.forward(AIS_seq, RF_seq)
    
    # Normalize the score to account for AIS sequence length
    adjusted_score = forward_score / len(AIS_seq)

    # Ground truth track
    is_true_match = RF_track_id == AIS_track_id
    
    end = time.time() - start

    return end, {
        "RF_signal_id": RF_signal_id,
        "RF_track_id": RF_track_id,
        "AIS_track_id": AIS_track_id,
        "forward_score": forward_score,
        "normalized_forward_score": adjusted_score,
        "is_true_match": is_true_match
    }
    
    
def compute_forward_score_parallel(df_train, df_preselection):
    """
    Match RF signals to AIS tracks using the Forward algorithm in parallel.
    
    Args: 
        df_train (pd.DataFrame): Dataset containing AIS and RF sequences 
        df_preselection (pd.DataFrame): Candidate AIS-RF pairs 
        
    Returns: 
        pd.DataFrame: Forward scores per AIS-RF pair
    """
    # Extract all RF observations (rows where RF coords are present)
    df_RF = df_train[df_train["RF"].notna()].copy()
    
    # Give each RF signal a sequential identifier within its track
    # Added because this study evaluates alignments per individual RF signal, rather than per full RF sequence
    df_RF["RF_signal_id"] = df_RF.groupby("ID").cumcount() + 1


    # Build fast lookup dictionaries for AIS and RF data
    ais_dict = {track_id: group for track_id, group in df_train.groupby("ID")}
    rf_dict = {(row["ID"], row["RF_signal_id"]): row for _, row in df_RF.iterrows()}

    tasks = []
    results = []
    
    # Build task list with AIS and RF sequences for alignment
    for row in tqdm(df_preselection.itertuples(index=False), total=len(df_preselection), desc="Building task list"):
        try:
            rf_row = rf_dict.get((row.RF_track_id, row.RF_signal_id))
            ais_data = ais_dict.get(row.AIS_track_id)

            if rf_row is None or ais_data is None:
                continue

            # Extract RF signal as a sequence (single timestamp–coordinate pair)
            RF_datetime = rf_row["RF_Timestamp"]
            RF_coordinates = rf_row["RF"]
            RF_seq = [(RF_datetime, RF_coordinates)]

            # Extract AIS sequence (list of timestamp–coordinate pairs)
            AIS_seq = [
                (dt, coord) for dt, coord in zip(ais_data["AIS_Timestamp"], ais_data["AIS"])
                if pd.notna(dt) and coord is not None
            ]

            tasks.append((AIS_seq, RF_seq, row.RF_signal_id, row.RF_track_id, row.AIS_track_id))

        except Exception as e:
            print(f"Error building task: {e}")

    times = []
    
    # Parallel alignment: one task per AIS-RF pair
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(compute_forward_scores, AIS_seq, RF_seq, RF_signal_id, RF_track_id, AIS_track_id)
            for (AIS_seq, RF_seq, RF_signal_id, RF_track_id, AIS_track_id) in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating Forward score for AIS-RF pair", unit="task"):
            try:
                time_iter, result = future.result()
                results.append(result)
                times.append(time_iter)
            except Exception as e:
                print(f"Error during processing: {e}")
    print(len(times), min(times), max(times))
    avg_time_per_iter = np.array(times).mean()
    return pd.DataFrame(results), avg_time_per_iter


# === MAIN SCRIPT ===
if __name__ == "__main__":
    print("Calculating Forward score for AIS-RF pair...")
    
    start_time = datetime.now()

    SOURCE_PATH1 = "./AIS_RF_matching/data/processed/train_data_sample_5000.pkl"
    SOURCE_PATH2 = "./AIS_RF_matching/data/processed/AIS_RF_preselection_data.pkl" 
    DESTINATION_PATH = "./AIS_RF_matching/data/processed/AIS_RF_forward_scores_data.pkl" 

    df_train = pd.read_pickle(SOURCE_PATH1)
    df_preselection = pd.read_pickle(SOURCE_PATH2)

    forward_score_df, avg_time_per_iter = compute_forward_score_parallel(df_train, df_preselection)
    forward_score_df.to_pickle(DESTINATION_PATH)
    
    processing_time = datetime.now() - start_time

    print("Data saved to", DESTINATION_PATH)
    print(f"Processing took {processing_time} (hh:mm:ss.ms)")
    print(f"Average processing time per forward alignment took {avg_time_per_iter} seconds")
