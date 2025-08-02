import pandas as pd
from haversine import haversine
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import math

from scripts.PHMM.viterbi_algorithm.states import *
from scripts.PHMM.viterbi_algorithm.forward import *

def compute_forward_scores(RF_data, AIS_data, RF_signal_id, RF_track_id, AIS_track_id, alpha=0.9):
    """
    Compute the forward score between a single RF signal and multiple AIS sequences

    Args:
        RF_signal (pd.Series): A single RF observation
        distance_threshold (float): Maximum distance (in kilometers) to consider an AIS point as a match candidate
        metadata (dict): Metadata per AIS track

    Returns:
        list: Alignment results for the given RF signal with all relevant AIS tracks
    """
    # Intialize PHMM with defined states
    states = [ BeginState(), AISState(), RFState(), MState(), EndState()]
    forward_model = PHMM_forward(states, EndState())

    # RF_data = df_RF[(df_RF["RF_signal_id"] == RF_signal_id) & (df_RF["ID"] == RF_track_id)]
    RF_datetime = RF_data["RF_Timestamp"].values[0]
    RF_coordinates = RF_data["RF"].values[0]
    
    # AIS_data = df_train[df_train["ID"] == AIS_track_id]
    
    AIS_seq = [
        (dt, coord) for dt, coord in zip(AIS_data["AIS_Timestamp"], AIS_data["AIS"])
        if pd.notna(dt) and coord is not None]
    RF_seq = [(RF_datetime, RF_coordinates)]
        
    # Calculate forward score
    forward_score = forward_model.forward(AIS_seq, RF_seq)
    adjusted_score = forward_score / len(AIS_seq)
    adjusted_score_alpha = forward_score / (len(AIS_seq)**alpha)
    
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
    Matches RF signals to AIS tracks using forward algorithm, in parallel

    Args:
        df (pd.DataFrame): Dataframe with AIS and RF sequence
        distance_threshold (float): Maximum distance (in kilometers) to consider an AIS point as a match candidate
        time_marge (timedelta): Time added before and after AIS timestamps

    Returns:
        pd.DataFrame: Alignment results with forward scores
    """
    
    results = []
    
    df_RF = df_train[df_train["RF"].notna()].copy()
    df_RF["RF_signal_id"] = df_RF.groupby("ID").cumcount() + 1 
    
    
    # In main thread, bouw tuples per taak met alleen de benodigde data
    tasks = []
    for row in tqdm(df_preselection.itertuples(index=False), total=len(df_preselection), desc="Building task list"):
        RF_data = df_RF[(df_RF["RF_signal_id"] == row.RF_signal_id) & (df_RF["ID"] == row.RF_track_id)]
        AIS_data = df_train[df_train["ID"] == row.AIS_track_id]
        tasks.append((RF_data, AIS_data, row.RF_signal_id, row.RF_track_id, row.AIS_track_id))
    
    # Run alignment in parallel using all cores
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(compute_forward_scores, RF_data, AIS_data, RF_signal_id, RF_track_id, AIS_track_id)
            for (RF_data, AIS_data, RF_signal_id, RF_track_id, AIS_track_id) in tasks
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aligning RF signals to AIS sequences"):
            try: 
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error during processing {e}")
                sys.stdout.flush()
                
    # # Flatten the list of results        
    # flatten_results = [item for sublist in results for item in sublist]
    
    return pd.DataFrame(results)
                    
if __name__ == "__main__":
    print("Calculating the Forward score for AIS and RF data...")

    SOURCE_PATH1 = "../test_scripts/train_data_sample_5000.pkl"
    SOURCE_PATH2 = "../test_scripts/AIS_RF_preselection_df.pkl"
    DESTINATION_PATH = "../test_scripts/forward_scores_df.pkl"
    
    df_train = pd.read_pickle(SOURCE_PATH1)
    df_preselection = pd.read_pickle(SOURCE_PATH2)
    forward_score_df = compute_forward_score_parallel(df_train, df_preselection)
    forward_score_df.to_pickle(DESTINATION_PATH)
    
    print("Data is saved to", DESTINATION_PATH)