import pandas as pd 

from scripts.PHMM.viterbi_algorithm.viterbi import *
from scripts.PHMM.viterbi_algorithm.states import *

def make_dataframe(df):
    # Get best predictions per RF signal
    df_best_preds = df.loc[
        df.groupby(["RF_signal_id", "RF_track_id"])["normalized_forward_score_alpha"].idxmax()
    ]

    # Select all incorrectly predicted matches (i.e., mismatches)
    mismatches = df_best_preds[df_best_preds["is_true_match"] == False]
    mismatch_keys = set(zip(mismatches['RF_track_id'], mismatches['RF_signal_id']))

    # From all forward results, find the correct (true) matches for the same RF signals that were incorrectly matched
    true_matches = df[
        (df['is_true_match'] == True) &
        df.apply(lambda row: (row['RF_track_id'], row['RF_signal_id']) in mismatch_keys, axis=1)
    ]

    # Merge to get both the best prediction and the ground truth per RF signal
    df_experiment_one = mismatches.merge(
        true_matches,
        on = ["RF_signal_id", "RF_track_id"],
        suffixes=("_pred", "_true")
    )
    
    return df_experiment_one
    
def get_AIS_seq(df_grouped, track_id):
    AIS_data = df_grouped.get_group(track_id)
    return [
        (dt, coord) for dt, coord in zip(AIS_data["AIS_Timestamp"], AIS_data["AIS"])
        if pd.notna(dt) and coord is not None
    ]
    
def get_closest_AIS_in_time(AIS_seq, RF_datetime):
    return min(AIS_seq, key=lambda x: abs((x[0] - RF_datetime).total_seconds()))
    
def process_experiment_one(df, df_subset):
    # Intialize PHMM with defined states
    states = [ BeginState(), AISState(), RFState(), MState(), EndState()]
    viterbi_model = PHMM_viterbi(states, EndState())

    # Group the training data so that we can quickly retrieve sequences per route
    df_grouped = df_subset.groupby("ID")

    df_RF = df_subset[df_subset["RF"].notna()].copy()
    df_RF["RF_signal_id"] = df_RF.groupby("ID").cumcount() + 1
    
    # Get dataframe needed for experiment one
    df_experiment_one = make_dataframe(df)

    results = []
    
    for idx, (_, row) in enumerate(df_experiment_one.iterrows(), 1):
        RF_signal_id = row["RF_signal_id"]
        RF_track_id = row["RF_track_id"]
        
        RF_signal_info = df_RF[(df_RF["RF_signal_id"] == RF_signal_id) & (df_RF["ID"] == RF_track_id)].iloc[0]
        RF_datetime = RF_signal_info["RF_Timestamp"]
        RF_coordinates = RF_signal_info["RF"]
        RF_seq = [(RF_datetime, RF_coordinates)]
        
        AIS_seq_pred = get_AIS_seq(df_grouped, row["AIS_track_id_pred"])
        AIS_seq_true = get_AIS_seq(df_grouped, row["AIS_track_id_true"])
        
        # Find nearest AIS point in time (may be before or after RF)
        pred_dt_nearest, pred_coord_nearest = get_closest_AIS_in_time(AIS_seq_pred, RF_datetime)
        true_dt_nearest, true_coord_nearest = get_closest_AIS_in_time(AIS_seq_true, RF_datetime)
        
        # Calculate difference in time and distance
        distance_pred = haversine(RF_coordinates, pred_coord_nearest)
        distance_true = haversine(RF_coordinates, true_coord_nearest)

        time_diff_pred = abs((RF_datetime - pred_dt_nearest).total_seconds())
        time_diff_true = abs((RF_datetime - true_dt_nearest).total_seconds()) 
        
        # Calculate viterbi score
        path_pred, viterbi_score_pred, log_probs_pred = viterbi_model.viterbi(AIS_seq_pred, RF_seq)
        adjusted_score_pred = viterbi_score_pred / (len(AIS_seq_pred) * 1.5) 
        path_true, viterbi_score_true, log_probs_true = viterbi_model.viterbi(AIS_seq_true, RF_seq)
        adjusted_score_true = viterbi_score_true / (len(AIS_seq_true) * 1.5)  

        # Save results
        results.append({
            "RF_signal_id": RF_signal_id,
            "RF_track_id": RF_track_id,
            "AIS_track_id_pred": row["AIS_track_id_pred"],
            "norm_forward_pred": row["normalized_forward_score_alpha_pred"],
            "norm_viterbi_pred": adjusted_score_pred,
            "distance_diff_pred": distance_pred,
            "time_diff_pred": time_diff_pred,
            "path_pred": path_pred,
            "AIS_track_id_true": row["AIS_track_id_true"],
            "norm_forward_true": row["normalized_forward_score_alpha_true"],
            "norm_viterbi_true": adjusted_score_true,
            "distance_diff_true": distance_true,
            "time_diff_true": time_diff_true,
            "path_true": path_true,
        })
        
    return pd.DataFrame(results) 
