import pandas as pd
from haversine import haversine
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

from scripts.PHMM.viterbi_algorithm.states import *
from scripts.PHMM.viterbi_algorithm.forward import *

def compute_alignments( RF_signal, distance_threshold, metadata):
    states = [ BeginState(), AISState(), RFState(), MState(), EndState()]
    forward_model = PHMM_forward(states, EndState())

    RF_datetime = RF_signal["RF_Timestamp"]
    RF_coordinates = RF_signal["RF"]
    RF_seq_id = RF_signal["UniqueRouteID"]
    RF_signal_id = RF_signal["RFSignalID"]
                
    results = []

    for AIS_seq_id, meta in metadata.items():
        # Use precomputed values
        if not (meta["min_time"] <= RF_datetime <= meta["max_time"]):
            continue
        
        # Fast filter 2: bounding box on lat/lon
        if not (meta["min_lat"] <= RF_coordinates[0] <= meta["max_lat"] and
                meta["min_lon"] <= RF_coordinates[1] <= meta["max_lon"]):
            continue

        in_range = False
        for coord in meta["coords"]:
            if haversine(RF_coordinates, coord) <= distance_threshold:
                in_range = True
                break

        if not in_range:
            continue

        AIS_data = meta["data"]
        AIS_seq = [
            (dt, coord) for dt, coord in zip(AIS_data["AIS_Timestamp"], AIS_data["AIS"])
            if pd.notna(dt) and coord is not None
        ]

        if not AIS_seq:
            continue

        RF_seq = [(RF_datetime, RF_coordinates)]
        forward_score = forward_model.forward(AIS_seq, RF_seq)
        adjusted_score = forward_score / len(AIS_seq)
        is_true_match = RF_seq_id == AIS_seq_id

        results.append({
            "RFSignalID": RF_signal_id,
            "RFRouteID": RF_seq_id,
            "AISRouteID": AIS_seq_id,
            "ForwardScore": forward_score,
            "NormalizedForwardScore": adjusted_score,
            "IsTrueMatch": is_true_match
        })

    return results

def compute_AIS_RF_alignments_parallel(
    df_train_set, 
    distance_threshold = 2.23, 
    time_marge = timedelta(minutes = 10)
    ):
    
    df_RF = df_train_set[df_train_set["RF"].notna()].copy()
    df_RF["RFSignalID"] = df_RF.groupby("UniqueRouteID").cumcount() + 1
    
    RF_list = [RF_signal for _, RF_signal in df_RF.iterrows()]
    
    df_train_set_grouped = df_train_set.groupby("UniqueRouteID")
    
    
    def preprocess_grouped_AIS(df_grouped, time_marge):
        meta = {}
        for route_id, group in df_grouped:
            coords = [coord for coord in group["AIS"] if coord is not None]
            if not coords:
                continue
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords]
            meta[route_id] = {
                "data": group,
                "coords": coords,
                "min_lat": min(lats),
                "max_lat": max(lats),
                "min_lon": min(lons),
                "max_lon": max(lons),
                "min_time": group["AIS_Timestamp"].min() - time_marge,
                "max_time": group["AIS_Timestamp"].max() + time_marge
            }
        return meta
    
    AIS_meta_dict = preprocess_grouped_AIS(df_train_set_grouped, time_marge)
    
    
    
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(compute_alignments, RF_signal, distance_threshold, AIS_meta_dict) 
            for RF_signal in RF_list
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aligning RF signals to AIS sequences"):
            try: 
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error during processing {e}")
                sys.stdout.flush()
            
    flatten_results = [item for sublist in results for item in sublist]
    return pd.DataFrame(flatten_results)
                    
if __name__ == "__main__":
    print("Aligning AIS and RF data...")

    SOURCE_PATH = "../test_scripts/train_set.pkl"
    DESTINATION_PATH = "../test_scripts/alignments_df.pkl"
    
    df = pd.read_pickle(SOURCE_PATH)
    alignments_df = compute_AIS_RF_alignments_parallel(df)
    alignments_df.to_pickle(DESTINATION_PATH)
    
    print("Data is saved to", DESTINATION_PATH)