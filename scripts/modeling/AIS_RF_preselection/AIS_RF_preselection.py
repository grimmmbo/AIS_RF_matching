import pandas as pd
from haversine import haversine
from datetime import timedelta, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import math
import time
import numpy as np

### RUNNING TIME ###
    # Total processing time 10:55:55.579386 (hh:mm:ss.ms)
    # Average processing time per AIS-RF pair took 0.014387352801028956 seconds

def compute_alignments(RF_signal, distance_threshold, metadata, time_window_hours):
    """
    Compute forward alignment candidates between a single RF signal and many AIS tracks

    The function performs three filters:
    1) time-range check against each AIS track's global min/max (with margin),
    2) spatial check using a precomputed bounding box per AIS track,
    3) precise distance check (Haversine) within a local temporal window around the RF timestamp. 
    
    Args:
        RF_signal (pd.Series): A single RF observation
        distance_threshold (float): Maximum distance (km) to consider an AIS point as a valid match candidate 
        metadata (dict): Pre-computed per-track metadata 
        time_window_hours (int): Temporal window size in hours for filtering AIS points around the RF timestamp. Defaults to 3

    Returns:
        list: Candidate list
    """
    start = time.time()
    # Extract RF information
    RF_datetime = RF_signal["RF_Timestamp"]
    RF_coordinates = RF_signal["RF"]
    RF_track_id = RF_signal["ID"]
    RF_signal_id = RF_signal["RF_signal_id"]
                
    results = []

    # Iterate over all AIS tracks using the precomputed metadata
    for AIS_track_id, meta in metadata.items():
        # Always include the ground-truth pair (same IDs) to ensure it appears in candidates
        if AIS_track_id == RF_track_id:
            is_true_match = True
            results.append({
                "RF_signal_id": RF_signal_id,
                "RF_track_id": RF_track_id,
                "AIS_track_id": AIS_track_id,
                "is_true_match": is_true_match
            })
            continue
        
        # (1) Time-range prefilter:
            # Skip early if RF timestamp lies outside the (min_time, max_time) of the AIS track
        if not (meta["min_time"] <= RF_datetime <= meta["max_time"]):
            continue
        
        # (2) Bounding box check: 
            # Skip if RF point lies outside the expanded lat/lon bounding box of this track
        if not (meta["min_lat"] <= RF_coordinates[0] <= meta["max_lat"] and
                meta["min_lon"] <= RF_coordinates[1] <= meta["max_lon"]):
            continue

        # (3) Precise spatial check within configurable time window
            # Skip if no AIS points fall within the time window or within the distance threshold of the RF point
        times = np.array(meta["times"], dtype='datetime64[ns]')
        coords = np.array(meta["coords"])
        RF_time = np.datetime64(RF_datetime)
        
        time_window = np.timedelta64(time_window_hours, 'h') 
        mask = np.abs(times - RF_time) <= time_window
        filtered_coords = coords[mask]
        
        in_range = any(haversine(RF_coordinates, coord) <= distance_threshold for coord in filtered_coords)
        if not in_range:
            continue

        is_true_match = RF_track_id == AIS_track_id
        results.append({
            "RF_signal_id": RF_signal_id,
            "RF_track_id": RF_track_id,
            "AIS_track_id": AIS_track_id,
            "is_true_match": is_true_match
        })
    end = time.time() - start

    return end, results

def make_AIS_metadata(df_grouped, distance_threshold, time_marge):
    """
    Precompute per-track metadata to accelerate alignment

    Args:
        df_grouped (pd.DataFrame): Grouped AIS data (data of one unique AIS track)
        distance_threshold (float): Maximum distance (in kilometers) to consider an AIS point as a match candidate
        time_marge (timedelta): Time added before and after AIS timestamps

    Returns:
        dict: Metadata for each AIS track 
    """
    meta = {}
    for track_id, group in df_grouped:
        # Collect valid AIS coordinates for this track
        coords = [coord for coord in group["AIS"] if coord is not None]
        if not coords:
            continue
        
        # Collect valid timestamps
        times = [timestamp for timestamp in group["AIS_Timestamp"] if not pd.isnull(timestamp)]
        if not times:
            continue
        
        # Compute raw bounding box
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)
        
        # Expand bounding box by distance_threshold converted from km to degrees
        lat_margin = distance_threshold / 111
        lon_margin_min = distance_threshold / (111 * math.cos(math.radians(min_lat)))
        lon_margin_max = distance_threshold / (111 * math.cos(math.radians(max_lat)))

        meta[track_id] = {
            "coords": coords,
            "times": times, 
            "min_lat": min_lat - lat_margin,
            "max_lat": max_lat + lat_margin,
            "min_lon": min_lon - lon_margin_min,
            "max_lon": max_lon + lon_margin_max,
            "min_time": group["AIS_Timestamp"].min() - time_marge,
            "max_time": group["AIS_Timestamp"].max() + time_marge
        }
        
    return meta

def compute_AIS_RF_alignments_parallel(df, distance_threshold = 6, time_marge = timedelta(minutes = 10), time_window_hours = 3): 
    """
    Match RF signals to AIS tracks in parallel (per RF observation)

    Args:
        df (pd.DataFrame): Dataset containing both AIS and RF data 
        distance_threshold (float): Max matching distance in km used in bbox expansion and precise Haversine checks. Defaults to 6 km
        time_marge (timedelta): Margin appended to the AIS track min/max timestamps. Defaults to 10 minutes
        time_window_hours (int): Temporal window size in hours for filtering AIS points around the RF timestamp. Defaults to 3

    Returns:
        pd.DataFrame: AIS-RF candidate pairs
    """
    # Extract all RF observations (rows where RF coords are present)
    df_RF = df[df["RF"].notna()].copy()
    
    # Give each RF signal a sequential identifier within its track
    # Added because this study evaluates alignments per individual RF signal, rather than per full RF sequence
    df_RF["RF_signal_id"] = df_RF.groupby("ID").cumcount() + 1
    RF_list = [RF_signal for _, RF_signal in df_RF.iterrows()]
    
    # Group AIS tracks by ID and precompute metadata
    df_grouped = df.groupby("ID")
    AIS_metadata = make_AIS_metadata(df_grouped, distance_threshold, time_marge)
    
    results = []
    times = []
    
    # Parallel alignment: submit one task per RF observation
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(compute_alignments, RF_signal, distance_threshold, AIS_metadata, time_window_hours) 
            for RF_signal in RF_list
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aligning RF signals to AIS sequences"):
            try: 
                time_iter, result = future.result()
                results.append(result)
                times.append(time_iter)
            except Exception as e:
                print(f"Error during processing {e}")
                
    # Flatten the nested lists and return a dataframe
    flatten_results = [item for sublist in results for item in sublist]
    
    avg_time_per_iter = np.array(times).mean()
    return pd.DataFrame(flatten_results), avg_time_per_iter
                    
if __name__ == "__main__":
    print("Aligning AIS and RF data...")

    start_time = datetime.now()
    
    SOURCE_PATH = "./AIS_RF_matching/data/processed/train_data_sample_5000.pkl" 
    DESTINATION_PATH = "./AIS_RF_matching/data/processed/AIS_RF_preselection_data.pkl"
    
    df = pd.read_pickle(SOURCE_PATH)
    alignments_df, avg_time_per_iter = compute_AIS_RF_alignments_parallel(df)
    alignments_df.to_pickle(DESTINATION_PATH)
    
    processing_time = datetime.now() - start_time
    
    print("Data is saved to", DESTINATION_PATH)
    print(f"Processing took {processing_time} (hh:mm:ss.ms)")
    print(f"Average processing time per RF iteration took {avg_time_per_iter} seconds")
