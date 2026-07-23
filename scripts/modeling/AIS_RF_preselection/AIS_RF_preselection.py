import argparse
import pandas as pd
from haversine import haversine
from datetime import timedelta, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import math
import pickle
import time
import numpy as np

### RUNNING TIME ###
    # Historical baseline: 10:55:55.579386 (hh:mm:ss.ms) on the 5000-track
    # sample dataset, dominated by re-pickling AIS_metadata over IPC for
    # every submitted task. With AIS_metadata now passed once per worker via
    # _init_worker() instead of per task, the same dataset runs in ~10 seconds.

### WORKER-PROCESS GLOBALS ###
# Populated once per worker by _init_worker() so that the (potentially
# large) AIS metadata dict is pickled and sent over IPC exactly once per
# worker process, rather than once per submitted RF signal.
_worker_metadata = None
_worker_distance_threshold = None
_worker_time_window_hours = None


def _init_worker(metadata, distance_threshold, time_window_hours):
    """
    ProcessPoolExecutor initializer: stashes the shared, read-only
    alignment inputs in this worker's globals so compute_alignments()
    can reuse them across every task the worker handles.

    Args:
        metadata (dict): Pre-computed per-track metadata
        distance_threshold (float): Maximum distance (km) to consider an AIS point as a valid match candidate
        time_window_hours (int): Temporal window size in hours for filtering AIS points around the RF timestamp
    """
    global _worker_metadata, _worker_distance_threshold, _worker_time_window_hours
    _worker_metadata = metadata
    _worker_distance_threshold = distance_threshold
    _worker_time_window_hours = time_window_hours


def compute_alignments(RF_signal, distance_threshold=None, metadata=None, time_window_hours=None):
    """
    Compute forward alignment candidates between a single RF signal and many AIS tracks

    The function performs three filters:
    1) time-range check against each AIS track's global min/max (with margin),
    2) spatial check using a precomputed bounding box per AIS track,
    3) precise distance check (Haversine) within a local temporal window around the RF timestamp.

    When run inside a ProcessPoolExecutor initialized with _init_worker(),
    distance_threshold/metadata/time_window_hours can be omitted and are
    read from this worker's globals instead, avoiding re-pickling the
    (shared, unchanged) metadata dict for every RF signal. They can still
    be passed explicitly, e.g. for direct/single-process calls in tests.

    Args:
        RF_signal (pd.Series): A single RF observation
        distance_threshold (float, optional): Maximum distance (km) to consider an AIS point as a valid match candidate
        metadata (dict, optional): Pre-computed per-track metadata
        time_window_hours (int, optional): Temporal window size in hours for filtering AIS points around the RF timestamp. Defaults to 3

    Returns:
        list: Candidate list
    """
    if metadata is None:
        metadata = _worker_metadata
        distance_threshold = _worker_distance_threshold
        time_window_hours = _worker_time_window_hours

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
        times = meta["times"]
        coords = meta["coords"]
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
            "coords": np.array(coords),
            "times": np.array(times, dtype='datetime64[ns]'),
            "min_lat": min_lat - lat_margin,
            "max_lat": max_lat + lat_margin,
            "min_lon": min_lon - lon_margin_min,
            "max_lon": max_lon + lon_margin_max,
            "min_time": group["AIS_Timestamp"].min() - time_marge,
            "max_time": group["AIS_Timestamp"].max() + time_marge
        }
        
    return meta

def _load_checkpoint(checkpoint_path):
    """
    Load previously computed RF-signal alignment results from a checkpoint file

    The checkpoint file holds a sequence of pickled
    ``(RF_key, result, time_iter)`` tuples, one per completed RF
    signal, appended as they finish rather than written all at once.
    This lets a run be resumed after a crash or interruption: RF
    signals whose key is already in the returned dict are skipped.

    Args:
        checkpoint_path (str): Path to the checkpoint file. Does not
            need to exist yet.

    Returns:
        tuple: (dict mapping RF key to its result list, list of
        per-signal processing times already recorded)
    """
    done_results = {}
    times = []
    if not os.path.exists(checkpoint_path):
        return done_results, times

    with open(checkpoint_path, "rb") as file:
        while True:
            try:
                key, result, time_iter = pickle.load(file)
            except EOFError:
                break
            done_results[key] = result
            times.append(time_iter)
    return done_results, times


def compute_AIS_RF_alignments_parallel(
    df,
    distance_threshold = 6,
    time_marge = timedelta(minutes = 10),
    time_window_hours = 3,
    checkpoint_path = None,
):
    """
    Match RF signals to AIS tracks in parallel (per RF observation)

    Args:
        df (pd.DataFrame): Dataset containing both AIS and RF data
        distance_threshold (float): Max matching distance in km used in bbox expansion and precise Haversine checks. Defaults to 6 km
        time_marge (timedelta): Margin appended to the AIS track min/max timestamps. Defaults to 10 minutes
        time_window_hours (int): Temporal window size in hours for filtering AIS points around the RF timestamp. Defaults to 3
        checkpoint_path (str, optional): Path to a checkpoint file used to persist
            each RF signal's result as soon as it completes. If given, an
            interrupted run can be restarted and will only recompute RF
            signals that are not yet in the checkpoint.

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

    done_results, times = (
        _load_checkpoint(checkpoint_path) if checkpoint_path else ({}, [])
    )

    # Skip RF signals that a previous, interrupted run already computed
    pending_RF_list = [
        RF_signal for RF_signal in RF_list
        if (RF_signal["ID"], RF_signal["RF_signal_id"]) not in done_results
    ]
    if len(pending_RF_list) < len(RF_list):
        print(
            f"Resuming from checkpoint: skipping "
            f"{len(RF_list) - len(pending_RF_list)} already-processed RF signals"
        )

    checkpoint_file = open(checkpoint_path, "ab") if checkpoint_path else None
    try:
        # Parallel alignment: submit one task per remaining RF observation.
        # AIS_metadata is handed to each worker once via the initializer
        # instead of being re-pickled on every submit() call.
        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=_init_worker,
            initargs=(AIS_metadata, distance_threshold, time_window_hours),
        ) as executor:
            futures = {
                executor.submit(compute_alignments, RF_signal): (RF_signal["ID"], RF_signal["RF_signal_id"])
                for RF_signal in pending_RF_list
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Aligning RF signals to AIS sequences"):
                key = futures[future]
                try:
                    time_iter, result = future.result()
                except Exception as e:
                    print(f"Error during processing {e}")
                    continue

                done_results[key] = result
                times.append(time_iter)
                if checkpoint_file is not None:
                    pickle.dump((key, result, time_iter), checkpoint_file)
                    checkpoint_file.flush()
    finally:
        if checkpoint_file is not None:
            checkpoint_file.close()

    # Flatten the nested lists and return a dataframe
    flatten_results = [item for result in done_results.values() for item in result]

    avg_time_per_iter = np.array(times).mean()
    result_df = pd.DataFrame(flatten_results)
    if not result_df.empty:
        # Row order otherwise follows worker-completion order, which is
        # non-deterministic across runs and produces spurious diffs on the
        # pickled output even when the underlying matches are unchanged.
        result_df = result_df.sort_values(
            ["RF_track_id", "RF_signal_id", "AIS_track_id"]
        ).reset_index(drop=True)
    return result_df, avg_time_per_iter
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align AIS and RF data")
    parser.add_argument(
        "--error-model", choices=["uniform", "gaussian"], default="uniform",
        help="RF bearing-error model whose data folder to read from and write to (default: uniform)",
    )
    args = parser.parse_args()

    print("Aligning AIS and RF data...")

    start_time = datetime.now()

    # "uniform" keeps the original flat layout for backward compatibility;
    # other error models live in their own subfolder under data/processed
    DATA_DIR = (
        "./data/processed" if args.error_model == "uniform"
        else f"./data/processed/{args.error_model}"
    )
    os.makedirs(DATA_DIR, exist_ok=True)

    SOURCE_PATH = f"{DATA_DIR}/train_data_sample_5000.pkl"
    DESTINATION_PATH = f"{DATA_DIR}/AIS_RF_preselection_data.pkl"
    CHECKPOINT_PATH = f"{DATA_DIR}/AIS_RF_preselection_checkpoint.pkl"

    df = pd.read_pickle(SOURCE_PATH)
    alignments_df, avg_time_per_iter = compute_AIS_RF_alignments_parallel(
        df, checkpoint_path=CHECKPOINT_PATH
    )
    alignments_df.to_pickle(DESTINATION_PATH)
    
    processing_time = datetime.now() - start_time
    
    print("Data is saved to", DESTINATION_PATH)
    print(f"Processing took {processing_time} (hh:mm:ss.ms)")
    print(f"Average processing time per RF iteration took {avg_time_per_iter} seconds")
