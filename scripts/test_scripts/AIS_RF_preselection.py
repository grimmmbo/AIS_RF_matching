import pandas as pd
from haversine import haversine
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import math

def compute_alignments(RF_signal, distance_threshold, metadata):
    """
    Compute the forward alignment score between a single RF signal and multiple AIS sequences

    Args:
        RF_signal (pd.Series): A single RF observation
        distance_threshold (float): Maximum distance (in kilometers) to consider an AIS point as a match candidate
        metadata (dict): Metadata per AIS track

    Returns:
        list: Alignment results for the given RF signal with all relevant AIS tracks
    """

    # Extract RF information
    RF_datetime = RF_signal["RF_Timestamp"]
    RF_coordinates = RF_signal["RF"]
    RF_track_id = RF_signal["ID"]
    RF_signal_id = RF_signal["RF_signal_id"]
                
    results = []

    for AIS_track_id, meta in metadata.items():
        # Check whether the is a true match (same track ids) --> always evaluate the true match
        if AIS_track_id == RF_track_id:
            
            is_true_match = RF_track_id == AIS_track_id

            # Save results
            results.append({
                "RF_signal_id": RF_signal_id,
                "RF_track_id": RF_track_id,
                "AIS_track_id": AIS_track_id,
                "is_true_match": is_true_match
            })
            
            continue
        
        # Step 1: Checks whether the timestamp of RF signal falls within the time range of the AIS track, with a small margin (defaults to 10 minutes) added before and after
        if not (meta["min_time"] <= RF_datetime <= meta["max_time"]):
            continue
        
        # Step 2, rough proximity check: 
        # Checks whether the location of the RF signal is within a bounding box (with a small margin of 2.23 km) that surrounds the AIS track 
        if not (meta["min_lat"] <= RF_coordinates[0] <= meta["max_lat"] and
                meta["min_lon"] <= RF_coordinates[1] <= meta["max_lon"]):
            continue

        # Step 3, more detailed check: 
        # Only if the RF point passes both checks, the precise haversine distance is computed to determine whether it is within the allowed range (2,23 km) of any AIS point in the sequence
        time_window = timedelta(hours=3)  # aangepast
        in_range = False
        for time, coord in zip(meta["times"],meta["coords"]): # aangepast
            if abs(time - RF_datetime) <= time_window:  #aangepast
                if haversine(RF_coordinates, coord) <= distance_threshold:
                    in_range = True
                    break
        if not in_range:
            continue
        
        is_true_match = RF_track_id == AIS_track_id

        # Save results
        results.append({
            "RF_signal_id": RF_signal_id,
            "RF_track_id": RF_track_id,
            "AIS_track_id": AIS_track_id,
            "is_true_match": is_true_match
        })

    return results

def make_AIS_metadata(df_grouped, distance_threshold, time_marge):
    """
    Precomputes metadata for each AIS track

    Args:
        df_grouped (pd.DataFrame): Grouped AIS data (data of one unique AIS track)
        distance_threshold (float): Maximum distance (in kilometers) to consider an AIS point as a match candidate
        time_marge (timedelta): Time added before and after AIS timestamps

    Returns:
        _type_: Metadata for each AIS track 
    """
    meta = {}
    for track_id, group in df_grouped:
        # Calculate the geographical bounding box for a single AIS track 
        coords = [coord for coord in group["AIS"] if coord is not None]
        if not coords:
            continue
        
        # aangepast
        times = [time for time in group["AIS_Timestamp"] if not pd.isnull(time)]
        if not times:
            continue
        
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        
        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)
        
        # The bounding box is expanded by a small margin (based on the distance threshold in kilometers)
        lat_margin = distance_threshold / 111
        lon_margin_min = distance_threshold / (111 * math.cos(math.radians(min_lat)))
        lon_margin_max = distance_threshold / (111 * math.cos(math.radians(max_lat)))

        # Store all precomputed metadata
        meta[track_id] = {
            "data": group,
            "coords": coords,
            "times": times, # aangepast
            "min_lat": min_lat - lat_margin,
            "max_lat": max_lat + lat_margin,
            "min_lon": min_lon - lon_margin_min,
            "max_lon": max_lon + lon_margin_max,
            "min_time": group["AIS_Timestamp"].min() - time_marge,
            "max_time": group["AIS_Timestamp"].max() + time_marge
        }
    return meta

def compute_AIS_RF_alignments_parallel(df, distance_threshold = 6, time_marge = timedelta(minutes = 10)): #aangepast van 2.380675025323449 naar 6
    """
    Matches RF signals to AIS tracks in parallel

    Args:
        df (pd.DataFrame): Dataframe with AIS and RF sequence
        distance_threshold (float): Maximum distance (in kilometers) to consider an AIS point as a match candidate
        time_marge (timedelta): Time added before and after AIS timestamps

    Returns:
        pd.DataFrame: Alignment results
    """
    # Extract all RF observations
    df_RF = df[df["RF"].notna()].copy()
    df_RF["RF_signal_id"] = df_RF.groupby("ID").cumcount() + 1
    RF_list = [RF_signal for _, RF_signal in df_RF.iterrows()]
    
    # Group AIS tracks by ID and precompute metadata
    df_grouped = df.groupby("ID")
    AIS_metadata = make_AIS_metadata(df_grouped, distance_threshold, time_marge)
    
    results = []
    
    # Run alignment in parallel using all cores
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(compute_alignments, RF_signal, distance_threshold, AIS_metadata) 
            for RF_signal in RF_list
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aligning RF signals to AIS sequences"):
            try: 
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error during processing {e}")
                
    # Flatten the list of results        
    flatten_results = [item for sublist in results for item in sublist]
    
    return pd.DataFrame(flatten_results)
                    
if __name__ == "__main__":
    print("Aligning AIS and RF data...")

    SOURCE_PATH = "../test_scripts/train_data_sample_5000.pkl"
    DESTINATION_PATH = "../test_scripts/AIS_RF_preselection_df_time_window_6km.pkl"
    
    df = pd.read_pickle(SOURCE_PATH)
    alignments_df = compute_AIS_RF_alignments_parallel(df)
    alignments_df.to_pickle(DESTINATION_PATH)
    
    print("Data is saved to", DESTINATION_PATH)