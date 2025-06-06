import pandas as pd
from haversine import haversine, Unit
import numpy as np

def compute_distance(row):
    """
    Computes the distance between two geographical points using the Haversine formula

    Args:
        row (pd.Series): A row of the DataFrame containing latitude and longitude of two points

    Returns:
        float: Distance in kilometers
    """
    if pd.notnull(row['prev_LAT']) and pd.notnull(row['prev_LON']):
        coord_prev = (row['prev_LAT'], row['prev_LON'])
        coord_curr = (row['LAT'], row['LON'])
        return haversine(coord_prev, coord_curr, unit=Unit.KILOMETERS)
    else:
        return np.nan

def filter_by_speed(df, min_speed = 1, max_speed = 60):
    """
    Filters the AIS data to include only vessels with a speed between min_speed and max_speed

    Args:
        df (pd.DataFrame): AIS data
        min_speed (float): Minimum speed threshold, defaults to 1
        max_speed (float): Maximum speed threshold, defaults to 60

    """
    # Sort the data and reset index for proper sequential processing
    df = df.sort_values(by=["MMSI", "SubTrackID", "BaseDateTime"]).copy()
    
    # Calculate distance between consecutive points
    df["prev_LAT"] = df.groupby(["MMSI", "SubTrackID"])["LAT"].shift(1)
    df["prev_LON"] = df.groupby(["MMSI", "SubTrackID"])["LON"].shift(1)
    df['distance_km'] = df.apply(compute_distance, axis=1)
    
    # Convert time difference to seconds
    df["TimeDiff"] = df["TimeDiff"].dt.total_seconds()
    
    # Calculate speed
    df["speed_kmph"] = 0.0
    valid_rows = (df["TimeDiff"] > 0) & (df["distance_km"] > 0)
    df.loc[valid_rows, "speed_kmph"] = (df.loc[valid_rows, 'distance_km'] / df.loc[valid_rows, 'TimeDiff']) * 3600 
    
    # Calculate average speed for each vessel
    df_avg_speeds = df.groupby(["MMSI", 'SubTrackID'])['speed_kmph'].mean().reset_index()
    
    # Filter vessels based on speed
    valid_tracks = df_avg_speeds[(df_avg_speeds['speed_kmph'] >= min_speed) & (df_avg_speeds['speed_kmph'] <= max_speed)][["MMSI", 'SubTrackID']]
    
    # Merge to get original data
    df_filterd = df.merge(valid_tracks, on=["MMSI", 'SubTrackID'], how='inner')
    
    return df_filterd

if __name__ == "__main__":
    print("Filtering AIS data based on average vessel speed...")
        
    SOURCE_PATH = "../../../../data/processed/segmented_tracks.parquet"
    DESTINATION_PATH = "../../../../data/processed/filtered_AIS_by_speed.parquet"
    
    df = pd.read_parquet(SOURCE_PATH)
    df_filtered = filter_by_speed(df)
    df_filtered.to_parquet(DESTINATION_PATH)
    
    print("Data is saved...")