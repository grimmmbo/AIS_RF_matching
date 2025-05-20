import random 
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
    df["speed_kmph"] = df['distance_km'] / df['TimeDiff'] * 3600
    
    # Calculate average speed for each vessel
    df_avg_speeds = df.groupby(["MMSI", 'SubTrackID'])['speed_kmph'].mean().reset_index()
    
    # Filter vessels based on speed
    df_filterd = df_avg_speeds[(df_avg_speeds['speed_kmph'] >= min_speed) & (df_avg_speeds['speed_kmph'] <= max_speed)]
    
    return df_filterd

def modify_AIS_data(df, choices=[1, 2, 3], probabilities=[0.5, 0.25, 0.25]):
    """
    Modifies AIS data by randomly removing blocks of data points from each vessel's journey to simulate missing data

    Args:
    df (pd.DataFrame): AIS data
    choices (list): Options for how many blocks to remove per vessel, defaults to [1, 2, 3]
    probabilities (list): The probabilities associated with each choice, defaults to [0.5, 0.25, 0.25]

    Returns:
        pd.DataFrame: AIS data with simulated data loss 
    """
    updated_dfs = []
    
    # Sort the data and reset index for proper sequential processing
    df = df.sort_values(["MMSI", "SubTrackID", "BaseDateTime"]).reset_index(drop=True)
    
    for vessel, vessel_data in df.groupby(["MMSI", "SubTrackID"]):
        length_of_journey = len(vessel_data)

        # Determine how many blocks to remove
        num_blocks = random.choices(choices, weights=probabilities)[0]
        points_to_remove = max(1, round(0.1 * length_of_journey))
        max_position = length_of_journey - (points_to_remove + 1)

        if max_position > 10:
            # Select starting positions for blocks
            block_positions = sorted(random.sample(range(max_position), num_blocks))
            block_ranges = [(pos, pos + points_to_remove) for pos in block_positions]

            # Remove rows for the specified ranges
            indices_to_remove = []
            for start, end in block_ranges:
                indices_to_remove.extend(vessel_data.iloc[start:end].index.tolist())
            
            vessel_data_cleaned = vessel_data.drop(index=indices_to_remove)
        else:
            # If max_position <= 10, do not remove any data, leave the data intact
            vessel_data_cleaned = vessel_data
        
        # Append the cleaned data (whether modified or not) to the list
        updated_dfs.append(vessel_data_cleaned)

    # Combine all cleaned vessel data
    manipulated_df = pd.concat(updated_dfs).reset_index(drop=True)
    
    return manipulated_df

if __name__ == "__main__":
    print("Modifying AIS data to simulate missing data has started...")
        
    SOURCE_PATH = "../../../../data/processed/segmented_tracks.parquet"
    DESTINATION_PATH = "../../../../data/processed/modified_AIS_data.parquet"
    
    df = pd.read_parquet(SOURCE_PATH)
    df_filtered = filter_by_speed(df)
    AIS_df = modify_AIS_data(df_filtered)
    AIS_df.to_parquet(DESTINATION_PATH)
    
    print("Data is saved...")