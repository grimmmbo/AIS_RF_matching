import random 
import pandas as pd

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
    DESTINATION_PATH = "data/processed/modified_AIS_data.parquet"
    
    df = pd.read_parquet(SOURCE_PATH)
    AIS_df = modify_AIS_data(df)
    AIS_df.to_parquet(DESTINATION_PATH)
    
    print("Data is saved...")