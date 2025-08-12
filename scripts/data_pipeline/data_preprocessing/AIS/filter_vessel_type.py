import pickle
import pandas as pd
import json
import os

def filter_on_vessel_type(data_path, json_path, save_path, vessel_name = "Cargo"):
    """
    Filter AIS data by vessel type using a vessel type mapping JSON
    
    Args:
        data_path (str): Path to the pickle file containing AIS data
        json_path (str): Path to the JSON file that maps vessel types to IDs
        vessel_name (str): The name of the vessel type to filter by, defaults to "Cargo"
    """
    # Extract the downloaded AIS data from a pickle file
    with open(data_path, 'rb') as file:  
        df = pickle.load(file)
    
    # Convert 'BaseDateTime' to datetime format
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    
    # Filter the AIS data to keep only rows with chosen vessel types
    # https://documentation.spire.com/ais-fundamentals/ship-type-mappings/
    with open(json_path, "r") as file:
        vessel_type_names = json.load(file)
    
    vessel_type_names = {int(k): v for k, v in vessel_type_names.items()}
    filtered_vessel_type_ids = [k for k, v in vessel_type_names.items() if v == vessel_name]
    
    filtered_vessel_df = df[df["VesselType"].isin(filtered_vessel_type_ids)].copy()
    
    # Save the filtered data to a .parquet file
    filtered_vessel_df.to_parquet(save_path)
    
if __name__ == "__main__":
    print("Filtering has started...")
    
    vessel_name = "Cargo"
    
    SOURCE_DATA_PATH = "../../../../data/raw/AIS_01_2024.pkl"
    SOURCE_JSON_PATH = "../../../../config/mappings/vessel_type_names.json"
    DESTINATION_PATH = f"../../../../data/processed/{vessel_name.lower()}_vessels.parquet"

    filter_on_vessel_type(SOURCE_DATA_PATH, SOURCE_JSON_PATH, DESTINATION_PATH, vessel_name)
    
    print("Data is saved...")
    