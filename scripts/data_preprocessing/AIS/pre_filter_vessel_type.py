import pickle
import pandas as pd
import json

def filter_on_vessel_type(data_path, json_path, save_path, vessel_name = "Cargo"):
    """
    Filter AIS data by vessel type using a vessel type mapping JSON

    Args:
        data_path (str): Path to the pickle file containing AIS data.
            The file holds a sequence of per-day DataFrames (one
            pickle.dump per day, as written by download_AIS_data.py),
            not a single combined DataFrame.
        json_path (str): Path to the JSON file that maps vessel types to IDs
        vessel_name (str): The name of the vessel type to filter by, defaults to "Cargo"
    """
    # Filter the AIS data to keep only rows with chosen vessel types
    # https://documentation.spire.com/ais-fundamentals/ship-type-mappings/
    with open(json_path, "r") as file:
        vessel_type_names = json.load(file)

    vessel_type_names = {int(k): v for k, v in vessel_type_names.items()}
    filtered_vessel_type_ids = [k for k, v in vessel_type_names.items() if v == vessel_name]

    # Read and filter one day at a time, so the full unfiltered month
    # is never held in memory at once - only the (much smaller)
    # filtered result accumulates across days.
    filtered_days = []
    with open(data_path, 'rb') as file:
        while True:
            try:
                day_df = pickle.load(file)
            except EOFError:
                break
            day_df["BaseDateTime"] = pd.to_datetime(day_df["BaseDateTime"])
            filtered_days.append(
                day_df[day_df["VesselType"].isin(filtered_vessel_type_ids)].copy()
            )

    filtered_vessel_df = pd.concat(filtered_days, ignore_index=True)

    # Save the filtered data to a .parquet file
    filtered_vessel_df.to_parquet(save_path)
    
if __name__ == "__main__":
    print("Filtering has started...")
    
    vessel_name = "Cargo"
    
    SOURCE_DATA_PATH = "./data/raw/AIS_01_2024.pkl"
    SOURCE_JSON_PATH = "./config/mappings/vessel_type_names.json"
    DESTINATION_PATH = f"./data/processed/{vessel_name.lower()}_vessels.parquet"

    filter_on_vessel_type(SOURCE_DATA_PATH, SOURCE_JSON_PATH, DESTINATION_PATH, vessel_name)
    
    print("Data is saved...")
    