import os
import requests
import zipfile
import io
import pandas as pd
from datetime import timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_AIS_data(start_date, end_date, base_url, output_file):
    """
    Downloads and processes AIS data from NOAA within the given date range

    Args:
        start_date (datetime): Start date of the data download
        end_date (datetime): End date of the data download
        base_url (str): Base URL for AIS data files
        output_file (str): Name of the output pickle file
    """
    all_data = []

    # Download & parse a single day
    def _fetch_one(day):
        date_str = day.strftime('%Y_%m_%d')
        day_url = f"{base_url}{date_str}.zip"
        
        # Download ZIP file
        response = requests.get(day_url)

        # If download is successful, process the ZIP file
        if response.status_code == 200:
            try:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    day_frames = []
                    for file_name in zip_ref.namelist():
                        # Extract and read the CSV content from the ZIP file
                        with zip_ref.open(file_name) as file:
                            df = pd.read_csv(file)
                            day_frames.append(df)
                            print(f"Processed {file_name} for {date_str}")
                    if day_frames:
                        return pd.concat(day_frames, ignore_index=True)
            except zipfile.BadZipFile:
                print(f"Corrupt ZIP: {day_url}")
        else:
            print(f"Failed to download {day_url}")
        return None

    # List of days
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    # Run in parallel
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_fetch_one, d) for d in dates]
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None:
                all_data.append(df)

    if not all_data:
        print("No data downloaded; nothing to save.")
        return

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Save DataFrame
    df.to_pickle(output_file)
    print(f"Data saved to {output_file}")
    
if __name__ == "__main__":
    print("Downloading has started...")
    
    DESTINATION_PATH = "./data/raw/AIS_01_2024.pkl"
    
    download_AIS_data(
        start_date = datetime(2024, 1, 1),
        end_date = datetime(2024, 1, 31),
        base_url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_',
        output_file = DESTINATION_PATH
    )