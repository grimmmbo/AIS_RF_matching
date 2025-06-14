import requests
import zipfile
import io
import pandas as pd
from datetime import timedelta, datetime

def download_AIS_data(start_date, end_date, url, output_file):
    """
    Downloads and processes AIS data from NOAA within the given date range

    Args:
        start_date (datetime): Start date of the data download
        end_date (datetime): End date of the data download
        base_url (str): Base URL for AIS data files
        output_file (str): Name of the output pickle file
    """
    all_data = []
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime('%Y_%m_%d')
        url = f"{url}{date_str}.zip"

        # Download ZIP file
        response = requests.get(url)

        # If download is successful, process the ZIP file
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                for file_name in zip_ref.namelist():
                    # Extract and read the CSV content from the ZIP file
                    with zip_ref.open(file_name) as file:
                        df = pd.read_csv(file)
                        all_data.append(df)
                        print(f"Processed {file_name} for {date_str}")
        else:
            print(f"Failed to download {url}")

        # Move to the next day
        current_date += timedelta(days=1)

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Save Dataframe
    df.to_parquet(output_file)
    print(f"Data saved to {output_file}")
    
if __name__ == "__main__":
    print("Downloading has started...")
    
    DESTINATION_PATH = "../../../../data/raw/AIS_01_2024.parquet"
    
    download_AIS_data(
        start_date = datetime(2024, 1, 1),
        end_date = datetime(2024, 1, 31),
        url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_',
        output_file = DESTINATION_PATH
    )