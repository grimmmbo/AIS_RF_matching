"""Download NOAA AIS vessel-position data for a date range into one pickle.

Each day is downloaded and cached to disk as soon as it is fetched, so
an interrupted run can be restarted without re-downloading cached days.

``output_file`` holds a sequence of pickled per-day DataFrames (each
appended with its own ``pickle.dump`` call), not one combined
DataFrame -- concatenating everything before writing doubled memory
use and exhausted it on the full date range. Read it back with a loop:

    with open(output_file, "rb") as f:
        while True:
            try:
                day_df = pickle.load(f)
            except EOFError:
                break
            ...
"""

import logging
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def _day_cache_path(cache_dir: Path, day: datetime) -> Path:
    """Return the cache file path for a single day's data."""
    return cache_dir / f"{day.strftime('%Y_%m_%d')}.pkl"


def _download_zip(day_url: str, dest_path: Path) -> bool:
    """Stream a day's ZIP file to disk instead of buffering it in memory.

    Returns:
        True if the file was downloaded successfully, False otherwise.
    """
    try:
        with requests.get(day_url, stream=True, timeout=60) as response:
            if response.status_code != 200:
                logger.warning(
                    "Failed to download %s (status %s)",
                    day_url,
                    response.status_code,
                )
                return False
            with open(dest_path, "wb") as dest_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    dest_file.write(chunk)
        return True
    except requests.RequestException as exc:
        logger.warning("Error downloading %s: %s", day_url, exc)
        return False


def _parse_zip_to_dataframe(zip_path: Path) -> pd.DataFrame | None:
    """Parse every CSV member of a day's ZIP into a single DataFrame."""
    try:
        with ZipFile(zip_path) as zip_ref:
            day_frames = []
            for file_name in zip_ref.namelist():
                with zip_ref.open(file_name) as csv_file:
                    day_frames.append(pd.read_csv(csv_file))
    except BadZipFile:
        logger.warning("Corrupt ZIP: %s", zip_path)
        return None

    if not day_frames:
        return None
    return pd.concat(day_frames, ignore_index=True)


def _fetch_and_cache_day(day: datetime, base_url: str, cache_dir: Path) -> None:
    """Download, parse, and cache a single day's AIS data.

    If a cache file for this day already exists, the download is
    skipped, which is what allows an interrupted run to resume
    cheaply instead of starting over.
    """
    cache_path = _day_cache_path(cache_dir, day)
    if cache_path.exists():
        logger.info("Skipping %s (already cached)", cache_path.name)
        return

    date_str = day.strftime("%Y_%m_%d")
    day_url = f"{base_url}{date_str}.zip"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".zip", dir=cache_dir, delete=False
    ) as tmp_zip:
        tmp_zip_path = Path(tmp_zip.name)

    try:
        if not _download_zip(day_url, tmp_zip_path):
            return
        day_df = _parse_zip_to_dataframe(tmp_zip_path)
        if day_df is None:
            logger.warning("No data extracted for %s", date_str)
            return
        day_df.to_pickle(cache_path)
        logger.info("Cached %s (%d rows)", cache_path.name, len(day_df))
    finally:
        tmp_zip_path.unlink(missing_ok=True)


def download_AIS_data(
    start_date: datetime,
    end_date: datetime,
    base_url: str,
    output_file: str,
    max_workers: int = 4,
) -> None:
    """Download daily AIS data and append it into a single pickle file.

    Each day is cached to
    ``<output_file's directory>/.<output stem>_cache/<YYYY_MM_DD>.pkl``
    as it's fetched; a re-run after a crash resumes from the cache. See
    the module docstring for the output format and how to read it back.

    Args:
        start_date: Start date of the data download.
        end_date: End date of the data download.
        base_url: Base URL for AIS data files.
        output_file: Path of the output file (a sequence of pickled
            per-day DataFrames, not one combined DataFrame).
        max_workers: Number of concurrent download threads.
    """
    output_path = Path(output_file)
    cache_dir = output_path.parent / f".{output_path.stem}_cache"

    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_fetch_and_cache_day, day, base_url, cache_dir)
            for day in dates
        ]
        for future in as_completed(futures):
            # Log and continue so one bad day doesn't discard the
            # progress already cached by the others.
            try:
                future.result()
            except Exception:
                logger.exception("A day-fetch task raised an exception")

    cached_files = sorted(cache_dir.glob("*.pkl")) if cache_dir.exists() else []
    if not cached_files:
        logger.warning("No data downloaded; nothing to save.")
        return

    total_rows = 0
    with open(output_file, "wb") as out_file:
        for file_path in cached_files:
            day_df = pd.read_pickle(file_path)
            pickle.dump(day_df, out_file)
            total_rows += len(day_df)

    logger.info(
        "Data saved to %s (%d total rows from %d days)",
        output_file,
        total_rows,
        len(cached_files),
    )


if __name__ == "__main__":
    logger.info("Downloading has started...")

    DESTINATION_PATH = "./data/raw/AIS_01_2024.pkl"

    download_AIS_data(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        base_url="https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_",
        output_file=DESTINATION_PATH,
    )
