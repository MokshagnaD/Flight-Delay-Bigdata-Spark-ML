#!/usr/bin/env python3
"""
Script to download multi-year real-world US BTS Airline Flight Performance Parquet dataset
from the open public mirror (blobs.duckdb.org).

Total size options:
- Default (2010-2023, 14 years): ~4.63 GB of compressed Parquet files (~40 GB uncompressed)
- Full (1987-2024, 38 years): ~11.0 GB of compressed Parquet files (~100+ GB uncompressed)
"""

import os
import sys
import urllib.request
import time

BASE_PARQUET_URL = "https://blobs.duckdb.org/flight-data-partitioned"
TARGET_DIR = os.path.abspath("data/flight_parquet")

DEFAULT_YEARS = list(range(2010, 2024))

def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def download_with_progress(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 1024 * 1024:
            print(f"  [Skipping] Already exists: {os.path.basename(dest_path)} ({format_size(size)})")
            return size

    print(f"  [Downloading] {url} -> {dest_path}")
    start_time = time.time()

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 256
            downloaded = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)
                if total_size > 0:
                    percent = min(100.0, downloaded * 100.0 / total_size)
                    speed = downloaded / max(1e-5, (time.time() - start_time))
                    sys.stdout.write(f"\r    {percent:5.1f}% | {format_size(downloaded)} / {format_size(total_size)} | {format_size(speed)}/s")
                    sys.stdout.flush()
        print()
        return os.path.getsize(dest_path)
    except Exception as e:
        print(f"\n    Error downloading {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return 0

def main():
    years = DEFAULT_YEARS
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        years = list(range(1987, 2025))
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        start_yr = int(sys.argv[1])
        end_yr = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else start_yr
        years = list(range(start_yr, end_yr + 1))

    print("=" * 70)
    print("US BTS Airline On-Time Performance (Parquet) Downloader")
    print(f"Target Years: {years[0]} - {years[-1]} ({len(years)} years)")
    print(f"Target Directory: {TARGET_DIR}")
    print("=" * 70)

    total_downloaded_bytes = 0

    for yr in years:
        url = f"{BASE_PARQUET_URL}/Year={yr}/data_0.parquet"
        dest = os.path.join(TARGET_DIR, f"Year={yr}", "data_0.parquet")
        size = download_with_progress(url, dest)
        total_downloaded_bytes += size

    print("\n" + "=" * 70)
    print("Download complete!")
    print(f"Total downloaded Parquet size: {format_size(total_downloaded_bytes)}")
    print(f"Location: {TARGET_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
