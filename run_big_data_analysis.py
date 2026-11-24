#!/usr/bin/env python3
"""
Big Data Flight Delay Analytics Runner
Scans multi-year Parquet partitions and computes deep analytical statistics across tens of millions of records.
"""

import sys
from src.ml.big_data_analysis import run_big_data_analysis

if __name__ == "__main__":
    max_years = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    run_big_data_analysis(max_years=max_years)
