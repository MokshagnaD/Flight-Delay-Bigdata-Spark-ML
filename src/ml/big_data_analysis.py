"""
High-Performance Out-of-Core Big Data Analytics Engine
Processes multi-year flight records across 4.63 GB of Parquet partitions
without overflowing memory, computing delay trends, airport bottlenecks,
carrier rankings, and cause breakdowns.
"""

import glob
import json
import os
import time
from collections import defaultdict
import pyarrow.parquet as pq
import pandas as pd

from src.ml.config import FLIGHT_PARQUET_DIR, REPORTS_DIR


def run_big_data_analysis(parquet_dir: str = FLIGHT_PARQUET_DIR, max_years: int = None):
    print("=" * 80)
    print("      US BTS AIRLINE ON-TIME PERFORMANCE - BIG DATA ANALYTICS SUITE      ")
    print("=" * 80)

    files = sorted(glob.glob(os.path.join(parquet_dir, "Year=*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet partitions found in {parquet_dir}")

    if max_years is not None and max_years > 0:
        files = files[-max_years:]

    print(f"Dataset Location:       {parquet_dir}")
    print(f"Partitions to Analyze:  {len(files)} yearly parquet files")
    print(f"Execution Mode:         Chunked Streaming Parquet Processing")
    print("-" * 80)

    # Accumulators for Big Data Analytics
    total_flights = 0
    total_delayed = 0
    total_ontime = 0
    total_cancelled = 0
    total_diverted = 0

    yearly_stats = defaultdict(lambda: {"total": 0, "delayed": 0, "cancelled": 0})
    monthly_delays = defaultdict(lambda: {"total": 0, "delayed": 0})
    hourly_delays = defaultdict(lambda: {"total": 0, "delayed": 0})
    dayofweek_delays = defaultdict(lambda: {"total": 0, "delayed": 0})

    carrier_stats = defaultdict(lambda: {"total": 0, "delayed": 0})
    origin_stats = defaultdict(lambda: {"total": 0, "delayed": 0})
    dest_stats = defaultdict(lambda: {"total": 0, "delayed": 0})

    # Cause decomposition accumulators
    delay_cause_counts = {
        "CarrierDelay": 0,
        "WeatherDelay": 0,
        "NASDelay": 0,
        "SecurityDelay": 0,
        "LateAircraftDelay": 0,
    }

    t0 = time.time()

    cols_to_read = [
        "Year", "Month", "DayOfWeek", "Reporting_Airline", "Origin", "Dest",
        "CRSDepTime", "ArrDel15", "Cancelled", "Diverted",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"
    ]

    for idx, f in enumerate(files, 1):
        t_file = time.time()
        year_str = os.path.basename(os.path.dirname(f)).replace("Year=", "")
        print(f"[{idx:02d}/{len(files):02d}] Scanning partition Year={year_str}...", end="", flush=True)

        try:
            pf = pq.ParquetFile(f)
            file_rows = 0

            # Stream row groups
            for rg_idx in range(pf.num_row_groups):
                table_batch = pf.read_row_group(rg_idx, columns=cols_to_read)
                df = table_batch.to_pandas()
                file_rows += len(df)
                total_flights += len(df)

                # Process cancellations and diversions
                cancelled_mask = df["Cancelled"] == 1.0
                diverted_mask = df["Diverted"] == 1.0
                n_canc = int(cancelled_mask.sum())
                n_div = int(diverted_mask.sum())
                total_cancelled += n_canc
                total_diverted += n_div

                # Filter valid completed flights
                valid_df = df[~cancelled_mask & ~diverted_mask & df["ArrDel15"].notna()]
                n_delayed = int((valid_df["ArrDel15"] == 1.0).sum())
                n_ontime = int((valid_df["ArrDel15"] == 0.0).sum())

                total_delayed += n_delayed
                total_ontime += n_ontime

                # Yearly
                yearly_stats[year_str]["total"] += len(df)
                yearly_stats[year_str]["delayed"] += n_delayed
                yearly_stats[year_str]["cancelled"] += n_canc

                # Monthly
                for m, cnt in valid_df.groupby("Month").size().items():
                    monthly_delays[int(m)]["total"] += int(cnt)
                for m, cnt in valid_df[valid_df["ArrDel15"] == 1.0].groupby("Month").size().items():
                    monthly_delays[int(m)]["delayed"] += int(cnt)

                # Day of Week
                for dow, cnt in valid_df.groupby("DayOfWeek").size().items():
                    dayofweek_delays[int(dow)]["total"] += int(cnt)
                for dow, cnt in valid_df[valid_df["ArrDel15"] == 1.0].groupby("DayOfWeek").size().items():
                    dayofweek_delays[int(dow)]["delayed"] += int(cnt)

                # Hourly (CRSDepTime // 100)
                dep_hours = (pd.to_numeric(valid_df["CRSDepTime"], errors="coerce").fillna(1200) // 100).clip(0, 23).astype(int)
                valid_df_dep = valid_df.copy()
                valid_df_dep["Hour"] = dep_hours
                for h, cnt in valid_df_dep.groupby("Hour").size().items():
                    hourly_delays[int(h)]["total"] += int(cnt)
                for h, cnt in valid_df_dep[valid_df_dep["ArrDel15"] == 1.0].groupby("Hour").size().items():
                    hourly_delays[int(h)]["delayed"] += int(cnt)

                # Carrier
                for carrier, cnt in valid_df.groupby("Reporting_Airline").size().items():
                    carrier_stats[carrier]["total"] += int(cnt)
                for carrier, cnt in valid_df[valid_df["ArrDel15"] == 1.0].groupby("Reporting_Airline").size().items():
                    carrier_stats[carrier]["delayed"] += int(cnt)

                # Origin
                for orig, cnt in valid_df.groupby("Origin").size().items():
                    origin_stats[orig]["total"] += int(cnt)
                for orig, cnt in valid_df[valid_df["ArrDel15"] == 1.0].groupby("Origin").size().items():
                    origin_stats[orig]["delayed"] += int(cnt)

                # Dest
                for dst, cnt in valid_df.groupby("Dest").size().items():
                    dest_stats[dst]["total"] += int(cnt)
                for dst, cnt in valid_df[valid_df["ArrDel15"] == 1.0].groupby("Dest").size().items():
                    dest_stats[dst]["delayed"] += int(cnt)

                # Delay Causes (flights with ArrDel15 == 1 and delay cause > 0)
                delayed_records = valid_df[valid_df["ArrDel15"] == 1.0]
                for cause in delay_cause_counts.keys():
                    if cause in delayed_records.columns:
                        delay_cause_counts[cause] += int((pd.to_numeric(delayed_records[cause], errors="coerce") > 0).sum())

            print(f" {file_rows:,d} records in {time.time() - t_file:.1f}s")
        except Exception as e:
            print(f" [Error reading {f}: {e}]")

    scan_duration = time.time() - t0
    overall_valid = total_delayed + total_ontime
    overall_delay_rate = (total_delayed / overall_valid * 100) if overall_valid > 0 else 0
    overall_cancel_rate = (total_cancelled / total_flights * 100) if total_flights > 0 else 0

    print("-" * 80)
    print(f"Analyzed {total_flights:,d} total flight records across all years in {scan_duration:.2f}s!")
    print(f"Processing Throughput: {total_flights / max(scan_duration, 0.001):,.0f} flights/second")
    print("=" * 80)

    # 1. Executive Summary
    print("\n1. GLOBAL OPERATIONAL PERFORMANCE SUMMARY:")
    print(f"   * Total Scheduled Flights:  {total_flights:,d}")
    print(f"   * Completed Flights:        {overall_valid:,d} ({overall_valid/total_flights*100:.1f}%)")
    print(f"   * Delayed Flights (>=15m):  {total_delayed:,d} ({overall_delay_rate:.2f}%)")
    print(f"   * On-Time Flights:          {total_ontime:,d} ({100 - overall_delay_rate:.2f}%)")
    print(f"   * Cancelled Flights:        {total_cancelled:,d} ({overall_cancel_rate:.2f}%)")
    print(f"   * Diverted Flights:         {total_diverted:,d} ({total_diverted/total_flights*100:.2f}%)")

    # 2. Carrier Rankings
    print("\n2. AIRLINE CARRIER RELIABILITY RANKING (Minimum 100,000 flights):")
    carrier_ranked = []
    for c, s in carrier_stats.items():
        if s["total"] >= 100_000:
            rate = s["delayed"] / s["total"] * 100
            carrier_ranked.append({"carrier": c, "total": s["total"], "delayed": s["delayed"], "delay_rate": rate})
    carrier_ranked.sort(key=lambda x: x["delay_rate"])

    print(f"   {'Carrier':<8} {'Total Flights':<15} {'Delayed Flights':<18} {'Delay Rate':<12} {'Reliability Tier'}")
    print("   " + "-" * 70)
    for c in carrier_ranked:
        tier = "PUNCTUAL" if c["delay_rate"] < 16.0 else ("AVERAGE" if c["delay_rate"] < 21.0 else "FREQUENT DELAYS")
        print(f"   {c['carrier']:<8} {c['total']:<15,d} {c['delayed']:<18,d} {c['delay_rate']:6.2f}%      {tier}")

    # 3. Airport Congestion & Delays
    print("\n3. TOP 15 ORIGIN AIRPORTS BY TRAFFIC & DELAYS:")
    origin_ranked = []
    for o, s in origin_stats.items():
        if s["total"] >= 100_000:
            rate = s["delayed"] / s["total"] * 100
            origin_ranked.append({"origin": o, "total": s["total"], "delayed": s["delayed"], "delay_rate": rate})
    origin_ranked.sort(key=lambda x: x["total"], reverse=True)

    print(f"   {'Airport':<8} {'Departures':<15} {'Delayed Flights':<18} {'Delay Rate':<12}")
    print("   " + "-" * 60)
    for o in origin_ranked[:15]:
        print(f"   {o['origin']:<8} {o['total']:<15,d} {o['delayed']:<18,d} {o['delay_rate']:6.2f}%")

    # 4. Hourly Delay Pattern
    print("\n4. DIURNAL (HOURLY) DELAY ACCUMULATION PATTERN:")
    print("   Departure Hour | Total Flights | Delayed Flights | Delay Rate (%) | Trend")
    print("   " + "-" * 70)
    for h in sorted(hourly_delays.keys()):
        h_tot = hourly_delays[h]["total"]
        h_del = hourly_delays[h]["delayed"]
        rate = (h_del / h_tot * 100) if h_tot > 0 else 0
        bar = "#" * int(rate / 1.5)
        print(f"   {h:02d}:00 - {h:02d}:59  | {h_tot:13,d} | {h_del:15,d} | {rate:12.2f}% | {bar}")

    # 5. Delay Cause Breakdown
    total_cause_incidents = sum(delay_cause_counts.values())
    print("\n5. PRIMARY DELAY CAUSE DECOMPOSITION:")
    for cause, cnt in delay_cause_counts.items():
        pct = (cnt / total_cause_incidents * 100) if total_cause_incidents > 0 else 0
        print(f"   * {cause:18s}: {cnt:,d} flights ({pct:.2f}% of delayed cause reports)")

    # Save structured report
    report_data = {
        "summary": {
            "total_flights": total_flights,
            "completed_flights": overall_valid,
            "delayed_flights": total_delayed,
            "ontime_flights": total_ontime,
            "cancelled_flights": total_cancelled,
            "diverted_flights": total_diverted,
            "delay_rate_pct": round(overall_delay_rate, 2),
            "cancel_rate_pct": round(overall_cancel_rate, 2),
            "analysis_duration_sec": round(scan_duration, 2),
        },
        "carrier_rankings": carrier_ranked,
        "top_airports": origin_ranked[:25],
        "hourly_delays": {str(k): hourly_delays[k] for k in sorted(hourly_delays.keys())},
        "monthly_delays": {str(k): monthly_delays[k] for k in sorted(monthly_delays.keys())},
        "dayofweek_delays": {str(k): dayofweek_delays[k] for k in sorted(dayofweek_delays.keys())},
        "yearly_trends": {str(k): yearly_stats[k] for k in sorted(yearly_stats.keys())},
        "delay_causes": delay_cause_counts,
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "big_data_analysis_summary.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[Saved] Full Big Data Analytics report saved to: {report_path}")
    print("=" * 80)
    return report_data


if __name__ == "__main__":
    run_big_data_analysis()
