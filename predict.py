#!/usr/bin/env python3
"""
Flight Delay Real-Time Inference Service
Scores flight queries for delay probability, risk tiers, and contributory factor breakdown.
"""

import sys
from src.ml.inference import FlightDelayPredictor, run_sample_inference

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        predictor = FlightDelayPredictor()
        print("=" * 70)
        print("FLIGHT DELAY INTERACTIVE PREDICTOR")
        print("=" * 70)
        carrier = input("Enter Carrier Code (e.g., DL, AA, UA, WN): ").strip().upper()
        origin = input("Enter Origin Airport Code (e.g., JFK, ORD, ATL): ").strip().upper()
        dest = input("Enter Destination Airport Code (e.g., LAX, SFO, MIA): ").strip().upper()
        month = int(input("Enter Month (1-12): ").strip())
        dow = int(input("Enter Day of Week (1=Mon, 7=Sun): ").strip())
        dep_time = int(input("Enter Scheduled Departure Time (HHMM, e.g., 1830): ").strip())
        arr_time = int(input("Enter Scheduled Arrival Time (HHMM, e.g., 2145): ").strip())
        distance = float(input("Enter Distance in Miles (e.g., 1000): ").strip() or "1000")

        res = predictor.predict_flight(
            carrier=carrier,
            origin=origin,
            dest=dest,
            month=month,
            day_of_week=dow,
            crs_dep_time=dep_time,
            crs_arr_time=arr_time,
            distance=distance
        )
        print("\n" + "=" * 70)
        print(f"Prediction for {carrier} ({origin} -> {dest}):")
        print(f"  * Delay Probability: {res['delay_probability_pct']:.2f}%")
        print(f"  * Status:            {res['predicted_status']}")
        print(f"  * Risk Tier:         {res['risk_tier']}")
        print("Contributing Factors:")
        for factor in res["contributing_factors"]:
            print(f"  - {factor}")
        print("=" * 70)
    else:
        run_sample_inference()

if __name__ == "__main__":
    main()
