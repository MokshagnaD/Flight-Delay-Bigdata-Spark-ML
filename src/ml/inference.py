"""
Real-Time Flight Delay Inference Service
Provides instant delay probability, risk scoring, and contributory factor breakdown
for scheduled flight queries.
"""

import os
import sys
import joblib
import pandas as pd

from src.ml.config import METADATA_PATH, RF_MODEL_PATH, XGBOOST_MODEL_PATH


class FlightDelayPredictor:
    """
    Inference service loading pre-trained XGBoost or Random Forest model
    and executing calibrated probability estimation and risk explanation.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            if os.path.exists(XGBOOST_MODEL_PATH):
                model_path = XGBOOST_MODEL_PATH
            elif os.path.exists(RF_MODEL_PATH):
                model_path = RF_MODEL_PATH
            else:
                raise FileNotFoundError(
                    "No trained model found! Please train a model first using `python train_model.py`."
                )

        meta_path = model_path.replace(".joblib", "_metadata.joblib")
        if not os.path.exists(meta_path):
            meta_path = METADATA_PATH

        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Model metadata not found at '{meta_path}'. Please train the model first."
            )

        self.model = joblib.load(model_path)
        self.metadata = joblib.load(meta_path)
        self.encoder = self.metadata["encoder"]
        self.threshold = self.metadata.get("optimal_threshold", 0.50)
        self.model_type = self.metadata.get("model_type", "ml_model")

    def predict_flight(
        self,
        carrier: str,
        origin: str,
        dest: str,
        month: int,
        day_of_week: int,
        crs_dep_time: int,
        crs_arr_time: int,
        distance: float = 500.0,
    ) -> dict:
        """
        Scores an individual flight query and decomposes delay risk factors.
        """
        record = {
            "Reporting_Airline": [carrier.upper().strip()],
            "Origin": [origin.upper().strip()],
            "Dest": [dest.upper().strip()],
            "Month": [month],
            "DayOfWeek": [day_of_week],
            "CRSDepTime": [crs_dep_time],
            "CRSArrTime": [crs_arr_time],
            "Distance": [distance],
        }

        df_query = pd.DataFrame(record)
        X_feat = self.encoder.transform(df_query)

        # Align features dynamically with what the model expects
        if hasattr(self.model, "feature_names_in_"):
            expected_cols = list(self.model.feature_names_in_)
        elif hasattr(self.model, "get_booster") and self.model.get_booster().feature_names:
            expected_cols = list(self.model.get_booster().feature_names)
        else:
            expected_cols = self.metadata.get("feature_names", list(X_feat.columns))

        # Ensure all required columns exist with default 0, and align column order
        for col in expected_cols:
            if col not in X_feat.columns:
                X_feat[col] = 0
        X_feat = X_feat[expected_cols]

        prob = float(self.model.predict_proba(X_feat)[0, 1])
        is_delayed = bool(prob >= self.threshold)

        # Calibrated Risk Tiering (benchmark: national average delay rate is ~18-20%)
        if prob >= 0.28:
            risk_tier = "HIGH RISK"
        elif prob >= 0.20:
            risk_tier = "ELEVATED RISK"
        elif prob >= 0.13:
            risk_tier = "MODERATE RISK"
        else:
            risk_tier = "LOW RISK"

        # Explain risk drivers or punctuality drivers
        dep_hour = (crs_dep_time // 100) % 24
        origin_rate = self.encoder.origin_map.get(origin, self.encoder.global_delay_rate) * 100
        dest_rate = self.encoder.dest_map.get(dest, self.encoder.global_delay_rate) * 100
        airline_rate = self.encoder.airline_map.get(carrier, self.encoder.global_delay_rate) * 100

        risk_factors = []
        positive_factors = []

        if 16 <= dep_hour <= 21:
            risk_factors.append(f"Evening Rush Departure ({dep_hour:02d}:00 hours, peak system congestion window)")
        elif 5 <= dep_hour <= 8:
            positive_factors.append(f"Early Morning Departure ({dep_hour:02d}:00 hours, lowest nationwide delay window)")

        if origin_rate >= 21.0:
            risk_factors.append(f"High-Congestion Origin Airport '{origin}' (historical delay rate: {origin_rate:.1f}%)")
        elif origin_rate < 16.0:
            positive_factors.append(f"High-Punctuality Origin Airport '{origin}' (historical delay rate: {origin_rate:.1f}%)")

        if dest_rate >= 21.0:
            risk_factors.append(f"Congested Destination Hub '{dest}' (arrival congestion rate: {dest_rate:.1f}%)")

        if airline_rate >= 21.0:
            risk_factors.append(f"Carrier '{carrier}' above-average historical delay rate ({airline_rate:.1f}%)")
        elif airline_rate < 16.0:
            positive_factors.append(f"Carrier '{carrier}' recognized for high on-time reliability ({airline_rate:.1f}%)")

        if month in [6, 7, 8]:
            risk_factors.append("Summer Convective Storm Season (nationwide thunderstorm ground stops)")
        elif month in [12, 1]:
            risk_factors.append("Winter Holiday Travel Volume & De-icing Operations")
        elif month in [4, 5, 9, 10]:
            positive_factors.append("Mild shoulder season with historically lower airspace congestion")

        # Output factors matched to the predicted status
        if is_delayed or prob >= 0.20:
            contributing = risk_factors if risk_factors else ["Route length and scheduled arrival window increase delay risk"]
        else:
            contributing = positive_factors if positive_factors else ["Off-peak flight timing with favorable historical on-time conditions"]

        status_text = "DELAYED (HIGH RISK)" if is_delayed else ("ON-TIME (MODERATE RISK)" if prob >= 0.18 else "ON-TIME / EARLY")

        return {
            "carrier": carrier,
            "origin": origin,
            "dest": dest,
            "dep_time": crs_dep_time,
            "arr_time": crs_arr_time,
            "distance_miles": distance,
            "delay_probability_pct": round(prob * 100, 2),
            "predicted_status": status_text,
            "risk_tier": risk_tier,
            "decision_threshold": self.threshold,
            "contributing_factors": contributing,
        }

    def predict_batch(self, flights: list) -> pd.DataFrame:
        """
        Scores a batch of flight dictionaries and returns formatted DataFrame.
        """
        results = []
        for f in flights:
            res = self.predict_flight(
                carrier=f.get("Carrier") or f.get("Reporting_Airline", "DL"),
                origin=f.get("Origin", "JFK"),
                dest=f.get("Dest", "LAX"),
                month=f.get("Month", 6),
                day_of_week=f.get("DayOfWeek", 3),
                crs_dep_time=f.get("CRSDepTime", 1200),
                crs_arr_time=f.get("CRSArrTime", 1430),
                distance=f.get("Distance", 500),
            )
            results.append(res)
        return pd.DataFrame(results)


def run_sample_inference():
    print("=" * 80)
    print("      REAL-TIME FLIGHT DELAY INFERENCE ENGINE (BIG DATA PREDICTOR)      ")
    print("=" * 80)

    try:
        predictor = FlightDelayPredictor()
    except Exception as e:
        print(f"[Error] Failed to initialize predictor: {e}")
        return

    sample_flights = [
        # Flight 1: Delta evening peak rush flight JFK -> LAX in December
        {"Carrier": "DL", "Origin": "JFK", "Dest": "LAX", "Month": 12, "DayOfWeek": 5, "CRSDepTime": 1830, "CRSArrTime": 2200, "Distance": 2475},
        # Flight 2: United early morning flight ORD -> SFO in May
        {"Carrier": "UA", "Origin": "ORD", "Dest": "SFO", "Month": 5,  "DayOfWeek": 2, "CRSDepTime": 630,  "CRSArrTime": 900,  "Distance": 1846},
        # Flight 3: Southwest late evening flight MDW -> MSY in July thunderstorm season
        {"Carrier": "WN", "Origin": "MDW", "Dest": "MSY", "Month": 7,  "DayOfWeek": 7, "CRSDepTime": 2115, "CRSArrTime": 2330, "Distance": 825},
        # Flight 4: American Airlines midday flight DFW -> MIA in March
        {"Carrier": "AA", "Origin": "DFW", "Dest": "MIA", "Month": 3,  "DayOfWeek": 3, "CRSDepTime": 1245, "CRSArrTime": 1630, "Distance": 1121},
        # Flight 5: Spirit Airlines late afternoon rush ATL -> LGA in November
        {"Carrier": "NK", "Origin": "ATL", "Dest": "LGA", "Month": 11, "DayOfWeek": 4, "CRSDepTime": 1730, "CRSArrTime": 2005, "Distance": 762},
        # Flight 6: Frontier morning flight DEN -> LAS in October
        {"Carrier": "F9", "Origin": "DEN", "Dest": "LAS", "Month": 10, "DayOfWeek": 1, "CRSDepTime": 815,  "CRSArrTime": 930,  "Distance": 628},
    ]

    print(f"Scoring {len(sample_flights)} Sample Flight Schedules:\n")
    print(f"{'Carrier':<8} {'Route':<14} {'DepTime':<9} {'Delay Prob':<13} {'Predicted Status':<20} {'Risk Tier'}")
    print("-" * 80)

    for f in sample_flights:
        res = predictor.predict_flight(
            carrier=f["Carrier"],
            origin=f["Origin"],
            dest=f["Dest"],
            month=f["Month"],
            day_of_week=f["DayOfWeek"],
            crs_dep_time=f["CRSDepTime"],
            crs_arr_time=f["CRSArrTime"],
            distance=f["Distance"]
        )
        route = f"{f['Origin']} -> {f['Dest']}"
        print(f"{f['Carrier']:<8} {route:<14} {f['CRSDepTime']:<9} {res['delay_probability_pct']:6.2f}%       {res['predicted_status']:<20} {res['risk_tier']}")
        for factor in res["contributing_factors"]:
            print(f"         * {factor}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    run_sample_inference()
