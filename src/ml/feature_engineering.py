"""
Advanced Feature Engineering for Flight Delay Prediction
Transforms raw flight records into predictive features for Machine Learning models.
"""

import numpy as np
import pandas as pd


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts high-signal temporal features from scheduled times.
    """
    df = df.copy()

    # Convert times to numeric
    dep_time = pd.to_numeric(df["CRSDepTime"], errors="coerce").fillna(1200).astype(int)
    arr_time = pd.to_numeric(df["CRSArrTime"], errors="coerce").fillna(1400).astype(int)

    df["CRSDepHour"] = (dep_time // 100) % 24
    df["CRSDepMin"] = dep_time % 100

    df["CRSArrHour"] = (arr_time // 100) % 24
    df["CRSArrMin"] = arr_time % 100

    # Scheduled duration in minutes
    dep_total_min = df["CRSDepHour"] * 60 + df["CRSDepMin"]
    arr_total_min = df["CRSArrHour"] * 60 + df["CRSArrMin"]
    raw_duration = arr_total_min - dep_total_min
    # Handle flights crossing midnight
    df["FlightDurationMin"] = np.where(raw_duration < 0, raw_duration + 1440, raw_duration)
    # Clip unreasonable outliers
    df["FlightDurationMin"] = df["FlightDurationMin"].clip(lower=15, upper=1200)

    # Rush hour flags (major contributors to tarmac & airspace delay)
    df["MorningRush"] = ((df["CRSDepHour"] >= 7) & (df["CRSDepHour"] <= 9)).astype(int)
    df["EveningRush"] = ((df["CRSDepHour"] >= 16) & (df["CRSDepHour"] <= 20)).astype(int)

    # Calendar features
    day_of_week = pd.to_numeric(df["DayOfWeek"], errors="coerce").fillna(3).astype(int)
    df["DayOfWeek"] = day_of_week
    df["IsWeekend"] = day_of_week.isin([6, 7]).astype(int)

    month = pd.to_numeric(df["Month"], errors="coerce").fillna(6).astype(int)
    df["Month"] = month

    distance = pd.to_numeric(df["Distance"], errors="coerce").fillna(500)
    df["Distance"] = distance

    return df


class FlightFeatureEncoder:
    """
    Fitted feature encoder that applies smoothed target encoding
    and one-hot / categorical mappings across high-cardinality airports and carriers.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.global_delay_rate = 0.20
        self.origin_map = {}
        self.dest_map = {}
        self.airline_map = {}
        self.top_airlines = []
        self.feature_names = []

    def fit(self, df: pd.DataFrame, target_col: str = "ArrDel15"):
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        self.global_delay_rate = float(y.mean()) if len(y) > 0 else 0.20

        # Smoothed target encoding for high-cardinality categorical variables
        # formula: (count * mean + smoothing * global_mean) / (count + smoothing)
        def compute_smoothed_rate(col_name):
            stats = df.groupby(col_name)[target_col].agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + self.smoothing * self.global_delay_rate) / (
                stats["count"] + self.smoothing
            )
            return smoothed.to_dict()

        self.origin_map = compute_smoothed_rate("Origin")
        self.dest_map = compute_smoothed_rate("Dest")
        self.airline_map = compute_smoothed_rate("Reporting_Airline")

        # Top airlines for one-hot indicators (sorted for deterministic column ordering)
        self.top_airlines = sorted(list(df["Reporting_Airline"].value_counts().nlargest(15).index))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat = extract_temporal_features(df)

        # Apply target encoding with fallback to global average
        df_feat["OriginDelayRate"] = df_feat["Origin"].map(self.origin_map).fillna(self.global_delay_rate)
        df_feat["DestDelayRate"] = df_feat["Dest"].map(self.dest_map).fillna(self.global_delay_rate)
        df_feat["AirlineDelayRate"] = df_feat["Reporting_Airline"].map(self.airline_map).fillna(self.global_delay_rate)

        # Add top airline indicators
        for carrier in self.top_airlines:
            col_name = f"is_carrier_{carrier}"
            df_feat[col_name] = (df_feat["Reporting_Airline"] == carrier).astype(int)

        feature_cols = [
            "Month",
            "DayOfWeek",
            "CRSDepHour",
            "CRSDepMin",
            "CRSArrHour",
            "CRSArrMin",
            "FlightDurationMin",
            "Distance",
            "IsWeekend",
            "MorningRush",
            "EveningRush",
            "OriginDelayRate",
            "DestDelayRate",
            "AirlineDelayRate",
        ] + [f"is_carrier_{c}" for c in self.top_airlines]

        self.feature_names = feature_cols
        return df_feat[feature_cols]

    def fit_transform(self, df: pd.DataFrame, target_col: str = "ArrDel15") -> pd.DataFrame:
        self.fit(df, target_col=target_col)
        return self.transform(df)
