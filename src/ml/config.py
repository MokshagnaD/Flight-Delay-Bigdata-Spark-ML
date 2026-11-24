"""
Configuration constants and paths for Big Data Analytics and ML Modeling
"""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
FLIGHT_PARQUET_DIR = os.path.join(DATA_DIR, "flight_parquet")
JOINED_PARQUET_DIR = os.path.join(DATA_DIR, "airline_and_plane", "joined.parquet")

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Model artifact paths
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "flight_delay_xgboost.joblib")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "flight_delay_rf_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.joblib")
SPARK_MODEL_PATH = os.path.join(MODELS_DIR, "flight_delay_spark_pipeline")

# Feature configurations
NUMERICAL_FEATURES = [
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
]

CATEGORICAL_FEATURES = [
    "Reporting_Airline",
    "Origin",
    "Dest",
]

TARGET_COL = "ArrDel15"

RAW_COLUMNS_TO_LOAD = [
    "Year",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "Distance",
    "Cancelled",
    "Diverted",
    "ArrDel15",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]
