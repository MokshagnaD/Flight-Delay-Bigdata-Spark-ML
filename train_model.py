#!/usr/bin/env python3
"""
Flight Delay Machine Learning Model Training Suite
Trains state-of-the-art Big Data ML models (XGBoost, Random Forest, or PySpark MLlib)
with advanced feature engineering, class imbalance compensation, and rigorous holdout evaluation.
"""

import argparse
import sys
from src.ml.model_trainer import train_model
from src.ml.spark_ml_trainer import train_spark_ml_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train Flight Delay Machine Learning Model on Big Data")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "xgb", "rf", "random_forest", "spark"],
        help="Model architecture: 'xgboost', 'rf' (Random Forest), or 'spark' (PySpark MLlib)"
    )
    parser.add_argument(
        "--records",
        type=int,
        default=1_500_000,
        help="Number of records to load for training (default: 1,500,000)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Holdout test partition fraction (default: 0.20)"
    )
    parser.add_argument(
        "--spark-fraction",
        type=float,
        default=0.25,
        help="Sampling fraction for PySpark MLlib (default: 0.25)"
    )

    args = parser.parse_args()

    if args.model.lower() == "spark":
        train_spark_ml_pipeline(
            dataset_path="data/flight_parquet",
            model_type="logistic_regression",
            sample_fraction=args.spark_fraction
        )
    else:
        train_model(
            model_type=args.model,
            max_records=args.records,
            test_size=args.test_size
        )


if __name__ == "__main__":
    main()
