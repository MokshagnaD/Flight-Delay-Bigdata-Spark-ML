"""
Distributed PySpark MLlib Pipeline Trainer for Big Data Flight Delay Analysis
Leverages Apache Spark's distributed computing to train scalable classification models
(LogisticRegression, RandomForestClassifier, GBTClassifier) across multi-million rows.
"""

import os
import sys
import time

def train_spark_ml_pipeline(
    dataset_path: str = "data/flight_parquet",
    model_type: str = "logistic_regression",
    sample_fraction: float = 0.5,
    save_path: str = "models/flight_delay_spark_pipeline"
):
    print("=" * 80)
    print("     PYSPARK MLLIB DISTRIBUTED FLIGHT DELAY CLASSIFICATION PIPELINE     ")
    print("=" * 80)
    print(f"Dataset Path:       {dataset_path}")
    print(f"Classifier:         {model_type.upper()}")
    print(f"Sampling Fraction:  {sample_fraction * 100:.0f}%")
    print(f"Target Save Path:   {save_path}")
    print("-" * 80)

    # Set explicit python executable for worker and driver
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when
    from pyspark.sql.types import DoubleType
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

    spark = (
        SparkSession.builder
        .appName("PySpark Flight Delay ML Pipeline")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    t0 = time.time()

    # 1. Ingest Data
    print("\n[Step 1/5] Ingesting Parquet flight dataset with PySpark SQL...")
    raw_df = spark.read.parquet(dataset_path)

    # Clean and filter completed flights
    filtered_df = (
        raw_df
        .filter(col("ArrDel15").isNotNull())
        .withColumn("label", col("ArrDel15").cast(DoubleType()))
        .withColumn("cleanCancelled", col("Cancelled").cast(DoubleType()))
        .filter(col("label").isin(0.0, 1.0))
        .filter(col("cleanCancelled").isNull() | (col("cleanCancelled") == 0.0))
        .select(
            col("label"),
            col("Reporting_Airline"),
            col("Origin"),
            col("Dest"),
            col("Month").cast(DoubleType()).alias("Month"),
            col("DayOfWeek").cast(DoubleType()).alias("DayOfWeek"),
            col("CRSDepTime").cast(DoubleType()).alias("CRSDepTime"),
            col("CRSArrTime").cast(DoubleType()).alias("CRSArrTime"),
            col("Distance").cast(DoubleType()).alias("Distance")
        )
        .na.drop()
    )

    if sample_fraction < 1.0:
        clean_df = filtered_df.sample(withReplacement=False, fraction=sample_fraction, seed=42).cache()
    else:
        clean_df = filtered_df.cache()

    total_records = clean_df.count()
    print(f"  Prepared {total_records:,d} valid flight records in {time.time() - t0:.2f}s")

    # 2. Train / Test Split
    print("\n[Step 2/5] Splitting into 80% Train and 20% Holdout Test...")
    train_data, test_data = clean_df.randomSplit([0.8, 0.2], seed=42)

    # 3. Construct ML Pipeline
    print("\n[Step 3/5] Building Distributed ML Pipeline Stages...")
    airline_indexer = StringIndexer(inputCol="Reporting_Airline", outputCol="airlineIdx", handleInvalid="keep")
    origin_indexer = StringIndexer(inputCol="Origin", outputCol="originIdx", handleInvalid="keep")
    dest_indexer = StringIndexer(inputCol="Dest", outputCol="destIdx", handleInvalid="keep")

    airline_encoder = OneHotEncoder(inputCols=["airlineIdx"], outputCols=["airlineVec"], handleInvalid="keep")

    feature_cols = [
        "airlineVec",
        "originIdx",
        "destIdx",
        "Month",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "Distance"
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

    # Classifier selection
    if model_type.lower() in ["rf", "random_forest"]:
        classifier = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=40,
            maxDepth=8,
            maxBins=360,
            seed=42
        )
    elif model_type.lower() in ["gbt", "gradient_boosting"]:
        classifier = GBTClassifier(
            labelCol="label",
            featuresCol="features",
            maxIter=30,
            maxDepth=6,
            seed=42
        )
    else:
        classifier = LogisticRegression(
            labelCol="label",
            featuresCol="features",
            maxIter=25,
            regParam=0.01,
            elasticNetParam=0.0
        )

    pipeline = Pipeline(stages=[
        airline_indexer,
        origin_indexer,
        dest_indexer,
        airline_encoder,
        assembler,
        classifier
    ])

    # 4. Train
    print(f"\n[Step 4/5] Fitting ML Pipeline on {model_type.upper()}...")
    t_train = time.time()
    pipeline_model = pipeline.fit(train_data)
    print(f"  Model trained in {time.time() - t_train:.2f}s")

    # 5. Evaluate
    print("\n[Step 5/5] Evaluating Distributed Model on Holdout Test Partition...")
    predictions = pipeline_model.transform(test_data).cache()

    binary_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    roc_auc = binary_eval.setMetricName("areaUnderROC").evaluate(predictions)
    pr_auc = binary_eval.setMetricName("areaUnderPR").evaluate(predictions)

    multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = multi_eval.setMetricName("accuracy").evaluate(predictions)
    f1 = multi_eval.setMetricName("f1").evaluate(predictions)
    precision = multi_eval.setMetricName("weightedPrecision").evaluate(predictions)
    recall = multi_eval.setMetricName("weightedRecall").evaluate(predictions)

    print("\n" + "=" * 80)
    print("                     SPARK ML EVALUATION RESULTS                        ")
    print("=" * 80)
    print(f"  * Area Under ROC (AUC-ROC):    {roc_auc:.4f}")
    print(f"  * Area Under PR  (AUC-PR):     {pr_auc:.4f}")
    print(f"  * Accuracy:                    {accuracy * 100:.2f}%")
    print(f"  * F1 Score:                    {f1:.4f}")
    print(f"  * Weighted Precision:          {precision * 100:.2f}%")
    print(f"  * Weighted Recall:             {recall * 100:.2f}%")

    print("\nSample Predictions:")
    predictions.select("Reporting_Airline", "Origin", "Dest", "CRSDepTime", "label", "probability", "prediction").show(5, False)

    # Save Pipeline
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pipeline_model.write().overwrite().save(save_path)
    print(f"\nTrained Spark Pipeline saved to: {save_path}")
    print("=" * 80)

    spark.stop()
    return roc_auc, accuracy, f1

if __name__ == "__main__":
    mtype = sys.argv[1] if len(sys.argv) > 1 else "logistic_regression"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    train_spark_ml_pipeline(model_type=mtype, sample_fraction=frac)
