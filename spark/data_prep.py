"""
Data preparation utilities for the PySpark e-commerce pipeline.

The repository ships with small CSV samples (campaigns.csv, holidays.csv,
product.csv, Dataset.csv). A real "big data" job would run against millions
of rows; since we don't have a multi-GB dataset available locally, this
module SYNTHETICALLY SCALES UP the existing campaigns.csv sample by
replicating rows with randomized jitter (different ids, perturbed counts,
shifted timestamps). This is clearly a demonstration technique, not real
production data, and is documented as such everywhere it's used.

Everything here uses Spark DataFrame APIs (no collecting to the driver,
no pandas) so the transformations genuinely run in a distributed fashion
under the Spark execution engine, even though local[*] mode is a single
machine.
"""
from __future__ import annotations

import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_spark(app_name: str = "ecommerce-bigdata") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_campaigns(spark: SparkSession, path: str | None = None) -> DataFrame:
    path = path or os.path.join(DATA_DIR, "campaigns.csv")
    return spark.read.csv(path, header=True, inferSchema=True)


def load_holidays(spark: SparkSession, path: str | None = None) -> DataFrame:
    path = path or os.path.join(DATA_DIR, "holidays.csv")
    return spark.read.csv(path, header=True, inferSchema=True)


def load_product(spark: SparkSession, path: str | None = None) -> DataFrame:
    path = path or os.path.join(DATA_DIR, "product.csv")
    df = spark.read.csv(path, header=True, inferSchema=True)
    # Assign a synthetic product_id (row order) so we have a join key -
    # the source CSV (a public wine-quality style dataset reused here as a
    # stand-in "product attributes" table) has no natural id column.
    return df.withColumn(
        "product_id", F.row_number().over(_row_number_window())
    )


def _row_number_window():
    from pyspark.sql.window import Window

    return Window.orderBy(F.monotonically_increasing_id())


def scale_up_campaigns(
    campaigns: DataFrame, replication_factor: int = 200, seed: int = 42
) -> DataFrame:
    """Synthetically scale the ~1900-row campaigns sample up to a much
    larger row count for a genuine "big data" style volume.

    Technique: cross-join the source rows against a small integer range
    [0, replication_factor), then perturb the numeric/timestamp columns
    per-copy so the replicated rows aren't exact duplicates. This is a
    synthetic-scaling demonstration technique only - it does not fabricate
    new "real" transactions, it multiplies the existing sample with noise
    so distributed groupBy/join/aggregation code has a realistically large
    volume to operate on.
    """
    spark = campaigns.sparkSession
    reps = spark.range(0, replication_factor).withColumnRenamed("id", "rep_idx")

    scaled = campaigns.crossJoin(F.broadcast(reps))

    scaled = scaled.withColumn(
        "synthetic_id",
        F.concat_ws("-", F.col("id").cast("string"), F.col("rep_idx").cast("string")),
    )

    # Perturb total_count with +/-15% multiplicative noise per replica so
    # the leading-digit distribution isn't just N identical copies of the
    # same digit pattern.
    noise = (F.rand(seed) * 0.30 + 0.85)  # in [0.85, 1.15)
    scaled = scaled.withColumn(
        "total_count_scaled",
        (F.col("total_count").cast("double") * noise).cast("long"),
    )

    # Shift started_at by rep_idx days so replicated rows spread across a
    # wider synthetic date range and can meaningfully join to holidays.
    scaled = scaled.withColumn(
        "started_at_scaled",
        F.expr("started_at + make_interval(0, 0, 0, rep_idx)"),
    )
    scaled = scaled.withColumn("event_date", F.to_date("started_at_scaled"))

    return scaled


def assign_synthetic_product(scaled_campaigns: DataFrame, n_products: int) -> DataFrame:
    """Deterministically assign each scaled campaign row a product_id in
    [1, n_products] so we can demonstrate a real distributed join against
    the product attributes table."""
    return scaled_campaigns.withColumn(
        "product_id",
        (F.abs(F.hash(F.col("synthetic_id"))) % F.lit(n_products) + 1),
    )
