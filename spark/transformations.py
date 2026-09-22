"""
Distributed transformations for the e-commerce campaigns dataset:
groupBy aggregations and joins across campaigns / holidays / product.

Kept separate from data_prep.py / benford.py so they're easy to unit
test in isolation (see spark/tests/test_transformations.py).
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def join_with_holidays(campaigns: DataFrame, holidays: DataFrame) -> DataFrame:
    """Left-join scaled campaign rows to the holidays table on event_date,
    flagging whether a campaign ran on a public holiday."""
    holidays_small = holidays.select(
        F.col("date").alias("holiday_date"), F.col("holiday")
    )
    joined = campaigns.join(
        F.broadcast(holidays_small),
        campaigns["event_date"] == holidays_small["holiday_date"],
        how="left",
    )
    return joined.withColumn("is_holiday", F.col("holiday").isNotNull())


def join_with_product(campaigns: DataFrame, product: DataFrame) -> DataFrame:
    """Inner-join scaled campaign rows to the product attributes table on
    the synthetic product_id key."""
    product_small = product.select(
        F.col("product_id"),
        F.col("quality").alias("product_quality"),
        F.col("total products").alias("product_total_stock")
        if "total products" in product.columns
        else F.lit(None).alias("product_total_stock"),
    )
    return campaigns.join(product_small, on="product_id", how="inner")


def aggregate_by_channel_and_type(df: DataFrame) -> DataFrame:
    """groupBy channel + campaign_type: volume and total_count stats."""
    return (
        df.groupBy("channel", "campaign_type")
        .agg(
            F.count("*").alias("n_campaigns"),
            F.sum("total_count_scaled").alias("total_reach"),
            F.avg("total_count_scaled").alias("avg_reach"),
            F.avg("subject_length").alias("avg_subject_length"),
        )
        .orderBy(F.desc("total_reach"))
    )


def aggregate_holiday_effect(df_with_holiday_flag: DataFrame) -> DataFrame:
    """groupBy is_holiday: does campaign reach differ on holidays?"""
    return (
        df_with_holiday_flag.groupBy("is_holiday")
        .agg(
            F.count("*").alias("n_campaigns"),
            F.avg("total_count_scaled").alias("avg_reach"),
        )
        .orderBy("is_holiday")
    )


def aggregate_product_quality_effect(df_with_product: DataFrame) -> DataFrame:
    """groupBy product_quality: average campaign reach for campaigns
    synthetically linked to higher vs lower quality-scored products."""
    return (
        df_with_product.groupBy("product_quality")
        .agg(
            F.count("*").alias("n_campaigns"),
            F.avg("total_count_scaled").alias("avg_reach"),
        )
        .orderBy("product_quality")
    )
