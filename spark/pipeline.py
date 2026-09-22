"""
Main entry point for the PySpark e-commerce analysis pipeline.

Run with:
    python spark/pipeline.py

or, using spark-submit:
    spark-submit spark/pipeline.py

See ../README.md for setup notes (Java requirement, Windows winutils.exe
gotcha, etc).

What this does, all via Spark DataFrame operations:
  1. Loads campaigns.csv, holidays.csv, product.csv.
  2. Synthetically scales campaigns.csv up (replication + perturbation) to
     a larger row count, to give the distributed operations below a
     realistic volume to chew on (see spark/data_prep.py for why/how).
  3. Runs distributed groupBy aggregations (channel/campaign_type,
     holiday effect, product-quality effect).
  4. Runs distributed joins across campaigns <-> holidays and
     campaigns <-> product.
  5. Re-implements the Benford's Law leading-digit fraud-detection check
     (originally done with pandas/numpy in statistical_analysis/BDF_review.ipynb)
     using Spark DataFrame groupBy/count instead.
  6. Writes results to spark/output/ (CSV, human-readable) and prints a
     summary to stdout.
"""
from __future__ import annotations

import os

from data_prep import (
    assign_synthetic_product,
    get_spark,
    load_campaigns,
    load_holidays,
    load_product,
    scale_up_campaigns,
)
from benford import benford_distribution, chi_square_from_distribution
from transformations import (
    aggregate_by_channel_and_type,
    aggregate_holiday_effect,
    aggregate_product_quality_effect,
    join_with_holidays,
    join_with_product,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _write_csv(df, name: str) -> None:
    """Write a (small, already-aggregated) result DataFrame to a single
    CSV file.

    We deliberately use `toPandas().to_csv()` here instead of Spark's own
    `DataFrame.write.csv(...)`. Spark's native CSV writer goes through
    Hadoop's output-committer machinery, which on Windows requires
    `winutils.exe`/`HADOOP_HOME` to be installed (see README) - without it,
    the write throws `FileNotFoundException: HADOOP_HOME and
    hadoop.home.dir are unset` even though every DataFrame *computation*
    (the groupBys, joins, and Benford aggregation) already ran correctly
    and fully distributed under Spark. Since these result tables are tiny
    (a handful to a few dozen rows - the heavy ~380k-row data never
    reaches the driver), collecting them once at the very end to write a
    plain CSV is a pragmatic, honest workaround for that Windows-only
    packaging gap rather than a shortcut around doing the real work in
    Spark.
    """
    out_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.toPandas().to_csv(out_path, index=False)


def run(replication_factor: int = 200) -> dict:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    campaigns = load_campaigns(spark)
    holidays = load_holidays(spark)
    product = load_product(spark)
    n_products = product.count()

    print(f"[pipeline] Loaded raw campaigns rows: {campaigns.count()}")
    print(f"[pipeline] Loaded product rows: {n_products}")
    print(f"[pipeline] Loaded holiday rows: {holidays.count()}")

    scaled = scale_up_campaigns(campaigns, replication_factor=replication_factor)
    scaled = assign_synthetic_product(scaled, n_products=n_products)
    scaled.cache()
    scaled_count = scaled.count()
    print(
        f"[pipeline] Synthetically scaled campaigns to {scaled_count} rows "
        f"(replication_factor={replication_factor})"
    )

    # --- Distributed joins -------------------------------------------------
    with_holiday = join_with_holidays(scaled, holidays)
    with_product = join_with_product(scaled, product)

    # --- Distributed groupBy aggregations -----------------------------------
    by_channel_type = aggregate_by_channel_and_type(scaled)
    holiday_effect = aggregate_holiday_effect(with_holiday)
    product_quality_effect = aggregate_product_quality_effect(with_product)

    print("\n[pipeline] Top rows: reach by channel/campaign_type")
    by_channel_type.show(10, truncate=False)

    print("[pipeline] Reach on holidays vs non-holidays")
    holiday_effect.show(truncate=False)

    print("[pipeline] Reach by linked product quality score")
    product_quality_effect.show(10, truncate=False)

    # --- Benford's Law re-implemented in Spark ------------------------------
    distribution = benford_distribution(scaled, "total_count_scaled")
    print("[pipeline] Benford's Law leading-digit distribution (Spark-native)")
    distribution.show(9, truncate=False)

    chi_result = chi_square_from_distribution(distribution, scaled_count)
    print(
        f"[pipeline] Chi-square={chi_result['chi2_statistic']:.4f} "
        f"p-value={chi_result['p_value']:.6f} n={chi_result['n']}"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _write_csv(by_channel_type, "by_channel_type")
    _write_csv(holiday_effect, "holiday_effect")
    _write_csv(product_quality_effect, "product_quality_effect")
    _write_csv(distribution, "benford_distribution")

    spark.stop()
    return chi_result


if __name__ == "__main__":
    run()
