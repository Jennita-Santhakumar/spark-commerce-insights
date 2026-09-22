"""
Benford's Law first-digit analysis implemented with PySpark DataFrame
operations (no pandas / numpy for the digit extraction or aggregation
step). This mirrors the statistical test used in the original notebook
(statistical_analysis/BDF_review.ipynb) but pushes the heavy per-row work
(digit extraction, counting) into Spark's distributed execution engine
instead of doing it in a single pandas process.

Only the tiny final 9-row result (observed vs expected frequency per
leading digit) is collected back to the driver for the chi-square test,
since scipy's chisquare has no Spark-native equivalent and the collected
data is 9 numbers, not the raw dataset.
"""
from __future__ import annotations

import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Benford's Law expected proportions for leading digits 1-9.
BENFORD_EXPECTED = {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def leading_digit_column(df: DataFrame, numeric_col: str, out_col: str = "leading_digit") -> DataFrame:
    """Add a column with the first significant (non-zero) digit of
    `numeric_col`, computed entirely with Spark SQL string functions."""
    abs_col = F.abs(F.col(numeric_col).cast("double"))
    # Drop rows that can't have a meaningful leading digit.
    filtered = df.filter(abs_col.isNotNull() & (abs_col > 0))

    # Cast to string, strip a leading "0." / sign / decimal point, then take
    # the first digit character. Using format_number/regexp so it works for
    # both large integers and small decimals without collecting to driver.
    as_str = F.format_number(abs_col, 10)  # fixed-format string, no exponent
    stripped = F.regexp_replace(as_str, "[^0-9]", "")
    stripped = F.regexp_replace(stripped, "^0+", "")
    first_char = F.substring(stripped, 1, 1)

    return filtered.withColumn(
        out_col, F.when(first_char == "", None).otherwise(first_char.cast("int"))
    ).filter(F.col(out_col).isNotNull())


def benford_distribution(df: DataFrame, numeric_col: str) -> DataFrame:
    """Return a small Spark DataFrame with columns:
    digit, observed_count, observed_pct, expected_pct
    computed via a distributed groupBy/count aggregation."""
    with_digit = leading_digit_column(df, numeric_col)
    total = with_digit.count()

    counts = (
        with_digit.groupBy("leading_digit")
        .agg(F.count("*").alias("observed_count"))
        .withColumnRenamed("leading_digit", "digit")
    )

    spark = df.sparkSession
    expected_rows = [(d, pct) for d, pct in BENFORD_EXPECTED.items()]
    expected_df = spark.createDataFrame(expected_rows, ["digit", "expected_pct"])

    result = (
        expected_df.join(counts, on="digit", how="left")
        .na.fill({"observed_count": 0})
        .withColumn("observed_pct", F.col("observed_count") / F.lit(total))
        .orderBy("digit")
    )
    return result


def chi_square_from_distribution(distribution_df: DataFrame, total_n: int | None = None) -> dict:
    """Collect the small (9-row) distribution and run a chi-square
    goodness-of-fit test against Benford's expected proportions.

    This is the one intentional collect() in the pipeline: the raw data
    stays distributed and only 9 aggregated numbers reach the driver.

    `total_n` is optional and, if given, is only used for reporting - the
    expected counts always sum to the actual observed total (the sum of
    `observed_count` across the 9 digit rows), since that's the row count
    that survived the leading-digit filter in `leading_digit_column`
    (rows with null/zero values are excluded there and would otherwise
    make the chi-square input sums disagree).
    """
    from scipy import stats

    rows = distribution_df.orderBy("digit").collect()
    observed = [r["observed_count"] for r in rows]
    observed_total = sum(observed)
    expected = [r["expected_pct"] * observed_total for r in rows]

    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "observed": observed,
        "expected": expected,
        "n": observed_total,
        "n_before_filtering": total_n,
    }
