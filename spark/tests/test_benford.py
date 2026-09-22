from benford import BENFORD_EXPECTED, benford_distribution, leading_digit_column


def test_leading_digit_extraction(spark):
    df = spark.createDataFrame(
        [(123.0,), (45.0,), (999.0,), (0.056,), (7.0,)], ["value"]
    )
    result = leading_digit_column(df, "value").collect()
    digits = sorted(r["leading_digit"] for r in result)
    assert digits == [1, 4, 5, 7, 9]


def test_leading_digit_ignores_zero_and_null(spark):
    df = spark.createDataFrame(
        [(0.0,), (None,), (10.0,)], schema="value double"
    )
    result = leading_digit_column(df, "value").collect()
    assert len(result) == 1
    assert result[0]["leading_digit"] == 1


def test_benford_distribution_shape_and_totals(spark):
    # A dataset engineered to roughly follow Benford's law: lots of
    # leading-1 values, fewer leading-9 values.
    values = [1, 10, 11, 12, 100, 123, 15] * 5 + [9, 91] * 1
    df = spark.createDataFrame([(v,) for v in values], ["value"])

    dist = benford_distribution(df, "value")
    rows = {r["digit"]: r for r in dist.collect()}

    assert set(rows.keys()) == set(range(1, 10))
    total_observed = sum(r["observed_count"] for r in rows.values())
    assert total_observed == len(values)

    for d in range(1, 10):
        assert abs(rows[d]["expected_pct"] - BENFORD_EXPECTED[d]) < 1e-9

    # Digit 1 should be the most common leading digit in this fixture.
    assert rows[1]["observed_count"] == max(r["observed_count"] for r in rows.values())
