from data_prep import scale_up_campaigns


def test_scale_up_campaigns_multiplies_rows(spark):
    df = spark.createDataFrame(
        [
            (1, "bulk", "email", 1000, "2021-01-01 00:00:00", 50),
        ],
        ["id", "campaign_type", "channel", "total_count", "started_at", "subject_length"],
    )
    df = df.withColumn("started_at", df["started_at"].cast("timestamp"))

    scaled = scale_up_campaigns(df, replication_factor=10)
    assert scaled.count() == 10

    rows = scaled.collect()
    # Perturbed counts should stay within the documented +/-15% noise band.
    for r in rows:
        assert 850 <= r["total_count_scaled"] <= 1150

    # synthetic_id should be unique per replica.
    ids = {r["synthetic_id"] for r in rows}
    assert len(ids) == 10
