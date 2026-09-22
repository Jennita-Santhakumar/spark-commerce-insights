from pyspark.sql import functions as F

from transformations import (
    aggregate_by_channel_and_type,
    aggregate_holiday_effect,
    join_with_holidays,
)


def _sample_campaigns(spark):
    return spark.createDataFrame(
        [
            ("mobile_push", "bulk", 100, 10.0, "2021-01-01"),
            ("email", "bulk", 200, 12.0, "2021-01-02"),
            ("mobile_push", "trigger", 50, 8.0, "2021-01-01"),
        ],
        ["channel", "campaign_type", "total_count_scaled", "subject_length", "event_date"],
    ).withColumn("event_date", F.to_date("event_date"))


def _sample_holidays(spark):
    return spark.createDataFrame(
        [("2021-01-01", "New Year")], ["date", "holiday"]
    ).withColumn("date", F.to_date("date"))


def test_aggregate_by_channel_and_type_groups_correctly(spark):
    df = _sample_campaigns(spark)
    result = {
        (r["channel"], r["campaign_type"]): r
        for r in aggregate_by_channel_and_type(df).collect()
    }
    assert result[("mobile_push", "bulk")]["n_campaigns"] == 1
    assert result[("mobile_push", "bulk")]["total_reach"] == 100
    assert result[("email", "bulk")]["total_reach"] == 200


def test_join_with_holidays_flags_correctly(spark):
    campaigns = _sample_campaigns(spark)
    holidays = _sample_holidays(spark)
    joined = join_with_holidays(campaigns, holidays)

    flags = {r["channel"] + "_" + r["campaign_type"]: r["is_holiday"] for r in joined.collect()}
    assert flags["mobile_push_bulk"] is True
    assert flags["email_bulk"] is False
    assert flags["mobile_push_trigger"] is True


def test_holiday_effect_aggregation(spark):
    campaigns = _sample_campaigns(spark)
    holidays = _sample_holidays(spark)
    with_holiday = join_with_holidays(campaigns, holidays)
    effect = {r["is_holiday"]: r for r in aggregate_holiday_effect(with_holiday).collect()}

    assert effect[True]["n_campaigns"] == 2
    assert effect[False]["n_campaigns"] == 1
