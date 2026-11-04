from datetime import timezone

from phase1b_rgb_utae.time_encoding import condition_timestamp, condition_time_features


def test_condition_timestamp_schedule():
    assert condition_timestamp("winter_morning").hour == 8
    assert condition_timestamp("spring_day").month == 4
    assert condition_timestamp("summer_evening").month == 7
    assert condition_timestamp("autumn_day").month == 10
    assert condition_timestamp("winter_morning").tzinfo == timezone.utc


def test_condition_time_features_have_season_and_daytime_one_hot_channels():
    features = condition_time_features("winter_morning")
    assert features.shape == (7,)
    assert features.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
