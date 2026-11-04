"""Categorical season/daytime features for U-TAE."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import torch
import torch.nn.functional as F


SEASON_MONTHS = {"winter": 1, "spring": 4, "summer": 7, "autumn": 10}
DAYTIME_HOURS = {"morning": 8, "day": 14, "evening": 21}


def condition_timestamp(condition: str, year: int = 2026) -> datetime:
    """Convert the project's season/daytime labels to a deterministic UTC time."""
    try:
        season, daytime = condition.split("_", 1)
        return datetime(year, SEASON_MONTHS[season], 1, DAYTIME_HOURS[daytime], tzinfo=timezone.utc)
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid condition {condition!r}") from error


def condition_time_features(condition: str, year: int = 2026,
                            reference: datetime | None = None) -> torch.Tensor:
    """Encode one condition as four season and three daytime one-hot values."""
    del year, reference
    season, daytime = condition.split("_", 1)
    seasons = ("winter", "spring", "summer", "autumn")
    daytimes = ("morning", "day", "evening")
    if season not in seasons or daytime not in daytimes:
        raise ValueError(f"invalid condition {condition!r}")
    result = torch.zeros(7, dtype=torch.float32)
    result[seasons.index(season)] = 1.0
    result[4 + daytimes.index(daytime)] = 1.0
    return result


def ids_to_time_features(season_ids: torch.Tensor, daytime_ids: torch.Tensor,
                         year: int = 2026) -> torch.Tensor:
    del year
    season = F.one_hot(season_ids.long(), num_classes=4)
    daytime = F.one_hot(daytime_ids.long(), num_classes=3)
    return torch.cat((season, daytime), dim=-1).to(torch.float32)


def encode_acquisition_times(timestamps: list[datetime], reference: datetime, device=None,
                             dtype=torch.float32) -> torch.Tensor:
    """Return the legacy continuous timestamp encoding for compatibility."""
    if reference.tzinfo is None or reference.utcoffset() is None:
        raise ValueError("reference must be timezone-aware")
    values = []
    for timestamp in timestamps:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        timestamp = timestamp.astimezone(timezone.utc)
        elapsed_days = (timestamp - reference.astimezone(timezone.utc)).total_seconds() / 86400.0
        signed_log_days = math.copysign(math.log1p(abs(elapsed_days)), elapsed_days)
        seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second + timestamp.microsecond / 1e6
        day_phase = seconds / 86400.0
        year_start = datetime(timestamp.year, 1, 1, tzinfo=timezone.utc)
        leap = timestamp.year % 4 == 0 and (timestamp.year % 100 != 0 or timestamp.year % 400 == 0)
        year_phase = ((timestamp - year_start).total_seconds() / 86400.0) / (366.0 if leap else 365.0)
        values.append([signed_log_days, math.sin(2 * math.pi * day_phase), math.cos(2 * math.pi * day_phase),
                       math.sin(2 * math.pi * year_phase), math.cos(2 * math.pi * year_phase), 1.0])
    return torch.tensor(values, device=device, dtype=dtype)
