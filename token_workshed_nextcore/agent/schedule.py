"""Small, dependency-free schedule parser used by NextCore's local scheduler."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _field_matches(expression: str, value: int, minimum: int, maximum: int) -> bool:
    for part in expression.split(","):
        base, slash, step_text = part.partition("/")
        step = int(step_text) if slash else 1
        if step < 1:
            raise ValueError("Cron step must be positive.")
        if base == "*":
            start, end = minimum, maximum
        elif "-" in base:
            first, last = base.split("-", 1)
            start, end = int(first), int(last)
        else:
            start = end = int(base)
        if start < minimum or end > maximum or start > end:
            raise ValueError(f"Cron value is outside {minimum}..{maximum}.")
        if start <= value <= end and (value - start) % step == 0:
            return True
    return False


def _cron_matches(parts: list[str], value: dt.datetime) -> bool:
    minute, hour, day, month, weekday = parts
    # Standard cron uses Sunday=0; Python uses Monday=0.
    cron_weekday = (value.weekday() + 1) % 7
    day_match = _field_matches(day, value.day, 1, 31)
    weekday_match = _field_matches(weekday, cron_weekday, 0, 6)
    calendar_match = (
        day_match and weekday_match
        if day == "*" or weekday == "*"
        else day_match or weekday_match
    )
    return (
        _field_matches(minute, value.minute, 0, 59)
        and _field_matches(hour, value.hour, 0, 23)
        and calendar_match
        and _field_matches(month, value.month, 1, 12)
    )


def next_run_time(schedule: str, timezone_name: str = "UTC", *, after: float) -> float | None:
    """Return next epoch time for an ISO one-shot, ``@every`` or five-field cron."""
    expression = str(schedule or "").strip()
    try:
        zone = ZoneInfo(str(timezone_name or "UTC"))
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc
    if expression.startswith("@every "):
        seconds = int(expression.split(maxsplit=1)[1])
        if seconds < 60 or seconds > 366 * 24 * 60 * 60:
            raise ValueError("@every must be between 60 seconds and one year.")
        return after + seconds
    if expression.startswith("once:"):
        raw = expression[5:].strip()
        try:
            timestamp = float(raw)
        except ValueError:
            parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=zone)
            timestamp = parsed.timestamp()
        return timestamp if timestamp > after else None
    fields = expression.split()
    if len(fields) != 5:
        raise ValueError("Schedule must be once:<ISO-8601>, @every <seconds>, or a five-field cron expression.")
    # Validate the expression before scanning forward.
    for field, low, high in zip(fields, (0, 0, 1, 1, 0), (59, 23, 31, 12, 6)):
        for part in field.split(","):
            base, slash, step = part.partition("/")
            if slash and int(step) < 1:
                raise ValueError("Cron step must be positive.")
            if base == "*":
                continue
            if "-" in base:
                first, last = map(int, base.split("-", 1))
                if first < low or last > high or first > last:
                    raise ValueError(f"Cron field must be within {low}..{high}.")
            else:
                number = int(base)
                if not low <= number <= high:
                    raise ValueError(f"Cron field must be within {low}..{high}.")
    candidate = dt.datetime.fromtimestamp(after, zone).replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
    last = candidate + dt.timedelta(days=366)
    while candidate <= last:
        if _cron_matches(fields, candidate):
            return candidate.timestamp()
        candidate += dt.timedelta(minutes=1)
    return None
