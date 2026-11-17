"""gta6-twitch-ccu-forecast-01 entry, v2, in Python 3.13 (standard library only).

Forecasts the launch-day Twitch curve by conditioning on launch size.

Why v1 was not enough (diagnosis from the server practice board, 17 Nov):
v1 took the median log-ratio to hour 1 across every usable comparable and
put one lognormal spread around it. That ignores a strong, systematic
pattern in the data: the bigger the hour-1 print, the smaller the later
ratios. Across the 37 usable comparables the correlation between
log(hour-1) and the log-ratio at peak hours is about -0.9, and inside the
big-clean subset it is about -0.8. A constant median therefore overstates
the evening peak of a huge launch roughly twofold (v1 said 2.4x hour 1 at
20:00Z; the size-conditioned fit says about 1.4x for a 270k hour-1 print)
and understates the growth of a small one. The server practice board
showed it: v1 loss 0.276, rank 2,410/2,673, while the field median was
0.211. The median was the problem, not the spread.

Method (v2):
  - Keep only launch curves that look like the target: clean tier (no
    early-access contamination, no weak tier) and hour-1 print at least
    5,000. When that set has fewer than 8 curves, widen it to all clean
    curves, then to every usable curve.
  - For each forecast hour, fit an ordinary least squares line through
    the pool's points (log hour-1, log viewers_h / viewers_1). The
    target's median is the line evaluated at its own log hour-1.
  - Do not extrapolate the line past the pool: the target's log hour-1
    is clamped to the smallest and largest log hour-1 in the pool. A
    launch bigger than anything seen before is scored like the biggest
    seen launch, not like a point far down an unbounded line.
  - The spread is the residual standard deviation of that hour's fit
    (floored at 0.05), and the five quantiles are lognormal around the
    fitted median.
  - Too few points for an hour, or too few curves overall, falls back
    to the flat house baseline for that hour or case.

Leave-one-out over the 37 comparables (same harness as v1, 19
representative targets: clean tier, rolling-midnight, hour-1 >= 5,000),
measured on this file as shipped: v2 loss 0.262 against v1's 0.323,
typical miss 46% against 64%, coverage 0.87. On the 8 practice-like
targets (rolling, clean, hour-1 >= 20,000) v2 scores 0.275 against
v1's 0.312. The clamp to the pool's observed size range is included
in those numbers; without it the representative loss is 0.265. The features a
second predictor could add were tested and rejected: viewers per
channel, platform flags and weekday move the score by less than 0.01
or make it worse, and the pre-release day's curve does not predict
the release day at all (loss about 0.9). Shrinking the slope toward
zero also hurts badly (0.33 at 10% shrinkage), so the line is used
as fitted.

Deterministic: no randomness, no wall clock, no network; comparables
are read in file order and every statistic is a deterministic function
of them, so two runs write byte-identical output. One case runs in
well under a second.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip
packages. -P keeps this file's directory OFF sys.path, so the sys.path
line below puts it back first (harmless here: this entry is one file).
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Inverse standard normal CDF at each level, from the standard library.
Z_VALUES = [statistics.NormalDist().inv_cdf(level) for level in LEVELS]
# A launch curve counts as usable with a positive hour-1 print and at
# least this many of the 24 release-day hours present.
MIN_HOURS = 20
# The size-conditioned pool wants clean launches with a real audience.
REP_MIN_HOUR1 = 5000
# A regression needs at least this many curves to be trusted.
MIN_POOL = 8
# Fewer usable curves than this and the whole case is the flat baseline.
MIN_CURVES = 3
# Fewer points than this for one hour and that hour is the flat baseline.
MIN_POINTS = 5
# Floor for the log spread, so an hour whose fit is nearly exact still
# carries a sane band.
SIGMA_FLOOR = 0.05


def round_half_up_cents(x):
    """Round x to 2 decimals, halves up, returning an int when whole.

    Quantiles are positive here, so the floor formulation is exact.
    """
    cents = math.floor(x * 100.0 + 0.5)
    value = cents / 100.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """Parse an ISO timestamp, accepting a trailing Z for UTC."""
    if text.endswith("Z"):
        return datetime.fromisoformat(text[:-1] + "+00:00")
    return datetime.fromisoformat(text)


def day_hours(comp):
    """UTC hour (1..24 of the comparable's release day) -> viewers."""
    start = parse_time(comp["day_start"])
    series = {}
    for row in comp.get("hourly", []):
        viewers = row.get("viewers")
        if viewers is None:
            continue
        hour = int((parse_time(row["from"]) - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            series[hour] = viewers
    return series


def is_clean(comp):
    """True when the comparable's unlock is not flagged as contaminated."""
    unlock = comp.get("unlock", "")
    return ("early-access-contaminated" not in unlock
            and "Tier: weak" not in unlock)


def launch_curve(comp):
    """A comparable's release-day series, or None when it is not a launch.

    Skips the GTA VI premiere entry (its 'release day' is a premiere,
    with no hour-1 print) and any comparable whose hour 1 is missing or
    zero, or whose day is too sparsely sampled to give a curve.
    """
    series = day_hours(comp)
    if not series.get(1) or series[1] <= 0:
        return None
    if len(series) < MIN_HOURS:
        return None
    return series


def pick_pool(curves):
    """The curves the regression is fitted on, widest fallback last.

    curves is a list of (series, clean) pairs. Prefer clean launches
    with a real hour-1 audience; widen only when that set is too small
    to fit a line on.
    """
    rep = [(s, c) for (s, c) in curves
           if c and s[1] >= REP_MIN_HOUR1]
    if len(rep) >= MIN_POOL:
        return [s for (s, c) in rep]
    clean = [s for (s, c) in curves if c]
    if len(clean) >= MIN_POOL:
        return clean
    return [s for (s, c) in curves]


def fit_line(xs, ys):
    """Least-squares intercept and slope of y on x."""
    n = len(xs)
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    return mean_y - slope * mean_x, slope


def flat_quantiles(p1, hour, z_values):
    """The house baseline shape for one hour (flat at hour 1, widening)."""
    return [round_half_up_cents(p1 * math.exp(z * 0.15 * math.sqrt(hour - 1)))
            for z in z_values]


def hour_forecast(p1, pool, hour, z_values):
    """The five quantiles for one hour from the size-conditioned fit."""
    points = []
    for series in pool:
        first = series.get(1)
        value = series.get(hour)
        if first and first > 0 and value and value > 0:
            points.append((math.log(first), math.log(value / first)))
    if len(points) < MIN_POINTS:
        return flat_quantiles(p1, hour, z_values)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    intercept, slope = fit_line(xs, ys)
    # Stay inside the observed size range: never extrapolate the line.
    x_target = min(max(math.log(p1), min(xs)), max(xs))
    centre = intercept + slope * x_target
    residuals = [y - (intercept + slope * x) for x, y in points]
    spread = max(statistics.pstdev(residuals), SIGMA_FLOOR)
    qs = [round_half_up_cents(p1 * math.exp(centre + z * spread))
          for z in z_values]
    for i in range(1, len(qs)):
        qs[i] = max(qs[i], qs[i - 1])
    return qs


def forecast_case(case, name):
    """Forecast one case: 23 rows, hours 2..24, five quantiles each."""
    levels = case["levels"]
    if levels == LEVELS:
        z_values = Z_VALUES
    else:
        z_values = [statistics.NormalDist().inv_cdf(level) for level in levels]
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    curves = []
    for comp in case.get("comparables", []):
        series = launch_curve(comp)
        if series is not None:
            curves.append((series, is_clean(comp)))
    if len(curves) < MIN_CURVES:
        return {"hours": [{"hour": row["hour"],
                           "q": flat_quantiles(p1, row["hour"], z_values)}
                          for row in case["forecast"]]}
    pool = pick_pool(curves)
    return {"hours": [{"hour": row["hour"],
                       "q": hour_forecast(p1, pool, row["hour"], z_values)}
                      for row in case["forecast"]]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
