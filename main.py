"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only):
launch-curve ratio forecast.

Method (chosen by leave-one-out evaluation over the 37 launch comparables
in the kit, each held out in turn and forecast from the rest):
  - For every comparable with a usable release day (hour-1 print > 0 and at
    least 20 of the 24 UTC hours present), take the per-hour log ratio
    lr_h = ln(viewers_h / viewers_1) for h = 2..24.
  - The forecast median for hour h is p1 * exp(median(lr_h)) and the spread
    is a lognormal with sigma_h = 1.4826 * MAD(lr_h) (robust: the median
    absolute deviation, so the handful of simultaneous-global launches
    whose hour-1 print was ~0 and whose ratios run into the thousands do
    not blow up the spread), shrunk by 0.9 (a mild calibration: without it
    the 90% band over-covers at ~0.93; with it ~0.91, close to honest).
  - Quantiles: q_tau(h) = p1 * exp(mu_h + z_tau * sigma_h), rounded to 2
    decimals, half up.

Why this shape: launch days are not flat. Across past big launches the
median curve dips to ~0.75x hour 1 in the small hours (03:00-05:00Z, before
the US unlocks), then climbs through the day to ~2.4x at 19:00-20:00Z (the
European evening + US afternoon peak) and eases to ~2.0x by 24:00Z. The
house baseline (flat at hour 1, sigma 0.15*sqrt(h-1)) misses that shape
entirely.

Leave-one-out results on the 19 representative pseudo-targets (clean tier,
rolling-midnight unlock, hour-1 >= 5,000 viewers -- the profile GTA VI
itself has): pinball loss 0.323 vs the starter baseline's 0.403, typical
miss 64% vs 83%, 90% coverage 0.91 vs 0.75. Variants tested and rejected
with their representative-target losses: UTC alignment beats
unlock-aligned (0.323 vs ~0.34-0.36); empirical quantiles of the full
pool (0.383) lose to the robust lognormal (0.323); trimming the pool to
clean-tier only (0.326) ties the full pool but uses fewer curves; sigma
multipliers 0.8-1.25 are flat around 0.9-1.0, so 0.9 is a mild calibration,
not a tuned spike; smoothing the median across hours and using the mean
instead of the median change nothing (0.326/0.331). The GTA VI premiere
comparable is not a launch (no release-day curve) and is skipped
automatically by the usability filter, like any comparable without a
valid hour 1.

Deterministic: no randomness, no wall clock, no network; the comparables
are read in file order and every statistic is a deterministic function of
them, so two runs write byte-identical output. One case runs in well
under a second.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so the sys.path line below puts
it back first (harmless here: this entry is a single file).
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
# The standard normal's quantile at each level (Acklam's approximation
# below reproduces these to ~1e-9; they are kept as the exact values the
# forecast uses when the case asks for exactly these levels).
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# Mild calibration of the robust spread (see module docstring).
SIGMA_SHRINK = 0.9
# A comparable counts as a usable launch curve only with a positive
# hour-1 print and at least this many of the 24 release-day hours present.
MIN_HOURS = 20
# Floor for the log spread, so an hour whose comparables agree almost
# exactly still carries a sane band.
SIGMA_FLOOR = 0.05


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.15e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day_series(comp: dict) -> dict:
    """UTC hour (1..24 of the comparable's release day) -> viewers."""
    day_start = parse_ts(comp["day_start"])
    out = {}
    for row in comp.get("hourly", []):
        viewers = row.get("viewers")
        if viewers is None:
            continue
        hour = int((parse_ts(row["from"]) - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            out[hour] = viewers
    return out


def usable_curve(comp: dict) -> dict | None:
    """A comparable's release-day series, or None when it is not a launch curve.

    Skips the GTA VI premiere entry (its 'release day' is a premiere, with
    no hour-1 print) and any comparable whose hour 1 is missing, zero, or
    whose day is too sparsely sampled to give a curve.
    """
    series = release_day_series(comp)
    if not series.get(1) or series[1] <= 0:
        return None
    if len(series) < MIN_HOURS:
        return None
    return series


def log_ratios(curves: list, hour: int) -> list:
    """ln(viewers_h / viewers_1) across the usable curves that have both hours."""
    ratios = []
    for series in curves:
        v1 = series.get(1)
        vh = series.get(hour)
        if v1 and v1 > 0 and vh and vh > 0:
            ratios.append(math.log(vh / v1))
    return ratios


def hour_quantiles(p1: float, curves: list, hour: int, z_values: list) -> list:
    """The five quantiles for one hour: p1 * exp(mu + z * sigma)."""
    ratios = log_ratios(curves, hour)
    if len(ratios) < 5:
        # Too few comparables cover this hour for a median/MAD to mean
        # anything (a sparse comparable set in the final, say): fall back
        # to the flat baseline shape for this hour only.
        return [round2(p1 * math.exp(z * 0.15 * math.sqrt(hour - 1))) for z in z_values]
    mu = statistics.median(ratios)
    mad = statistics.median([abs(x - mu) for x in ratios])
    sigma = max(1.4826 * mad, SIGMA_FLOOR) * SIGMA_SHRINK
    qs = [round2(p1 * math.exp(mu + z * sigma)) for z in z_values]
    for i in range(1, len(qs)):
        qs[i] = max(qs[i], qs[i - 1])
    return qs


def forecast_case(case: dict, name: str) -> dict:
    levels = case["levels"]
    if levels == LEVELS:
        z_values = Z
    else:
        z_values = [norm_ppf(level) for level in levels]
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    curves = []
    for comp in case.get("comparables", []):
        series = usable_curve(comp)
        if series is not None:
            curves.append(series)
    if len(curves) < 3:
        # Too few launch curves to learn a shape from: the flat baseline.
        return {"hours": [{"hour": row["hour"],
                           "q": [round2(p1 * math.exp(z * 0.15 * math.sqrt(row["hour"] - 1)))
                                 for z in z_values]}
                          for row in case["forecast"]]}
    return {"hours": [{"hour": row["hour"],
                       "q": hour_quantiles(p1, curves, row["hour"], z_values)}
                      for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
