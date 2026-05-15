"""
tokenizer.py — Encode conditions and dates as numbers for the model.

Input conditions:  [WED] [JAN] [False] [196]
Output date:       3-12-1962
"""

# ── Vocabularies ──────────────────────────────────────────────────────────────

DAYS   = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

MIN_YEAR   = 1800
MAX_YEAR   = 2200
MIN_DECADE = 180   # decade prefix of 1800
MAX_DECADE = 220   # decade prefix of 2200


# ── Parse a raw line ──────────────────────────────────────────────────────────

def parse_line(line):
    """
    Parse one line from data.txt.

    Example input:  "[WED] [JAN] [False] [196] 3-12-1962"

    Returns a dict:
        {
          "day": "WED", "month": "JAN",
          "leap": "False", "decade": 196,
          "d": 3, "m": 12, "y": 1962      ← only if date present
        }
    """
    tokens = line.strip().replace("[", "").replace("]", "").split()
    # tokens = ["WED", "JAN", "False", "196", "3-12-1962"]

    result = {
        "day":    tokens[0],
        "month":  tokens[1],
        "leap":   tokens[2],
        "decade": int(tokens[3]),
    }

    if len(tokens) == 5:          # full line (has date)
        d, m, y = tokens[4].split("-")
        result["d"] = int(d)
        result["m"] = int(m)
        result["y"] = int(y)

    return result


# ── Encode conditions → list of numbers (length 22) ──────────────────────────

def encode_conditions(day, month, leap, decade):
    """
    Convert conditions into a flat list of 22 floats.

    Layout:
        [0..6]   one-hot day of week  (7 values)
        [7..18]  one-hot month        (12 values)
        [19]     leap = 1.0 if True else 0.0
        [20]     decade normalised to [0, 1]
        [21]     unused (padding to keep dim=22)   ← kept for future use

    Returns list of 22 floats.
    """
    vec = [0.0] * 22

    # one-hot day
    vec[DAYS.index(day)] = 1.0

    # one-hot month
    vec[7 + MONTHS.index(month)] = 1.0

    # leap as a single float
    vec[19] = 1.0 if leap == "True" else 0.0

    # decade normalised
    vec[20] = (decade - MIN_DECADE) / (MAX_DECADE - MIN_DECADE)

    return vec


# ── Encode date → list of 3 floats ───────────────────────────────────────────

def encode_date(d, m, y):
    """
    Normalise day / month / year each to [0, 1].

    Returns list of 3 floats.
    """
    return [
        (d - 1) / 30.0,                              # day  in [1,31]
        (m - 1) / 11.0,                              # month in [1,12]
        (y - MIN_YEAR) / (MAX_YEAR - MIN_YEAR),      # year in [1800,2200]
    ]


# ── Decode model output → date string ────────────────────────────────────────

def decode_date(vec):
    """
    Convert model output (3 floats) back to a date string "d-m-yyyy".

    Clamps values to valid range automatically.
    """
    d = round(max(0.0, min(1.0, vec[0])) * 30) + 1
    m = round(max(0.0, min(1.0, vec[1])) * 11) + 1
    y = round(max(0.0, min(1.0, vec[2])) * (MAX_YEAR - MIN_YEAR)) + MIN_YEAR

    # clamp day to valid range for that month/year
    import calendar
    _, max_day = calendar.monthrange(y, m)
    d = max(1, min(d, max_day))

    return f"{d}-{m}-{y}"


# ── Check if a generated date satisfies the conditions ───────────────────────

def check_date(date_str, conditions):
    """
    Check whether date_str satisfies all 4 conditions.

    Parameters
    ----------
    date_str   : str   e.g. "3-12-1962"
    conditions : dict  output of parse_line()

    Returns dict with True/False for each condition.
    """
    import datetime
    import calendar

    try:
        d, m, y = [int(x) for x in date_str.split("-")]

        # weekday
        weekday_idx = datetime.date(y, m, d).weekday()  # 0=Mon … 6=Sun
        day_ok      = DAYS[weekday_idx] == conditions["day"]

        # month
        month_ok = MONTHS[m - 1] == conditions["month"]

        # leap year
        is_leap  = calendar.isleap(y)
        leap_ok  = str(is_leap) == conditions["leap"]

        # decade
        decade_start = conditions["decade"] * 10
        decade_ok    = decade_start <= y <= decade_start + 9
    except ValueError:
        # If date is invalid (e.g. Feb 30), all checks fail
        return {"day_ok": False, "month_ok": False, "leap_ok": False, "decade_ok": False, "all_ok": False}

    return {
        "day_ok":    day_ok,
        "month_ok":  month_ok,
        "leap_ok":   leap_ok,
        "decade_ok": decade_ok,
        "all_ok":    day_ok and month_ok and leap_ok and decade_ok,
    }
