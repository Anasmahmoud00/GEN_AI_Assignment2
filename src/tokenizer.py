"""
tokenizer.py — Encode conditions and dates as numbers for the model.

Input conditions:  [WED] [JAN] [False] [196]
Output date:       3-12-1962
"""

DAYS   = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

MIN_YEAR   = 1800
MAX_YEAR   = 2200
MIN_DECADE = 180
MAX_DECADE = 220


def parse_line(line):
    clean_line = line.strip()
    
    no_brackets_1 = clean_line.replace("[", "")
    no_brackets_2 = no_brackets_1.replace("]", "")
    
    tokens = no_brackets_2.split()

    result = {
        "day":    tokens[0],
        "month":  tokens[1],
        "leap":   tokens[2],
        "decade": int(tokens[3]),
    }

    if len(tokens) == 5:
        date_str = tokens[4]
        date_parts = date_str.split("-")
        
        day_part = date_parts[0]
        month_part = date_parts[1]
        year_part = date_parts[2]
        
        result["d"] = int(day_part)
        result["m"] = int(month_part)
        result["y"] = int(year_part)

    return result


def encode_conditions(day, month, leap, decade):
    vec = [0.0] * 22

    day_index = DAYS.index(day)
    vec[day_index] = 1.0

    month_index = MONTHS.index(month)
    target_index = 7 + month_index
    vec[target_index] = 1.0

    if leap == "True":
        vec[19] = 1.0
    else:
        vec[19] = 0.0

    top = decade - MIN_DECADE
    bottom = MAX_DECADE - MIN_DECADE
    vec[20] = top / bottom

    return vec


def encode_date(d, m, y):
    day_norm = (d - 1) / 30.0
    month_norm = (m - 1) / 11.0
    
    year_top = y - MIN_YEAR
    year_bottom = MAX_YEAR - MIN_YEAR
    year_norm = year_top / year_bottom
    
    return [day_norm, month_norm, year_norm]


def decode_date(vec):
    val0 = vec[0]
    if val0 < 0.0:
        val0 = 0.0
    if val0 > 1.0:
        val0 = 1.0
    d = round(val0 * 30) + 1
    
    val1 = vec[1]
    if val1 < 0.0:
        val1 = 0.0
    if val1 > 1.0:
        val1 = 1.0
    m = round(val1 * 11) + 1
    
    val2 = vec[2]
    if val2 < 0.0:
        val2 = 0.0
    if val2 > 1.0:
        val2 = 1.0
    y = round(val2 * (MAX_YEAR - MIN_YEAR)) + MIN_YEAR

    import calendar
    _, max_day = calendar.monthrange(y, m)
    
    if d > max_day:
        d = max_day
    if d < 1:
        d = 1

    return f"{d}-{m}-{y}"


def check_date(date_str, conditions):
    import datetime
    import calendar

    try:
        parts = date_str.split("-")
        d = int(parts[0])
        m = int(parts[1])
        y = int(parts[2])

        dt_obj = datetime.date(y, m, d)
        weekday_idx = dt_obj.weekday()
        day_name = DAYS[weekday_idx]
        
        target_day = conditions["day"]
        day_ok = (day_name == target_day)

        month_name = MONTHS[m - 1]
        target_month = conditions["month"]
        month_ok = (month_name == target_month)

        is_leap = calendar.isleap(y)
        target_leap = conditions["leap"]
        leap_ok = (str(is_leap) == target_leap)

        decade_start = conditions["decade"] * 10
        decade_end = decade_start + 9
        decade_ok = (decade_start <= y <= decade_end)
        
    except ValueError:
        return {
            "day_ok": False, 
            "month_ok": False, 
            "leap_ok": False, 
            "decade_ok": False, 
            "all_ok": False
        }

    all_ok = day_ok and month_ok and leap_ok and decade_ok
    
    return {
        "day_ok":    day_ok,
        "month_ok":  month_ok,
        "leap_ok":   leap_ok,
        "decade_ok": decade_ok,
        "all_ok":    all_ok,
    }

