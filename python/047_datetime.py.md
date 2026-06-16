# Concept: datetime Module

## Concept ID

PYT-047

## Difficulty

Intermediate

## Domain

Python

## Module

Intermediate Python

## Learning Objectives

- Understand the `datetime` module and its core classes
- Create and manipulate `datetime`, `date`, `time`, and `timedelta` objects
- Format dates and times with `strftime()` and parse with `strptime()`
- Handle time zones with `timezone` and aware datetime objects
- Perform date arithmetic with `timedelta`
- Convert between timestamps and datetime objects
- Use `dateutil` for advanced date parsing
- Apply datetime concepts to real-world scheduling and logging

## Prerequisites

- Python numbers and basic arithmetic
- String formatting with f-strings
- Understanding of time zones and UTC

## Definition

The **`datetime`** module supplies classes for manipulating dates, times, and time intervals. The core classes are `date` (year, month, day), `time` (hour, minute, second, microsecond, tzinfo), `datetime` (combined date and time), `timedelta` (duration), `timezone` (fixed offset time zone), and `tzinfo` (abstract base class for time zone information).

## Intuition

Think of `datetime` as a calendar and clock combined into code. Instead of manually converting between seconds, hours, days, and months, the module handles all the complexity of calendar arithmetic — including leap years, month lengths, and Daylight Saving transitions. You work with meaningful units (days, months, years) rather than raw seconds.

## Why This Concept Matters

Date and time handling is notoriously error-prone in programming. Leap years, varying month lengths, time zones, and Daylight Saving Time create countless edge cases. The `datetime` module provides a robust, well-tested foundation for all time-related operations. It is essential for logging, scheduling, data analysis with timestamped data, financial calculations, and any application that deals with real-world time.

## Real World Examples

- Logging events with timestamps
- Scheduling tasks and recurring jobs
- Data analysis with time series
- Calculating age, durations, and deadlines
- Converting between time zones for international applications
- Expiration dates for sessions, caches, and tokens

## AI/ML Relevance

The `datetime` module in AI/ML:
- Handling timestamped dataset records and time series
- Logging training start/end times and durations
- Scheduling model retraining and evaluation jobs
- Working with temporal features (hour of day, day of week, month)
- Parsing date columns from CSV and database data
- Experiment tracking with timestamps

## Code Examples

### Example 1: Creating datetime objects

```python
from datetime import datetime, date, time, timedelta

# Current date and time
now = datetime.now()
print(f'Now: {now}')

# Create specific datetime
dt = datetime(2024, 12, 25, 10, 30, 0)
print(f'Christmas: {dt}')

# Create date and time separately
d = date(2024, 12, 25)
t = time(10, 30, 0)
print(f'Date: {d}, Time: {t}')

# Combine date and time
combined = datetime.combine(d, t)
print(f'Combined: {combined}')

# Output:
# Now: 2026-06-16 12:00:00.123456
# Christmas: 2024-12-25 10:30:00
# Date: 2024-12-25, Time: 10:30:00
# Combined: 2024-12-25 10:30:00
```

### Example 2: Accessing components

```python
from datetime import datetime

dt = datetime(2024, 12, 25, 10, 30, 45, 123456)

print(f'Year: {dt.year}')
print(f'Month: {dt.month}')
print(f'Day: {dt.day}')
print(f'Hour: {dt.hour}')
print(f'Minute: {dt.minute}')
print(f'Second: {dt.second}')
print(f'Microsecond: {dt.microsecond}')
print(f'Weekday (Mon=0): {dt.weekday()}')
print(f'Weekday (Mon=1): {dt.isoweekday()}')
print(f'ISO format: {dt.isoformat()}')

# Output:
# Year: 2024
# Month: 12
# Day: 25
# Hour: 10
# Minute: 30
# Second: 45
# Microsecond: 123456
# Weekday (Mon=0): 2
# Weekday (Mon=1): 3
# ISO format: 2024-12-25T10:30:45.123456
```

### Example 3: timedelta arithmetic

```python
from datetime import datetime, timedelta

now = datetime(2024, 1, 15, 10, 0, 0)

# Add time
future = now + timedelta(days=7, hours=3)
print(f'Future: {future}')

# Subtract time
past = now - timedelta(weeks=2)
print(f'Past: {past}')

# Difference between datetimes
diff = future - now
print(f'Difference: {diff}')
print(f'Days: {diff.days}')
print(f'Total seconds: {diff.total_seconds()}')

# Chaining timedelta operations
three_days_ago = now - timedelta(days=3)
print(f'Three days ago: {three_days_ago}')

# Output:
# Future: 2024-01-22 13:00:00
# Past: 2024-01-01 10:00:00
# Difference: 7 days, 3:00:00
# Days: 7
# Total seconds: 615600.0
# Three days ago: 2024-01-12 10:00:00
```

### Example 4: strftime and strptime

```python
from datetime import datetime

dt = datetime(2024, 12, 25, 10, 30, 45)

# Format datetime to string
print(dt.strftime('%Y-%m-%d'))
print(dt.strftime('%B %d, %Y at %I:%M %p'))
print(dt.strftime('%A, %B %d'))
print(dt.strftime('%c'))

# Parse string to datetime
date_str = '2024-12-25 10:30:45'
parsed = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
print(parsed)

# Common format codes
print(dt.strftime('%Y/%m/%d'))  # 2024/12/25
print(dt.strftime('%d-%b-%Y'))  # 25-Dec-2024

# Output:
# 2024-12-25
# December 25, 2024 at 10:30 AM
# Wednesday, December 25
# Wed Dec 25 10:30:45 2024
# 2024-12-25 10:30:45
# 2024/12/25
# 25-Dec-2024
```

### Example 5: Time zone handling

```python
from datetime import datetime, timezone, timedelta

# UTC time
utc_now = datetime.now(timezone.utc)
print(f'UTC: {utc_now}')

# Create time zones with fixed offsets
est = timezone(timedelta(hours=-5))
eastern = utc_now.astimezone(est)
print(f'Eastern: {eastern}')

# Convert between time zones
pst = timezone(timedelta(hours=-8))
pacific = utc_now.astimezone(pst)
print(f'Pacific: {pacific}')

# Naive vs aware datetime
naive = datetime(2024, 1, 1, 12, 0, 0)
aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
print(f'Naive timezone: {naive.tzinfo}')
print(f'Aware timezone: {aware.tzinfo}')

# Output:
# UTC: 2026-06-16 12:00:00+00:00
# Eastern: 2026-06-16 07:00:00-05:00
# Pacific: 2026-06-16 04:00:00-08:00
# Naive timezone: None
# Aware timezone: UTC
```

### Example 6: Timestamp conversion

```python
from datetime import datetime

# Unix timestamp (seconds since epoch)
timestamp = 1703500000
dt = datetime.fromtimestamp(timestamp)
print(f'From timestamp: {dt}')

utc_dt = datetime.utcfromtimestamp(timestamp)
print(f'From timestamp (UTC): {utc_dt}')

# Convert datetime to timestamp
now = datetime(2024, 12, 25, 10, 30, 0)
ts = now.timestamp()
print(f'Timestamp: {ts}')

# Round trip
reconstructed = datetime.fromtimestamp(ts)
print(f'Round trip: {reconstructed}')

# Output:
# From timestamp: 2024-12-25 10:26:40
# From timestamp (UTC): 2024-12-25 10:26:40
# Timestamp: 1735115400.0
# Round trip: 2024-12-25 10:30:00
```

### Example 7: Date range generation

```python
from datetime import datetime, timedelta

def date_range(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

start = datetime(2024, 1, 28)
end = datetime(2024, 2, 3)

for day in date_range(start, end):
    print(day.strftime('%a %Y-%m-%d'))

# Output:
# Sun 2024-01-28
# Mon 2024-01-29
# Tue 2024-01-30
# Wed 2024-01-31
# Thu 2024-02-01
# Fri 2024-02-02
# Sat 2024-02-03
```

### Example 8: AI/ML training logger with timestamps

```python
from datetime import datetime, timezone
import time

class TrainingLogger:
    def __init__(self):
        self.start_time = None
        self.logs = []

    def start_training(self):
        self.start_time = datetime.now(timezone.utc)
        self._log('Training started')

    def log_epoch(self, epoch, loss, accuracy):
        timestamp = datetime.now(timezone.utc)
        elapsed = timestamp - self.start_time
        self._log(
            f'Epoch {epoch}: loss={loss:.4f}, acc={accuracy:.4f}, '
            f'elapsed={elapsed.total_seconds():.1f}s'
        )

    def end_training(self):
        elapsed = datetime.now(timezone.utc) - self.start_time
        self._log(f'Training completed in {elapsed.total_seconds():.1f}s')

    def _log(self, message):
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = f'[{timestamp}] {message}'
        self.logs.append(entry)
        print(entry)

logger = TrainingLogger()
logger.start_training()
time.sleep(0.1)
logger.log_epoch(1, 0.5432, 0.8234)
time.sleep(0.05)
logger.log_epoch(2, 0.3210, 0.9145)
logger.end_training()

# Output:
# [2026-06-16T12:00:00.123456+00:00] Training started
# [2026-06-16T12:00:00.234567+00:00] Epoch 1: loss=0.5432, acc=0.8234, elapsed=0.1s
# [2026-06-16T12:00:00.289012+00:00] Epoch 2: loss=0.3210, acc=0.9145, elapsed=0.2s
# [2026-06-16T12:00:00.345678+00:00] Training completed in 0.2s
```

## Common Mistakes

1. Mixing naive and aware datetime objects (comparing or subtracting them raises TypeError)
2. Assuming `datetime.now()` returns UTC — it returns local time; use `datetime.now(timezone.utc)`
3. Forgetting that `timedelta` only stores days, seconds, and microseconds internally
4. Using `strptime` with mismatched format strings, causing ValueError
5. Confusing `datetime.fromtimestamp` (local time) with `datetime.utcfromtimestamp` (UTC)

## Interview Questions

### Beginner - 5

1. What is the difference between `date`, `time`, and `datetime` in the datetime module?
2. How do you get the current date and time?
3. What is `timedelta` and how do you use it?
4. How do you format a datetime object as a string?
5. How do you parse a date string into a datetime object?

### Intermediate - 5

1. What is the difference between naive and aware datetime objects?
2. How do you convert between time zones?
3. How do you get the Unix timestamp from a datetime and vice versa?
4. What is the difference between `strftime` and `strptime`?
5. How does `dateutil` extend the capabilities of the standard `datetime` module?

### Advanced - 3

1. How would you handle recurring events with Daylight Saving Time transitions?
2. Explain the `tzinfo` abstract base class and how to implement a custom time zone.
3. How would you implement a calendar scheduling system that handles recurrence rules and time zones correctly?

## Practice Problems

### Easy - 5

1. Print the current date and time in the format "YYYY-MM-DD HH:MM:SS".
2. Calculate how many days until your next birthday.
3. Create a datetime for New Year's Day 2025 at midnight.
4. Add 30 days to the current date and print the result.
5. Extract the day of the week from a given date.

### Medium - 5

1. Generate a list of all Mondays in a given month.
2. Write a function that converts a UTC datetime to multiple time zones.
3. Calculate the age of a person given their birth date.
4. Parse a log file timestamp and compute the time between entries.
5. Create a recurring event scheduler with daily, weekly, and monthly intervals.

### Hard - 3

1. Implement a calendar class that handles holidays, weekends, and business day arithmetic.
2. Build a time zone-aware meeting scheduler that finds overlapping availability across time zones.
3. Design a complete datetime parsing and validation system for a data pipeline that handles multiple date formats.

## Solutions

### Easy 1

```python
from datetime import datetime
now = datetime.now()
print(now.strftime('%Y-%m-%d %H:%M:%S'))

# Output:
# 2026-06-16 12:00:00
```

### Medium 1

```python
from datetime import datetime, timedelta

def get_mondays(year, month):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)

    mondays = []
    current = start
    while current < end:
        if current.weekday() == 0:
            mondays.append(current)
        current += timedelta(days=1)
    return mondays

mondays = get_mondays(2024, 1)
for m in mondays:
    print(m.strftime('%Y-%m-%d'))

# Output:
# 2024-01-01
# 2024-01-08
# 2024-01-15
# 2024-01-22
# 2024-01-29
```

### Hard 1 (simplified)

```python
from datetime import datetime, timedelta

class BusinessCalendar:
    def __init__(self, holidays=None):
        self.holidays = set(holidays or [])

    def is_business_day(self, dt):
        if dt.weekday() >= 5:
            return False
        date_only = dt.date()
        return date_only not in self.holidays

    def add_business_days(self, dt, n):
        current = dt
        added = 0
        while added < n:
            current += timedelta(days=1)
            if self.is_business_day(current):
                added += 1
        return current

    def business_days_between(self, start, end):
        days = []
        current = start
        while current <= end:
            if self.is_business_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days

cal = BusinessCalendar(holidays=[datetime(2024, 1, 1).date()])
result = cal.add_business_days(datetime(2024, 1, 5), 3)
print(f'3 business days from Jan 5: {result.date()}')

# Output:
# 3 business days from Jan 5: 2024-01-10
```

## Related Concepts

- Unix timestamps and epoch time
- ISO 8601 date/time format
- The `time` module (low-level time functions)
- The `calendar` module
- The `dateutil` third-party library
- Time zones and UTC

## Next Concepts

- JSON serialization (PYT-048)
- Logging (PYT-049)
- Unit testing (PYT-050)
- Time series analysis

## Summary

The `datetime` module provides comprehensive tools for working with dates, times, and durations in Python. With `date`, `time`, `datetime`, `timedelta`, and `timezone` classes, it handles everything from simple date arithmetic to complex time zone conversions. Proper datetime handling is essential for logging, scheduling, time series analysis, and any application that deals with real-world time.

## Key Takeaways

- `datetime` combines date and time; `date` and `time` are separate components
- `timedelta` represents durations and supports arithmetic with datetime objects
- `strftime()` formats datetimes to strings; `strptime()` parses strings to datetimes
- Aware datetimes include time zone info; naive ones do not
- Always use `timezone.utc` or a specific time zone to create aware datetimes
- Unix timestamps bridge Python datetime and external systems
- AI/ML: use datetime for experiment logging, time series features, and training duration tracking
