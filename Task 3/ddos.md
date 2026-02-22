Below is one complete script you can copy–paste into ddos_analysis.py.
It:

Reads NT.csv,

Runs regression-based DDoS detection,

Prints attack intervals,

Saves task_3/ddos_regression_plot.png,

Writes a detailed report task_3/ddos.md you can commit to GitHub.

python
"""
ddos_analysis.py

Regression-based DDoS detection on HTTP access logs.

Requirements:
    pip install pandas numpy scikit-learn matplotlib
"""

import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ============================================================
# 0. Configuration
# ============================================================

# Path to the event log file.
# Place NT.csv next to this script (or adjust the path).
LOG_PATH = "NT.csv"

# Output folder required by the task.
OUTPUT_DIR = "task_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Load and inspect data
# ============================================================

# NT.csv is a plain text HTTP access log (one entry per line).
# We read it as a single-column CSV.
logs = pd.read_csv(LOG_PATH, header=None, names=["raw"])

# Example row (for reference):
# 183.143.217.177 - - [2024-03-22 18:02:26+04:00] "GET /usr/login HTTP/1.0" 303 5046 ...


# ============================================================
# 2. Parse timestamps from log lines
# ============================================================

# We extract the part inside [...] which is an ISO-like datetime string.
# Example: [2024-03-22 18:02:26+04:00]

timestamp_pattern = re.compile(r"\[(?P<ts>[^\]]+)\]")


def parse_ts(line):
    """
    Extract timestamp from a raw log line and convert to datetime.

    Returns:
        datetime object if successful, otherwise None.
    """
    line = str(line)
    match = timestamp_pattern.search(line)
    if not match:
        return None
    ts_str = match.group("ts")
    # ts_str example: "2024-03-22 18:02:26+04:00"
    # datetime.fromisoformat can parse this.
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


# Apply timestamp parsing to each row.
logs["ts"] = logs["raw"].apply(parse_ts)

# Drop rows where timestamp could not be parsed.
logs = logs.dropna(subset=["ts"])

# If there is no data after parsing, stop early.
if logs.empty:
    raise RuntimeError("No valid timestamps parsed from NT.csv. Check file format.")


# ============================================================
# 3. Aggregate requests by second
# ============================================================

# We floor each timestamp to the second to build a time series of request counts.
logs["ts_floor"] = logs["ts"].dt.floor("S")

# Count number of requests in each second.
counts = (
    logs.groupby("ts_floor")
    .size()
    .reset_index(name="requests")
    .sort_values("ts_floor")
)

# Create a numeric time variable t = seconds from start.
t0 = counts["ts_floor"].min()
counts["t"] = (counts["ts_floor"] - t0).dt.total_seconds()

# Features X (time) and target y (request count).
X = counts["t"].values.reshape(-1, 1)
y = counts["requests"].values


# ============================================================
# 4. Fit a linear regression baseline
# ============================================================

# We fit a simple linear model:
#   y_hat(t) = a * t + b
# This models the "baseline" trend of requests.
reg = LinearRegression().fit(X, y)

# Predicted baseline request counts.
counts["pred"] = reg.predict(X)

# Residuals = actual - predicted
counts["resid"] = counts["requests"] - counts["pred"]


# ============================================================
# 5. Detect anomalies (DDoS candidates) using residuals
# ============================================================

# We define a threshold based on the residual distribution:
#   threshold = mean(resid) + 3 * std(resid)
# Any second where residual > threshold is treated as suspiciously high.
resid_mean = counts["resid"].mean()
resid_std = counts["resid"].std()
threshold = resid_mean + 3 * resid_std

counts["ddos_flag"] = counts["resid"] > threshold


# ============================================================
# 6. Group consecutive anomalous seconds into attack intervals
# ============================================================

intervals = []
current_start = None
prev_time = None

for ts, flag in zip(counts["ts_floor"], counts["ddos_flag"]):
    if flag:
        # If we just entered an anomalous region, mark its start.
        if current_start is None:
            current_start = ts
        prev_time = ts
    else:
        # If we just left an anomalous region, close the interval.
        if current_start is not None:
            intervals.append((current_start, prev_time))
            current_start = None
            prev_time = None

# Close last interval if file ends during an attack.
if current_start is not None:
    intervals.append((current_start, prev_time))


# ============================================================
# 7. Print detected intervals (for console)
# ============================================================

print("Detected DDoS intervals (regression-based):")
if not intervals:
    print("- No intervals exceeded the threshold with the current settings.")
else:
    for s, e in intervals:
        print(f"- {s} to {e}")


# ============================================================
# 8. Visualization: Requests, regression, anomalies
# ============================================================

plt.figure(figsize=(10, 5))

# Plot raw request counts per second.
plt.plot(
    counts["ts_floor"],
    counts["requests"],
    label="Requests per second",
    alpha=0.7,
)

# Plot linear regression baseline.
plt.plot(
    counts["ts_floor"],
    counts["pred"],
    label="Linear regression (baseline)",
    color="orange",
    linewidth=2,
)

# Highlight points flagged as potential DDoS.
ddos_points = counts[counts["ddos_flag"]]
plt.scatter(
    ddos_points["ts_floor"],
    ddos_points["requests"],
    color="red",
    label="DDoS candidate",
    zorder=5,
)

plt.xlabel("Time")
plt.ylabel("Requests per second")
plt.title("HTTP Request Rate and Regression-based DDoS Detection")
plt.legend()
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "ddos_regression_plot.png")
plt.savefig(plot_path, dpi=150)
plt.close()


# ============================================================
# 9. Create Markdown report ddos.md
# ============================================================

# Format intervals for Markdown.
if intervals:
    intervals_str = "\n".join(f"- {s} to {e}" for s, e in intervals)
else:
    intervals_str = "- No intervals exceeded the threshold with the current settings."

# Link to the event log file as required by the task.
# (Assuming you will upload NT.csv into the same task_3 folder in GitHub.)
log_link = "NT.csv"

report_md = f"""# DDoS Detection in Web Server Logs (Regression Approach)

This report describes how a distributed denial-of-service (DDoS) attack was detected
in the provided HTTP access log using regression-based analysis of request rates.

## Data

The analysis is based on the event log file [`NT.csv`]({log_link}), which should be
stored in the same `task_3` folder of the GitHub repository.

Each line corresponds to a single HTTP request in a common log format and includes
a timestamp such as:

```text
183.143.217.177 - - [2024-03-22 18:02:26+04:00] "GET /usr/login HTTP/1.0" 303 5046 ...
Methodology
Parsing timestamps
We extract the datetime value from the square brackets in each log line and
convert it to a Python datetime object.

Aggregation by second
We floor each timestamp to the nearest second and count how many requests
occur in that second, obtaining a time series of request counts.

Regression baseline
We define a numeric time variable \(t\) (seconds since the beginning of the log)
and fit a simple linear regression model

\[
\hat{{y}}(t) = a t + b
\]

where \(y(t)\) is the number of requests in second \(t\). This models the
expected baseline activity over time.

Residual analysis
For each second we compute the residual

\[
r(t) = y(t) - \hat{{y}}(t),
\]

which measures how much the actual traffic deviates from the baseline.
Large positive residuals indicate unusually high activity.

Anomaly threshold and intervals
We compute the mean and standard deviation of the residuals and define

\[
\text{{threshold}} = \text{{mean}}(r) + 3 \cdot \text{{std}}(r).
\]

All seconds where \(r(t)\) exceeds this threshold are marked as
DDoS candidates. Consecutive candidate seconds are merged into
continuous attack intervals.

Detected DDoS time interval(s)
The following time interval(s) were identified as DDoS attack periods based on
the regression residuals exceeding the chosen threshold:

{intervals_str}

All timestamps are in the original log timezone (e.g., +04:00).

Main source code fragments
Below are the key parts of the Python code used for this analysis. The full script
is provided in the same task_3 folder as ddos_analysis.py.

Loading and parsing the log
python
import pandas as pd
import re
from datetime import datetime

logs = pd.read_csv("NT.csv", header=None, names=["raw"])

timestamp_pattern = re.compile(r"\\[(?P<ts>[^\\]]+)\\]")

def parse_ts(line):
    line = str(line)
    match = timestamp_pattern.search(line)
    if not match:
        return None
    ts_str = match.group("ts")
    return datetime.fromisoformat(ts_str)

logs["ts"] = logs["raw"].apply(parse_ts)
logs = logs.dropna(subset=["ts"])
Aggregating by second and fitting regression
python
logs["ts_floor"] = logs["ts"].dt.floor("S")
counts = logs.groupby("ts_floor").size().reset_index(name="requests")
counts = counts.sort_values("ts_floor")

counts["t"] = (counts["ts_floor"] - counts["ts_floor"].min()).dt.total_seconds()

from sklearn.linear_model import LinearRegression
import numpy as np

X = counts["t"].values.reshape(-1, 1)
y = counts["requests"].values

reg = LinearRegression().fit(X, y)
counts["pred"] = reg.predict(X)
counts["resid"] = counts["requests"] - counts["pred"]
Flagging anomalies and extracting attack intervals
python
thr = counts["resid"].mean() + 3 * counts["resid"].std()
counts["ddos_flag"] = counts["resid"] > thr

intervals = []
current_start = None
prev_time = None
for ts, flag in zip(counts["ts_floor"], counts["ddos_flag"]):
    if flag:
        if current_start is None:
            current_start = ts
        prev_time = ts
    else:
        if current_start is not None:
            intervals.append((current_start, prev_time))
            current_start = None
            prev_time = None
if current_start is not None:
    intervals.append((current_start, prev_time))
Visualization
The figure ddos_regression_plot.png (stored in task_3) shows:

Request counts per second (blue line),

The fitted linear regression baseline (orange line),

Points flagged as potential DDoS activity (red dots).

This provides a visual confirmation of the detected spikes above the baseline.

Reproducibility
To reproduce the analysis:

Place NT.csv and ddos_analysis.py in the task_3 folder.

Install the required Python packages:

bash
pip install pandas numpy scikit-learn matplotlib
Run the script:

bash
python ddos_analysis.py
Check the console output for the list of detected DDoS intervals and inspect
task_3/ddos_regression_plot.png and task_3/ddos.md.
"""

Write the report to task_3/ddos.md
report_path = os.path.join(OUTPUT_DIR, "ddos.md")
with open(report_path, "w", encoding="utf-8") as f:
f.write(report_md)

text

After running:

```bash
pip install pandas numpy scikit-learn matplotlib
python ddos_analysis.py
you’ll have everything required in task_3 to commit to GitHub.

