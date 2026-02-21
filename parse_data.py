import pandas as pd
from datetime import datetime, timedelta

CSV_PATH = "data/output.csv"
df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
now = datetime.utcnow()

# Last hour data
last_hour_df = df[df['timestamp'] > now - timedelta(hours=1)]
last_hour_df.to_csv("data/last_hour.csv", index=False)

# Day summary
day_df = df[df['timestamp'].dt.date == now.date()]
day_df.to_csv("data/day.csv", index=False)

# Week summary
week_df = df[df['timestamp'] > now - timedelta(days=7)]
week_df.to_csv("data/week.csv", index=False)

# Stats: observations per hour (last day)
last_day_df = df[df['timestamp'] > now - timedelta(days=1)]
stats = last_day_df.groupby(last_day_df['timestamp'].dt.hour).size()
stats.to_csv("data/stats.csv")
