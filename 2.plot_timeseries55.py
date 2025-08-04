# -*- coding: utf-8 -*-
"""
Created on Aug2025

@author: Angelos
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta


# File paths
wd = r"E:\your\path\IDFs"
input_file = os.path.join(wd, "timeseries55.csv")


# Read only the Rain column to ensure that any date format won't be read in a problematic way.
df_rain = pd.read_csv(input_file, usecols=['Rain (mm)'])


# Create a complete daily date index
# (manually defining now start-end dates to ensure they're read properly)
start = datetime(1970, 1, 1)
end = datetime(2024, 12, 31)
date_index = pd.date_range(start, end, freq='D')


# Check:
# Ensure the rain series matches the date range length
if len(df_rain) != len(date_index):
    raise ValueError(f"Length mismatch: Rain series has {len(df_rain)} rows but date range has {len(date_index)} days.")


# Build a new DataFrame with correct dates
df = pd.DataFrame({
    'Date': date_index,
    'Rain (mm)': pd.to_numeric(df_rain['Rain (mm)'], errors='coerce')
})




##########  PLOTS  ################

# DAILY RAINFALL plot
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
ax.plot(df['Date'], df['Rain (mm)'], linewidth=0.5, color='tab:blue')
ax.set_title('Daily Rainfall Time Series (1970–2024)')
ax.set_xlabel('Year')
ax.set_ylabel('Rain (mm)')
ax.set_xlim(start, end)

# Tick every 10 years
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.YearLocator(10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()
ymax = df['Rain (mm)'].max()
ax.set_yticks([0, ymax*0.25, ymax*0.5, ymax*0.75, ymax])
ax.set_ylim(0, ymax*1.05)

plt.tight_layout()
plt.savefig(os.path.join(wd, 'daily_rainfall55.png'), dpi=300)
plt.close()



# MONTHLY mean + min-max plot
df['Month'] = df['Date'].dt.month
grp = df.groupby('Month')['Rain (mm)']
mn = grp.min()
mx = grp.max()
mean = grp.mean()

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
months = np.arange(1, 13)
ax.errorbar(months, mean,
            yerr=[mean - mn, mx - mean],
            fmt='o-', capsize=5, markersize=5, color='tab:blue')
ax.set_xticks(months)
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_title('Monthly Rainfall: Average 1970-2024 with Min–Max Range')
ax.set_xlabel('Month')
ax.set_ylabel('Rain (mm)')
plt.tight_layout()
plt.savefig(os.path.join(wd, 'monthly_mean_minmax55.png'), dpi=300)
plt.close()


# ANNUAL TOTAL precipitation with ticks every 10 years
df['Year'] = df['Date'].dt.year
annual = df.groupby('Year')['Rain (mm)'].sum()

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.bar(annual.index, annual.values, width=0.8, color='tab:purple')
ax.set_title('Annual Total Precipitation (1970–2024)')
ax.set_xlabel('Year')
ax.set_ylabel('Total Rain (mm)')

# Numeric ticks every 10 years
years = np.arange(1970, 2025, 10)
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years], rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(wd, 'annual_total55.png'), dpi=300)
plt.close()

print("Plots saved to:", wd)
