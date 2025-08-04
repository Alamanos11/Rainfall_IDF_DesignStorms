# -*- coding: utf-8 -*-
"""
Created on Aug2025

@author: Angelos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import gumbel_r, genextreme, weibull_min, gamma


# File paths
wd = r"E:\your\pth\distributions"
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


################

# Extract Annual Maxima Series (AMS)
df['Year'] = df['Date'].dt.year
ams = df.groupby('Year')['Rain (mm)'].max().reset_index()
print("Annual Maxima Series:")
print(ams.head(10))


# Fit Gumbel (EV1) distribution  - because we chosed this one as a best-fitting one
params_gumbel = gumbel_r.fit(ams['Rain (mm)'].dropna())
print(f"\nGumbel fit parameters (loc, scale): {params_gumbel}")


### FOR PLOTS AND GOODNESS OF FIT TESTS, SEE THE OTHER SCRIPT "fitting.py"  ###############

#  Compute and plot 24h IDF for Gumbel

# Fitted Gumbel parameters
mu, beta = params_gumbel

# Selected return periods
T = np.array([2, 5, 10, 25, 50, 100])


# Inverse CDF for Gumbel: xT = mu - beta * ln[-ln(1 - 1/T)]
p = 1 - 1/T
xT = mu - beta * np.log(-np.log(p))


# Build and print the IDF table
idf_table = pd.DataFrame({
    'Return period (yr)': T,
    'Depth 24 h (mm)': np.round(xT, 1),
    'Intensity (mm/h)': np.round(xT / 24, 3)
})
print("\n24 h IDF Table (Gumbel):")
print(idf_table.to_string(index=False))


# Plot IDF curve
plt.figure(figsize=(6,4))
plt.plot(T, xT, marker='o', lw=2)
plt.xscale('log')
plt.xticks(T, T)
plt.xlabel('Return period T (years)')
plt.ylabel('24 hour- Rainfall depth (mm)')
plt.title('Limassol 24 h IDF Curve (Gumbel, 1970–2024)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(wd, "IDF_24h_Gumbel55.png"), dpi=300)
plt.show()


