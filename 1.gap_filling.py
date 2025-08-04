# -*- coding: utf-8 -*-
"""
Created on Aug2025

@author: Angelos
"""

import pandas as pd
import numpy as np
import os

# Define working directory and file paths
wd = r"E:\your\path\filling_missing_data"
input_file = os.path.join(wd, "timeseries_missing.xlsx")
output_mean    = os.path.join(wd, "filled_mean.xlsx")
output_interp  = os.path.join(wd, "filled_interp.xlsx")
output_movav   = os.path.join(wd, "filled_movav.xlsx")
output_summary = os.path.join(wd, "summary_table.xlsx")


# Read the Excel file, parsing the Date column
df = pd.read_excel(input_file, parse_dates=['Date'])


# Identify which rows have missing rainfall
mask_missing = df['Rain (mm)'].isna()



# Technique 1 = Mean (Global) Imputation
global_mean = df['Rain (mm)'].mean(skipna=True)
df_mean = df.copy()
df_mean['Rain (mm)'] = df['Rain (mm)'].fillna(global_mean)

# Technique 2 =Linear Interpolation
df_interp = df.copy()
df_interp['Rain (mm)'] = df['Rain (mm)'].interpolate(method='linear', limit_direction='both')

# Technique 3 = Moving-Average Imputation
window = 3  # e.g., 3-day window
rolling_mean = df['Rain (mm)'].rolling(window=window, min_periods=1, center=True).mean()
df_movav = df.copy()
df_movav['Rain (mm)'] = df['Rain (mm)'].fillna(rolling_mean)


# Save each filled series as its own Excel file
df_mean.to_excel(output_mean, index=False)
df_interp.to_excel(output_interp, index=False)
df_movav.to_excel(output_movav, index=False)



# Create and save summary table of only the imputed values
summary = pd.DataFrame({
    'Date': df.loc[mask_missing, 'Date'],
    'Filled by Mean': df_mean.loc[mask_missing, 'Rain (mm)'],
    'Filled by Interp': df_interp.loc[mask_missing, 'Rain (mm)'],
    'Filled by MovAv': df_movav.loc[mask_missing, 'Rain (mm)']
})
summary.to_excel(output_summary, index=False)

print("All files have been written to:", wd)



##########  Part B: Visualization  ######################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Raw data, pasted after I provided the original (correct) values (manually)
data = """
Date,Filled by Mean,Filled by Interp,Filled by MovAv,Original
9/27/1975,1.164,0,0,0
11/22/1975,1.164,4.1,4.1,1.3
10/8/1976,1.164,0,0,0
1/2/1978,1.164,14.65,14.65,7.2
6/14/1978,1.164,0,0,0
6/15/1978,1.164,0,0,0
6/16/1978,1.164,0,0,0
"""

# Load the data
df = pd.read_csv(StringIO(data), parse_dates=['Date'])

# Methods to compare
methods = ['Filled by Mean', 'Filled by Interp', 'Filled by MovAv']

# Calculate errors
errors = {m: df[m] - df['Original'] for m in methods}

# Compute MAE and RMSE
for m in methods:
    e = errors[m]
    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e**2))
    print(f"{m}: MAE = {mae:.3f}, RMSE = {rmse:.3f}")

# Bar chart: imputed vs observed
x = np.arange(len(df))
width = 0.2
plt.figure()
for i, m in enumerate(methods):
    plt.bar(x + (i-1)*width, df[m], width, label=m)
plt.scatter(x, df['Original'], color='black', marker='x', label='Original')
plt.xticks(x, df['Date'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
plt.ylabel('Rain (mm)')
plt.title('Imputed vs Observed Rainfall')
plt.legend()
plt.tight_layout()
plt.show()

# Signed-error plot
plt.figure()
for m in methods:
    plt.plot(df['Date'], errors[m], marker='o', label=m)
plt.axhline(0, linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Error (mm)')
plt.title('Signed Error of Imputation Methods')
plt.legend()
plt.tight_layout()
plt.show()
