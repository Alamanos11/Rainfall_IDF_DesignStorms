# -*- coding: utf-8 -*-
"""
Created on Aug2025

@author: Angelos
"""

"""
Standalone script to:
 1. Read 35-year daily rainfall series (1990–2024)
 2. Extract Annual Maxima Series (AMS)
 3. Fit a Gumbel distribution
 4. Bootstrap for 95% confidence bands on the IDF curve
 5. Plot the 24 h IDF curve with shaded uncertainty
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
from datetime import datetime

# ─── USER PARAMETERS ────────────────────────────────────────────────────────────

# Working directory and input file
wd = r"E:\ana8eseis ergon\4.IDF_Limassol\5.distributions"
input_file = os.path.join(wd, "timeseries55.csv")

# Column name in the CSV
rain_col = "Rain (mm)"

# Date range of the full series
start = datetime(1970, 1, 1)
end   = datetime(2024,12,31)

# Return periods to compute
T = np.array([2, 5, 10, 25, 50, 100])
p_T = 1 - 1/T

# Bootstrap settings
n_boot = 1000
rng = np.random.default_rng(seed=42)



# ─── 1) LOAD DAILY DATA ────────────────────────────────────────────────────────

# Read only the rain column
df_rain = pd.read_csv(input_file, usecols=[rain_col])

# Build full daily index
date_index = pd.date_range(start, end, freq='D')

# Check length match
if len(df_rain) != len(date_index):
    raise ValueError(
        f"Length mismatch: Rain series has {len(df_rain)} rows but "
        f"date range has {len(date_index)} days."
    )

# Assemble DataFrame with proper dates
df = pd.DataFrame({
    'Date': date_index,
    rain_col: pd.to_numeric(df_rain[rain_col], errors='coerce')
})


# ─── 2) EXTRACT ANNUAL MAXIMA SERIES ───────────────────────────────────────────

df['Year'] = df['Date'].dt.year
ams = df.groupby('Year')[rain_col].max().reset_index()
ams_values = ams[rain_col].dropna().values


# ─── 3) FIT GUMBEL DISTRIBUTION ────────────────────────────────────────────────

mu_full, beta_full = gumbel_r.fit(ams_values)
print(f"Gumbel parameters: loc (μ) = {mu_full:.3f}, scale (β) = {beta_full:.3f}")


# ─── 4) BOOTSTRAP FOR UNCERTAINTY ──────────────────────────────────────────────

xT_boot = np.zeros((n_boot, len(T)))
n = len(ams_values)

for i in range(n_boot):
    sample = rng.choice(ams_values, size=n, replace=True)
    loc_i, scale_i = gumbel_r.fit(sample)
    xT_boot[i, :] = loc_i - scale_i * np.log(-np.log(p_T))

# Compute percentiles
lower = np.percentile(xT_boot, 2.5, axis=0)
median = np.percentile(xT_boot, 50, axis=0)
upper = np.percentile(xT_boot, 97.5, axis=0)

# Full-sample IDF
xT_full = mu_full - beta_full * np.log(-np.log(p_T))

# ─── 5) PLOT IDF WITH CONFIDENCE BANDS ────────────────────────────────────────

plt.figure(figsize=(7,5))

# Shaded 95% confidence band
plt.fill_between(T, lower, upper, color='lightgray', label='95% confidence band')

# Median bootstrap curve
plt.plot(T, median, color='blue', lw=2, label='Median IDF (bootstrap)')

# Full-sample curve
plt.plot(T, xT_full, 'r--', lw=2, label='Full-sample IDF')

# Customize axes
plt.xscale('log')
plt.xticks(T, T)
plt.xlabel('Return period $T$ (years)')
plt.ylabel('24 h depth $x_T$ (mm)')
plt.title('24 h IDF Curve with 95% Confidence Bands\n(Gumbel fit, 1970–2024)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save and show
out_file = os.path.join(wd, "IDF_24h_Gumbel_with_CI55.png")
plt.savefig(out_file, dpi=300)
print(f"Plot saved to: {out_file}")
plt.show()
