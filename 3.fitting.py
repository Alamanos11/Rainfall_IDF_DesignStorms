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
wd = r"E:\your\path\distributions"
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

# 1. Extract Annual Maxima Series (AMS)
df['Year'] = df['Date'].dt.year
ams = df.groupby('Year')['Rain (mm)'].max().reset_index()
print("Annual Maxima Series:")
print(ams.head(10))


# 2. Fit Gumbel (EV1) distribution
params_gumbel = gumbel_r.fit(ams['Rain (mm)'].dropna())
print(f"\nGumbel fit parameters (loc, scale): {params_gumbel}")


# 3. Fit Generalized Extreme Value (GEV) distribution
# Note: scipy's genextreme takes c = -shape parameter
params_gev = genextreme.fit(ams['Rain (mm)'].dropna())
print(f"\nGEV fit parameters (shape, loc, scale): {params_gev}")



# 4. Fit Log‐Pearson III distribution
# Transform AMS by log10
log_ams = np.log10(ams['Rain (mm)'].dropna())
# Fit Gamma distribution to log10 data
# Gamma fit gives alpha, loc, scale
params_lp3 = gamma.fit(log_ams, floc=0)  # force loc=0 for Pearson III
print(f"\nLog-Pearson III (gamma) fit on log10(data) (alpha, loc, scale): {params_lp3}")




# 5. Fit Weibull distribution
params_weibull = weibull_min.fit(ams['Rain (mm)'].dropna(), floc=0)  # force loc=0
print(f"\nWeibull fit parameters (shape, loc, scale): {params_weibull}")



# Summary table of parameter estimates
dist_table = pd.DataFrame({
    'Distribution': ['Gumbel', 'GEV', 'Log-Pearson III', 'Weibull'],
    'Parameters': [
        f"loc={params_gumbel[0]:.3f}, scale={params_gumbel[1]:.3f}",
        f"shape={params_gev[0]:.3f}, loc={params_gev[1]:.3f}, scale={params_gev[2]:.3f}",
        f"alpha={params_lp3[0]:.3f}, loc={params_lp3[1]:.3f}, scale={params_lp3[2]:.3f}",
        f"shape={params_weibull[0]:.3f}, loc={params_weibull[1]:.3f}, scale={params_weibull[2]:.3f}"
    ]
})
print("\nSummary of fitted distribution parameters:")
print(dist_table.to_string(index=False))


########## PLOTS  ############

#  Plot fitted distributions vs empirical AMS ====

# Get your AMS array and fitted params (from previous code)
ams_values = ams['Rain (mm)'].dropna().values

# Define x-grid spanning the AMS range
x = np.linspace(ams_values.min()*0.8, ams_values.max()*1.2, 200)

# PDF for each fitted distribution
pdf_gumbel    = gumbel_r.pdf(x, *params_gumbel)
pdf_gev       = genextreme.pdf(x, *params_gev)
pdf_weibull   = weibull_min.pdf(x, *params_weibull)

# Log-Pearson III: transform x→y=log10(x), then back-transform PDF
# Only consider x>0
mask = x > 0
y = np.log10(x[mask])
pdf_gamma_y   = gamma.pdf(y, *params_lp3)                # f_Y(y)
pdf_lp3       = pdf_gamma_y / (x[mask] * np.log(10))     # f_X(x)

# Plotting
plt.figure(figsize=(8,5))
# Empirical histogram (density)
plt.hist(ams_values, bins=10, density=True, alpha=0.3, color='gray', label='Empirical AMS')

# Fitted PDFs
plt.plot(x, pdf_gumbel,    '-', lw=2, label='Gumbel (EV1)')
plt.plot(x, pdf_gev,       '--', lw=2, label='GEV')
plt.plot(x[mask], pdf_lp3, ':', lw=2, label='Log-Pearson III')
plt.plot(x, pdf_weibull,   '-.', lw=2, label='Weibull')

plt.xlabel('Annual max 24 h precipitation (mm)')
plt.ylabel('Probability density')
plt.title('Fitted Distributions vs Empirical AMS (1970–2024)')
plt.legend()
plt.tight_layout()

# Save the plot at high resolution
plt.savefig(os.path.join(wd, "comparison_plot55.png"), dpi=300)

plt.show()



#########  GOODNESS OF FIT TEST  ####################


from scipy.stats import chisquare, kstest

# --- Prepare observed counts for Chi-Square ---
# Choose number of bins (e.g. 8–12); here 10
k = 10
obs_counts, bin_edges = np.histogram(ams_values, bins=k)

# Container for results
results = []

# Define helper to run tests for a given dist name, CDF & PDF
def evaluate_fit(name, cdf_func, args):
    # Expected probabilities per bin
    expected_probs = cdf_func(bin_edges[1:], *args) - cdf_func(bin_edges[:-1], *args)
    expected_counts = expected_probs * len(ams_values)

    # Normalize expected counts to match total observed
    expected_counts *= obs_counts.sum() / expected_counts.sum()

    # Chi-Square
    chi2_stat, chi2_p = chisquare(obs_counts, f_exp=expected_counts)

    # K–S Test
    ks_stat, ks_p = kstest(ams_values, cdf_func, args=args)

    results.append({
        'Distribution': name,
        'Chi2 stat': f"{chi2_stat:.2f}",
        'Chi2 p-value': f"{chi2_p:.3f}",
        'KS stat': f"{ks_stat:.3f}",
        'KS p-value': f"{ks_p:.3f}"
    })


# Evaluate each fitted distribution
evaluate_fit(
    'Gumbel',
    gumbel_r.cdf,
    params_gumbel
)
evaluate_fit(
    'GEV',
    genextreme.cdf,
    params_gev
)
# Log-Pearson III: CDF of Gamma on log10(x)
def lp3_cdf(x, a, loc, scale):
    # Return F_Y(log10(x))
    return gamma.cdf(np.log10(x), a, loc, scale)

evaluate_fit(
    'Log-Pearson III',
    lp3_cdf,
    params_lp3
)
evaluate_fit(
    'Weibull',
    weibull_min.cdf,
    params_weibull
)

# Show results in a table
import pandas as pd
fit_table = pd.DataFrame(results)
print("\nGoodness-of-Fit Test Results:")
print(fit_table.to_string(index=False))



