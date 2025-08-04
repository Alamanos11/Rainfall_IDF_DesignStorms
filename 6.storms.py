# -*- coding: utf-8 -*-
"""
Created on Aug2025

@author: Angelos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ─── Configuration ───────────────────────────────────────────────────────────
output_folder = r"E:\your\path\DesignStorms"
os.makedirs(output_folder, exist_ok=True)

# Input depths for each return period (input manually follwoing previous step)
depths = {
    10: 63.7,  # mm / 24 h
    25: 74.8,
    50: 83.1
}
D = 24  # duration in hours
hours = np.arange(1, D + 1)  # 1…24

# Prepare container for DataFrames
storms = {}

# ─── Generate hyetographs ─────────────────────────────────────────────────────
for T, xT in depths.items():
    avg = xT / D

    # 1) Uniform
    uni = np.full(D, avg)

    # 2) Triangular (symmetric peak at mid-storm)
    peak = 2 * avg
    tri = np.where(
        hours <= D/2,
        peak * (hours / (D/2)),
        peak * ((D - hours) / (D/2))
    )

    # 3) Chicago (Gamma PDF with mean D)
    alpha = 2.5
    theta = D / alpha
    f = (hours**(alpha - 1) * np.exp(-hours/theta)) / (np.math.gamma(alpha) * theta**alpha)
    chf = xT * (f / f.sum())

    storms[T] = pd.DataFrame({
        'Hour': hours,
        'Uniform (mm/h)': np.round(uni, 3),
        'Triangular (mm/h)': np.round(tri, 3),
        'Chicago (mm/h)': np.round(chf, 3)
    })

# ─── Save to Excel ─────────────────────────────────────────────────────────────

excel_path = os.path.join(output_folder, "DesignStorms.xlsx")
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    for T, df_storm in storms.items():
        sheet_name = f"T{T}yr"
        df_storm.to_excel(writer, sheet_name=sheet_name, index=False)
print(f"Excel file saved to: {excel_path}")



# ─── Plot & Save Figures ──────────────────────────────────────────────────────

for T, df_storm in storms.items():
    plt.figure(figsize=(8, 4))
    plt.plot(df_storm['Hour'], df_storm['Uniform (mm/h)'],
             label='Uniform', lw=2)
    plt.plot(df_storm['Hour'], df_storm['Triangular (mm/h)'],
             label='Triangular', lw=2)
    plt.plot(df_storm['Hour'], df_storm['Chicago (mm/h)'],
             label='Chicago', lw=2)
    plt.title(f'Design Storm (T={T} yr, 24 h depth={depths[T]} mm)')
    plt.xlabel('Hour of storm')
    plt.ylabel('Intensity (mm/h)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(output_folder, f"DesignStorm_T{T}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {fig_path}")



