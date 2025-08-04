# Rainfall_IDF_DesignStorms
Python workflow &amp; tools for hydro-meteorological analysis from rain to storms:

- We merged two publicly available sources (national meteorological archives and the ECA&D homogenized series) to assemble a continuous 108-year (1916–2024) daily rainfall time‐series for Limassol, Cyprus.
- 1.gap_filling.py - A tool providing options (mean, linear interpolation, moving-average) for filling missing values. It save the results as excel files, compares them through MAE and RMSE, and provides the respective plots.
- 2.plot_timeseries55.py - A tool for plotting the daily, monthly and annual timeseries. Makes sure any date format would work.
- 3.fitting.py - A tool fitting Gumbel, GEV, Log-Pearson III, and Weibull distributions, evaluating goodness-of-fit (using two different tests: Chi-Square and Kolmogorov-Smirnov). High-resolution plots and excel summary tables are saved.
- 4.idf.py - A tool to create the IDF Curve using the best-fitted distribution, and plotting it.
- 5.uncertainty.py - Draws uncertainty bands to the IDF curve (95% confidence bounds via bootstrap resampling). 
- 6.storms.py - Using this IDF curve, this script generates 24-hour design storms at multiple return periods (T=2,5,10,25,50,100). Each hyetograph is produced by three methods (uniform, triangular, and Chicago). High-resolution plots and excel summary tables are saved.


References:

1. Alamanos, A. & Nisiforou, O. (2025). A 108-year rainfall dataset, Intensity-Duration-Frequency Curves under uncertainty, and design storm generation for Limassol, Cyprus through an automated Python workflow. The 6th International Electronic Conference on Applied Sciences (ASEC2025). Online. 9-11 December, 2025.

2. Alamanos, A. (2025). An automated Python workflow for hydro-meteorological, IDF Curves and design storm analysis. DOI: . Available from: https://github.com/Alamanos11/Rainfall_IDF_DesignStorms
