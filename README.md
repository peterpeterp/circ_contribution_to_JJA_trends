# Decomposition of summer temperature trends in the northern hemisphere mid-latitudes into thermodynamic and circulation induced

## Main evaluation

The evaluation of decomposition methods including skill statistics and plots are created with `1_main_evaluation.ipynb`.

The plots for the application to ERA5 are created with `2_application_to_ERA5.ipynb`.

In `nudging_overview` the nudging experiments used for the evaluation of decomposition methods are analyzed.


## Decomposition methods:

### Ridge regression

Scripts in `ridge/`

Use `apply_ridge_to_CESM2.py` to apply the ridge regression to a CESM2 run and estimate trend contributions for some CESM2 piControl nudged run.

Use `apply_ridge_to_ERA5.py` to apply the ridge regression to ERA5.

### Circulation analogues

Please refer to methodology described in:

Merrifield, A., Lehner, F., Xie, S.-P., & Deser, C. (2017). Removing Circulation Effects to Assess Central U.S. Land-Atmosphere Interactions in the CESM Large Ensemble. Geophysical Research Letters, 44(19), 9938–9946. https://doi.org/10.1002/2017GL074831

### Direct effect analysis (DEA)

Scripts in `DEA/`

Use `apply_DEA_to_CESM2.py` to apply the DEA to a CESM2 run and estimate trend contributions for some CESM2 piControl nudged run.

Use `apply_DEA_to_ERA5.ipynb` to apply the DEA to ERA5.

### UNET

Please refer to code published with:

Cariou, E., Cattiaux, J., Qasmi, S., Ribes, A., Cassou, C., & Doury, A. (2025). Linking European Temperature Variations to Atmospheric Circulation With a Neural Network: A Pilot Study in a Climate Model. Geophysical Research Letters, 52(9), e2024GL113540. https://doi.org/10.1029/2024GL113540
