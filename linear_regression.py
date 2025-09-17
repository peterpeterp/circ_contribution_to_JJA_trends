import scipy
import xarray as xr
import numpy as np

def get_slope_and_pval(Y_):

    def linregress_on_pixel(y, x):
        slope, _, _, p_value, _ = scipy.stats.linregress(x, y)
        return slope, p_value

    slope, p_value = xr.apply_ufunc(
        linregress_on_pixel,
        Y_,
        Y_.year,
        input_core_dims=[['year'], ['year']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',  # optional, if using Dask
        output_dtypes=[float, float]
    )
    return slope, p_value