import os,glob,sys
import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
import dask
from direct_effect_analysis import DirectEffectAnalysis
sys.path.append('../')
from linear_regression import get_slope_and_pval

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run_train", type=str)
parser.add_argument("--run_test", type=str)
parser.add_argument("--period", type=str)
args = parser.parse_args()
for k,v in vars(args).items():
    globals()[k] = v

# Define summer months (June, July, August)
summer_months = [6, 7, 8]
gmst_rolling_window_size_in_days = 30
y1,y2 = period.split('-')

def preprocessing(nc):
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        nc = nc.sel(time=nc['time.year'].isin(np.arange(int(y1), int(y2)+1, 1)))
    return nc

def get_histssp370_files(var, run):
    fls = []
    if int(y1) < 2015 and int(y2) >= 2015:
        compsets = ['b.e212.BHISTcmip6.f09_g17','b.e212.BSSP370cmip6.f09_g17']
    elif int(y1) >= 2015:
        compsets = ['b.e212.BSSP370cmip6.f09_g17']

    for compset in compsets:
        fl = f"/climca/data/CESM2-ETH/{compset}.{run}/{var}_day_{compset}.{run}.nc"
        if os.path.isfile(fl):
            fls += [fl]
    return fls

def get_nudged_files(var, run):
    compset = 'b.e212.B1850cmip6.f09_g17.001.nudge-1850-2100-SSP370'
    fl = f'/climca/data/CESM2-ETH/{compset}.{run}.linear-weak/{var}_day_{compset}.{run}.linear-weak.nc'
    if os.path.isfile(fl):
        return fl

with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    nc_trefht_recent = xr.open_mfdataset(get_histssp370_files('TREFHT', run_train), preprocess=preprocessing)
    trefht_recent = nc_trefht_recent.sel(time=nc_trefht_recent['time.month'].isin(summer_months))
    
    nc_trefht_nudge = xr.open_mfdataset(get_nudged_files('TREFHT', run_train), preprocess=preprocessing)
    trefht_nudge = nc_trefht_nudge.sel(time=nc_trefht_nudge['time.month'].isin(summer_months))
    
    nc_z500_recent = xr.open_mfdataset(get_histssp370_files('Z500', run_train), preprocess=preprocessing)
    c = 6 # Coarsening the atmospheric circulation to avoid overfitting
    z500_recent = nc_z500_recent['Z500'].sel(time=nc_z500_recent['time.month'].isin(summer_months)).coarsen(lat=c, lon=c, boundary='trim').mean()
    z500_global_mean = nc_z500_recent['Z500'].sel(time=nc_z500_recent['time.month'].isin(summer_months)).weighted(np.cos(np.radians(nc_z500_recent.lat))).mean(('lat','lon'))
    z500_recent -= z500_global_mean

# select relevant grid-cells
trefht_recent = trefht_recent.sel(lat=slice(30, 70))
trefht_nudge = trefht_nudge.sel(lat=slice(30, 70))
z500_recent = z500_recent.sel(lat=slice(-10, 90))

# remove seasonality
z500_recent = (z500_recent.groupby('time.month') - z500_recent.groupby('time.month').mean())
trefht_recent = (trefht_recent.groupby('time.month') - trefht_recent.groupby('time.month').mean())
trefht_nudge = (trefht_nudge.groupby('time.month') - trefht_nudge.groupby('time.month').mean())

gmst_recent = nc_trefht_recent.weighted(np.cos(np.radians(nc_trefht_recent.lat))).mean(('lat','lon'))
gmst_recent = gmst_recent.rolling(time=gmst_rolling_window_size_in_days, center=True).mean()
gmst_recent = gmst_recent.sel(time=nc_trefht_recent['time.month'].isin(summer_months))['TREFHT'].data[:, None]

gmst_nudge = nc_trefht_nudge.weighted(np.cos(np.radians(nc_trefht_nudge.lat))).mean(('lat','lon'))
gmst_nudge = gmst_nudge.rolling(time=gmst_rolling_window_size_in_days, center=True).mean()
gmst_nudge = gmst_nudge.sel(time=nc_trefht_recent['time.month'].isin(summer_months))['TREFHT'].data[:, None]

gmst_recent = gmst_recent - gmst_recent.mean()
gmst_nudge = gmst_nudge - gmst_nudge.mean()

# Saving lat, lon and time
lats = trefht_recent.lat.data
lons = trefht_recent.lon.data
time = trefht_recent.time.data

# Converting xarray to numpy array of correct dimensions
X_2d = z500_recent.values.reshape((len(time), -1))
Y_2d = trefht_recent.TREFHT.values.reshape((len(time), -1))

# training
X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X_2d, Y_2d, gmst_recent, test_size=0.2)
n_cps = np.logspace(0.15, 2.2, 20).astype('int')
dea = DirectEffectAnalysis(n_components='optimal', n_cps=n_cps, k_fold=5)
dea.fit(X_train, Y_train, Z_train, fit_test=False)

def transform_to_xarray(Y_):
    Y_ = xr.DataArray(
    Y_.reshape((-1, len(lats), len(lons))),
        dims = trefht_nudge.dims,
        coords = trefht_nudge.coords)
    Y_ = Y_.assign_coords(year=Y_.time.dt.year)
    Y_ = Y_.swap_dims({'time': 'year'})
    Y_ = Y_.groupby('year').mean()
    return Y_

# prediction
Y_dyn, Y_thermo = dea.counterfactual(Y_2d, gmst_nudge)

slope, pval = get_slope_and_pval(transform_to_xarray(Y_dyn))
xr.Dataset(
    {'slope':slope, 'pval':pval}
).to_netcdf(f'/climca/people/ppfleiderer/decomposition/DEA_homer/train{run_train}_test{run_train}_trend_{period}_dyn.nc')

slope, pval = get_slope_and_pval(transform_to_xarray(Y_thermo))
xr.Dataset(
    {'slope':slope, 'pval':pval}
).to_netcdf(f'/climca/people/ppfleiderer/decomposition/DEA_homer/train{run_train}_test{run_train}_trend_{period}_thermo.nc')

####################
# Cross-validation #
####################
with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    nc_trefht_recent = xr.open_mfdataset(get_histssp370_files('TREFHT', run_test), preprocess=preprocessing)
    trefht_recent = nc_trefht_recent.sel(time=nc_trefht_recent['time.month'].isin(summer_months))
    
    nc_trefht_nudge = xr.open_mfdataset(get_nudged_files('TREFHT', run_test), preprocess=preprocessing)
    trefht_nudge = nc_trefht_nudge.sel(time=nc_trefht_nudge['time.month'].isin(summer_months))

# remove seasonality
trefht_recent = trefht_recent.sel(lat=slice(30, 70))
seasonality = trefht_recent.groupby('time.month').mean()
trefht_recent = (trefht_recent.groupby('time.month') - seasonality)

gmst_nudge = nc_trefht_nudge.weighted(np.cos(np.radians(nc_trefht_nudge.lat))).mean(('lat','lon'))
gmst_nudge = gmst_nudge.rolling(time=gmst_rolling_window_size_in_days, center=True).mean()
gmst_nudge = gmst_nudge.sel(time=nc_trefht_recent['time.month'].isin(summer_months))['TREFHT'].data[:, None]

Y_2d = trefht_recent.TREFHT.values.reshape((len(time), -1))
Y_dyn, Y_thermo = dea.counterfactual(Y_2d, gmst_nudge)
Y_rec = Y_dyn + Y_thermo

slope, pval = get_slope_and_pval(transform_to_xarray(Y_dyn))
xr.Dataset(
    {'slope':slope, 'pval':pval}
).to_netcdf(f'/climca/people/ppfleiderer/decomposition/DEA_homer/train{run_train}_test{run_test}_trend_{period}_dyn.nc')

slope, pval = get_slope_and_pval(transform_to_xarray(Y_thermo))
xr.Dataset(
    {'slope':slope, 'pval':pval}
).to_netcdf(f'/climca/people/ppfleiderer/decomposition/DEA_homer/train{run_train}_test{run_test}_trend_{period}_thermo.nc')
