import xarray as xr
import xcdat as xc
import numpy as np
import pandas as pd

import itertools

import glmnet_python
from glmnet import glmnet; from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict

from sklearn_som.som import SOM
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score,r2_score

import multiprocessing

import scipy 

import os, cftime

import statsmodels.api as sm
lowess = sm.nonparametric.lowess

from scipy.interpolate import splrep, BSpline

#if os.path.isdir('/run/user/1567/cdo_tmp'):
from cdo import *
cdo_tmp_dir = '/climca/people/ppfleiderer/cdo_tmp'
cdo = Cdo(tempdir=cdo_tmp_dir)
cdo.forceOutput = False

import warnings

import matplotlib
import matplotlib.pyplot as plt
import cartopy

from rich.table import Table

from _data_minimal import *

def lon_to_360(dlon: float) -> float:
    return ((360 + (dlon % 360)) % 360)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)



class decomp():
    def __init__(self, target_variable, months, period):
        self._target_variable = target_variable
        self._months = np.array(months)
        self._data_raw = {}
        self._data = {}
        self._period = period
        self._data_tag = f"{self._name}_{self._run}_{'m'.join([str(m) for m in self._months])}"

    def copy(self):
        '''
        duplicates a decomp object
        can be useful to safe time for loading data
        '''
        if self._name == 'cc':
            c = decomp_CESM2_cc(self._run, self._target_variable, self._months, self._period)
        if self._name == 'piN':
            c = decomp_CESM2_piN(self._run, self._target_variable, self._months, self._period)
        if self._name == 'ERA5':
            c = decomp_ERA5(self._target_variable, self._months, self._period)
        for k,v in self.__dict__.items():
            c.__dict__[k] = v
        return c

    def subselect_time(self, y1, y2):
        '''
        select data between year (y1) and year (y2)
        for all dataset in _data_raw
        the data is then stored in _data
        '''
        self._data = {}
        for k,v in self._data_raw.items():
            new = v.copy()
            new.subselect_time(y1, y2)
            self._data[k] = new

    ###############
    # DATA        #
    ###############

    def get_cdo_input_string(self, **args):
        fls = self.get_raw_file_names(**args)
        if len(fls) == 1:
            return fls[0]
        if len(fls) > 1:
            return '-mergetime ' + ' '.join(fls)

    def inflate_monthly(self, y, mont_shift=0):
        x = self._data['target']._x.copy() * 0
        for year in np.unique(self._data['time']._x.time.dt.year):
            for month in self._months:
                val = y[(y.time.dt.year.values == year) & (y.time.dt.month.values == month - mont_shift)].values
                x[(self._data['time']._x.time.dt.year.values == year) & (self._data['time']._x.time.dt.month.values == month)] = val
        return x

    def set_location(self, lon, lat):
        self._lon = lon
        self._lat = lat

    def select_location(self, var_name):
        self._data[var_name] = data_1D(self._data_raw[var_name].sel(dict(lat=self._lat, lon=self._lon), method='nearest'))

    def extract_region(self, var_name, lat, lon, lon_extent=20, lat_extent=20, lon_shift=0, lat_shift=0):
        lon1,lon2 = lon-lon_extent+lon_shift, lon+lon_extent+lon_shift
        lat1,lat2 = lat-lat_extent+lat_shift, lat+lat_extent+lat_shift
        lons = self._data_raw[var_name].lon.values.copy()
        if lon2 > 360 and lon1 < 360:
            lons = lons[(lons >= lon1) | (lons <= lon2 - 360)]
            roll = np.sum(lons > 180)
        elif lon2 > 0 and lon1 < 0:
            lons = lons[(lons >= lon1 +360) | (lons <= lon2)]
            roll = np.sum(lons > 180)
        else:
            lons = lons[(lons >= lon1) & (lons <= lon2)]
            roll = 0
        lats = self._data_raw[var_name].lat.values.copy()
        lats = lats[(lats>lat1) & (lats<lat2)]
        x = self._data_raw[var_name].loc[:, lats, lons].copy()
        x = x.roll(lon = roll, roll_coords=True)
        return data_gridded(x)

    def select_region(self, var_name, lon_extent=20, lat_extent=20, lon_shift=0, lat_shift=0):
        self._data[var_name] = self.extract_region(var_name, self._lat, self._lon, lon_extent, lat_extent, lon_shift, lat_shift)

    ##############
    # target     #
    ##############

    def target_open(self, lon1, lon2, lat1, lat2, var_name_in_file=None, cdo_options=''):
        if var_name_in_file is None:
            var_name_in_file = self._target_variable
        fl = self.get_cdo_input_string(var=self._target_variable)
        tmp_file = cdo.sellonlatbox(f"{lon1},{lon2},{lat1},{lat2}", 
                input=f"{cdo_options} -selmon,{','.join([str(m) for m in self._months])} {fl}", 
                output=f"{cdo_tmp_dir}/{self._target_variable}_{self._data_tag}_{lon1},{lon2},{lat1},{lat2}.nc")
        target_raw = xr.open_dataset(tmp_file)[var_name_in_file].loc[str(self._period[0]):str(self._period[1])].squeeze()
        if target_raw.mean() > 200 and self._target_variable == 'TREFHT':
            target_raw -= 273.15

        # select only full seasons (relevant for DJF)
        self._days_per_year = target_raw.time[target_raw.time.dt.year == target_raw.time.dt.year[1000]].shape[0]
        l = []
        self._years = np.array([])
        for year in np.unique(target_raw.time.dt.year):
            if np.diff(self._months).max() > 1:
                x = target_raw[
                    (target_raw.time.dt.year == year) & np.isin(target_raw.time.dt.month,self._months[self._months < 6]) |\
                    (target_raw.time.dt.year == year - 1) & np.isin(target_raw.time.dt.month,self._months[self._months > 6])
                ]
            else:
                x = target_raw[target_raw.time.dt.year == year]
            if len(x) == self._days_per_year:
                l.append(x.time)
                self._years = np.append(self._years, year)
        self._time = xr.concat(l, dim='time')
        self._data_raw['target'] = target_raw.loc[self._time]
        self._data['time'] = data_1D(target_raw.loc[self._time].time)

    ##############
    # GMT        #
    ##############

    def gmt_open(self):
        '''
        load gmt and build an array of the same dimensions of target
        '''
        fls = self.get_raw_file_names(var='TREFHT', time_freq='ann', additional_tag='_GL')
        gmt_yearly = xr.open_mfdataset(fls)['TREFHT'].squeeze()
        self.gmt_bring_to_same_time_axis_as_target(gmt_yearly)

    def gmt_compute(self, var_name_in_file=None):
        '''
        load gmt and build an array of the same dimensions of target
        '''
        if var_name_in_file is None:
            var_name_in_file = 'TREFHT'
        fl = self.get_cdo_input_string(var='TREFHT')
        tmp_file = cdo.fldmean(input=f"-yearmean {fl}", output=f"{cdo_tmp_dir}/GMT_{self._data_tag}.nc")
        gmt_yearly = xr.open_dataset(tmp_file)[var_name_in_file].loc[str(self._period[0]):str(self._period[1])].squeeze()
        self.gmt_bring_to_same_time_axis_as_target(gmt_yearly)

    def gmt_bring_to_same_time_axis_as_target(self, gmt_yearly):
        gmt = self._data_raw['target'][:,0,0].squeeze().copy() * 0
        for year in np.unique(self._data_raw['target'].time.dt.year):
            gmt[self._data_raw['target'].time.dt.year.values == year] = gmt_yearly.loc[str(year)].values
        self._data['gmt'] = data_1D(gmt.loc[self._time].drop_vars(['lat', 'lon']))

    def gmt_adjust_to_pi(self):
        if np.unique(self._data['gmt']._x.loc[:'1950'].time.dt.year).shape[0] > 50:
            gmt = self._data['gmt']._x
            gmt -= gmt.loc[:'1950']
            self._data['gmt'] = data_1D(gmt)

    ##############
    # covariate  #
    ##############

    def cov_open(self, var_name, lon1, lon2, lat1, lat2, var_name_in_file=None, new_var_name=None, cdo_options=''):
        fl = self.get_cdo_input_string(var=var_name)
        if var_name_in_file is None:
            var_name_in_file = var_name
        if new_var_name is None:
            new_var_name = var_name        
        tmp_file = cdo.sellonlatbox(f"{lon1},{lon2},{lat1},{lat2}", 
                input=f"-selmon,{','.join([str(m) for m in self._months])} {cdo_options} {fl}", 
                output=f"{cdo_tmp_dir}/{var_name_in_file}_{self._data_tag}_{lon1},{lon2},{lat1},{lat2}_{cdo_options}.nc")
        self._data_raw[new_var_name] = xr.open_dataset(tmp_file)[var_name_in_file].loc[self._time].squeeze()

    def cov_remove_global_mean(self, var_name, var_name_in_file=None):
        if var_name_in_file is None:
            var_name_in_file = var_name
        x = self._data_raw[var_name]
        fl = self.get_cdo_input_string(var=var_name)
        tmp_file = cdo.fldmean(input=f'-selmon,{",".join([str(m) for m in self._months])} {fl}', 
                    output=f"{cdo_tmp_dir}/{var_name_in_file}_{self._data_tag}_GL.nc")
        x -= xr.open_dataset(tmp_file)[var_name_in_file].loc[self._time].squeeze()
        self._data_raw[var_name] = x

    ####################
    # Ridge regression #
    ####################

    def extract_X(self, variables: dict):
        # construct X
        labels = []
        columns = []
        for name,variable in variables.items():
            y = variable.with_feature_dim()
            columns.append(y.values)
            labels += [f'{name}'] * y.shape[-1] 

        X = xr.DataArray(np.concatenate(columns, axis=-1).reshape(-1, len(labels)), dims=['time','feature'], 
                                    coords=dict(time=self._data['time']._seasonal.values.flatten(), feature=np.array(labels).flatten()))

        return X

    def construct_X(self, variables, lags=[0]):
        target_days = self._data['time']._seasonal.day.values
        target_days = target_days[target_days >= np.max(lags)]
        labels = []
        columns = []
        for variable in variables:
            for lag in lags:
                y = self._data[variable].with_feature_dim().loc[:, target_days - lag, :]
                columns.append(y.values)
                labels += [f'{variable}_lag{lag}'] * y.shape[-1] 

        return xr.DataArray(np.concatenate(columns, axis=-1).reshape(-1, len(labels)), dims=['time','feature'], 
                                    coords=dict(time=self._data['time']._seasonal[:, target_days].values.flatten(), feature=np.array(labels).flatten()))
                                    
    def construct_target(self, target_variable='target', lags=[0]):
        target_days = self._data[target_variable]._seasonal.day.values
        target_days = target_days[target_days >= np.max(lags)]
        target = xr.DataArray(self._data[target_variable]._seasonal.loc[:, target_days].values.flatten(),
                                            dims=['time'], coords=dict(time=self._data['time']._seasonal[:, target_days].values.flatten()))
        return data_1D(target)

    def construct_penalty(self, penalty_modifiers):
        self._penalty_modifiers = penalty_modifiers
        self._penalty = np.ones([self._X.shape[1]])
        for v,mod in self._penalty_modifiers.items():
            self._penalty[np.array([v in var for var in self._X.feature.values])] = mod

    def fit(self, variables, target_variable='target', lags=[0], penalty_modifiers={'gmt':0}, cv=True, parallel_cores=4):
        self._variables = variables
        self._lags = lags
        self._X = self.construct_X(self._variables, lags=lags)
        self._target = self.construct_target(target_variable=target_variable, lags=lags)
        self.construct_penalty(penalty_modifiers)
        self.fit_only(cv=cv, parallel_cores=parallel_cores)

    def fit_only(self, cv, parallel_cores):
        self._cv = cv
        weights = self._target._seasonal.squeeze().copy().values.flatten() * 0. + 1.
        if self._cv:
            self._fit = cvglmnet(
                x = self._X.values.astype('float64').copy(), 
                y = self._target._x.values.astype('float64').copy(), 
                penalty_factor=self._penalty.astype('float64').copy(), 
                weights = weights.astype('float64').copy(),
                alpha = 0,
                parallel = parallel_cores,
                )
        else:
            self._fit = glmnet(
                x = self._X.values.astype('float64').copy(), 
                y = self._target._x.values.astype('float64').copy(), 
                penalty_factor=self._penalty.astype('float64').copy(), 
                weights = weights.astype('float64').copy(),
                alpha = 0
                )


    def sel_alpha(self, alpha=10**3):
        self._alpha=alpha
        if self._cv:
            self._coefs = cvglmnetCoef(self._fit, s = scipy.float64([self._alpha]))[:,0]
        else:
            self._coefs = glmnetCoef(self._fit, s = scipy.float64([self._alpha]), exact=False)[:,0]


    def reproduce(self, X):
        reproduction = self._target._x.copy()
        if self._cv:
            reproduction[:] = cvglmnetPredict(self._fit, X, ptype = 'response', s = scipy.float64([self._alpha]))[:,0].squeeze()
        else:
            reproduction[:] = glmnetPredict(self._fit, X, ptype = 'response', s = scipy.float64([self._alpha]))[:,0].squeeze()
        return data_1D(reproduction)


    def print_fit_summary_markdown(self):
        if np.unique([f.split('lag')[1] for f in self._X.feature.values]) == ['0']:
            features = np.unique([f.split('_lag')[0] for f in self._X.feature.values])
        else:
            features = np.unique(self._X.feature.values)
        t = pd.DataFrame([{
            'variable':f, 
            #'regularized':{0:' ', 1:'yes'}[self._penalty_modifiers[f]], 
            'features':f'{np.sum([f in ff for ff in self._X.feature.values])}'} for f in features])
        t.set_index('variable', inplace=True)
        print(t.to_markdown())
        print(f"\nalpha: {self._alpha}")
        print(f"\nvariance explained: {explained_variance_score(self._target._x.values, self.reproduce(self._X)._x.values)}")        

    def plot_coefs(self, var_name, lag=0, ax=None, maxabs=None):
        if ax is None:
            fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), subplot_kw={'projection': cartopy.crs.PlateCarree()})
        x = self._data[var_name].reconstruct_grid(self._coefs[1:][self._X.feature.values == f'{var_name}_lag{lag}'])
        ax.coastlines()
        if maxabs is None:
            maxabs = np.nanmax(np.abs(x))
        ax.pcolormesh(x.lon, x.lat, x, transform=cartopy.crs.PlateCarree(), cmap='RdBu_r', vmin=-maxabs, vmax=maxabs)
        ax.scatter(self._lon,self._lat, transform=cartopy.crs.PlateCarree(), color='g', marker='x')
        return ax

    ####################
    # Weather patterns #
    ####################

    def wp_som(self, m, n, circ_var, r=1, overwrite=False):
        self._m = m
        self._n = n
        self._circ_var = circ_var
        X = self.construct_X([self._circ_var])
        self._som = SOM(m=self._m, n=self._n, dim=X.shape[1])
        self._som.fit(X)
        centers = self._som.cluster_centers_
        self._wp_centers = xr.DataArray(
            centers.reshape((self._m*self._n, self._data[self._circ_var]._x.shape[1], self._data[self._circ_var]._x.shape[2])), 
            dims=['wp','lat','lon'], 
            coords=dict(wp=np.arange(self._m*self._n), lat=self._data[self._circ_var]._x.lat, lon=self._data[self._circ_var]._x.lon))
        self._wp_labels = data_1D(xr.DataArray(self._som.predict(X), dims=['time'], coords=dict(time=X.time)))
        #self.wp_select()

    def wp_select(self):
        self._labels = self._wp_labels
        self._label_dict = {}
        self._label_colors = {}
        self._label_map = pd.DataFrame(index=range(self._m), columns=range(self._n))
        for i,rc in enumerate(itertools.product(range(self._m),range(self._n))):
            r,c = rc
            self._label_dict[i] = dict(r=r, c=c)
            self._label_map.loc[r,c] = i
            self._label_colors[i] = matplotlib.cm.get_cmap('RdBu_r')(i / self._labels._x.max())

    def wp_plot_centers(self, anomalies=True):
        fig,axes = plt.subplots(nrows=self._m, ncols=self._n, figsize=(self._n*2,self._m*2), subplot_kw={'projection': cartopy.crs.PlateCarree()})
        circ_time_mean = self._data[self._circ_var]._x.mean('time')
        maxabs = np.nanmax(np.abs(self._wp_centers.values - circ_time_mean.values))
        for i,rc in self._label_dict.items():
            r,c = rc.values()
            ax = axes[r,c]
            ax.coastlines()
            Z = self._wp_centers[i]
            if anomalies:
                Z = Z - circ_time_mean
            ax.pcolormesh(self._wp_centers.lon, self._wp_centers.lat, Z, transform=cartopy.crs.PlateCarree(), 
                            cmap='RdBu_r', vmin=-maxabs, vmax=maxabs)
            ax.annotate(i, xy = (0.05,0.95), xycoords = 'axes fraction', ha='left', va='top')
            ax.scatter(self._lon,self._lat, transform=cartopy.crs.PlateCarree(), color='g', marker='x')
        return fig,axes
        
    def plot_table(self, t, centered=True, cmap='RdBu_r'):
        vals = np.array(t.values, float).round(2)
        if centered:
            maxabs = np.max(np.abs(vals))
            norm = plt.Normalize(-maxabs, maxabs)
        else:
            norm = plt.Normalize(vals.min(), vals.max())
        colours = matplotlib.cm.get_cmap(cmap)(norm(vals))
        fig = plt.figure(figsize=(t.shape[0], t.shape[1] / 3))
        ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
        the_table = plt.table(cellText=vals, rowLabels=t.index, colLabels=t.columns, 
                              #colWidths = [0.1] * t.shape[1], 
                              loc='center', 
                              cellColours=colours)
        plt.tight_layout()


class decomp_CESM2_cc(decomp):
    def __init__(self, run, target_variable, months, period):
        self._run = run
        self._name = 'cc'
        super().__init__(target_variable, months, period)

    def get_raw_file_names(self, var, time_freq='day', additional_tag=''):
        fls = []
        for source in ['/climca/data/CESM2-ETH', '/climca/people/ppfleiderer/CESM2-ETH_postprocessed']:
            for compset in ['b.e212.BHISTcmip6.f09_g17','b.e212.BSSP370cmip6.f09_g17']:
                fl = f"{source}/{compset}.{self._run}/{var}_{time_freq}_{compset}.{self._run}{additional_tag}.nc"
                if os.path.isfile(fl):
                    fls += [fl]
            if len(fls) > 0:
                return fls

class decomp_CESM2_piN(decomp):
    def __init__(self, run, target_variable, months, period):
        self._run = run
        self._name = 'piN'
        super().__init__(target_variable, months, period)

    def get_raw_file_names(self, var, time_freq='day', additional_tag=''):
        compset = 'b.e212.B1850cmip6.f09_g17.001.nudge-1850-2100-SSP370'
        for source in ['/climca/data/CESM2-ETH', '/climca/people/ppfleiderer/CESM2-ETH_postprocessed']:
            fl = f'{source}/{compset}.{self._run}.linear-weak/{var}_{time_freq}_{compset}.{self._run}.linear-weak{additional_tag}.nc'
            if os.path.isfile(fl):
                return [fl]

class decomp_CESM2_cc322(decomp):
    def __init__(self, run, target_variable, months, period):
        self._run = run
        self._name = 'cc322'
        super().__init__(target_variable, months, period)

    def get_raw_file_names(self, var, time_freq='day', additional_tag=''):
        case = f"b.e212.BHISTcmip6.f09_g17.{self._run}.nudge-HIST-1950-2014-322hPa.{self._run}"
        return [f'/climca/data/CESM2-ETH/{case}/{var}_{time_freq}_{case}{additional_tag}.nc']

class decomp_CESM2_cc691(decomp):
    def __init__(self, run, target_variable, months, period):
        self._run = run
        self._name = 'cc691'
        super().__init__(target_variable, months, period)

    def get_raw_file_names(self, var, time_freq='day', additional_tag=''):
        case = f"b.e212.BHISTcmip6.f09_g17.{self._run}.nudge-HIST-1950-2014-691hPa.{self._run}"
        return [f'/climca/data/CESM2-ETH/{case}/{var}_{time_freq}_{case}{additional_tag}.nc']





class decomp_ERA5(decomp):
    def __init__(self, target_variable, months, period):
        self._run = 'E5'
        self._name = 'ERA5'
        super().__init__(target_variable, months, period)

    def get_raw_file_names(self, var, time_freq='day', additional_tag=''):
        var = {
            'TREFHT':'t2m',
            'Z500'  :'geopot500hpa',
            }[var]
        search_string = f"/climca/data/ERA5/daily/{var}/{self._run}*_1D_*.nc"
        fls = glob.glob(search_string)
        return fls












































