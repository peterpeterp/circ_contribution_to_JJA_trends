import sys,itertools

import xarray as xr
import xcdat as xc
import numpy as np
import pandas as pd

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

sys.path.append('ridge')
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












































