import sys, os, glob

import xarray as xr
import numpy as np

class _data():
    def __init__(self, x):
        self.preprocess(x)

    def preprocess(self, x):
        self._x = x
        self.time = self._x.time
        self._years = np.unique(self._x.time.dt.year.values)
        self._days_per_year = self.time[self.time.dt.year == self._years[2]].shape[0]
        # relevant for DJF
        if self._years.shape[0] != self._x.shape[0] / self._days_per_year:
            self._years = self._years[1:]

        self._months = np.unique(self._x.time.dt.month.values)
        
        self.preprocess_specific()

    def subselect_time(self, y1, y2):
        self.preprocess(self._x.loc[str(y1):str(y2)])

    def copy(self):
        return self._class_func(self._x)

    def with_feature_dim(self):
        if 'feature' not in self._seasonal.dims:
            return xr.DataArray(self._seasonal.values.reshape((len(self._years),self._days_per_year, 1)), dims=['year','day','feature'], 
                                                coords=dict(year=self._years, day=range(self._days_per_year), feature=[0]))
        else:
            return self._seasonal

    def year_mean(self):
        return self._seasonal.mean('day').squeeze()

    def year_max(self):
        return self._seasonal.max('day').squeeze()

    def year_min(self):
        return self._seasonal.min('day').squeeze()

    def mX5(self):
        return self._seasonal.rolling(day=5).mean().max('day').squeeze()

    def summer_days(self):
        return(self._seasonal > 25).sum('day').squeeze()
        

class data_gridded(_data):
    def __init__(self, x):
        self._class_func = data_gridded
        super().__init__(x)

    def preprocess_specific(self):
        self._flat_all = self._x.values.reshape((self._x.shape[0],-1))
        self._lat = self._x.lat
        self._lon = self._x.lon
        self._used_cells = np.where(np.isfinite(self._flat_all.mean(axis=0)))[0]
        self._flat = self._flat_all[:,self._used_cells]
        self._seasonal = xr.DataArray(self._flat.reshape((len(self._years),self._days_per_year, self._flat.shape[1])), dims=['year','day','feature'], 
                                                coords=dict(year=self._years, day=range(self._days_per_year), feature=np.arange(0,self._flat.shape[1],1,'int')))

    def reconstruct_grid(self, x):
        a = self._flat_all[0].copy() * np.nan
        a[self._used_cells] = x
        a = xr.DataArray(a.reshape((self._lat.shape[0],-1)), dims=['lat','lon'], coords=dict(lat=self._lat, lon=self._lon))
        return a

class data_1D(_data):
    def __init__(self, x):
        self._class_func = data_1D
        super().__init__(x)

    def preprocess_specific(self):
        self._flat = self._x.values.reshape((-1, 1))
        self._seasonal = xr.DataArray(self._flat.reshape((len(self._years),self._days_per_year)), dims=['year','day'], 
                                            coords=dict(year=self._years, day=range(self._days_per_year)))

class data_emu(_data):
    def __init__(self, x):
        self._class_func = data_emu
        super().__init__(x)

    def preprocess_specific(self):
        self._days_per_year = int(self._x.shape[1] / self._years.shape[0])
        self._seasonal = xr.DataArray(self._x.values.reshape((self._x.shape[0], len(self._years),self._days_per_year)), dims=['run','year','day'], 
                                                coords=dict(run=self._x.run, year=self._years, day=range(self._days_per_year)))


