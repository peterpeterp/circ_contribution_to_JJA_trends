import sys, os, glob, datetime, cftime, pickle
import xarray as xr
import numpy as np
import pandas as pd
from scipy import float64
from dill.source import getsource
import multiprocessing

from _ridge import *
from _data_minimal import *

def write_line(file, line):
    lock.acquire()
    with open(file, 'a') as fl:
        fl.write(line + '\n')
    lock.release()

def do_grid_cell(coords, alpha=1):
    index,lat,lon = coords

    oo.set_location(lon,lat)
    oo.select_location('target')
    oo.select_region(cov_variable)
    oo.fit(['gmt', cov_variable], penalty_modifiers={'gmt':0, cov_variable:1}, cv=False)
    oo.sel_alpha(alpha)
    prediction = {}
    prediction['self'] = oo.reproduce(oo._X)
    prediction['thermo'] = data_1D(oo._data['gmt']._x * oo._coefs[1] + oo._coefs[0])

    X = oo._X.copy()
    X[:,0] = 0
    prediction['circ'] = oo.reproduce(X)

    for version in ['self','circ','thermo']:
        write_line(
            file = f'{out_path}/{tag}/ERA5_predict_{version}.csv',
            line = ','.join([str(e) for e in [index, lat, lon, alpha] + list(prediction[version]._x.values)])
        )
        y = prediction[version]._x.squeeze()
        slope = sm.OLS(y.values, sm.add_constant(y.time.dt.year.values)).fit().params[1]
        write_line(
            file = f'{out_path}/{tag}/ERA5_trend_predict_{version}.csv',
            line = ','.join([str(e) for e in [index, lat, lon, slope]])
        )

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_variable", type=str)
parser.add_argument("--months", type=int, nargs='+')
parser.add_argument("--period", type=int, nargs='+')
parser.add_argument("--overwrite", action='store_true')
args = parser.parse_args()


for k,v in vars(args).items():
    globals()[k] = v

out_path = '/climca/people/ppfleiderer/decomposition/decomp_out'


cov_variable = 'stream'
tag = f"ERA5_{target_variable}_{'m'.join([str(m) for m in months])}_{cov_variable}"
tag += ''
tag += f"_{period[0]}-{period[1]}"

os.system(f'mkdir -p {out_path}/{tag}')
os.system(f'cp _decomp.py {out_path}/{tag}/')
_ = os.system(f'cp ERA5_multiple_v0.py {out_path}/{tag}/')

oo = decomp_ERA5(target_variable=target_variable, months=months, period=period)
oo.target_load(0, 360, 30, 70, var_name_in_file='var167', cdo_options='-remapcon,r360x180')
oo.gmt_compute(var_name_in_file='var167')
with xr.open_dataset('/climca/people/ppfleiderer/ERA5_streamfunction/E5pl00_1D_stream_500hPa_JJA.nc') as nc:
    stream = nc.stream
oo._data_raw['stream'] = stream[:,0,:,:].squeeze().loc[str(period[0]):str(period[1])]

grid_cells = np.array([(int(i),c[0],c[1]) for i,c in enumerate([(y,x) 
                        for y,x in itertools.product(oo._data_raw['target'].lat.values, oo._data_raw['target'].lon.values)])])

np.savetxt('grid_cells.csv', grid_cells)

for version in ['self','circ','thermo']:
    file_name = f'{out_path}/{tag}/ERA5_predict_{version}.csv'
    os.system(f'rm {file_name}')
    with open(file_name, 'w') as fl:
        fl.write('index,lat,lon,alpha,' + ','.join([str(t)[:10] for t in oo._data_raw['target'].time.values]) + '\n')

    file_name = file_name.replace('predict_','trend_predict_')
    os.system(f'rm {file_name}')
    with open(file_name, 'w') as fl:
        fl.write(','.join(['index','lat','lon','trend']) + '\n')


def init(l):
    global lock
    lock = l
    
p = multiprocessing.Pool(64, initializer=init, initargs=(multiprocessing.Lock(),))
p.map(do_grid_cell, grid_cells)