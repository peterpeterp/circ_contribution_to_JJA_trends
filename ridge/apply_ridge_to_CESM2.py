import sys, os, glob, datetime, cftime, pickle, re
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

def do_grid_cell(coords):
    index,lat,lon = coords

    cc_train.set_location(lon,lat)
    cc_train.select_location('target')
    cc_train.select_region(cov_variable)
    cc_train.fit(['gmt', cov_variable], penalty_modifiers={'gmt':0, cov_variable:1}, cv=False)
    cc_train.sel_alpha(alpha)

    # save coeffs
    write_line(
        file = coefs_file_name,
        line = ','.join([str(e) for e in [int(index), lat, lon, alpha] + list(cc_train._coefs)])
    )

    # prepare X of test
    cc_test.set_location(lon,lat)
    cc_test.select_region(cov_variable)
    cc_test._X = cc_test.construct_X(['gmt', cov_variable])

    for run_name,oo,oo_piN in zip(
        [run_train,run_test],
        [cc_train,cc_test],
        [piN_train,piN_test]
        ):
          
        prediction = {}
        prediction['target'] = cc_train.reproduce(oo._X)._x
        prediction['thermo'] = oo._data['gmt']._x * cc_train._coefs[1] + cc_train._coefs[0]

        X = oo._X.copy()
        X[:,0] = 0
        prediction['circOnly'] = cc_train.reproduce(X)._x

        X = oo._X.copy()
        X[:,0] = oo_piN._data['gmt']._x.values
        prediction['circ'] = cc_train.reproduce(X)._x

        for version in ['target','circOnly','circ','thermo']:
            write_line(
                file = f'{out_path}/{tag}/train{run_train}_test{run_name}_predict_{version}.csv',
                line = ','.join([str(e) for e in [int(index), lat, lon, alpha] + list(prediction[version].values)])
            )
            y = prediction[version].squeeze()
            slope = sm.OLS(y.values, sm.add_constant(y.time.dt.year.values)).fit().params[1]
            write_line(
                file = f'{out_path}/{tag}/train{run_train}_test{run_name}_trend_{version}.csv',
                line = ','.join([str(e) for e in [int(index), lat, lon, slope]])
            )

# interpret command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_variable", type=str)
parser.add_argument("--cov_variable", type=str)
parser.add_argument("--alpha", type=str)
parser.add_argument("--run_train", type=str)
parser.add_argument("--run_test", type=str)
parser.add_argument("--months", type=int, nargs='+')
parser.add_argument("--period", type=int, nargs='+')
parser.add_argument("--overwrite", action='store_true')
args = parser.parse_args()
for k,v in vars(args).items():
    globals()[k] = v

# prepare out dir
out_path = '/climca/people/ppfleiderer/decomposition/decomp_out'
tag = f"{target_variable}_{'m'.join([str(m) for m in months])}_{cov_variable}_vX_{run_train}_{period[0]}-{period[1]}"
os.system(f'mkdir -p {out_path}/{tag}')

# initialize data objects
cc_train = decomp_CESM2_cc(run=run_train, target_variable=target_variable, months=months, period=period)
cc_test = decomp_CESM2_cc(run=run_test, target_variable=target_variable, months=months, period=period)
piN_train = decomp_CESM2_piN(run=run_train, target_variable=target_variable, months=months, period=period)
piN_test = decomp_CESM2_piN(run=run_test, target_variable=target_variable, months=months, period=period)

for oo in [cc_train, cc_test]:
    oo.target_open(0, 360, 30, 70)
    oo.gmt_open()
    if cov_variable == 'Z500':
        oo.cov_open(cov_variable, 0, 360, 0, 90)
        oo.cov_remove_global_mean(cov_variable)
    elif cov_variable == 'stream':
        oo.cov_open('streamVelopot_500hPa', 0, 360, 0, 90, var_name_in_file='stream', new_var_name=cov_variable, cdo_options="-shifttime,+2days")
    else:
        assert False, 'cov_variable needs to be Z500 or stream'

for oo in [piN_train, piN_test]:
    oo.target_open(0, 360, 30, 70)
    oo.gmt_open()


# prepare coefs file
# required to get the region size
cc_train.set_location(0,40)
cc_train.select_location('target')
cc_train.select_region(cov_variable)

coefs_file_name = f'{out_path}/{tag}/train{run_train}_coefs.csv'
if os.path.isfile(coefs_file_name) == False or overwrite:
    with open(coefs_file_name, 'w') as fl:
        fl.write('index,lat,lon,alpha,' + ','.join(['const'] + list(cc_train.construct_X(['gmt', cov_variable]).feature.values)) + '\n')

# prepare other files
for run_name in [run_train,run_test]:
    for version in ['target','circOnly','thermo','circ']:
        file_name = f'{out_path}/{tag}/train{run_train}_test{run_name}_predict_{version}.csv'
        if os.path.isfile(file_name) == False or overwrite:
            with open(file_name, 'w') as fl:
                fl.write('index,lat,lon,alpha,' + ','.join([str(t.strftime("%m/%d/%Y")) for t in cc_train._data_raw['target'].time.values]) + '\n')

        file_name = file_name.replace('predict','trend')
        if os.path.isfile(file_name) == False or overwrite:
            with open(file_name, 'w') as fl:
                fl.write(','.join(['index','lat','lon','trend']) + '\n')



# load data that is going to be used in parallel
# cc train
cc_train._data_raw['target'].load()
cc_train._data['gmt'] = data_1D(cc_train._data['gmt']._x.load())
cc_train._data_raw[cov_variable].load()
# cc test
#cc_test._data_raw['target'].load()
cc_test._data['gmt'] = data_1D(cc_test._data['gmt']._x.load())
cc_test._data_raw[cov_variable].load()
# piN train
#piN_train._data_raw['target'].load()
piN_train._data['gmt'] = data_1D(piN_train._data['gmt']._x.load())
#piN_train._data_raw[cov_variable].load()
# piN test
#piN_test._data_raw['target'].load()
piN_test._data['gmt'] = data_1D(piN_test._data['gmt']._x.load())
#piN_test._data_raw[cov_variable].load()




# make list of all grid-cells to do
# only do missing grid-cells -> remove grid-cells from list that have been done
# open last file in which reslauts are written in do_grid_cell()
file_name = f'{out_path}/{tag}/train{run_train}_test{run_test}_trend_circ.csv'

txt = open(file_name, 'r').read()
grid_cells = np.array([
    (int(i),c[0],c[1]) 
    for i,c in enumerate([
        (y,x) for y,x in itertools.product(
            cc_train._data_raw['target'].lat.values.round(3), 
            cc_train._data_raw['target'].lon.values.round(3)
            )
        ])
        if len(re.findall(f"\n{i},{c[0]},{c[1]},", txt)) == 0
    ])

def init(l):
    global lock
    lock = l
    
p = multiprocessing.Pool(64, initializer=init, initargs=(multiprocessing.Lock(),))
p.map(do_grid_cell, grid_cells)