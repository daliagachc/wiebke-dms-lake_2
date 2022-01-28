# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python [conda env:q5]
#     language: python
#     name: conda-env-q5-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt

# from useful_scit.imps2.defs import *
# from sklearn.linear_model import ElasticNet,ElasticNetCV
# import flexpart_management.modules.flx_array as fa
import warnings
warnings.filterwarnings('ignore',category=UserWarning)


import sys
sys.path.insert(0,'../../wiebke_dms_lake/..')
import wiebke_dms_lake.funss as fu
import wiebke_dms_lake.flx_array as fa



# %%
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import QuantileTransformer,StandardScaler
from sklearn.cluster import KMeans
import warnings


from sklearn.linear_model import (ElasticNetCV, BayesianRidge, LinearRegression,
                                  RidgeCV, Ridge, Lasso, LassoCV, ARDRegression, ElasticNet)

import sklearn.preprocessing



#
# %% Constants and Functions
# Constants

NAME = 'CLUS_CON'

PATH = 'multi_fit2'

PATH_dat = 'multi_fit2_data'
SEED = 34345

# %%
# !mkdir {PATH}

# %%
# !mkdir {PATH_dat}

# %%

# %% Constants and Functions
# Functions 
def add_lat_lon_corners(ds):
    ds = ds.reset_coords(drop=True)
    r = ds['R_CENTER']
    th = ds['TH_CENTER']
    ROUND_R_LOG = np.log(ds['R_CENTER'][3:15]).diff('R_CENTER').mean().item()
    ROUND_TH_RAD = ds['TH_CENTER'].diff('TH_CENTER').mean().item()
    rM = r * np.exp(ROUND_R_LOG / 2)
    rm = r * np.exp(-ROUND_R_LOG / 2)
    thm = th - ROUND_TH_RAD / 2
    thM = th + ROUND_TH_RAD / 2
    val_list = [
        ['LAT_00', 'LON_00', rm, thm],
        ['LAT_10', 'LON_10', rM, thm],
        ['LAT_11', 'LON_11', rM, thM],
        ['LAT_01', 'LON_01', rm, thM],
    ]
    for v in val_list:
        ds = ds.assign_coords({v[0]: r_th_to_lat(fa.CHC_LAT, v[2], v[3])})
        ds = ds.assign_coords({v[1]: r_th_to_lon(fa.CHC_LON, v[2], v[3])})

    ds['XLAT'] = (ds['LAT_00'] + ds['LAT_11']) / 2
    ds['XLONG'] = (ds['LON_00'] + ds['LON_11']) / 2

    return ds


def r_th_to_lon(lon_center, rr, th):
    return r_th_to_ll(lon_center, rr, th, np.sin)


def r_th_to_lat(lat_center, rr, th):
    return r_th_to_ll(lat_center, rr, th, np.cos)


def r_th_to_ll(center, rr, th, fun):
    res = rr * fun(
        th
    ) + center
    return res


def fun_flattend_z_level(_clus_ds):
    f1 = _clus_ds[{'R_CENTER': 29, 'ZMID': slice(1, None)}]

    last = _clus_ds['R_CENTER'].max().item()
    last = last + _clus_ds['R_CENTER'][{'R_CENTER':[28,29]}].diff('R_CENTER').item()/2

    z = len(f1['ZMID'])

    ll = [last + i/5000 for i in f1['ZMID']]
    

    f1['ZMID'] = f1['ZMID'] * 0 + ll

    f2 = f1.drop('R_CENTER').rename({'ZMID': 'R_CENTER'})
    _clus_ds_flatz = add_lat_lon_corners(f2)

    # _clus_ds_flatz.plot(x='XLONG', y='XLAT')
    

    _clus_ds.name = NAME
    _clus_ds_flatz.name = NAME
    g1=  _clus_ds[{'ZMID': 0}]
    g2 = _clus_ds_flatz
    
    gg2 = g2.reset_coords(drop=True)
    gg1 = g1.reset_coords(drop=True)

    g3 = xr.concat([gg1,gg2],dim='R_CENTER')

    g4 = add_lat_lon_corners(g3)

    _clus_ds_flatz1 = add_lat_lon_corners(g4)
    return _clus_ds_flatz1

def get_clus_lab_sum_ss(_clus_lab_sum):
        SS = StandardScaler(with_mean=False)
        SS = SS.fit(_clus_lab_sum)
        _clus_lab_sum_ss = _clus_lab_sum * 0 + SS.transform(_clus_lab_sum)
        return _clus_lab_sum_ss

def get_zreduced_flx_ds():
    # open xarray and transform local time
    da_ = fu.get_conc_localtime_array()
    # select bins to reduce z dim size
    bns = [0, 500, 1_500, 4_000, 6_000, 8_000, 12_000, 15_000]
    labs = [(i + j) / 2 for i, j in zip(bns[:-1], bns[1:])]
    da = da_.groupby_bins('ZMID', bins=bns, labels=labs).sum().rename({'ZMID_bins': 'ZMID'})
    return da

def get_flx_df(da,index):
        # combine both
        s1 = da[{'ZMID': [0]}].to_series()
        s2 = da[{'ZMID': slice(1, None), 'R_CENTER': [29]}].to_series()
        ss = pd.concat([s1, s2])
        ss1 = ss.unstack('localtime')[index]
        return ss1

def get_flx_df_qt(_flx_df):
    qt = QuantileTransformer()
    #         qt = StandardScaler(with_mean=False)
    #         qt = sklearn.preprocessing.MaxAbsScaler()
    _flx_df_qt = _flx_df.iloc[:, :] * 0 + qt.fit_transform(_flx_df.iloc[:, :].T).T
    return _flx_df_qt

def get_cluster_flx_df(A, B, _flx_df_qt):
    km = KMeans(A * B, random_state=SEED)
    km.fit(_flx_df_qt)
    labels = pd.Series(km.labels_, index=_flx_df_qt.index)
    return labels

# %%
def main():
    pass


# %%

# %%

# %%

# %%

# %% tags=[]
ratios = [.0001,.001,.01,.1,.5,.9,.99,.999]
alphas = [.0001,.001,.01,.05,.1,.5,1,10]
inter  = [True,False]
norms = ['mean_norm','mean_norm_sum','mean_norm_sqr']
pars = ['CH4SO', 'C2H4OS', 'C2H6OS', 'CH4O2S', 'C2H6O2S']

ls = []
for p in pars:
    for r in ratios:
        for a in alphas:
            for i in inter:
                for n in norms:
                    di = dict(p=p,r=r,a=a,i=i,n=n)
                    ls.append(di)


ll = len(ls)
s=pd.Series(range(ll))
lll = list(s.sample(ll))

for i in lll:
    print(i)
    r = ls[i]
    A = B = 11

    PAR = r['p']
    
    # ignore warnings for sklearn and pandas
    warnings.filterwarnings('ignore', category=FutureWarning)


    _ptr_df = fu.open_ptr_data()

    _index_ptr = _ptr_df[_ptr_df[PAR].notna()].index

    _da = get_zreduced_flx_ds()

    _flx_df = get_flx_df(_da,_index_ptr)

    _flx_df_qt = get_flx_df_qt(_flx_df)


    _clus_flx_lab = get_cluster_flx_df(A, B, _flx_df_qt)

    _clus_lab_sum = (_flx_df.groupby(_clus_flx_lab).sum().T/3600)

    _ptr_par = _ptr_df[PAR].loc[_index_ptr]

    _clus_lab_sum_ss = get_clus_lab_sum_ss(_clus_lab_sum)

    # get cluster ds
    _clus_ds = _da[{'localtime':0}]*0 + _clus_flx_lab.replace(_clus_lab_sum.sum()).to_xarray()



    _clus_ds_flat_z = fun_flattend_z_level(_clus_ds)

    des = _clus_lab_sum.describe().T

    np.mean(_clus_lab_sum.values.flat)
    np.median(_clus_lab_sum.values.flat)


    # des['mean/std'] = des['mean']/des['std']

    # for m in ['mean','50%','max','std','mean/std']:
    #     f,ax=plt.subplots()
    #     des[m].sort_values().plot.barh()
    #     ax.set_xscale('log')
    #     ax.set_title(m)

    normer=_clus_lab_sum.mean()
    _norm_mean = _clus_lab_sum/normer
    _norm_mean.mean().mean()

    normer_sum = np.mean(_clus_lab_sum.values.flat)
    _norm_mean_sum = _clus_lab_sum/normer_sum
    _norm_mean_sum.mean().mean()

    normer_sqr = ((_clus_lab_sum.mean()**.5)*np.mean((_clus_lab_sum/_clus_lab_sum.mean()**.5).values.flat))
    _norm_mean_sqr = _clus_lab_sum/normer_sqr

    _norm_mean_sqr.mean().mean()



    _norm_vals = {
        'mean_norm'     :{'data':_norm_mean    , 'normer':normer     },
        'mean_norm_sum' :{'data':_norm_mean_sum, 'normer':normer_sum },
        'mean_norm_sqr' :{'data':_norm_mean_sqr, 'normer':normer_sqr },
    } 

    # for i,v in _norm_vals.items():
    #     _,(ax1,ax2)=plt.subplots(1,2)
    #     v['data'].mean().sort_values().plot.barh(ax=ax1)
    #     (v['data']*v['normer']).mean().sort_values().plot.barh(ax=ax2)
    #     ax1.set_title(i)
    #     ax2.set_title('data*normer')




    en = ElasticNet(
        positive=True, 
        fit_intercept=r['i'],
        alpha=r['a'],
        l1_ratio=r['r'],
        random_state=SEED
    )

    en.fit(_norm_vals[r['n']]['data'],_ptr_par)

    r['en'] = en 





    pred = _ptr_par*0+r['en'].predict(_norm_vals[r['n']]['data'])

    r['pred']=pred

    r['coef']=pd.Series(r['en'].coef_,index=_norm_vals[r['n']]['data'].columns)

    # r['coef_n']=r['coef']/_norm_vals[r['n']]['normer']
    r['res'] =(r['coef']*_norm_vals[r['n']]['data']).sum()
    res = (r['coef']*_norm_vals[r['n']]['data'])

    import cartopy

    path = PATH

    # f,ax = plt.subplots(subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
    
    f = plt.figure(figsize=(5,7),constrained_layout=True)
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(10, 10, figure=f)
    ax = f.add_subplot(gs[:7,:9],projection=cartopy.crs.PlateCarree())
    ax1 = f.add_subplot(gs[7:,:])
    cax = f.add_subplot(gs[:7,9])
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.2)
    ax = fa.get_ax_bolivia(
        ax=ax,lola_extent=[-90,-45,5,-40],lake_face_color='none',
        grid_alpha=0
    )
    _=_da[{'localtime':0}]*0 + _clus_flx_lab.replace(r['res']).to_xarray()
    g4=fun_flattend_z_level(_)
#     g4.plot(ax=ax,x='XLONG',y='XLAT')
    _1 = g4.reset_coords(fu.LL).reset_coords(drop=True).to_dataframe()
    gp = fu.df2geopandas(_1)
    gg = gp.plot(
    'CLUS_CON',ax=ax,zorder=0,legend=True,cax=cax,
    edgecolor = 'w', lw = .1
            )

    plt.gcf().set_dpi(300)
    tit = f"p={r['p']}|r={r['r']}|a={r['a']}|i={r['i']}|n={r['n']}"
    ax.set_title(tit,fontdict=dict(fontsize=5))
    _ptr_par.resample('1H').mean().plot(ax=ax1)
    (res.sum(axis=1)+r['en'].intercept_).plot(ax=ax1)
    f.tight_layout()
    f.savefig(f"{path}/{tit}.pdf")
    g4.to_netcdf(f"{PATH_dat}/{tit}.nc")
    plt.close(f)
    ls[i] = None

# %%

# %%

# %%












