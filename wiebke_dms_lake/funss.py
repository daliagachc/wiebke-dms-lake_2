# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python [conda env:b37]
#     language: python
#     name: conda-env-b37-py
# ---

# %%
"""
8/25/21

diego.aliaga at helsinki dot fi
"""
# %%

import matplotlib.pyplot as plt
# %%
# from useful_scit.imps2.defs import *
# import os
# import seaborn as sns
import pandas as pd
import xarray as xr
from sklearn.linear_model import ElasticNetCV

from wiebke_dms_lake import flx_array as fa

# import cartopy as crt
# import numpy as np
# import matplotlib as mpl
# import flexpart_management.modules.flx_array as fa


# %%
LL = ['LAT_00',
      'LON_00',
      'LAT_10',
      'LON_10',
      'LAT_11',
      'LON_11',
      'LAT_01',
      'LON_01', ]

PATH_FLX = '/Volumes/mbProD/Documents_mbProD_DA/Work_DA/flexpart_management_data/flexpart_management/tmp_data/new_log_pol_ds_agl.nc'
PATH_DMS = '/Users/aliaga/Documents/Work_DA/Py-packs/wiebke-dms-lake/wiebke_dms_lake/data/DMSoxidationProducts.csv'

def open_ptr_data():
    path = PATH_DMS
    data = pd.read_csv(path, skiprows=1)
    data['localtime'] = pd.to_datetime(data['localtime'])
    d1 = data.set_index('localtime')
    d2 = d1.resample('1H').mean()
    return d2


def open_flx_data():


    da = get_conc_localtime_array()
    das = get_surface(da)

    # dan = normalize_da(da)
    df = da_to_dataframe(da)


    dfs = da_to_dataframe(das)

    return da,das,df,dfs


def open_flx_data2():
    '''includes boundary vertical srr in the last ring'''

    da = get_conc_localtime_array()
    das = get_surface(da)

    rsum = da[{'R_CENTER': -1}].sum('ZMID')
    das[{'R_CENTER': -1}] = rsum

    # dan = normalize_da(da)
    df = da_to_dataframe(da)

    dfs = da_to_dataframe(das)

    return da, das, df, dfs

def get_surface(dan):
    dans = dan[{ 'ZMID': 0 }]
    return dans

def da_to_dataframe(dan):
    ds1 = dan.reset_coords(drop=True)
    ds2 = ds1.to_dataframe()['CONC'].unstack('localtime').T
    return ds2

# def normalize_da(da02):
#     all_ds_norm = da02 / da02.sum('localtime')
#     adn1 = all_ds_norm.where(~all_ds_norm.isnull(), 0)
#     return adn1

def get_conc_localtime_array():
    path = PATH_FLX
    ds = xr.open_dataset(path)
    lt = ds['releases'] - pd.Timedelta(hours=4)
    ds['localtime'] = lt
    ds01 = ds.set_coords('localtime')
    da02 = ds01.swap_dims({ 'releases': 'localtime' })['CONC']
    return da02

def combine_data(data_ptr, data_flx):
    mer_index = pd.merge(data_flx, data_ptr, left_index=True, right_index=True).index

    d_flx = data_flx.loc[mer_index]
    d_ptr = data_ptr.loc[mer_index]
    return d_ptr,d_flx,

def apply_elastic_net(com_ptr, com_flx, da_flx, par, cv=5 , ww=2 ):

    regr = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            fit_intercept=False,
            positive=True,
            cv=cv
            )



    #not na index
    nna_index = com_ptr[~com_ptr[par].isna()].index

    cf1 = com_flx.loc[nna_index]
    com_flx_norm = normalize_df(cf1,ww=ww)

    regr.fit(com_flx_norm, com_ptr[par][nna_index])

    res = regr.predict(com_flx_norm)

    pred_ser = pd.Series(res,index=nna_index).resample('1H').mean()

    flx_df = pd.DataFrame(regr.coef_,index = com_flx.columns,columns=['coef'])

    # flx_df = (flx_df['coef']/cf1.sum()).to_frame(name='coef')

    flx_ds_coef = flx_df.to_xarray()

    coef_da = xr.zeros_like(da_flx) + flx_ds_coef


    return pred_ser, coef_da, regr

def normalize_df(cf1,ww=2):
    cf2 = cf1 / ((cf1**ww).sum()**(1/ww))
    # cf2 = cf1
    cf3 = cf2.where(~cf2.isna(), 0)
    return cf3

def plt_map_bol(com_ptr, da_coef):
    plt_map_bol_lp(com_ptr, da_coef, fa.get_ax_bolivia)

def plt_map_lp(com_ptr, da_coef):
    plt_map_bol_lp(com_ptr, da_coef, fa.get_ax_lapaz)
    ax = plt.gca()
    ax.set_range(fa.LOLA_LAPAZ)
    
def plt_map_bol_lp(com_ptr, da_coef, ax_fun):
#     ax = fa.get_ax_bolivia(lake_face_color='none')
    ax = ax_fun(lake_face_color='none')
    ll = list(set(da_coef.dims) - { 'R_CENTER', 'TH_CENTER' })

    dd = da_coef.loc[{ 'localtime': com_ptr.index }]
    dd1 = dd
    for l in ll:
        dd1 = dd1.sum(l)
    dd1.plot(x='XLONG', y='XLAT', ax=ax, zorder=0, robust=True)


def plt_map_bol2(com_ptr, da_coef):
    plt_map_bol_lp2(com_ptr, da_coef, fa.get_ax_bolivia)


def plt_map_lp2(com_ptr, da_coef):
    plt_map_bol_lp2(com_ptr, da_coef, fa.get_ax_lapaz)


def plt_map_bol_lp2(com_ptr, da_coef, ax_fun):
    #     ax = fa.get_ax_bolivia(lake_face_color='none')
    # ax = ax_fun(lake_face_color='none')
    # import cartopy as crt


    _ds = da_coef.loc[{'localtime': com_ptr.index}].mean('localtime')
    _ds.name = 'coef'
    _df = _ds.reset_coords(LL).reset_coords(drop=True).to_dataframe()

    _d1 = df2geopandas(_df)

    ax = ax_fun(lake_face_color='none')

    q1, q2 = _d1['coef'].quantile([.0, .99])

    _d1.plot(ax=ax, cmap='viridis', column='coef', vmin=q1, vmax=q2, zorder=-10)


def df2geopandas(_df):
    _df['geometry'] = _df.apply(polygon_from_row, axis=1)
    import geopandas
    _d1 = geopandas.GeoDataFrame(_df)
    return _d1


def polygon_from_row(r):
    from shapely.geometry import Polygon
    #     kw = {'closed': True, **kwargs}
    pol = Polygon([
        [r['LON_00'], r['LAT_00']],
        [r['LON_10'], r['LAT_10']],
        [r['LON_11'], r['LAT_11']],
        [r['LON_01'], r['LAT_01']],
    ])
    return pol
def plot_ts(data_ptr, df_pred, par):
    f,ax = plt.subplots()
    data_ptr[par].plot(label=par,ax=ax)
    df_pred.plot(label='reconstruction',ax=ax)
    ax.legend()

