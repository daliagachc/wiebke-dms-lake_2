# load and autoreload
from IPython import get_ipython

# noinspection PyBroadException
from astropy.io.misc import asdf

try:
    _magic = get_ipython().magic
    _magic('load_ext autoreload')
    _magic('autoreload 2')
except:
    pass

import os
import glob
import sys
import pprint
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import cartopy as crt

def get_flx_ds(path) -> xr.Dataset:
    '''
    get_flx_ds: opens data set. Flexpart
    '''


    return None


def get_ptr_df(path) -> pd.DataFrame:
    '''
    get_ptr_df: return dataframe from ptr
    '''


    return None

def main():
    ds_flx = get_flx_ds()
    df_ptr = get_ptr_df()






