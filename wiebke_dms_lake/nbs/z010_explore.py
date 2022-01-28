# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
"""
8/25/21

diego.aliaga at helsinki dot fi
"""

# %%
import matplotlib.pyplot as plt

# from useful_scit.imps2.defs import *
# from sklearn.linear_model import ElasticNet,ElasticNetCV
# import flexpart_management.modules.flx_array as fa
import warnings
warnings.filterwarnings('ignore',category=UserWarning)


import sys
sys.path.insert(0,'../..')
import wiebke_dms_lake.funs as fu

# %%

# def main():
#     pass

# %%
data_ptr = fu.open_ptr_data()
da_flx, das_flx, df_flx, dfs_flx  = fu.open_flx_data()


# %%
com_ptr,com_flx = fu.combine_data(data_ptr, dfs_flx)
for par in list(com_ptr.columns):
#     for par in list(com_ptr.columns)[-1:]:

    df_pred,ds_coef, regr = fu.apply_elastic_net(com_ptr,com_flx, das_flx, par )

    fu.plot_ts(data_ptr, df_pred, par)
    plt.gca().set_title(f'{par}')

    fu.plt_map_bol(com_ptr, ds_coef['coef'])
    plt.gca().set_title(f'reconstructed {par}')

    fu.plt_map_bol(com_ptr, das_flx)
    plt.gca().set_title('all surface influence')

    fu.plt_map_lp(com_ptr, ds_coef['coef'])
    plt.gca().set_title(f'reconstructed --zoom {par}')

    fu.plt_map_lp(com_ptr, das_flx)
    plt.gca().set_title('all surface influence --zoom')

# %%
# !jupyter-nbconvert --to=html z010_explore.ipynb

# %%
