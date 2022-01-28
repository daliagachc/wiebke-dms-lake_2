'''functions used when analyzing flexpart output
Examples:
    >>> euristic_import_flexpart(dir_path='path_to_dir', dd=D2)
'''

# project name: flexpart_management
# created by diego aliaga daliaga_at_chacaltaya.edu.bo

from typing import List
import cartopy
import cartopy.mpl.geoaxes

# from flexpart_management.modules import constants as co

import numpy as np
# import matplotlib as mpl

import matplotlib.pyplot as plt

# from flexpart_management.modules import constants as co
import matplotlib.ticker as m_ticker
import datetime as dt


PROJ = cartopy.crs.PlateCarree()


LOLA_LAPAZ = [ -70, -66, -18, -14 ]
LOLA_BOL = [ -83, -43, -35, 2 ]

CHC_LAT = -16.350427
CHC_LON = -68.131335
LPB_LAT = -16.507125
LPB_LON = -68.129299

def get_ax_lapaz( ax=False ,
                  fig_args=None ,
                  lalo_extent=LOLA_LAPAZ ,
                  chc_lp_legend=True ,
                  lola_ticks=None
                  , draw_labels=True
                  , plot_cities=True
                  , grid_alpha=0.5 , grid_color='k' , grid_style='--' ,
                  lake_face_color='none' ,
                  map_line_alpha=1
                  , y_left_lab=False , y_right_lab=False , borders=True ):
    if fig_args is None:
        fig_args = { }
    import matplotlib.ticker as mticker
    fig_ops = dict( figsize=(15 , 10) )
    fig_ops = { **fig_ops , **fig_args }
    if ax is False:
        fig = plt.figure( **fig_ops )
        ax = fig.add_subplot( 1 , 1 , 1 , projection=PROJ , )
    
    ax.set_extent( lalo_extent , crs=PROJ )
    ax.add_feature( cartopy.feature.COASTLINE.with_scale( '10m' ) )
    ax.add_feature(
            cartopy.feature.LAKES.with_scale( '10m' ) ,
            facecolor=lake_face_color ,
            edgecolor=cartopy.feature.COLORS[ 'water' ] ,
            alpha=map_line_alpha ,
            zorder=0
            )
    if borders:
        ax.add_feature( cartopy.feature.BORDERS.with_scale( '10m' ) ,
                        alpha=map_line_alpha ,
                        )
    if plot_cities:
        ax.add_feature( cartopy.feature.STATES.with_scale( '10m' ) ,
                        alpha=map_line_alpha ,
                        linestyle=':' )
    gl = ax.gridlines(
            crs=PROJ , alpha=grid_alpha ,
            linestyle=grid_style ,
            draw_labels=draw_labels ,
            color=grid_color
            )
    
    gl.xlabels_top = False
    gl.ylabels_right = y_right_lab
    gl.ylabels_left = y_left_lab
    rr = 2
    if lola_ticks is None:
        lo1 = np.round( lalo_extent[ 0 ] / rr ) * rr - 4 * rr
        # print(lo1)
        lo2 = lalo_extent[ 1 ] + 4 * rr
        la1 = np.round( lalo_extent[ 2 ] / rr ) * rr - 4 * rr
        la2 = lalo_extent[ 3 ] + 4 * rr
        lolo = np.arange( *(lo1 , lo2 , rr) )
        lala = np.arange( *(la1 , la2 , rr) )
    else:
        lala = lola_ticks[ 1 ]
        lolo = lola_ticks[ 0 ]
    
    gl.ylocator = m_ticker.FixedLocator( lala )
    gl.xlocator = m_ticker.FixedLocator( lolo )
    
    if chc_lp_legend:
        add_chc_lpb( ax )
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # ax:plt.Axes
    # ax.text(-.1,-)
    
    return ax

def add_chc_lpb(ax):
    ax.scatter(CHC_LON, CHC_LAT, marker='.', color='b',
               transform=PROJ, label='CHC', zorder=100)
    ax.scatter(LPB_LON, LPB_LAT, marker='.', color='g',
               transform=PROJ, label='La Paz', zorder=100)
    ax.legend()


def get_ax_bolivia(
        ax=False ,
        fig_args: dict = None ,
        lola_extent: List[ float ] = LOLA_BOL ,
        proj=PROJ ,
        chc_lp_legend=False ,
        plot_cities=False ,
        plot_ocean=False ,
        lola_ticks=None ,
        draw_labels=False ,
        grid_alpha=0.5 ,
        grid_color='w' ,
        grid_style='--' ,
        map_res='50m' ,
        lake_face_color=cartopy.feature.COLORS[ 'water' ] ,
        map_line_alpha=1
        , xlab_top=False , ylab_right=False , xlab_bot=True ,
        ylab_left=True ):
    """
    returns a geo ax object with the area of bolivia

    Parameters
    ----------
    proj
        projection to use
    ax
        ax to use. if false create a new one
    fig_args
        args passed to the figure
    lola_extent
        extent of the lalo

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes
        returns a cartopy geoax
        #todo check this

    """
    if fig_args is None:
        fig_args = { }
    fig_ops = dict( figsize=(15 , 10) )
    fig_ops = { **fig_ops , **fig_args }
    if ax is False:
        fig = plt.figure( **fig_ops )
        ax = fig.add_subplot( 1 , 1 , 1 , projection=proj , )
    
    ax.set_extent( lola_extent , crs=proj )
    ax.add_feature( cartopy.feature.LAKES.with_scale( map_res ) ,
                    alpha=1 , linestyle='-' ,
                    facecolor=lake_face_color ,
                    edgecolor=cartopy.feature.COLORS[ 'water' ] ,
                    zorder=10,
                    lw=.5
                    )
    if plot_ocean:
        ax.add_feature( cartopy.feature.OCEAN.with_scale( '50m' ) )
    ax.add_feature( cartopy.feature.COASTLINE.with_scale( map_res ) ,
                    alpha=map_line_alpha
                    )
    ax.add_feature( cartopy.feature.BORDERS.with_scale( map_res ) ,
                    alpha=map_line_alpha
                    )
    if plot_cities:
        ax.add_feature( cartopy.feature.STATES.with_scale( map_res ) ,
                        alpha=0.5 ,
                        linestyle=':' )
    gl = ax.gridlines( crs=proj , alpha=grid_alpha , linestyle=grid_style ,
                       draw_labels=draw_labels , color=grid_color )
    gl.xlabels_top = xlab_top
    gl.ylabels_right = ylab_right
    gl.xlabels_bottom = xlab_bot
    gl.ylabels_left = ylab_left
    
    if lola_ticks is None:
        lo1 = np.round( lola_extent[ 0 ] / 5 ) * 5 - 5
        # print(lo1)
        lo2 = lola_extent[ 1 ] + 5
        la1 = np.round( lola_extent[ 2 ] / 5 ) * 5 - 5
        la2 = lola_extent[ 3 ] + 5
        lolo = np.arange( *(lo1 , lo2 , 5) )
        lala = np.arange( *(la1 , la2 , 5) )
    else:
        lala = lola_ticks[ 1 ]
        lolo = lola_ticks[ 0 ]
    
    gl.ylocator = m_ticker.FixedLocator( lala )
    gl.xlocator = m_ticker.FixedLocator( lolo )
    
    if chc_lp_legend:
        add_chc_lpb( ax )
    
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    
    return ax