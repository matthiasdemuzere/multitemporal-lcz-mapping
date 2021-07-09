import yaml
from typing import Dict
from typing import Any
import fiona
fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default
import geopandas as gpd
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from zipfile import ZipFile
import shutil
import traceback
import os, glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import seaborn as sns
import xarray as xr
from bokeh import palettes
import ee
ee.Initialize()

## *********************************************** ##
##***********           VISUALS         ***********##
## *********************************************** ##

def print_oa(out_dir:str,
             roi_name:str):

    print(f" ----- {roi_name} -----")
    list_years = np.arange(1986, 2017, 5).tolist()

    for year in list_years:
        df = pd.read_csv(f'{out_dir}/'
                         f'{roi_name}_{year}_filter_final_oa_df.csv',
                         index_col=0)

        mean = df.mean(axis=0).round(2)
        std = df.std(axis=0).round(2)

        print(f'{year} : '
              f'{mean[0]} ± {std[0]} | {mean[1]} ± {std[1]} | '
              f'{mean[2]} ± {std[2]} | {mean[3]} ± {std[3]}')

        #print(mean)


def plot_lcz_change(out_dir, roi_name):

    figfile = os.path.join(
        out_dir,
        f'{roi_name}_lcz_change.jpg')


    list_years = np.arange(1986, 2017, 5).tolist()

    df_count = pd.DataFrame(index=np.arange(0,18,1).tolist())

    col_choice = palettes.Colorblind
    # https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    colors = list(col_choice[len(list_years)])

    # Loop over all years
    for year in list_years:

        # Read geotif to plot map
        lczTif = xr.open_rasterio(
            os.path.join(
                out_dir,
                f"{roi_name}_{year}_filter_final.tif"),
            )[0,:,:]

        counts = np.array(np.unique(lczTif.values, return_counts=True)).T
        df_tmp = pd.DataFrame(index=counts[:,0])
        df_tmp[str(year)] = counts[:,1]

        df_count = pd.concat([df_count, df_tmp], axis=1)
        #df_count.set_index = list(range(1, 18, 1))

    # Drop the few missing values
    data = df_count.iloc[1:,:]

    LCurb = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    urbsum = data.iloc[LCurb, :].sum(axis=0)
    totsum = data.iloc[:,:].sum(axis=0)

    lcz_tot = data/totsum*100
    lcz_urb = data.iloc[LCurb,:]/urbsum*100

    plt.close('all')
    plt.rcParams.update({'font.size': 10})
    bar_width = 0.8

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.3])
    ax0 = plt.subplot(gs[0])
    #plt.subplots_adjust(top=0.95,bottom=0.2,left=0.055,right=0.985,hspace=0.1,wspace=0.05)
    xtickname = ['1', '2', '3', '4', '5', '6','7', '8', '9', '10',
                 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    xtickname_urb = ['1', '2', '3', '4', '5', '6','7', '8', '10']

    lcz_tot.plot(kind='bar',color=colors,ax=ax0,
                 zorder=3, width=bar_width)
    ax0.set_ylabel('# $LCZ_{i}$ / # $LCZ_{all}$ x100 [%]')
    ax0.grid(linestyle='dotted', color='0.8',zorder=2)
    ax0.set_xlabel('')
    #ax0.text(-0.3,31,'(a)')
    ax0.set_xticklabels(xtickname,rotation=0)
    ax0.legend(loc='upper left')
    ax0.set_xlabel('LCZ Class')

    ax1 = plt.subplot(gs[1])
    lcz_urb.plot(kind='bar',color=colors,ax=ax1,
                             zorder=3, width=bar_width,legend=None)
    ax1.set_ylabel('# $LCZ_{i}$ / # $LCZ_{u}$ x100 [%]')
    ax1.grid(linestyle='dotted', color='0.8',zorder=2)
    #ax1.set_xticks(data.index)
    ax1.set_xticklabels(xtickname_urb,rotation=0)
    ax1.set_xlabel('LCZ Class')
    #ax1.text(-0.3,43,'(b)')
    plt.tight_layout()
    plt.savefig(figfile,dpi=300)
    plt.close('all')


def table_urban_area(out_dir, cities):

    urban_area_csv = os.path.join(
        out_dir,
        'cities_urban_area.csv')
    missing_area_csv = os.path.join(
        out_dir,
        'cities_missing_area.csv')

    list_years = np.arange(1986, 2017, 5).tolist()

    df_urban_area = pd.DataFrame(index=cities, columns=list_years)
    df_missing = pd.DataFrame(index=cities, columns=list_years)

    # Loop over all years and cities
    for city in cities:

        for year in list_years:

            # Read geotif to plot map
            lczTif = xr.open_rasterio(
                os.path.join(
                    out_dir,
                    f"{city}_{year}_filter_final.tif"),
                )[0,:,:]

            # Only select urban classes
            #LC = [1,2,3,4,5,6,7,8,9,10]
            LC = [1, 2, 3, 4, 5, 6, 7, 8, 10]
            mask = xr.DataArray(np.in1d(lczTif, LC).reshape(lczTif.shape),
                         dims=lczTif.dims, coords=lczTif.coords)
            area = int(lczTif.where(mask).count().values) * 0.01

            df_urban_area.loc[city,year] = area

            # Also count missing values
            missing = int(lczTif.where(lczTif==0).count().values) * 0.01
            df_missing.loc[city,year] = missing

    # Store
    df_urban_area.to_csv(urban_area_csv)
    df_missing.to_csv(missing_area_csv)

    # Print
    print(df_urban_area)
    print(df_missing)

