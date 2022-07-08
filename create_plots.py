import yaml
from typing import Dict, Any
import geopandas as gpd
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import xarray as xr
import rasterio
import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(
    description="PURPOSE: Prepare TA set that contains TAs for all cities\n \n"
                "OUTPUT:\n"
                "- PUT HERE ...",
    formatter_class=RawTextHelpFormatter
)

# Required arguments
parser.add_argument(type=str, dest='CITY',
                    help='City to classify',
                    )
parser.add_argument(type=str, dest='TA_VERSION',
                    help='Version of TA set (default is "v1")',
                    default="v1",
                    )
args = parser.parse_args()

# Arguments to script
CITY       = args.CITY
TA_VERSION = args.TA_VERSION

# For testing
#CITY       = 'Hyderabad'
#TA_VERSION = 'v1'

# Set files and folders:
fn_loc_dir = f"/home/demuzmp4/Nextcloud/data/wudapt/dynamic-lcz/{CITY}"



def _read_config(CITY) -> Dict[str, Dict[str, Any]]:
    with open(
        os.path.join(
            '/home/demuzmp4/Nextcloud/scripts/wudapt/dynamic-lcz/config',
            f'{CITY.lower()}.yaml',
        ),
    ) as ymlfile:
        pm = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return pm

# Make stacked bar plot, color per available city.
def plt_ta_freq_year(info, CITY, TA_VERSION) -> None:

    """
    Create TA frequency barchart, with stacked LCZ classes per year.
    """

    df = gpd.read_file(os.path.join(
        fn_loc_dir,
        "input",
        f"TA_{TA_VERSION}.shp"

    ))

    # Initialize figure
    fig, ax = plt.subplots(1,1,figsize=(8, 7))

    df2 = df.groupby(['Year', 'Class'])['Class'].count()\
        .unstack('Class').fillna(0)

    # In case LCZ class numbers or missing, add zero's.
    for i in range(1,18,1):
        if not i in df2.columns:
            df2[i] = 0

    # Sort columns, so that this is in line with LCZ colors
    df2 = df2.reindex(sorted(df2.columns), axis=1)

    df2.plot(kind='bar', stacked=True, ax=ax,
             color=info['LCZ']['COLORS'])

    ax.set_ylabel('# Training areas')
    ax.set_xlabel('Year')
    ax.grid(zorder=0, ls=':', color='0.5')
    ax.set_title(CITY)

    # Legend
    ax.legend(title='LCZ class',
             bbox_to_anchor=(1.03, 0),
             loc='lower left')

    plt.tight_layout()

    FIGFILE = os.path.join(
        fn_loc_dir,
        "output",
        f"plot_TA_FREQ.jpg"

    )
    fig.savefig(FIGFILE, transparent=False,dpi=150)
    plt.close('all')


def _get_oa_df(info, CITY, TA_VERSION, YEAR):

    natClass = [i-1 for i in [11, 12, 13, 14, 16, 17]]

    # Read the raw confusion matrix
    cm_raw = f"CM_{CITY}_" \
                f"{TA_VERSION}_" \
                f"{YEAR}_" \
                f"CC{info['CC']}_" \
                f"ED{info['EXTRA_DAYS']}_" \
                f"JDs{info['JD_START']}_{info['JD_END']}_" \
                f"L7{info['ADD_L7']}"

    ## Read OA weights file
    oaw_file = info['CM_WEIGHTS']
    oaw = pd.read_csv(oaw_file, sep=',', header=None)

    ## Get all the accuracy metrics
    df = pd.read_csv(f'{fn_loc_dir}/output/{cm_raw}.csv')
    mStr = df.iloc[:, 1][0].replace("[", "").replace("]", "")
    mList = [int(e) for e in mStr.split(',')]

    arr = np.array(mList)
    arr_oa = arr.reshape(info['LCZ']['NRBOOT'],
                         info['LCZ']['NRLCZ'],
                         info['LCZ']['NRLCZ'])

    ## Initialize an empty dataframe
    index = range(info['LCZ']['NRBOOT'])
    df_oa = pd.DataFrame(index=index, columns=range(5 + info['LCZ']['NRLCZ']))

    ## Loop over all available bootstraps
    for i in range(info['LCZ']['NRBOOT']):
        diag = arr_oa[i].diagonal()
        diagOaurb = arr_oa[i, :10, :10].diagonal()

        sumColumns = arr_oa[i].sum(0)
        sumRows = arr_oa[i].sum(1)

        sumDiag = diag.sum()
        sumDiagOaurb = diagOaurb.sum()

        sumTotal = arr_oa[i].sum()
        sumTotalOaurb = arr_oa[i, :10, :10].sum()

        ## weighted cm
        cmw = oaw * arr_oa[i]
        sumDiagOAW = np.nansum(cmw)
        sumOAWTotal = np.nansum(arr_oa[i])

        pa = diag / sumColumns  # PA or Precision
        ua = diag / sumRows  # UA or Recall

        df_oa.loc[i, 0] = sumDiag / sumTotal  # OA
        df_oa.loc[i, 1] = sumDiagOaurb / sumTotalOaurb  # OA_urb
        df_oa.loc[i, 2] = (arr_oa[i, :10, :10].sum() + arr_oa[i, natClass, natClass].sum()) / \
                         (arr_oa[i, :10, natClass].sum() + arr_oa[i, natClass, :10].sum() +
                          arr_oa[i, :10, :10].sum() + arr_oa[i, natClass, natClass].sum())
        df_oa.loc[i, 3] = sumDiagOAW / sumOAWTotal  # OA_weighted
        df_oa.loc[i, 5:] = 2 * ((pa * ua) / (pa + ua))

    # create formated confusion matrix, average over all bootstraps
    dfC = pd.DataFrame(arr_oa[i],
                       columns=np.arange(1, info['LCZ']['NRLCZ'] + 1, 1),
                       index=np.arange(1, info['LCZ']['NRLCZ'] + 1, 1)
                       ).astype(int)
    dfC.loc['Total'] = sumColumns
    dfC['Total'] = dfC.sum(1)
    dfC.loc['PA (%)'] = np.append(np.round((diag / sumColumns) * 100, 1), np.nan)
    dfC['UA (%)'] = np.append(np.round((diag / sumRows) * 100, 1),
                              [np.nan, np.round((diag.sum() / sumColumns.sum()) * 100, 1)])

    ## Store dataframe to file, for futher processing
    dfC.to_csv(os.path.join(
        fn_loc_dir,
        "output",
        f"{cm_raw}_oa_df.csv"))

    return df_oa


# Accuracy assessment
def plot_oa_multiplot(info, CITY, TA_VERSION, DPI=150):

    # Get the years in the dataset
    years = list(info['TA'][TA_VERSION].keys())

    xlabels = ['$OA$', '$OA_{u}$', '$OA_{bu}$', '$OA_{w}$', '', \
               '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    lczcol = ['m', 'm', 'm', 'm', '#FFFFFF'] + info['LCZ']['COLORS']

    fontsize = 10

    # General figure settings
    meanpointprops = dict(marker='o', markerfacecolor='w',
                          markeredgecolor='0.8',markersize=2)
    whiskerprops = dict(color='0.7', linestyle='-')
    capprops = dict(color='0.7')
    boxprops = dict(linewidth=0)
    flierprops = dict(marker='o', markerfacecolor='0.7',
                      alpha=0.5)

    fig, axes = plt.subplots(1,len(years),
                             figsize=(4*len(years),4), sharey=True)

    # Set figure name
    FIGFILE = os.path.join(
        fn_loc_dir,
        "output",
        f"plot_OA_BOXPLOT.jpg"
    )

    # Start loop over years
    for y, year in enumerate(years):

        df_oa = _get_oa_df(
            info = info,
            CITY=CITY,
            TA_VERSION=TA_VERSION,
            YEAR=year
            )

        # Make subplot
        sns.boxplot(data=df_oa.astype(float), palette=lczcol,
                    boxprops=boxprops, flierprops=flierprops,
                    meanline=False, showmeans=True, meanprops=meanpointprops,
                    whis=[5, 95], whiskerprops=whiskerprops, capprops=capprops, ax=axes[y])
        axes[y].set_axisbelow(True)
        axes[y].set_title(year, fontsize=fontsize)

        ## add visuals to improve clarity
        axes[y].axvline(x=4, color="gray", linewidth=1)
        axes[y].grid(axis='y', linestyle=':', linewidth=1.5, color='0.4')
        axes[y].tick_params(axis="y", labelsize=fontsize)
        axes[y].set_ylim((0, 1.1))
        axes[y].text(len(lczcol)/2, 1.05, 'F1 metric',fontsize=fontsize,color='0.4')
        axes[y].set_xticklabels(xlabels, rotation='vertical', fontsize=fontsize)
        axes[y].text(len(lczcol) / 2, -0.16, 'LCZ Class', fontsize=fontsize)
        axes[y].set_ylabel('Accuracy', fontsize=fontsize)

    ## Save image
    plt.tight_layout()
    plt.savefig(FIGFILE, dpi=DPI, bbox_inches='tight')
    plt.close('all')


# Make lczmap
def plot_lczmap_multiplot(info, CITY, TA_VERSION, BAND_TO_PLOT, DPI):

    # Get the years in the dataset
    years = list(info['TA'][TA_VERSION].keys())

    band_labels = {
        0 : 'LCZ',
        1: 'lczFilter'
    }

    cb_labels = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        'A', 'B', 'C', 'D', 'E', 'F', 'G',
    ]

    cmap = mpl.colors.ListedColormap(info['LCZ']['COLORS'])
    cmap.set_bad(color='white')
    cmap.set_under(color='white')

    # Initialize figure here.
    fig, axes = plt.subplots(1,len(years), figsize=(4*len(years),4), sharey=True)

    # Set figure name
    FIGFILE = os.path.join(
        fn_loc_dir,
        "output",
        f"plot_LCZ_MAP.jpg"
    )

    for y, year in enumerate(years):

        print(f"Mapping {year}")

        # Read geotif to plot map
        tif_file = f"LCZ_{CITY}_" \
                f"{TA_VERSION}_" \
                f"{year}_" \
                f"CC{info['CC']}_" \
                f"ED{info['EXTRA_DAYS']}_" \
                f"JDs{info['JD_START']}_{info['JD_END']}_" \
                f"L7{info['ADD_L7']}.tif"

        lczTif = xr.open_rasterio(os.path.join(
            fn_loc_dir,
            "output",
            tif_file)
        )

        #lczTif_clean = lczTif[0, :, :].fillna(0)
        im = lczTif[BAND_TO_PLOT, :, :].plot(
            cmap=cmap, vmin=1, vmax=info['LCZ']['NRLCZ'],
            ax=axes[y], add_colorbar=False,
        )
        axes[y].set_title(year)

        # Remove all axes thicks and labels
        axes[y].set_xlabel('')
        axes[y].set_ylabel('')
        axes[y].set_xticklabels([])
        axes[y].set_yticklabels([])
        axes[y].set_xticks([])
        axes[y].set_yticks([])

    # Save image
    plt.tight_layout()
    plt.savefig(
        fname=FIGFILE,
        dpi=DPI, bbox_inches='tight',
    )
    plt.close('all')

###############################################################################
##### __main__  scope
###############################################################################

info = _read_config(CITY)

print("TA frequency plot")
plt_ta_freq_year(info=info, CITY=CITY, TA_VERSION=TA_VERSION)

print("Plot OA per city")
plot_oa_multiplot(info=info, CITY=CITY, TA_VERSION=TA_VERSION,
                  DPI=150)

print("Plot LCZ map per city")
plot_lczmap_multiplot(info=info, CITY=CITY, TA_VERSION=TA_VERSION,
                      BAND_TO_PLOT=1, DPI=150)

###############################################################################
