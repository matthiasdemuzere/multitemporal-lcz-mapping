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

#version = "NO_L7"
#version = "WITH_L7"
version = "FINAL"

def main(version):

    info = _read_config()
    info['VERSION'] = version

    # print("TA frequency plot")
    # plt_ta_freq_city_year(info)

    # print("Plot OA & LCZ map per city / all years")
    # #year=2005
    # #city='Shanghai'
    # cities = ['Guangzhou', 'Shanghai'] #'Beijing',
    # years = [2000, 2005, 2010, 2015, 2020]
    # for city in cities:
    #     for year in years:
    #
    #         plot_oa(info,  year=year, city=city)
    #         plot_lczmap(info,  year=year, city=city)

    print("Plot OA & LCZ map per city - multipanel")
    #city = 'Guangzhou'
    #city = 'Shanghai'
    city = 'Beijing'
    plot_oa_multiplot(info=info, city=city,DPI=300)
    plot_lczmap_multiplot(info=info, city=city, BAND_TO_PLOT=0, DPI=300)
    plot_lczmap_multiplot(info=info, city=city, BAND_TO_PLOT=1, DPI=300)

def _read_config() -> Dict[str, Dict[str, Any]]:
    with open(
        os.path.join(
            '/home/demuzmp4/Nextcloud/scripts/wudapt/dynamic-lcz-china',
            'param_config.yaml',
        ),
    ) as ymlfile:
        pm = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return pm

# Make stacked bar plot, color per available city.
def plt_ta_freq_city_year(info) -> None:

    """
    Create TA frequency barchart, per city and year,
    with stacked LCZ classes.
    """

    df = gpd.read_file(info['TA_SHP'])

    # Initialize figure
    fig, ax = plt.subplots(1,3,figsize=(15, 6), sharey=True)

    # Map LCZ counts per year and city
    cities = ['Beijing', 'Guangzhou', 'Shanghai']

    for c_i, city in enumerate(cities):

        df_sel = df[df.City == city]

        df2 = df_sel.groupby(['Year', 'Class'])['Class'].count()\
            .unstack('Class').fillna(0)
        df2.plot(kind='bar', stacked=True, ax=ax[c_i],
                 color=info['lcz']['COLORS'])

        ax[c_i].set_ylabel('# Training areas')
        ax[c_i].set_xlabel('Year')
        ax[c_i].grid(zorder=0, ls=':', color='0.5')
        ax[c_i].set_title(city)

    # Legend on last panel only
    ax[0].legend().set_visible(False)
    ax[1].legend().set_visible(False)

    ax[2].legend(title='LCZ class',
                 bbox_to_anchor=(1.03, 0),
                 loc='lower left')

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            info['FIG_DIR'],
            'ta_percity_peryear.pdf',
        ),
        transparent=False,dpi=300
    )
    plt.close('all')


def _get_oa_df(info, cm_file):

    natClass = [10, 11, 12, 13, 15, 16]

    # Read the raw confusion matrix
    cm_raw = os.path.join(info['OUT_DIR'],cm_file)

    ## Read OA weights file
    cm_weights = info['CM_WEIGHTS']
    oaFile = os.path.join(cm_weights)
    oaw = pd.read_csv(oaFile, sep=',', header=None)

    ## Get all the accuracy metrics
    df = pd.read_csv('{}'.format(cm_raw))
    mStr = df.iloc[:, 1][0].replace("[", "").replace("]", "")
    mList = [int(e) for e in mStr.split(',')]
    arr = np.array(mList)
    arr_oa = arr.reshape(info['lcz']['NRBOOT'],
                         info['lcz']['NRLCZ'],
                         info['lcz']['NRLCZ'])

    ## Initialize an empty dataframe
    index = range(info['lcz']['NRBOOT'])
    df_oa = pd.DataFrame(index=index, columns=range(5 + info['lcz']['NRLCZ']))

    ## Loop over all available bootstraps
    for i in range(info['lcz']['NRBOOT']):
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

    ## Store dataframe to file, for futher processing
    df_oa.to_csv("{}/{}".format(info['OUT_DIR'], cm_file.replace('.csv', '_oa_df.csv')))

    ## create formated confusion matrix, average over all bootstraps
    dfC = pd.DataFrame(arr_oa[i], columns=np.arange(1, info['lcz']['NRLCZ'] + 1, 1),
                       index=np.arange(1, info['lcz']['NRLCZ'] + 1, 1)).astype(int)
    dfC.loc['Total'] = sumColumns
    dfC['Total'] = dfC.sum(1)
    dfC.loc['PA (%)'] = np.append(np.round((diag / sumColumns) * 100, 1), np.nan)
    dfC['UA (%)'] = np.append(np.round((diag / sumRows) * 100, 1),
                              [np.nan, np.round((diag.sum() / sumColumns.sum()) * 100, 1)])
    dfC.to_csv(
        os.path.join(
            info['OUT_DIR'],
            cm_file.replace('.csv','_cm_average_formatted.csv')))

    return df_oa


# Accuracy assessment
def plot_oa(info, city, year):

    xlabels = ['$OA$', '$OA_{u}$', '$OA_{bu}$', '$OA_{w}$', '', \
               '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    lczcol = ['m', 'm', 'm', 'm', '#FFFFFF'] + info['lcz']['COLORS']

    fontsize = 12

    # General figure settings
    meanpointprops = dict(marker='o', markerfacecolor='w', markeredgecolor='0.8',markersize=2)
    whiskerprops = dict(color='0.7', linestyle='-')
    capprops = dict(color='0.7')
    boxprops = dict(linewidth=0)
    flierprops = dict(marker='o', markerfacecolor='0.7', alpha=0.5)

    fig, axes = plt.subplots(1,1, figsize=(7,7))
    # Set figure name
    figfile = os.path.join(
        info['FIG_DIR'],
        f"{city}_{year}_SCALE{info['lcz']['SCALE']}_CC{info['CC']}_DAYS{info['EXTRA_DAYS']}_oa.pdf"
    )
    # Get accuracies.
    cm_file = os.path.join(
        info['VERSION'],
        f"{city}_{year}_SCALE{info['lcz']['SCALE']}_CC{info['CC']}_DAYS{info['EXTRA_DAYS']}_cm.csv"
    )
    df_oa = _get_oa_df(
        info = info,
        cm_file = cm_file)

    # Make subplot
    sns.boxplot(data=df_oa.astype(float), palette=lczcol,
                boxprops=boxprops, flierprops=flierprops,
                meanline=False, showmeans=True, meanprops=meanpointprops,
                whis=[5, 95], whiskerprops=whiskerprops, capprops=capprops, ax=axes)
    axes.set_axisbelow(True)
    axes.set_title(year, fontsize=fontsize)

    ## add visuals to improve clarity
    axes.axvline(x=4, color="gray", linewidth=1)
    axes.grid(axis='y', linestyle=':', linewidth=1.5, color='0.4')
    axes.tick_params(axis="y", labelsize=fontsize)
    axes.set_ylim((0, 1.1))
    axes.text(len(lczcol)/2, 1.05, 'F1 metric',fontsize=fontsize,color='0.4')
    axes.set_xticklabels(xlabels, rotation='vertical', fontsize=fontsize)
    axes.text(len(lczcol) / 2, -0.16, 'LCZ Class', fontsize=fontsize)
    axes.set_ylabel('Accuracy', fontsize=fontsize)

    ## Save image
    plt.tight_layout()
    plt.savefig(figfile, dpi=300, bbox_inches='tight')
    plt.close('all')


# Make lczmap
def plot_lczmap(info, city, year):

    cb_labels = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        'A', 'B', 'C', 'D', 'E', 'F', 'G',
    ]

    cmap = mpl.colors.ListedColormap(info['lcz']['COLORS'])
    cmap.set_bad(color='white')
    cmap.set_under(color='white')

    fig, axes = plt.subplots(1,1, figsize=(12,12))

    # Initialize figure here.
    figfile = os.path.join(
        info['FIG_DIR'],
        f"{city}_{year}_SCALE{info['lcz']['SCALE']}_CC{info['CC']}_DAYS{info['EXTRA_DAYS']}_lczmap.jpg"
    )

    print(f"Mapping {city} for {year}")

    # Read geotif to plot map
    lczTif = xr.open_rasterio(
        os.path.join(
            info['OUT_DIR'],
            info['VERSION'],
            f"{city}_{year}_SCALE{info['lcz']['SCALE']}_CC{info['CC']}_DAYS{info['EXTRA_DAYS']}.tif",
        ),
    )

    #lczTif_clean = lczTif[0, :, :].fillna(0)
    im = lczTif[1, :, :].plot(
        cmap=cmap, vmin=1, vmax=info['lcz']['NRLCZ'],
        ax=axes, add_colorbar=False,
    )

    # #cbar_ax = fig.add_axes([0.45, 0.25, 0.45, 0.025])
    # cbar_ax = fig.add_axes([0.35, 0.02, 0.02, 0.29])
    # cb = plt.colorbar(
    #     im,
    #     ticks=np.linspace(1.5, info['lcz']['NRLCZ']-0.5, info['lcz']['NRLCZ']),
    #     orientation='vertical', pad=0.08,
    #     cax=cbar_ax
    #     )
    # cb.set_ticklabels(cb_labels)
    # cb.set_label(label='LCZ Class', fontsize=11)
    # cb.ax.tick_params(labelsize=11)

    axes.set_title(year)
    # #ax.tick_params(axis='both', which='major', labelsize=9)
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    # ax.set_title(list_years[ax_i])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticks([])
    # ax.set_yticks([])

    # Save image
    plt.tight_layout()
    plt.savefig(
        fname=figfile,
        dpi=150, bbox_inches='tight',
    )
    plt.close('all')


def plot_oa_multiplot(info, city, DPI):

    years = list(range(2000, 2021, 5))
    #years = list(range(2010, 2021, 5))

    xlabels = ['$OA$', '$OA_{u}$', '$OA_{bu}$', '$OA_{w}$', '', \
               '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    lczcol = ['m', 'm', 'm', 'm', '#FFFFFF'] + info['lcz']['COLORS']

    fontsize = 10

    # General figure settings
    meanpointprops = dict(marker='o', markerfacecolor='w', markeredgecolor='0.8',markersize=2)
    whiskerprops = dict(color='0.7', linestyle='-')
    capprops = dict(color='0.7')
    boxprops = dict(linewidth=0)
    flierprops = dict(marker='o', markerfacecolor='0.7', alpha=0.5)

    fig, axes = plt.subplots(1,len(years), figsize=(4*len(years),4), sharey=True)

    # Set figure name
    figfile = os.path.join(
        info['FIG_DIR'],
        f"{city}_SCALE{info['lcz']['SCALE']}_CC{info['CC']}_"
        f"DAYS{info['EXTRA_DAYS']}_multi_oa.jpg"
    )

    # Start loop over years
    for y, year in enumerate(years):

        # Get accuracies.
        cm_file = os.path.join(
            info['VERSION'],
            f"{city}_{year}_SCALE{info['lcz']['SCALE']}_CC{info['CC']}_"
            f"DAYS{info['EXTRA_DAYS']}_cm.csv"
        )
        df_oa = _get_oa_df(
            info = info,
            cm_file = cm_file)

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
    plt.savefig(figfile, dpi=DPI, bbox_inches='tight')
    plt.close('all')


# Make lczmap
def plot_lczmap_multiplot(info, city, BAND_TO_PLOT, DPI):

    years = list(range(2000,2021,5))

    band_labels = {
        0 : 'lcz',
        1: 'lczFilter'
    }

    cb_labels = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        'A', 'B', 'C', 'D', 'E', 'F', 'G',
    ]

    cmap = mpl.colors.ListedColormap(info['lcz']['COLORS'])
    cmap.set_bad(color='white')
    cmap.set_under(color='white')

    fig, axes = plt.subplots(1,len(years), figsize=(4*len(years),4), sharey=True)

    # Initialize figure here.
    figfile = os.path.join(
        info['FIG_DIR'],
        f"{city}_SCALE{info['lcz']['SCALE']}_CC{info['CC']}_"
        f"DAYS{info['EXTRA_DAYS']}_{band_labels[BAND_TO_PLOT]}_multi_map.jpg"
    )

    for y, year in enumerate(years):

        print(f"Mapping {city} for {year}")

        # Read geotif to plot map
        lczTif = xr.open_rasterio(
            os.path.join(
                info['OUT_DIR'],
                info['VERSION'],
                f"{city}_{year}_SCALE{info['lcz']['SCALE']}_CC{info['CC']}_DAYS{info['EXTRA_DAYS']}.tif",
            ),
        )

        #lczTif_clean = lczTif[0, :, :].fillna(0)
        im = lczTif[BAND_TO_PLOT, :, :].plot(
            cmap=cmap, vmin=1, vmax=info['lcz']['NRLCZ'],
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
        fname=figfile,
        dpi=DPI, bbox_inches='tight',
    )
    plt.close('all')

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

        main(version)

###############################################################################
