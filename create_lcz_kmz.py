import yaml
import io
import zipfile
import time
from typing import Dict, Any
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
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
# CITY       = 'Melbourne'
# TA_VERSION = 'v1'

def _read_config(CITY) -> Dict[str, Dict[str, Any]]:
    with open(
        os.path.join(
            '/home/demuzmp4/Nextcloud/scripts/wudapt/dynamic-lcz/config',
            f'{CITY.lower()}.yaml',
        ),
    ) as ymlfile:
        pm = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return pm


def make_kmz(info, CITY, TA_VERSION, BAND_TO_PLOT=1) -> None:
    """
    Function to create a kmz file with the lcz map in the correct color map
    """

    # Get the years in the dataset
    years = list(info['TA'][TA_VERSION].keys())


    for y, year in enumerate(years):

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

        # Get corner coordinates
        xmin = float(lczTif.x.min())
        xmax = float(lczTif.x.max())
        ymin = float(lczTif.y.min())
        ymax = float(lczTif.y.max())

        # Make png figure from the geotif.
        cmap = mpl.colors.ListedColormap(info['LCZ']['COLORS'])
        cmap.set_bad(color='white')
        cmap.set_under(color='white')
        figsize = (lczTif.shape[1] / 100, lczTif.shape[2] / 100)
        fig, ax = plt.subplots(figsize=figsize)
        lczTif[BAND_TO_PLOT].plot(
            cmap=cmap,
            vmin=1,
            vmax=info['LCZ']['NRLCZ'],
            ax=ax,
            add_colorbar=False
        )
        ax.set_title('')
        plt.axis('off')

        # Info on removing white boundary
        # https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image-in-matplotlib
        # pad_inches does the trick
        with io.BytesIO() as figfile, io.BytesIO() as kmlfile:
            plt.savefig(
                fname=figfile,
                facecolor=fig.get_facecolor(),
                transparent=True,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0,
            )
            plt.close('all')
            # Basic xml text needed to create kmz.
            txt = f'''\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
    <GroundOverlay>
        <name>{tif_file.replace('.tif', '')}</name>
        <Icon>
            <href>{tif_file.replace('.tif', '')}.png</href>
            <viewBoundScale>1</viewBoundScale>
        </Icon>
        <LatLonBox>
            <north>{ymax}</north>
            <south>{ymin}</south>
            <east>{xmax}</east>
            <west>{xmin}</west>
        </LatLonBox>
    </GroundOverlay>
</kml>
'''
            kmlfile.write(bytes(txt, encoding='utf-8'))
            contents = (
                (
                    zipfile.ZipInfo(
                        filename=f"{tif_file.replace('.tif', '')}.png",
                        date_time=time.gmtime()[:6],
                    ),
                    figfile.getvalue(),
                ),
                (
                    zipfile.ZipInfo(
                        filename=f"{tif_file.replace('.tif', '')}.kml",
                        date_time=time.gmtime()[:6],
                    ),
                    kmlfile.getvalue(),
                ),
            )

    OFILE_KMZ = os.path.join(
        fn_loc_dir,
        'output',
        tif_file.replace('.tif', '.kmz'),
    )
    with zipfile.ZipFile(OFILE_KMZ, 'w') as zf:
        for f in contents:
            zf.writestr(*f)

    print(OFILE_KMZ)

###############################################################################
##### __main__  scope
###############################################################################
info = _read_config(CITY)
# Set files and folders:
fn_loc_dir = f"/home/demuzmp4/Nextcloud/data/wudapt/dynamic-lcz/{CITY}"
make_kmz(info, CITY, TA_VERSION, BAND_TO_PLOT=1)
###############################################################################