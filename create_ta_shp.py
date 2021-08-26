import yaml
from typing import Dict
from typing import Any
from zipfile import ZipFile
import shutil
import os
import fiona
fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default
import geopandas as gpd
import numpy as np
import traceback
import argparse
from argparse import RawTextHelpFormatter
import zipfile


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
parser.add_argument(type=str, dest='EE_ACCOUNT',
                    help="Which EE account to use?",
                    )
parser.add_argument(type=str, dest='TA_VERSION',
                    help='Version of TA set (default is "v1")',
                    default="v1",
                    )
args = parser.parse_args()

# Arguments to script
CITY       = args.CITY
EE_ACCOUNT = args.EE_ACCOUNT
TA_VERSION = args.TA_VERSION

# For testing
CITY       = 'Melbourne'
EE_ACCOUNT = 'mdemuzere'
TA_VERSION = 'v1'


# Set files and folders:
fn_loc_dir = f"/home/demuzmp4/Nextcloud/data/wudapt/dynamic-lcz/{CITY}"
fn_ee_dir  = f"projects/WUDAPT/LCZ_L0/dynamic-lcz/{CITY}"
fn_ee_acc  = "/home/demuzmp4/Nextcloud/scripts/tools/set_ee_account.sh"

print("> Setting requested EE acount first ...")
os.system(f"bash {fn_ee_acc} {EE_ACCOUNT}")

print("> Create EE folder if it does not already exists ...")
try:
    os.system(f"earthengine create folder {fn_ee_dir}")
except Exception:
    err = traceback.format_exc()
    print(f"WARNING, unable to create EE folder: \n {err}")
    pass

# ************** HELPER FUNCTIONS ***********************

def _read_config(CITY) -> Dict[str, Dict[str, Any]]:
    with open(
        os.path.join(
            '/home/demuzmp4/Nextcloud/scripts/wudapt/dynamic-lcz/config',
            f'{CITY.lower()}.yaml',
        ),
    ) as ymlfile:
        pm = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return pm


def _convert_kmz2kml(up_dir, taName) -> str:

    """
    Helper function to convert .kmz to .kml
    """

    kmz = ZipFile(os.path.join(up_dir,taName), 'r')
    kmz.extract('doc.kml', up_dir)

    taName_kml = taName.replace('.kmz','.kml')
    shutil.move(os.path.join(up_dir,'doc.kml'),
                os.path.join(up_dir,taName_kml))

    return taName_kml


def _fix_lczFolders(lczFolders: list,
                    lczDict: dict) -> list:

    """
    Helper function to fix LCZ folders
    """

    lczFoldersFixed = []
    for i in lczFolders:
        if i in [str(i) for i in range(1, 11, 1)] + \
                list(lczDict[0].keys()) + \
                list(lczDict[1].keys()) + \
                list(lczDict[2].keys()):
            lczFoldersFixed.append(i)
        else:
            print(f'Skipping folder {i}')

    return lczFoldersFixed


def _check_clean_ta(ifile: str):

    """
    Initial check of TA format:
    1) should be kmz or kml file
    2) should contain folders of TA classes, as expected from the template file

    Template file: http://www.wudapt.org/wp-content/uploads/2020/08/WUDAPT_L0_Training_template.kml

    """

    # Get the parts of the file dir and name
    up_dir = '/'.join(ifile.split('/')[:-1])
    taName = ifile.split('/')[-1]

    ## Check if extension is correct
    if not taName.lower().endswith(('.kmz', '.kml')):
        print("File not .kmz nor .kml.")

    ## If .kmz file, convert to kml.
    if taName.lower().endswith('.kmz'):
        taName = _convert_kmz2kml(up_dir, taName)

    ## Check if file can be read, according to LCZ TA template with folders.
    lczFolders = fiona.listlayers("{}/{}".format(up_dir,taName))

    if len(lczFolders) > 1:
        continueValue = True

    else:
        continueValue = False

    ## All good so far, continue cleaning
    if continueValue:

        ## Copy file from upload to projectdir, rename to hashId
        taNameNew = os.path.join(up_dir,taName)
        #taNameNew = "/home/demuzmp4/Nextcloud/data/wudapt/dynamic-lcz/Melbourne/input/v1/[Melbourne][2019][JamesBennie].kml"

        lczConversionDict = \
            {
                0: {'101': 11, '102': 12, '103': 13, '104': 14,
                    '105': 15, '106': 16, '107': 17, '108': 18, '109': 19},
                1: {'A': 11, 'B': 12, 'C': 13, 'D': 14,
                    'E': 15, 'F': 16, 'G': 17, 'H': 19, 'W': 18},
                2: {'11': 11, '12': 12, '13': 13, '14': 14,
                    '15': 15, '16': 16, '17': 17, '19': 19, '18': 18},
            }

        ## Remove LCZ folders that are not LCZ classes
        lczFoldersTmp = fiona.listlayers(f"{taNameNew}")
        lczFolders = _fix_lczFolders(lczFoldersTmp, lczConversionDict)

        # Initialize GeoDataFrame to store clean training area info
        df = gpd.GeoDataFrame()

        try:

            # iterate over valid LCZ layers
            for lczFolder in lczFolders:

                #lczFolder = '2'
                print(lczFolder)

                ## Read content per folder (layer)
                ## If it fails, read with fiona, take out unvalid ones, and write to temp file
                ## Read that again with gpd and append to file.
                try:
                    a = gpd.read_file(f"{taNameNew}", driver='KML', layer=lczFolder)

                    ## Remove style place holder or empty geometry if present
                    try:
                        a = a[~a.is_empty]

                    except:
                        pass

                except Exception:
                    ## Sometime the above procedure is not sufficient.
                    ## Could also be a geometry that does not have a proper type
                    ## CHECK THIS: https://github.com/Toblerity/Fiona/blob/master/examples/with-shapely.py
                    source = fiona.open(f"{taNameNew}", 'r', layer=lczFolder)

                    ## Write temp file with lczFolder extention
                    tmpFile = f"{taNameNew.replace('.kml',f'_{lczFolder}.kml')}"
                    with fiona.open(tmpFile, 'w', **source.meta, layer=lczFolder) as sink:
                        for f in source:
                            coords = f['geometry']['coordinates']
                            if len(coords) > 0:
                                if len(coords[0]) > 3:
                                    sink.write(f)
                    a = gpd.read_file(tmpFile, driver='KML', layer=lczFolder)
                    os.system('rm {}'.format(tmpFile))

                ## Add Class field that contains folder name
                if lczFolder in list(lczConversionDict[0].keys()):
                    a['Class'] = lczConversionDict[0][lczFolder]
                elif lczFolder in list(lczConversionDict[1].keys()):
                    a['Class'] = lczConversionDict[1][lczFolder]
                elif lczFolder in list(lczConversionDict[2].keys()):
                    a['Class'] = lczConversionDict[2][lczFolder]
                else:
                    a['Class'] = int(lczFolder)

                ## Remove the description and Name columns
                if 'Description' in a.columns:
                    a.drop('Description', axis=1, inplace=True)

                if 'Name' in a.columns:
                    a.drop('Name', axis=1, inplace=True)

                ## Put all layers together
                df = df.append(a, ignore_index=True)


            # convert multi part features to single part features
            # reset the index to get rid of the now multi part index
            df = df.explode().reset_index(drop=True)

            ## Again set Class to integers, info seems to be lost?
            df['Class'] = [int(i) for i in df['Class']]
            df['City']  = taName.split('_')[0]

            #ofile = "{}/{}".format(up_dir,taName).replace(".kml",f"{ext_name}.kml")
            #df.to_file(ofile,driver='KML')

        except:
            print('something went wrong')

        return df


def _add_geometry(df):

    try:
        df = df.set_crs("epsg:4326")
        df_m = df.copy()
        df_m = df_m.to_crs("epsg:3395")
    except Exception:
        print("Unable to reproject:")

    try:
        df['area']      = df_m['geometry'].area / 1e6 # km²
        df['perimeter'] = df_m['geometry'].length / 1e3 #km
        df['shape']     = (df['perimeter'] * df['perimeter'])/\
                          (4 * np.pi * df['area'])
        df['vertices'] = [
            len(i[1].geometry.exterior.coords) - 1 for i in df.iterrows()
        ]
    except Exception:
        print("Unable to calculate geometric information:")

    return df

# *************** Launch the process ********************
# Set the info
info = _read_config(CITY)

# Initialize dataframe
df_all = gpd.GeoDataFrame()

# Loop TA years
for year in list(info['TA'][TA_VERSION].keys()):

    print(f"Processing {year}")

    ## Read and process TAs for one city, all years
    ifile = os.path.join(
        fn_loc_dir,
        "input",
        TA_VERSION,
        info['TA'][TA_VERSION][year]
    )

    df = _check_clean_ta(ifile)

    # Add geometry
    df = _add_geometry(df)

    # Add additional data
    df['Author'] = info['AUTHOR']
    df['City'] = CITY
    df['Year'] = year

    df_all = df_all.append(df)

df_all['Class'] = [int(i) for i in df_all.Class]

SHP_FILE = os.path.join(
        fn_loc_dir,
        "input",
        f'TA_{TA_VERSION}.shp'
    )
df_all.to_file(SHP_FILE, driver='ESRI Shapefile')

# Zip to shp for upload
shp_file_base = SHP_FILE.replace('.shp','')
shp_list = [f"{shp_file_base}.{i}" for i in ['cpg','dbf','shp','shx']]
zip_out = SHP_FILE.replace('.shp','.zip')
with zipfile.ZipFile(zip_out, 'w') as zipMe:
    for file in shp_list:
        zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)

print(" ++++++++++++++  IMPORTANT ++++++++++++++++")
print(f" --> Manually upload SHP ZIP to EE: {zip_out}")
