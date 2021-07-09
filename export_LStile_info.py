import yaml
from typing import Dict
from typing import Any
import traceback
import os
import ee
import pandas as pd
ee.Initialize()

def _read_config() -> Dict[str, Dict[str, Any]]:
    with open(
        os.path.join(
            '/home/demuzmp4/Nextcloud/scripts/wudapt/dynamic-lcz-china',
            'param_config.yaml',
        ),
    ) as ymlfile:
        pm = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return pm


def _get_roi(info, city):
    roi = ee.FeatureCollection(os.path.join(
        info['EE_TA_DIR'],
        "cai_zhi_all"))\
        .filter(ee.Filter.eq("City",city)) \
        .geometry().bounds().buffer(info['lcz']['ROIBUFFER']).bounds()
    return roi

def _mask_clouds(img):
  # Bits 3 and 5 are cloud shadow and cloud, respectively.
  cloudShadowBitMask = 1 << 3
  cloudsBitMask = 1 << 5

  # Get the pixel QA band.
  qa = img.select('pixel_qa')

  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
      .And(qa.bitwiseAnd(cloudsBitMask).eq(0))

  # Return the masked image, scaled to reflectance, without the QA bands.
  # .select("B[0-9]*")\
  return img.updateMask(mask)\
      .copyProperties(img, img.propertyNames())


# Set path row info per city
def _get_pathrow_dict():
    idict = {
        'Beijing': {'PATH': 123,
                    'ROW': 32,
                    'S_DOY': 60,
                    'E_DOY': 335},
        'Shanghai': {'PATH': 118,
                     'ROW': 38,
                     'S_DOY': 60,
                     'E_DOY': 335},
        'Guangzhou': {'PATH': 122,
                      'ROW': 44,
                      'S_DOY': 0,
                      'E_DOY': 366},
    }

    return idict

## Band names depending on sensor
def _l8rename(img):
    return img.select(['B2', 'B3', 'B4', 'B5', 'B6',
                       'B7', 'B10', 'pixel_qa'],
                      ['blue', 'green', 'red', 'nir', 'swir1',
                       'swir2', 'tirs1', 'pixel_qa'])

def _l5_7rename(img):
    return img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                       'B7', 'pixel_qa'],
                      ['blue', 'green', 'red', 'nir', 'swir1', 'tirs1',
                       'swir2', 'pixel_qa'])

def _get_all_ls(city, year, CLOUD_COVER, EXTRA_DAYS, CLOUD_MASKING, ADD_L7):

    print(f"Gathering the appropriate Landsat images for {city} - {year} ...")

    # Get the roi
    roi = _get_roi(info, city)

    # Sample 0.5 year before / after year of interest
    start_date = ee.Date(str(year) + '-01-01')
    end_date   = ee.Date(str(year) + '-12-31')

    idict = _get_pathrow_dict()

    _ls8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
                .filterDate(start_date.advance(-EXTRA_DAYS,"days"),
                            end_date.advance(EXTRA_DAYS,"days")) \
                .filterBounds(roi) \
                .filterMetadata('CLOUD_COVER', 'not_greater_than', CLOUD_COVER) \
                .filterMetadata('WRS_PATH', 'equals', idict[city]['PATH']) \
                .filterMetadata('WRS_ROW', 'equals', idict[city]['ROW']) \
                .filter(ee.Filter.dayOfYear(idict[city]['S_DOY'], idict[city]['E_DOY'])) \
                .map(_l8rename)
    _ls5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')\
                .filterDate(start_date.advance(-EXTRA_DAYS,"days"),
                            end_date.advance(EXTRA_DAYS,"days")) \
                .filterBounds(roi) \
                .filterMetadata('CLOUD_COVER', 'not_greater_than', CLOUD_COVER) \
                .filterMetadata('WRS_PATH', 'equals', idict[city]['PATH']) \
                .filterMetadata('WRS_ROW', 'equals', idict[city]['ROW']) \
                .filter(ee.Filter.dayOfYear(idict[city]['S_DOY'], idict[city]['E_DOY'])) \
                .map(_l5_7rename)

    if year < 2003 and ADD_L7:
        _ls7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR') \
            .filterDate(start_date.advance(-EXTRA_DAYS, "days"),
                        end_date.advance(EXTRA_DAYS, "days")) \
            .filterBounds(roi) \
            .filterMetadata('CLOUD_COVER', 'not_greater_than', CLOUD_COVER) \
            .filterMetadata('WRS_PATH', 'equals', idict[city]['PATH']) \
            .filterMetadata('WRS_ROW', 'equals', idict[city]['ROW']) \
            .filter(ee.Filter.dayOfYear(idict[city]['S_DOY'], idict[city]['E_DOY'])) \
            .map(_l5_7rename)

        _ls_all = _ls5.merge(_ls7).merge(_ls8)
    else:
        _ls_all = _ls5.merge(_ls8)

    if CLOUD_MASKING:
        print("Mask the clouds")
        ls_all = _ls_all.map(_mask_clouds)
    else:
        ls_all = _ls_all

    return ls_all


def extract_ls_tile_info(city, year, CLOUD_COVER, EXTRA_DAYS, CLOUD_MASKING, ADD_L7):

    # Get the roi
    roi = _get_roi(info, city)

    # Get the image collection
    ls_all = _get_all_ls(city, year, CLOUD_COVER, EXTRA_DAYS, CLOUD_MASKING, ADD_L7)

    # Set filmstrip settings
    stripVis = {
        'dimensions':1000,
        'min':0.0,
        'max':3000,
        'bands':['red', 'green', 'blue'],
        'region':roi,
    }

    ls_all_count = ls_all.size().getInfo()
    ls_all_url = ls_all.getFilmstripThumbURL(stripVis)
    ls_all_ids = ls_all.aggregate_array('LANDSAT_ID').getInfo()

    return ls_all_count, ls_all_url, ls_all_ids


###############################################################################
# _Execute code
###############################################################################

print('Set info')
info = _read_config()

print('Set proper EE account, kode as default')
os.system(f"bash {info['EE_SET_ACCOUNT']} kode")

cities = ['Beijing', 'Guangzhou', 'Shanghai']
years  = [2000, 2005, 2010, 2015, 2020]

CLOUD_COVER=10
EXTRA_DAYS=180
CLOUD_MASKING = True
ADD_L7=True

for city in ['Beijing', 'Guangzhou', 'Shanghai']:
#for city in ['Beijing']:

    # Initialize dataframe to store information, per city
    df = pd.DataFrame(index=years,columns=['CNT','URL','IDs'])
    ofile = os.path.join(
        info['OUT_DIR'],
        f"{city}_ids_{CLOUD_COVER}_{EXTRA_DAYS}_{CLOUD_MASKING}_{ADD_L7}.csv"
    )

    #for year in [2000]:
    for year in years:

        ls_all_count, ls_all_url, ls_all_ids = extract_ls_tile_info(
            city, year, CLOUD_COVER, EXTRA_DAYS, CLOUD_MASKING, ADD_L7)

        df.loc[year, 'CNT'] = ls_all_count
        df.loc[year,'URL'] = ls_all_url
        df.loc[year, 'IDs'] = ls_all_ids

    df.to_csv(ofile)

###############################################################################