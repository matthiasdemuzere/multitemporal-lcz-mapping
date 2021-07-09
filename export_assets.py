import yaml
from typing import Dict
from typing import Any
import numpy as np
import os
import ee
ee.Initialize()


def main(year, city):

    print('Set info')
    info = _read_config()

    print('Set proper EE account, kode as default')
    os.system(f"bash {info['EE_SET_ACCOUNT']} kode")

    print(f"Exporting assets for {city} - {year}")
    export_all_ls(info, city, year)

    return 0

def _read_config() -> Dict[str, Dict[str, Any]]:
    with open(
        os.path.join(
            '/home/demuzmp4/Nextcloud/scripts/wudapt/dynamic-lcz-china',
            'param_config.yaml',
        ),
    ) as ymlfile:
        pm = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return pm



## Mask clouds
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

## Function to create BCI index
def _add_bci(img):

    b = img.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).divide(10000);

    ## Coefficients from Table S1 in De Vries et al., 2016
    brightness_= ee.Image([0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303]);
    greenness_= ee.Image([-0.1603, 0.2819, -0.4934, 0.7940, -0.0002, -0.1446]);
    wetness_= ee.Image([0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]);
    sum = ee.call("Reducer.sum");

    brightness = b.multiply(brightness_).reduce(sum).rename('br');
    greenness = b.multiply(greenness_).reduce(sum).rename('gr');
    wetness = b.multiply(wetness_).reduce(sum).rename('we');

    combined_ = ee.Image(brightness).addBands(greenness).addBands(wetness);

    ## Calculate BCI (Deng & Wu, 2012)
    bci_ = (brightness.add(wetness)).divide(ee.Image(2));
    bci = (bci_.subtract(greenness)).divide((bci_.add(greenness))).rename('bci').toFloat();

    combined = combined_.addBands(bci);
    return img.addBands(combined);

## Function to calculate other bands ratios
def _add_ratios(img):

  ndbai = img.normalizedDifference(['swir1','tirs1']).rename('ndbai')
  ndbi = img.normalizedDifference(['swir1','nir']).rename('ndbi')
  ebbi = (img.select('swir1').subtract(img.select('nir')))\
               .divide(ee.Image(10).multiply((img.select('swir1').add(img.select('tirs1'))).sqrt()))\
               .rename('ebbi').toFloat()
  ndvi = img.normalizedDifference(['nir','red']).rename('ndvi')
  ndwi = img.normalizedDifference(['green','nir']).rename('ndwi')

  return img\
            .addBands(ndbai) \
            .addBands(ndbi) \
            .addBands(ebbi) \
            .addBands(ndvi)\
            .addBands(ndwi)\
            .toFloat()

# Helper function to get ROI based on name.
def _get_roi(info, city):

    roi = ee.FeatureCollection(os.path.join(
        info['EE_TA_DIR'],
        "cai_zhi_all"))\
        .filter(ee.Filter.eq("City",city)) \
        .geometry().buffer(15000).bounds()

    return roi


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


def export_all_ls(info, city, year):

    # Get the roi
    roi = _get_roi(info, city)

    # Sample 0.5 year before / after year of interest
    start_date = ee.Date(str(year) + '-01-01')
    end_date   = ee.Date(str(year+1) + '-01-01')

    idict = _get_pathrow_dict()

    # Get Landsat collections
    _ls5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
    _ls7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')\
            .filterDate('1999-01-01','2002-12-31') # Avoid scan line error
    _ls8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

    # Merge collections
    l5 = _ls5.map(_l5_7rename)
    l7 = _ls7.map(_l5_7rename)
    l8 = _ls8.map(_l8rename)
    _ls_all = l5.merge(l7).merge(l8)

    # Filter collection for clouds, dates, roi & add band rations
    ls_all = _ls_all \
        .filterDate(start_date.advance(-info['EXTRA_DAYS'], "days"),
                    end_date.advance(info['EXTRA_DAYS'], "days")) \
        .filterMetadata('CLOUD_COVER', 'not_greater_than', info['CC']) \
        .filter(ee.Filter.dayOfYear(idict[city]['S_DOY'], idict[city]['E_DOY'])) \
        .filterBounds(roi) \
        .map(_mask_clouds) \
        .map(_add_bci) \
        .map(_add_ratios) \
        .select(['blue', 'green', 'red', 'nir', 'swir1', 'tirs1', 'swir2',
                 'bci', 'ndbai', 'ndbi', 'ebbi', 'ndvi', 'ndwi']) \

    # Apply the reducers
    fI_pct = ls_all.reduce(ee.Reducer.percentile([10, 50, 90]))
    fI_std = ls_all.reduce(ee.Reducer.stdDev())

    # Merge all into one image
    finalImage = fI_pct.addBands(fI_std)

    # Add information on orography
    dtm = ee.Image('NASA/ASTER_GED/AG100_003').select('elevation')
    slope = ee.Terrain.slope(dtm).clip(roi)
    finalImage = finalImage.addBands(slope).clip(roi)

    # If asset already exists, remove
    try:
        ee.data.deleteAsset(f"{info['EE_IN_DIR']}/{city}_{year}")
    except:
        pass

    # Export as asset
    task_asset_export = ee.batch.Export.image.toAsset( \
        image=finalImage.clip(roi), \
        description=f"{city}_{year}", \
        assetId=f"{info['EE_IN_DIR']}/{city}_{year}", \
        scale=info['lcz']['SCALE'], \
        region=roi, \
        maxPixels=1e13)
    task_asset_export.start()


###############################################################################
##### __main__  scope
###############################################################################

cities = ['Beijing']

if __name__ == "__main__":

    for year in list(np.arange(2000,2021,5)):

        for city in cities:

            main(year, city)

###############################################################################