import yaml
from typing import Dict
from typing import Any
import traceback
import os
import ee
ee.Initialize()
from datetime import datetime
import traceback
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
# CITY       = 'Hyderabad'
# EE_ACCOUNT = 'mdemuzere'
#TA_VERSION = 'v1'

# Set files and folders:
fn_loc_dir = f"/home/demuzmp4/Nextcloud/data/wudapt/dynamic-lcz/{CITY}"
fn_ee_dir  = f"projects/WUDAPT/LCZ_L0/dynamic-lcz/{CITY}"
fn_ee_acc  = "/home/demuzmp4/Nextcloud/scripts/tools/set_ee_account.sh"

print("> Setting requested EE acount first ...")
os.system(f"bash {fn_ee_acc} {EE_ACCOUNT}")

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

def _get_roi(info, TA_VERSION):
    roi = ee.FeatureCollection(os.path.join(
        fn_ee_dir,
        f"TA_{TA_VERSION}"))\
        .geometry().bounds().buffer(info['LCZ']['ROIBUFFER']).bounds()
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
  return img.updateMask(mask).divide(10000)\
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

def _get_all_ls(info, CITY, TA_VERSION, YEAR):

    print("Gathering the appropriate Landsat images ...")

    # Get the roi
    roi = _get_roi(info, TA_VERSION)

    # Sample 0.5 year before / after year of interest
    start_date = ee.Date(str(YEAR) + '-01-01')
    end_date   = ee.Date(str(YEAR) + '-12-31')

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
        .filter(ee.Filter.dayOfYear(info['JD_START'], info['JD_END'])) \
        .filterBounds(roi) \
        .map(_mask_clouds) \
        .map(_add_bci) \
        .map(_add_ratios) \
        .select(['blue', 'green', 'red', 'nir', 'swir1', 'tirs1', 'swir2',
                 'bci', 'ndbai', 'ndbi', 'ebbi', 'ndvi', 'ndwi'])

    print("Extracting the Landsat IDs for future reference ...")
    def _get_id(element):
        return ee.Feature(None, {'id': element})

    ids = ls_all.aggregate_array('system:index')
    ftColl_ids = ee.FeatureCollection(ids.map(_get_id))

    print("Export selected LS IDs to drive")
    ids_ofile = f"IDs_" \
                f"{CITY}_" \
                f"{YEAR}_" \
                f"CC{info['CC']}_" \
                f"ED{info['EXTRA_DAYS']}_" \
                f"JDs{info['JD_START']}_{info['JD_END']}_" \
                f"L7{info['ADD_L7']}"
    task_export_ids = ee.batch.Export.table.toDrive( \
        collection=ftColl_ids, \
        description=f"{ids_ofile}_LS_IDs", \
        folder=info['GD_FOLDER'], \
        fileFormat='CSV'
    );
    task_export_ids.start()

    # Apply the reducers
    fI_pct = ls_all.reduce(ee.Reducer.percentile([10, 50, 90]))
    fI_std = ls_all.reduce(ee.Reducer.stdDev())

    # Merge all into one image
    finalImage = fI_pct.addBands(fI_std)

    # Add information on orography
    dtm = ee.Image('NASA/ASTER_GED/AG100_003').select('elevation')
    slope = ee.Terrain.slope(dtm).clip(roi)
    finalImage = finalImage.addBands(slope).clip(roi)

    if eval(info['EXPORT_TO_ASSET']):
        # Export as asset
        task_asset_export = ee.batch.Export.image.toAsset( \
            image=finalImage.clip(roi), \
            description=f"{CITY}_{YEAR}_EO_input", \
            assetId=f"{info['EE_IN_DIR']}/{CITY}_{YEAR}_EO_input", \
            scale=info['EXPORT_SCALE'], \
            region=roi, \
            maxPixels=1e13)
        task_asset_export.start()

    # Print the band nalmes
    print(f" Available bands: {finalImage.bandNames().getInfo()}")

    return finalImage.clip(roi)

def _buffer_polygons(info, ta):

    def _addArea(feature):
        area = feature.area()
        return feature.set({'myArea': area})

    def _getCentroidBuffer(feature):
        return feature.centroid().buffer(info['LCZ']['BUFFERSIZE'])

    ## Reduce large polygons
    polyarea = ta.map(_addArea);
    bigPoly = polyarea.filterMetadata('myArea', 'not_less_than', info['LCZ']['POLYSIZE']);
    smaPoly = polyarea.filterMetadata('myArea', 'less_than', info['LCZ']['POLYSIZE']);
    bigPolyRed = bigPoly.map(_getCentroidBuffer);
    polyset = smaPoly.merge(bigPolyRed);

    return polyset

## Sample regions functions to extract pixel values from polygons
def _sample_regions(image,
                   polygons,
                   properties,
                   SCALE):
    """
    Helper function to sample pixel values from EO input assets using TA polygons.
    """

    def reducePoly(f):
        col = (image.reduceRegion(geometry= f.geometry(), \
                                  reducer= reducer, \
                                  scale= SCALE, \
                                  tileScale= 16).get('features'))
        #def copyProp(g):
        #    return g.copyProperties(f.select(properties))

        return ee.FeatureCollection(col).map(lambda x: x.copyProperties(f.select(properties)))

    reducer = ee.Reducer.toCollection(image.bandNames())
    return polygons.map(reducePoly).flatten();


# Perform the LCZ classification
def _gaussian_filter(image,roi):

    ## Rename image back to 'remapped'
    image = image.rename('remapped').clip(roi)

    ## Pre-define list of lcz classes
    LCZclasses = ee.List(["1","2","3","4","5","6","7","8","9","10",\
                          "11","12","13","14","15","16","17"])

    ## Set the radius and sigmas for the gaussian kernels
    ## Currently set as used in Demuzere et al., NSD, 2020.
    dictRadius = ee.Dictionary({
        1: 200, 2: 300, 3: 300, 4: 300, 5: 300, 6: 300, 7: 300, 8: 500, 9: 300, 10: 500,
        11: 200, 12: 200, 13: 200, 14: 200, 15: 300, 16: 200, 17: 50
    });
    dictSigma = ee.Dictionary({
        1: 100, 2: 150, 3: 150, 4: 150, 5: 150, 6: 150, 7: 150, 8: 250, 9: 150, 10: 250,
        11: 75, 12: 75, 13: 75, 14: 75, 15: 150, 16: 75, 17: 25
    });

    def applyKernel(i):
        i_int = ee.Number.parse(i).toInt()
        radius = dictRadius.get(i)
        sigma = dictSigma.get(i)
        kernel = ee.Kernel.gaussian(radius, sigma, 'meters')

        try:
            lcz = image.eq(i_int).convolve(kernel).addBands(ee.Image(i_int).toInt().rename('lcz'))
        except Exception:
            print("Problem with gaussian filtering ...")
            pass

        return lcz

    ## Make mosaic from collection
    coll = ee.ImageCollection.fromImages(LCZclasses.map(applyKernel))

    ## Select highest value per pixel
    mos = coll.qualityMosaic('remapped')

    ## Select lcz bands again to obtain filtered LCZ map
    lczF = mos.select('lcz')

    return lczF.rename('lczFilter')

## LCZ mapping script - random boot, per year.
def lcz_mapping(info, CITY, TA_VERSION, YEAR):

    try:
        print("Get ROI")
        roi = _get_roi(info, TA_VERSION)

        print("Remap the TAs, from 1-17 to 0-16")
        ta = ee.FeatureCollection(os.path.join(
                fn_ee_dir,
                f"TA_{TA_VERSION}"))\
            .filter(ee.Filter.eq("City",CITY)) \
            .filter(ee.Filter.eq("Year", YEAR)) \
            .remap(ee.List.sequence(1,info['LCZ']['NRLCZ'],1),
                   ee.List.sequence(0,info['LCZ']['NRLCZ']-1,1), 'Class')\
            .filterBounds(roi)
        print(f"TA size for {CITY} ({YEAR}): {ta.size().getInfo()}")

        ## Reduce large polygons
        polyset = _buffer_polygons(info, ta)

        # Get the EO input assets to classify
        finalImage = _get_all_ls(info, CITY, TA_VERSION, YEAR).clip(roi)
        print(f"Check bands: {finalImage.bandNames().getInfo()}")

        ## function to do the actual classificaion
        def _do_classify(seed):

            rand = polyset.randomColumn('random', seed)
            ta = rand.filter(ee.Filter.lte('random', info['LCZ']['BOOTTRESH']))
            va = rand.filter(ee.Filter.gt('random', info['LCZ']['BOOTTRESH']))

            training = _sample_regions(finalImage, ta, ['Class'], info['LCZ']['SCALE'])
            validation = _sample_regions(finalImage, va, ['Class'], info['LCZ']['SCALE'])

            classifier = ee.Classifier.smileRandomForest(info['LCZ']['RFNRTREES'])\
                            .train(training, 'Class', finalImage.bandNames())
            validated = validation.classify(classifier);

            ## Confusion matrix
            cm = validated.errorMatrix('Class', 'classification').array()

            ## Force matrix to be NRLCZ x NRLCZ
            ## See email conversations Noel and link: http://bit.ly/2szEOqM
            height = cm.length().get([0])
            width = cm.length().get([1])

            fill1 = ee.Array([[0]]).repeat(0, ee.Number(info['LCZ']['NRLCZ']).subtract(height))\
                .repeat(1, width)
            fill2 = ee.Array([[0]]).repeat(0, info['LCZ']['NRLCZ'])\
                .repeat(1, ee.Number(info['LCZ']['NRLCZ']).subtract(width))

            a = ee.Array.cat([cm, fill1], 0)
            cmfinal = ee.Array.cat([a, fill2], 1)

            return {
                "confusionMatrix": cmfinal,
            }

        print("Start the classification")
        ## Apply the bootstrapping, get confusion matrices
        bootstrap = ee.List.sequence(1, info['LCZ']['NRBOOT']).map(_do_classify)
        matrices = bootstrap.map(lambda d: ee.Dictionary(d).get('confusionMatrix'))

        ## Make a final map, using all TAs, no bootstrap
        training_final = _sample_regions(finalImage, polyset, ['Class'], info['LCZ']['SCALE'])
        classifier_final = ee.Classifier.smileRandomForest(info['LCZ']['RFNRTREES']) \
                            .train(training_final, 'Class', finalImage.bandNames())
        lczMap = finalImage.classify(classifier_final)
        lczMap = lczMap.remap(ee.List.sequence(0, info['LCZ']['NRLCZ'] - 1, 1), \
                              ee.List.sequence(1, info['LCZ']['NRLCZ'], 1)) \
                       .int8()\
                       .rename('lcz')\
                       .clip(roi)

        ## Apply Gaussian filter
        lczMap_filter = _gaussian_filter(lczMap, roi)

        # Choose what to export
        lczMap_Final = lczMap.addBands(lczMap_filter).toInt()
        #lczMap_Final = lczMap_filter.toInt()

        # Slightly smaller export ROI (reduce 500m all sides)
        # to remove artifacts from boundaries of Filtered map.
        roi_export = roi.buffer(-500).bounds()

        print("Set output file name")
        ofile = f"{CITY}_" \
                f"{TA_VERSION}_" \
                f"{YEAR}_" \
                f"CC{info['CC']}_" \
                f"ED{info['EXTRA_DAYS']}_" \
                f"JDs{info['JD_START']}_{info['JD_END']}_" \
                f"L7{info['ADD_L7']}"
        print(ofile)

        print("Start all exports")
        ## Export confusion matrix to Google Cloud Storage.
        task_lcz_cm = ee.batch.Export.table.toDrive(\
            collection=ee.FeatureCollection(ee.Feature(None, {'matrix': matrices})),\
            description= f"CM_{ofile}",\
            folder= info['GD_FOLDER'],\
            fileFormat= 'CSV'
        );
        task_lcz_cm.start()

        # ## Export LCZ map as EE asset
        # task_lcz_ee = ee.batch.Export.image.toAsset(\
        #     image            = lczMap_Final.clip(roi_export),\
        #     description      = f"{CITY}_{YEAR}",\
        #     assetId          = f"{info['EE_OUT_DIR']}/{CITY}_{YEAR}",\
        #     scale            = info['LCZ']['SCALE'],\
        #     region           = roi_export,\
        #     maxPixels        = 1e13,\
        #     pyramidingPolicy = {".default":"mode"})
        # task_lcz_ee.start()

        ## Export LCZ map as to drive
        task_lcz_gd = ee.batch.Export.image.toDrive(\
            image            = lczMap_Final.clip(roi_export),\
            description      = f"LCZ_{ofile}",\
            folder           = info['GD_FOLDER'],\
            scale            = info['LCZ']['SCALE'],\
            region           = roi_export,\
            maxPixels        = 1e13)
        task_lcz_gd.start()

    except Exception:
        err = traceback.format_exc()
        print(err)

###############################################################################
# _Execute code
###############################################################################

info = _read_config(CITY)

for YEAR in list(info['TA'][TA_VERSION].keys()):

    print(f"Create LCZ map for {YEAR}, start the clock ------------")
    start = datetime.now()

    lcz_mapping(
        info=info,
        CITY=CITY,
        TA_VERSION=TA_VERSION,
        YEAR=YEAR,
    )

    print(f' LCZ for {YEAR} took', datetime.now()-start, 'seconds  ------------')
###############################################################################