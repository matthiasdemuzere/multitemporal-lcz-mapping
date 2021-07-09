import yaml
from typing import Dict
from typing import Any
import traceback
import os
import ee
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
  ebbi = (img.select('swir1').subtract(img.select('nir'))) \
              .divide(ee.Image(10).multiply((img.select('swir1').add(img.select('tirs1'))).sqrt())) \
              .rename('ebbi').toFloat()
  ndvi = img.normalizedDifference(['nir','red']).rename('ndvi')
  ndwi = img.normalizedDifference(['green','nir']).rename('ndwi')

  return img\
            .addBands(ndbai) \
            .addBands(ndbi) \
            .addBands(ebbi) \
            .addBands(ndvi) \
            .addBands(ndwi) \
            .toFloat()

def _get_all_ls(city, year, info):

    print("Gathering the appropriate Landsat images ...")

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

    #print(f"Final bandnames: {finalImage.bandNames().getInfo()}")

    return finalImage


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
def lcz_mapping_rboot(city, year, info):

    ## Helper functions
    def _addArea(feature):
        area = feature.area()
        return feature.set({'myArea': area})

    def _getCentroidBuffer(feature):
        return feature.centroid().buffer(info['lcz']['BUFFERSIZE'])

    try:
        print("Get ROI")
        roi = _get_roi(info, city)

        print("Remap the TAs, from 1-17 to 0-16")
        ta = ee.FeatureCollection(os.path.join(
                info['EE_TA_DIR'],
                "cai_zhi_all"))\
            .filter(ee.Filter.eq("City",city)) \
            .filter(ee.Filter.eq("Year", year)) \
            .remap(ee.List.sequence(1,info['lcz']['NRLCZ'],1),
                   ee.List.sequence(0,info['lcz']['NRLCZ']-1,1), 'Class')\
            .filterBounds(roi)

        print("Reduce sample set for Beijing, otherwise computational error")
        print("For 2000/5, 70% is used, for 2010-20, 50%")
        print("Finalmap (RFNRTREES=30) and Accuracies (RFNRTREES=15) are done in two steps.")
        if city == 'Beijing' and year in [2010,2015,2020]:
            rand = ta.randomColumn('random', 0)
            ta = rand.filter(ee.Filter.lte('random', 0.5))
        elif city == 'Beijing' and year in [2000, 2005]:
            rand = ta.randomColumn('random', 0)
            ta = rand.filter(ee.Filter.lte('random', 0.7))

        print(f"TA size for {city} ({year}): {ta.size().getInfo()}")

        ## Reduce large polygons
        polyarea   = ta.map(_addArea);
        bigPoly    = polyarea.filterMetadata('myArea', 'not_less_than', info['lcz']['POLYSIZE']);
        smaPoly    = polyarea.filterMetadata('myArea', 'less_than', info['lcz']['POLYSIZE']);
        bigPolyRed = bigPoly.map(_getCentroidBuffer);
        polyset    = smaPoly.merge(bigPolyRed);

        # Get the finalImage, use pre-processed one for Beijing.
        if city == 'Beijing':
            finalImage = ee.Image(f"{info['EE_IN_DIR']}/{city}_{year}").clip(roi)
        else:
            finalImage = _get_all_ls(city, year, info).clip(roi)

        print(f"Available bands: {finalImage.bandNames().getInfo()}")

        ## function to do the actual classificaion
        def _do_classify(seed):

            rand = polyset.randomColumn('random', seed)
            ta = rand.filter(ee.Filter.lte('random', info['lcz']['BOOTTRESH']))
            va = rand.filter(ee.Filter.gt('random', info['lcz']['BOOTTRESH']))

            training = _sample_regions(finalImage, ta, ['Class'], info['lcz']['SCALE'])
            validation = _sample_regions(finalImage, va, ['Class'], info['lcz']['SCALE'])

            classifier = ee.Classifier.smileRandomForest(info['lcz']['RFNRTREES'])\
                            .train(training, 'Class', finalImage.bandNames())
            validated = validation.classify(classifier);

            ## Confusion matrix
            cm = validated.errorMatrix('Class', 'classification').array()

            ## Force matrix to be NRLCZ x NRLCZ
            ## See email conversations Noel and link: http://bit.ly/2szEOqM
            height = cm.length().get([0])
            width = cm.length().get([1])

            fill1 = ee.Array([[0]]).repeat(0, ee.Number(info['lcz']['NRLCZ']).subtract(height))\
                .repeat(1, width)
            fill2 = ee.Array([[0]]).repeat(0, info['lcz']['NRLCZ'])\
                .repeat(1, ee.Number(info['lcz']['NRLCZ']).subtract(width))

            a = ee.Array.cat([cm, fill1], 0)
            cmfinal = ee.Array.cat([a, fill2], 1)

            return {
                "confusionMatrix": cmfinal,
            }

        print("Start the classification")

        ## Apply the bootstrapping, get confusion matrices
        bootstrap = ee.List.sequence(1, info['lcz']['NRBOOT']).map(_do_classify)
        matrices = bootstrap.map(lambda d: ee.Dictionary(d).get('confusionMatrix'))

        ofile = f"{city}_{year}_SCALE{info['lcz']['SCALE']}_CC{info['CC']}_DAYS{info['EXTRA_DAYS']}"
        print(f"Set output file name: {ofile}")

        ## Export confusion matrix to Google Cloud Storage.
        task_lcz_cm = ee.batch.Export.table.toDrive(\
            collection=ee.FeatureCollection(ee.Feature(None, {'matrix': matrices})),\
            description= f"{ofile}_cm",\
            folder= info['GF_FOLDER'],\
            fileFormat= 'CSV'
        );
        task_lcz_cm.start()

        ## Make a final map, using all TAs, no bootstrap
        training_final = _sample_regions(finalImage, polyset, ['Class'], info['lcz']['SCALE'])
        classifier_final = ee.Classifier.smileRandomForest(info['lcz']['RFNRTREES']) \
                            .train(training_final, 'Class', finalImage.bandNames())
        lczMap = finalImage.classify(classifier_final)
        lczMap = lczMap.remap(ee.List.sequence(0, info['lcz']['NRLCZ'] - 1, 1), \
                              ee.List.sequence(1, info['lcz']['NRLCZ'], 1)) \
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

        # ## Export LCZ map as EE asset
        # task_lcz_ee = ee.batch.Export.image.toAsset(\
        #     image            = lczMap_Final.clip(roi_export),\
        #     description      = f"{city}_{year}",\
        #     assetId          = f"{info['EE_OUT_DIR']}/{city}_{year}",\
        #     scale            = info['lcz']['SCALE'],\
        #     region           = roi_export,\
        #     maxPixels        = 1e13,\
        #     pyramidingPolicy = {".default":"mode"})
        # task_lcz_ee.start()

        ## Export LCZ map as to drive
        task_lcz_gd = ee.batch.Export.image.toDrive(\
            image            = lczMap_Final.clip(roi_export),\
            description      = f"{ofile}",\
            folder           = info['GF_FOLDER'],\
            scale            = 30,\
            region           = roi_export,\
            maxPixels        = 1e13)
        task_lcz_gd.start()

    except Exception:
        err = traceback.format_exc()
        print(err)

###############################################################################
# _Execute code
###############################################################################

print('Set info')
info = _read_config()

print('Set proper EE account, kode as default')
os.system(f"bash {info['EE_SET_ACCOUNT']} kode")

cities = ['Beijing', 'Guangzhou', 'Shanghai']
years = [2000, 2005, 2010, 2015, 2020]
years = [2000, 2005]
#cities = ['Guangzhou']
#years = [2020]

for city in ['Beijing']:
#for city in cities:
    #for year in [2005]:
    for year in years:

        lcz_mapping_rboot(city, year, info)

###############################################################################
