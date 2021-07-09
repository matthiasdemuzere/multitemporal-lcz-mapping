import yaml
from typing import Dict
from typing import Any
import traceback
import os
import ee
ee.Initialize()
from datetime import datetime


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
  return img.updateMask(mask).divide(10000)\
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

def _get_all_lst_list(info, city, year):

    # Define the collections
    _ls5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
    _ls7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')
    _ls8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

    if year == 2000:
        if city == 'Beijing':
            l5 = _ls5.filter(ee.Filter.inList("system:index", info[city][year]['L5']))
            l7 = _ls7.filter(ee.Filter.inList("system:index", info[city][year]['L7']))
            ls_all = l5.merge(l7).map(_l5_7rename)
        else:
            l7 = _ls7.filter(ee.Filter.inList("system:index", info[city][year]['L7']))
            ls_all = l7.map(_l5_7rename)
    elif year in [2005, 2010]:
        l5 = _ls5.filter(ee.Filter.inList("system:index", info[city][year]['L5']))
        ls_all = l5.map(_l5_7rename)
    else:
        l8 = _ls8.filter(ee.Filter.inList("system:index", info[city][year]['L8']))
        ls_all = l8.map(_l8rename)

    # print("Mask the clouds and select bands")
    # ls_all = ls_all.map(_mask_clouds) \
    #     .select(info['lcz']['BANDS_FLY'])
    print("Clouds are NOT masked, bands are selected")
    ls_all = ls_all.select(info['lcz']['BANDS_FLY'])
    print(ls_all.size().getInfo())

    print("Convert to image for LCZ mapping")
    # .toBands uses img ID in name, making it useless for RF
    def coll_to_img(img,previous):
        return ee.Image(previous).addBands(img)

    first = ee.Image(ls_all.first()).select([])
    ls_all_img = ee.Image(ls_all.iterate(coll_to_img, first))

    # Work with mosaic. FAILS
    #ls_all_img = ee.Image(ls_all.mosaic())

    return ls_all_img


def _get_all_ls(city, year, CLOUD_COVER, EXTRA_DAYS, ADD_L7):

    print("Gathering the appropriate Landsat images ...")

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

    print("Extracting the Landsat IDs for future reference ...")
    # THIS PROCEDURE IS TOO SLOW.
    # id_list = pd.DataFrame(_ls_all.aggregate_array('system:index').getInfo())
    # print(f"{len(id_list)} images selected: {id_list}")
    #
    # id_list.to_csv(os.path.join(
    #     info['OUT_DIR'],
    #     f"Landsat_IDS_{city}_{year}.csv"
    # ))
    def _get_id(element):
        return ee.Feature(None, {'id': element})

    ids = _ls_all.aggregate_array('system:index')
    ftColl_ids = ee.FeatureCollection(ids.map(_get_id))

    print("Export selected LS IDs to drive")
    ids_ofile = f"{city}_{year}_on_the_fly_{CLOUD_COVER}_{EXTRA_DAYS}_{ADD_L7}"
    task_export_ids = ee.batch.Export.table.toDrive( \
        collection=ftColl_ids, \
        description=f"{ids_ofile}_LS_IDs", \
        folder=info['GF_FOLDER'], \
        fileFormat='CSV'
    );
    task_export_ids.start()

    print("Mask the clouds and select bands")
    ls_all = _ls_all.map(_mask_clouds) \
        .select(info['lcz']['BANDS_FLY'])

    #return ls_all

    print("Convert to image for LCZ mapping")
    # .toBands uses img ID in name, making it useless for RF
    def coll_to_img(img,previous):
        return ee.Image(previous).addBands(img)

    first = ee.Image(ls_all.first()).select([])
    ls_all_img = ee.Image(ls_all.iterate(coll_to_img, first))
    #ls_all_img = ls_all.iterate(coll_to_img, ee.Image([]))

    # Print the band nalmes
    print(f" Selected bands: {ls_all_img.bandNames().getInfo()}")

    return ls_all_img.clip(roi)


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
def lcz_mapping_rboot(city, year, info,
                      SRC_INPUT='on_the_fly',
                      CLOUD_COVER=10,
                      EXTRA_DAYS=180,
                      ADD_L7=True):

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
        print(f"TA size for {city} ({year}): {ta.size().getInfo()}")

        ## Reduce large polygons
        polyarea   = ta.map(_addArea);
        bigPoly    = polyarea.filterMetadata('myArea', 'not_less_than', info['lcz']['POLYSIZE']);
        smaPoly    = polyarea.filterMetadata('myArea', 'less_than', info['lcz']['POLYSIZE']);
        bigPolyRed = bigPoly.map(_getCentroidBuffer);
        polyset    = smaPoly.merge(bigPolyRed);

        if SRC_INPUT == 'on_the_fly':
            print("Produce input features on the fly")
            finalImage = _get_all_ls(city, year, CLOUD_COVER, EXTRA_DAYS, ADD_L7).clip(roi)

        elif SRC_INPUT == 'id_list':
            finalImage = _get_all_lst_list(info, city, year).clip(roi)

        else:
            print("Read the available input features")
            ## Add slope and latitude
            dtm = ee.Image('NASA/ASTER_GED/AG100_003').select('elevation')
            slope = ee.Terrain.slope(dtm).clip(roi)

            inputImage = ee.Image(f"{info['EE_IN_DIR']}/{city}_{year}") \
                .addBands(slope)

            finalImage = inputImage.select(info['lcz']['BANDS']).clip(roi)

        print(f"Check bands: {finalImage.bandNames().getInfo()}")

        ## function to do the actual classificaion
        def _do_classify(seed):

            rand = polyset.randomColumn('random', seed)
            ta = rand.filter(ee.Filter.lte('random', info['lcz']['BOOTTRESH']))
            va = rand.filter(ee.Filter.gt('random', info['lcz']['BOOTTRESH']))

            training = _sample_regions(finalImage, ta, ['Class'], info['lcz']['SCALE'])
            validation = _sample_regions(finalImage, va, ['Class'], info['lcz']['SCALE'])

            classifier = ee.Classifier.smileRandomForest(info['lcz']['RFNRTREES'])\
                            .train(training, 'Class', info['lcz']['BANDS_FLY'])
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

        ## Make a final map, using all TAs, no bootstrap
        training_final = _sample_regions(finalImage, polyset, ['Class'], info['lcz']['SCALE'])
        classifier_final = ee.Classifier.smileRandomForest(info['lcz']['RFNRTREES']) \
                            .train(training_final, 'Class', info['lcz']['BANDS_FLY'])
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

        print("Set output file name")
        if SRC_INPUT == 'id_list':
            ofile = f"{city}_{year}_{SRC_INPUT}"
        else:
            ofile = f"{city}_{year}_{SRC_INPUT}_{CLOUD_COVER}_{EXTRA_DAYS}_{ADD_L7}"
        print(ofile)

        print("Start all exports")
        ## Export confusion matrix to Google Cloud Storage.
        task_lcz_cm = ee.batch.Export.table.toDrive(\
            collection=ee.FeatureCollection(ee.Feature(None, {'matrix': matrices})),\
            description= f"{ofile}_cm",\
            folder= info['GF_FOLDER'],\
            fileFormat= 'CSV'
        );
        task_lcz_cm.start()

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

        ## Export LCZ map as to drive - 30m
        task_lcz_gd_30m = ee.batch.Export.image.toDrive(\
            image            = lczMap.toInt().clip(roi_export),\
            description      = f"{ofile}_30m",\
            folder           = info['GF_FOLDER'],\
            scale            = 100,\
            region           = roi_export,\
            maxPixels        = 1e13)
        task_lcz_gd_30m.start()

        ## Export LCZ map as to drive
        task_lcz_gd = ee.batch.Export.image.toDrive(\
            image            = lczMap_Final.clip(roi_export),\
            description      = f"{ofile}",\
            folder           = info['GF_FOLDER'],\
            scale            = info['lcz']['SCALE'],\
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


#TODO: check if clous or masked or not. Currently NOT for 'id_list' setting.

#SRC_INPUT='on_the_fly'
SRC_INPUT='id_list'
CLOUD_COVER=10
EXTRA_DAYS=120
ADD_L7=True

#for city in ['Guangzhou', 'Shanghai']:
for city in ['Beijing']:
     for year in [2000]:
#for city in cities:
#   for year in [2000, 2005, 2010, 2015, 2020]:

        #print(f"Check IDs for {city} - {year}")
        #_get_all_ls(city, year, CLOUD_COVER, EXTRA_DAYS, ADD_L7)

        print(f"Create LCZ map for {city} - {year}, start the clock ------------")
        start = datetime.now()

        lcz_mapping_rboot(
            city=city,
            year=year,
            info=info,
            SRC_INPUT=SRC_INPUT,
            CLOUD_COVER=CLOUD_COVER,
            EXTRA_DAYS=EXTRA_DAYS,
            ADD_L7=ADD_L7)

        print(f'{city} | {year} took', datetime.now()-start, 'seconds  ------------')
###############################################################################