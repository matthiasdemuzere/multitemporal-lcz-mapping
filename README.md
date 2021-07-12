# Dynamic LCZ maps: generic approach

## Done for:
- Hyderabad: 2003, 2015, 2017, 2019 (Siva Ram Edupuganti)
- Kathmandu: 1990 (Lilu Thapa) ==> NOT DONE, VERY POOR TA SET.



## Procedure

**0. Set python environment / Prepare CITY config & Data folders **

```bash
cd /home/demuzmp4/Nextcloud/scripts/wudapt/dynamic-lcz
con && conda activate dynamic-lcz
```

Create `.yaml` file by making copy from existing one (`./config/`):
* Fill in TA version, year and file names
* Set Author name(s)
* Check if other settings need to be changed, e.g.:
  * CC: Cloud Cover (DEFAULT: 70% Cloud Cover allowed)
  * EXTRA_DAYS: Extra days outside the year of interest? (DEFAULT: half a year before and after)
  * ADD_L7: include Landsat 7 sensor or not (bool as string) (DEFAULT: "True")
  * JD _START/_END: Should julian days be excluded, e.g. because of snow cover? (DEFAULT: all days included)

Create a CITY folder under `/home/demuzmp4/Nextcloud/data/wudapt/dynamic-lcz`:
* with `input` and `output`
* store .kml files under proper TA_VERSION: `input/TA_VERSION`

**1. Create TA set**

```bash
python create_ta_shp.py CITY EE_ACCOUNT TA_VERSION

# E.g.:
python create_ta_shp.py Hyderabad mdemuzere v1
```

**2. Upload TA set**

* Manually upload to EE, in respective CITY folder:
```bash
> projects/WUDAPT/LCZ_L0/dynamic-lcz/CITY/
```

**3. LCZ mapping**

Create the LCZ map, including:
- processing of EO input features on the fly (store to asset option possible via `EXPORT_TO_ASSET` in `.yaml` namelist.)
- store Landsat ID list as csv file
- 25 Bootstrap to get the confusion matrices for the accuracy assessment.
- Final LCZ map using all polygons (band `lcz`)
- Gaussian filtered version added (band `lczFilter`)

```bash
python create_lcz_map.py CITY EE_ACCOUNT TA_VERSION

# E.g.:
python create_lcz_map.py Hyderabad mdemuzere v1
```

## LCZ Process

In General, follow the procedure as outlined in Demuzere et al. ([2020](http://doi.org/10.31219/osf.io/h5tm6), [2021](https://www.frontiersin.org/articles/10.3389/fenvs.2021.637455/)). Yet keep in mind the following changes:

* only Landsat-based input features: from Landsat 5, 7 (years prior to 2003 because of the scan line error) and 8
* Select all images with cloud cover < 70%, and for year +/- 90 days (e.g 01-10-2002 to 01-04-2004 for the year 2003)
* Check whether periods should be excluded, example because of potential snow cover?
* all images are masked for clouds.
* selected bands: 
    * blue, green, red, near infrared (nir), shortwave infrared (swir) 1/2 and thermal infrared (tirs), 
    * band rations bci, ndbai, ndbi, ebbi, ndvi, ndwi 
    * from all bands, derive p10, p50, p90 and standard deviation
    * add slope from ASTER DEM to account for topography
* LCZs are mapped at a resolution of 30m. Afterwards, the default Gaussian filter is applied
* resulting resolution is 30m, yet thematic resolution is lower due to the Gaussian filter.
* a buffer of 15km is applied around the bounding box of all TAs for a city. This defines the mapped Region of Interest.
* all other LCZ mapping settings from the LCZ Generator are kept.



