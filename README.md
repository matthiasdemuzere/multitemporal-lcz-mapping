# Dynamic LCZ maps: generic approach

## Done for:
- Hyderabad: 2003, 2015, 2017, 2019 (Siva Ram Edupuganti)
- Melbourne (Australia): 2006, 2016, 2019 (James Bennie)
- Kathmandu: 1990 (Lilu Thapa) ==> **NOT DONE, VERY POOR TA SET**.


## Procedure

**0. Set python environment / Prepare CITY config & Data folders**

```bash
python3.8 -m venv ~/python-environments/dynamic-lcz
source ~/python-environments/dynamic-lcz/bin/activate

# Install packages first time
source ~/python-environments/dynamic-lcz/bin/activate
pip install -r /home/demuzmp4/Nextcloud/scripts/wudapt/dynamic-lcz
```

Create `./config/CITY.yaml` file by making copy from existing one:
* Fill in TA version, year and file names
* Set Author name(s)
* Check if other settings need to be changed, e.g.:
  * CC: Cloud Cover (DEFAULT: `70%` Cloud Cover allowed)
  * EXTRA_DAYS: Extra days outside the year of interest? (DEFAULT: ` 180 days` = half a year before and after)
  * ADD_L7: include Landsat 7 sensor or not (bool as string) (DEFAULT: `"True"`)
  * JD _START/_END: Should julian days be excluded, e.g. because of snow cover? (DEFAULT: `0` and `366` = all days included)

Create a CITY folder under `/home/demuzmp4/Nextcloud/data/wudapt/dynamic-lcz`:
* with `input` and `output`
* store .kml files under proper TA_VERSION: `input/TA_VERSION`

**1. Create TA set**

* This routine combines all years into one .shp and afterwards .zip.
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

**4. Plotting**

For a quick assessment of the results, create the following plots:
- TA frequencies per year, in a stacked barchart
- Accuracy boxplots, one per year, in multipanel: (1 x len(years))
- lczFilter map, one per year, in multipanel: (1 x len(years))

```bash
python create_plots.py CITY TA_VERSION

# E.g.:
python create_plots.py Hyderabad v1
```


# DESCRIPTION
<hr>

## General 
* Keep in mind that this is still **an experimental procedure**. It has first been applied over Canadian cities ([Demuzere et al., 2020](https://osf.io/h5tm6)), and has been developed and tested over Chinese mega-cities (Zhi, C., Demuzere, M., Tang, Y., Wan, Y, Exploring the characteristics and transformation of 3D urban morphology in three Chinese mega-cities. Cities, Under Review).
* Because of this, it also requires quite a bit of time investment from my side. So please keep this in mind when thinking about acknowledging this contribution. 

## LCZ mapping 
The dynamic LCZ mapping procedure in general follows the procedure of the LCZ Generator ([Demuzere et al. 2021](https://www.frontiersin.org/articles/10.3389/fenvs.2021.637455/)). Yet keep in mind the following changes:

* only Landsat-based input features: from Landsat 5, 7 (years prior to 2003 because of the scan line error) and 8
* this limited set of input features might result in classification errors, which might be most noticeable around edges of features, e.g. along coast lines / edges of water bodies
* Using all images with cloud cover < 70%, and for year +/- 180 days (e.g 01-07-2002 to 01-07-2004 for the year 2003)
* all images are masked for clouds.
* selected bands: 
    * blue, green, red, near infrared (nir), shortwave infrared (swir) 1/2 and thermal infrared (tirs), 
    * band rations bci, ndbai, ndbi, ebbi, ndvi, ndwi 
    * for all bands and all images, derive 10th, 50th and 90th percentile and standard deviation
    * add slope from ASTER DEM to account for topography
* LCZs are mapped at a resolution of 30m. Afterwards, the default Gaussian filter is applied, as described in [Demuzere et al. (2020)](https://doi.org/10.1038/s41597-020-00605-z).
* resulting resolution is 100m, yet thematic resolution is slightly lower due to the Gaussian filter.
* a buffer of 15km is applied around the bounding box of all TAs (all years). This defines the mapped Region of Interest, being the same for all years.
* all other TA filters and LCZ mapping settings from the LCZ Generator are retained.


## Output

All outputs are available HERE > output. (MAKE SURE TO ADD LINK!!)

* The outputs are similar as those provided by the LCZ Generator (see [here](https://lcz-generator.rub.de/submissions)). 
* The FILENAME reflect the settings used in the production:
  * XX_: CM or IDs or LCZ (type of output, see below)
  * CITYNAME_
  * v1_: TA version (in case multiple iterations will be done)
  * YEAR_
  * CC_: Cloud Cover
  * ED_: Extra Days before and after relevant year
  * JDs_: Julian days considered (0_366 = all year)
  * L7True: Landsat 7 info is used prior to 2003.
  
* Data output:
  * IDs_FILENAME_LS_IDs.csv: The IDs of the Landsat images that were used to create the Landsat mosaics as input to the random forest classifier.
  * CM_FILENAME.csv: Raw confusion matrix, can be ignored
  * CM_FILENAME_oa_df.csv: Overall accuracies and F1 metrics per bootstrap. This is the data used to make the plot_OA_BOXPLOT.jpg
  * LCZ_FILENAME.tif: Actual LCZ map, with two bands: "lcz" is raw map, "lczFilter" is Gaussian filtered map.
  
* Figures:
  * plot_TA_FREQ.jpg: stacked bar plot showing the number of TAs per class and year
  * plot_OA_BOXPLOT.jpg: accuracy boxplots per year
  * plot_LCZ_MAP.jpg: LCZ map (using lczFilter band) per year


