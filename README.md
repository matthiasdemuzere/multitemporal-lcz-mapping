# Dynamic LCZ maps: generic approach

## Done for:
- Hyderabad: 2003, 2015, 2017, 2019 (Siva Ram Edupuganti)
- Melbourne (Australia): 2006, 2016, 2019 (James Bennie)
- Kathmandu: 1990 (Lilu Thapa) ==> **NOT DONE, VERY POOR TA SET**.
- Pune: 2006, 2021 (Labani Saha) ==> **In progress, waiting for TA sets between 1994 - 2016**.
- Lagos: 2000 (Benjamin Obe) ==> Seems to have been used in [Obe et al. (2023 - preprint)](https://www.researchsquare.com/article/rs-2869856/v1)
 

## Procedure
The procedure is based upon previous experiences:
- Canadian cities: [Demuzere et al. (2020)](https://osf.io/h5tm6/)
- Beijing, Shanghai, Guangzhou: 2000-2020, every 5 years. Used in [Cai et al., 2023](https://doi.org/10.1016/j.cities.2022.103988).

**0. Set python environment**

```bash
python3.11 -m venv ~/python-environments/dynamic-lcz
source ~/python-environments/dynamic-lcz/bin/activate

# Install packages first time
pip install -r /home/demuzmp4/Nextcloud/scripts/wudapt/dynamic-lcz/requirements.txt
```

**1. Prepare CITY config & Data folders**

Create `./config/CITY.yaml` file by making copy from existing one:
* Fill in TA version (e.g. v1), year (e.g. 2003) and file names (...)
* Set Author name(s)
* Check if other settings need to be changed, e.g.:
  * CC: Cloud Cover (DEFAULT: `70%` Cloud Cover allowed)
  * EXTRA_DAYS: Extra days outside the year of interest? (DEFAULT: ` 180 days` = half a year before and after)
  * ADD_L7: include Landsat 7 sensor or not (bool as string) (DEFAULT: `"True"`)
  * JD _START/_END: Should julian days be excluded, e.g. because of snow cover? (DEFAULT: `0` and `366` = all days included)

Create a CITY folder under `/home/demuzmp4/Nextcloud/data/wudapt/dynamic-lcz`:
* with `input/` and `output/`
* store .kml files under proper filename, e.g.: `input/TA_XX`, with XX referring to the version (same as in CITY.yaml)


**2. Create TA set**

* This routine combines all years into one .shp and afterwards .zip.
```bash
python create_ta_shp.py CITY EE_ACCOUNT TA_VERSION

# E.g.:
python create_ta_shp.py Hyderabad mdemuzere v1
```

**3. Upload TA set**

* Manually upload to EE, in respective CITY folder:
```bash
> projects/WUDAPT/LCZ_L0/dynamic-lcz/CITY/
```

**4. LCZ mapping**

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

**5. Plotting**

Once EE has finished the processing, and the data is available in Google Drive, then copy all files to its default output location, e.g. `/home/demuzmp4/Nextcloud/data/wudapt/dynamic-lcz/CITY/output`

For a quick assessment of the results, create the following plots:
- TA frequencies per year, in a stacked barchart
- Accuracy boxplots, one per year, in multipanel: (1 x len(years))
- lczFilter map, one per year, in multipanel: (1 x len(years))

```bash
python create_plots.py CITY TA_VERSION

# E.g.:
python create_plots.py Hyderabad v1
```

**6. LCZ map as kmz**

Also store the LCZ GeoTIFF map as `.kmz`, that can be opened in Google Earth

```bash
python create_lcz_kmz.py CITY TA_VERSION

# E.g.:
python create_lcz_kmz.py Hyderabad v1
```


# DESCRIPTION
<hr>

## General 
* Keep in mind that this is still **an experimental procedure**. It has first been applied over Canadian cities ([Demuzere et al., 2020](https://osf.io/h5tm6)), and has been further developed, fine-tuned and tested over Chinese mega-cities ([Zhi, C., Demuzere, M., Tang, Y., Wan, Y, Exploring the characteristics and transformation of 3D urban morphology in three Chinese mega-cities. Cities](https://doi.org/10.1016/j.cities.2022.103988)). 
* In contrast to the other LCZ mapping efforts from Demuzere et al. 2019a,b, 2020a,b, 2021, 2022, this work only uses input from the Landsat sensor. As such, the quality of such a landsat-only map might be inferior to one produced by the [LCZ-Generator](https://lcz-generator.rub.de/), which also has access to more recent Sentinel-1, 2 and other earth observation products. 
* No temporal filtering has been applied on the resulting LCZ maps. That means that LCZ labels might jump from one LCZ class to the other in subsequent years, which might be artificial and due to the nature of the random forest classifier. However, temporal filtering has been applied in Qi et al., Mapping urban form into local climate zones for the continental U.S. from 1986 to 2020 (under review).
* Note that the creation of these dynamic LCZ maps requires quite a bit of time investment from my side. So please keep this in mind when thinking about acknowledging this contribution. 

## Requirements
* Training area polygons representative for each year of interest. Ideally the TA sets across years share the same polygons (for areas that did not change over time), to increase consistency across the dynamic LCZ maps
* Make sure to make the training area sets according to this template (https://www.wudapt.org/wp-content/uploads/2020/08/WUDAPT_L0_Training_template.kml) and these guidelines (https://www.wudapt.org/digitize-training-areas/)!! If not, I will be unable to process them.


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

All outputs are available in `output/`.

* The outputs are similar as those provided by the LCZ Generator (see [here](https://lcz-generator.rub.de/submissions)). 
* The FILENAME reflects the settings used in the LCZ map production:
  * XX_: CM or IDs or LCZ (type of output, see below)
  * CITYNAME_
  * v1_: TA version (in case multiple iterations will be done)
  * YEAR_
  * CC_: Cloud Cover threshold
  * ED_: Extra Days before and after relevant year
  * JDs_: Julian days considered (0_366 = all year)
  * L7True: Landsat 7 info is used prior to 2003.
  
* Data output:
  * IDs_FILENAME_LS_IDs.csv: The IDs of the Landsat images that were used to create the Landsat mosaics as input to the random forest classifier.
  * CM_FILENAME.csv: Raw confusion matrix, can be ignored
  * CM_FILENAME_oa_df.csv: Overall accuracies and F1 metrics per bootstrap. This is the data used to make the plot_OA_BOXPLOT.jpg
  * LCZ_FILENAME.tif: Actual LCZ map, with two bands: "lcz" is raw map, "lczFilter" is Gaussian filtered map.
  * LCZ_FILENAME.kmz: LCZ map ("lczFilter"), in kmz format, which can be opened in Google Earth.
  
* Figures:
  * plot_TA_FREQ.jpg: stacked bar plot showing the number of TAs per class and year
  * plot_OA_BOXPLOT.jpg: accuracy boxplots per year
  * plot_LCZ_MAP.jpg: LCZ map (using lczFilter band) per year


## References

* Demuzere M, Bechtel B, Mills G. Global transferability of local climate zone models. Urban Clim. 2019a;27:46-63. doi:10.1016/j.uclim.2018.11.001
* Demuzere M, Bechtel B, Middel A, Mills G. Mapping Europe into local climate zones. Mourshed M, ed. PLoS One. 2019b;14(4):e0214474. doi:10.1371/journal.pone.0214474
* Demuzere M, Hankey S, Mills G, Zhang W, Lu T, Bechtel B. Combining expert and crowd-sourced training data to map urban form and functions for the continental US. Sci Data. 2020a;7(1):264. doi:10.1038/s41597-020-00605-z
* Demuzere M, Mihara T, Redivo CP, Feddema J, Setton E. Multi-temporal LCZ maps for Canadian functional urban areas. OSF Prepr. December 2020b. doi:10.31219/osf.io/h5tm6
* Demuzere M, Kittner J, Bechtel B. LCZ Generator: A Web Application to Create Local Climate Zone Maps. Front Environ Sci. 2021;9. doi:10.3389/fenvs.2021.637455
* Demuzere M, Kittner J, Martilli A, et al. A global map of Local Climate Zones to support earth system modelling and urban scale environmental science. Earth Syst Sci Data Discuss. 2022.
* Cai Z, Demuzere M, Tang Y, Wan Y. The characteristic and transformation of 3D urban morphology in three Chinese mega-cities. Cities. 2022;(July 2021):103988. doi:10.1016/j.cities.2022.103988

