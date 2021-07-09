# Generic approach to make dynamic LCZ maps

## Done for:
- Hyderabad: 2003, 2015, 2017, 2019 (Siva Ram Edupuganti)


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



