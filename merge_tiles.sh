#!bin/bash

tile1=$1
tile2=$2
tile3=$3
out_name=$4

echo "Merging tiles  ---------------"
gdal_merge.py -a_nodata 0 -o $out_name $tile1 $tile2 $tile3

echo "Compressing file  ---------------"
gdal_translate -co "COMPRESS=LZW" $out_name "${out_name}.tif"

echo "Deleting uncompressed file -----------"
rm $out_name
