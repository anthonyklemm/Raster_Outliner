# Raster Outliner
This tool will create a generalized outline polygon of one or more rasters. The input rasters can be BAG or GeoTiff (or a combination of the two). It also supports VR BAG. 
The output filetypes are a shapefile and a geojson.
The shapefile will be in NAD83, and will be populated with the following attributes: 
1. Area_SQNM: Area of polygon in square nautical miles
2. Platform: the name of the vessel that collected the data
3. State
4. Scale
5. CATCOV: Usually populated as "1" to denote coverage in that area
6. Survey_ID: usually the survey number (like HXXXXX or W00000)

This tool uses a combination of resampling of the raster to coarser resolution, morphological dilation and erosion on a binary mask, and polygon simplification to provide a polygon survey outline that
is intended to be a compromise between processing speed, outline accuracy compared to the raster, and a minimization of polygon vertices. 
***Note: This is the first iteration of this tool, and is in the test/evaluation phase.***

***Contact Anthony Klemm anthony.r.klemm@noaa.gov for any questions regarding this tool*** 
![image (3)](https://github.com/anthonyklemm/Raster_Outliner/assets/76973843/fefc4fc1-de30-42c3-b85b-fc6f157149e4)
