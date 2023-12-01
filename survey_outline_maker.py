# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:56:53 2023

@author: Anthony.R.Klemm
"""
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.features import shapes
from rasterio.enums import Resampling
from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
import geopandas as gpd
import numpy as np
from skimage.morphology import binary_dilation, binary_erosion
import tkinter as tk
from tkinter import filedialog, Label, Entry, Button
import time
import argparse



def resample_raster(raster, desired_resolution):
    # Extract the original resolution (average of x and y resolutions)
    original_resolution = (raster.res[0] + raster.res[1]) / 2

    # Check if the original resolution is finer than the desired resolution
    if original_resolution < desired_resolution:
        # Calculate the new dimensions
        new_height = int(raster.height * raster.res[0] / desired_resolution)
        new_width = int(raster.width * raster.res[1] / desired_resolution)

        # Resample the raster - only the first band
        data = raster.read(
            1,  # Reading only the first band
            out_shape=(new_height, new_width),
            resampling=Resampling.bilinear
        )

        # Calculate the new transform
        new_transform = raster.transform * raster.transform.scale(
            (raster.width / data.shape[-1]),
            (raster.height / data.shape[-2])
        )

        # Update raster profile
        profile = raster.profile
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            'count': 1  # Working with only one band
        })

        # Create an in-memory raster file
        memfile = MemoryFile()
        dataset = memfile.open(**profile)
        dataset.write(data, 1)
        return dataset, memfile
    else:
        # No resampling needed, return the original raster
        return raster, None

def calculate_iterations(resolution, max_resolution=15, min_iterations=3, max_iterations=20):
    """
    Calculate the number of dilation and erosion iterations based on raster resolution.
    
    Args:
    resolution (float): The resolution of the input raster.
    max_resolution (float): The resolution threshold for maximum iterations.
    min_iterations (int): Minimum number of iterations for large resolutions.
    max_iterations (int): Maximum number of iterations for resolutions finer than max_resolution.
    
    Returns:
    int: Number of dilation iterations.
    int: Number of erosion iterations.
    """
    if resolution <= max_resolution:
        return max_iterations, max_iterations - 1
    else:
        # Scale down iterations based on resolution
        scale_factor = (resolution - max_resolution) / (100 - max_resolution)
        iterations = max(min_iterations, int(max_iterations - scale_factor * (max_iterations - min_iterations)))
        erosion_iterations = iterations - 1
        return iterations, erosion_iterations


def apply_dilation_erosion(binary_array, dilation_iterations, erosion_iterations):
    # Apply dilation for the specified number of iterations
    for _ in range(dilation_iterations):
        binary_array = binary_dilation(binary_array)

    # Apply erosion for the specified number of iterations
    for _ in range(erosion_iterations):
        binary_array = binary_erosion(binary_array)

    return binary_array

def create_dilated_binary_mask(raster, nodata, dilation_iterations=20, erosion_iterations=18):
    binary_array = np.where(raster == nodata, 0, 1).astype(bool)

    # Apply dilation
    for _ in range(dilation_iterations):
        binary_array = binary_dilation(binary_array)

    # Apply erosion
    for _ in range(erosion_iterations):
        binary_array = binary_erosion(binary_array)

    return binary_array.astype('uint8')


def read_rasters(raster_paths):
    rasters = [rasterio.open(path) for path in raster_paths]
    return rasters

def mosaic_rasters(rasters):
    # Merging rasters and extracting the transform
    mosaic, out_transform = merge(rasters)
    return mosaic, out_transform

def create_binary_mask(raster):
    mask = (raster < 1000000).astype('uint8')
    return mask

def raster_to_polygons(binary_mask, transform):
    polygons = [shape(geom) for geom, val in shapes(binary_mask, mask=binary_mask, transform=transform) if val]
    return polygons

def check_and_fix_geometry(polygon):
    if not polygon.is_valid:
        return polygon.buffer(0)
    return polygon

def generalize_polygons(polygons, tolerance=100):
    simplified_polygons = []
    for polygon in polygons:
        if isinstance(polygon, Polygon):
            fixed_polygon = check_and_fix_geometry(polygon)
            simplified_polygon = fixed_polygon.simplify(tolerance)
            if isinstance(simplified_polygon, (Polygon, MultiPolygon)):
                simplified_polygons.append(simplified_polygon)
    return simplified_polygons
    

def save_polygons(polygons, filename, crs, platform, state, scale, catcov, survey_number):
    try:
        union_polygon = unary_union(polygons)

        if isinstance(union_polygon, GeometryCollection):
            valid_geoms = []
            for geom in union_polygon:
                if isinstance(geom, (Polygon, MultiPolygon)):
                    valid_geoms.append(geom)
                elif isinstance(geom, GeometryCollection):
                    # Handle nested GeometryCollections
                    for sub_geom in geom:
                        if isinstance(sub_geom, (Polygon, MultiPolygon)):
                            valid_geoms.append(sub_geom)

            if valid_geoms:
                union_polygon = MultiPolygon(valid_geoms)
            else:
                raise ValueError("No valid Polygon or MultiPolygon geometries found.")

        gdf = gpd.GeoDataFrame({'geometry': [union_polygon]}, crs=crs)



        # Calculate area in square nautical miles (assuming the CRS is in meters)
        # 1 square meter = 0.0000002915533496 square nautical miles
        gdf['Area_SQNM'] = gdf['geometry'].area * 0.0000002915533496

        # Add other attributes
        gdf['Platform'] = platform
        gdf['State'] = state
        gdf['Scale'] = scale
        gdf['CATCOV'] = catcov
        gdf['Survey_ID'] = survey_number

        # Reproject to NAD83
        gdf = gdf.to_crs("EPSG:4269")

        # Save as Shapefile
        shapefile_path = f"{filename}.shp"
        gdf.to_file(shapefile_path, driver='ESRI Shapefile')
        
        # Reproject to WGS84 for GeoJSON
        gdf = gdf[['geometry', 'Survey_ID']]
        gdf = gdf.to_crs("EPSG:4326")

        # Save as GeoJSON
        geojson_path = f"{filename}.geojson"
        gdf.to_file(geojson_path, driver='GeoJSON')

    except ValueError as e:
        print(f"Error in creating geometry: {e}")


def process_rasters(raster_paths, output_filename, platform, state, scale, catcov, survey_number, resample_threshold=15):
    # Convert resample_threshold to float
    start_time = time.time()  # Start the timer

    resample_threshold = float(resample_threshold)
    print('reading raster data')
    rasters = read_rasters(raster_paths)
    print('resampling raster to coarser resolution')

    # Check and resample rasters only if their resolution is finer than the resample_threshold
    resampled_datasets_memfiles = []
    original_resolutions = []  # List to store original resolutions
    for raster in rasters:
        original_resolutions.append(raster.res[0])  # Store the original resolution
        if raster.res[0] < resample_threshold or raster.res[1] < resample_threshold:
            resampled_dataset, memfile = resample_raster(raster, resample_threshold)
            resampled_datasets_memfiles.append((resampled_dataset, memfile))
        else:
            resampled_datasets_memfiles.append((raster, None))

    print('combining rasters if needed')
    resampled_rasters = [item[0] for item in resampled_datasets_memfiles]
    mosaic, transform = mosaic_rasters(resampled_rasters)
    nodata = rasters[0].nodatavals[0]

    # Create a binary mask for the entire mosaic and apply dilation and erosion
    single_band_mosaic = mosaic[0]  # Extract the single band from the mosaic

    # Calculate iterations based on the original resolution of the first raster
    dilation_iterations, erosion_iterations = calculate_iterations(original_resolutions[0], max_resolution=resample_threshold)

    print('creating and dilating binary mask of data')
    dilated_binary_mask = create_dilated_binary_mask(single_band_mosaic, nodata, dilation_iterations, erosion_iterations)

    print('converting binary mask to polygon')
    polygons = raster_to_polygons(dilated_binary_mask, transform)
    print('generalizing and simplifying the polygons')
    generalized_polygons = generalize_polygons(polygons)
    print('saving the polygon to shapefile (NAD83) and geojson (WGS84)')
    save_polygons(generalized_polygons, output_filename, rasters[0].crs, platform, state, scale, catcov, survey_number)

    # For each memfile, close it if it is not None
    for _, memfile in resampled_datasets_memfiles:
        if memfile is not None:
            memfile.close()
            
    end_time = time.time()
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)
    print(f"***** Done! Enjoy your outline in shapefile and geojson format.*****\nTotal processing time: {int(minutes)} minutes and {seconds:.1f} seconds")

# Example usage
'''
raster_files = [r"E:\erode_test_datasets\H13709_MB_1m_MLLW_Final.bag", r"E:\erode_test_datasets\H13709_SSSAB_1m_600kHz_3of3.tiff", r"E:\erode_test_datasets\H13709_SSSAB_1m_455kHz_2of3.tiff", r"E:\erode_test_datasets\H13709_SSSAB_1m_455kHz_1of3.tiff" ]  
output_file = r"E:\erode_test_datasets\outline_tests\output_polygon3"  # Output filename without file extensions
process_rasters(raster_files, output_file)
'''
def select_raster_files():
    file_paths = filedialog.askopenfilenames(
        title="Select Raster Files",
        filetypes=[("Raster Files", "*.bag *.tiff *.tif"), ("All Files", "*.*")]
    )
    if file_paths:  # Only update if files were selected
        current_files = raster_files_var.get()
        new_files = ', '.join(file_paths)
        raster_files_var.set(f"{current_files}, {new_files}" if current_files else new_files)

def clear_raster_selection():
    raster_files_var.set('')


def select_output_directory():
    directory = filedialog.askdirectory(title="Select Output Directory")
    output_dir_var.set(directory)


def run_processing():
    raster_files = raster_files_var.get().split(', ')
    output_directory = output_dir_var.get()
    output_filename = output_filename_entry.get()
    platform = platform_entry.get()
    state = state_entry.get()
    scale = scale_entry.get()
    catcov = catcov_entry.get()
    survey_number = survey_number_entry.get()

    if raster_files and output_directory and output_filename:
        full_output_path = f"{output_directory}/{output_filename}"
        process_rasters(raster_files, full_output_path, platform, state, scale, catcov, survey_number)

root = tk.Tk()
root.title("Survey Outline Creator - Lite")

raster_files_var = tk.StringVar()
output_dir_var = tk.StringVar()

# Raster files selection
raster_files_label = Label(root, text="Selected Raster Files:")
raster_files_label.pack()
raster_files_button = Button(root, text="Select Raster Files", command=select_raster_files)
raster_files_button.pack()
clear_rasters_button = Button(root, text="Clear Selection", command=clear_raster_selection)
clear_rasters_button.pack()
selected_raster_files_label = Label(root, textvariable=raster_files_var)
selected_raster_files_label.pack()

# Output directory selection
output_dir_label = Label(root, text="Output Directory:")
output_dir_label.pack()
output_dir_button = Button(root, text="Select Output Directory", command=select_output_directory)
output_dir_button.pack()
selected_output_dir_label = Label(root, textvariable=output_dir_var)
selected_output_dir_label.pack()

# Output filename entry
output_filename_label = Label(root, text="Output Filename (without extension):")
output_filename_label.pack()
output_filename_entry = Entry(root)
output_filename_entry.pack()

# Additional attribute entries
platform_label = Label(root, text="Platform:")
platform_label.pack()
platform_entry = Entry(root)
platform_entry.pack()

state_label = Label(root, text="State:")
state_label.pack()
state_entry = Entry(root)
state_entry.pack()

scale_label = Label(root, text="Scale:")
scale_label.pack()
scale_entry = Entry(root)
scale_entry.pack()

catcov_label = Label(root, text="CATCOV:")
catcov_label.pack()
catcov_entry = Entry(root)
catcov_entry.pack()

survey_number_label = Label(root, text="Survey Number:")
survey_number_label.pack()
survey_number_entry = Entry(root)
survey_number_entry.pack()


# Run processing
run_button = Button(root, text="Run Processing", command=run_processing)
run_button.pack(pady=20)

root.geometry("600x600")  
root.mainloop()

def main():
    parser = argparse.ArgumentParser(description='Process Raster Files')
    parser.add_argument('--rasters', nargs='+', help='Paths to raster files')
    parser.add_argument('--output', help='Output file path (without extension)')
    parser.add_argument('--platform', help='Platform name')
    parser.add_argument('--state', help='State')
    parser.add_argument('--scale', help='Scale')
    parser.add_argument('--catcov', help='CATCOV')
    parser.add_argument('--survey_number', help='Survey Number')
    args = parser.parse_args()

    if args.rasters and args.output:
        process_rasters(args.rasters, args.output, args.platform, args.state, args.scale, args.catcov, args.survey_number)

if __name__ == "__main__":
    main()