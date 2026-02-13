Download images to local storage from Google Earth Engine in format of geotiff for a given AOI (geojson) and date.
Warns: it can be very slow if AOI is too large because the maximum number of pixels and dimensions of extent that can be downloaded from
GEE are limited.

Python requirements:  
`rasterio, earthengine-api, google-cloud-bigquery, db_dtypes`  

Install the `gcloud CLI` on your machine (https://cloud.google.com/sdk/docs/install)  

In the browser, create a new project on google cloud, and enable big query + Google Earth Engine in your gcloud project 
(also register your project to use G.E.E.: https://code.earthengine.google.com/register)

run these commands to authenticate gcloud in the cli:  
(inside the gee-downloader directory)  
(you might need to set the project id as created in the browser)  
`gcloud auth login`  
`gcloud auth application-default login`


usage:  
`python main.py -c download.yaml`

## Workflow Guide

The complete workflow for downloading Sentinel-2 imagery consists of the following steps:

### Step 1: Create AOI Shapefile

Create a shapefile representing your Area of Interest (AOI) that will be used throughout the download process. This shapefile will define the geographic extent for metadata extraction and image download.

### Step 2: Metadata Extraction

Extract basic information (cloud percentage, snow/ice percentage, etc.) for each Sentinel-2 image and each grid cell over a specified date range using gee_downloader.

- The shapefile generated in Step 1 is used as the AOI in the configuration file (`download.yaml`), with `mode=info`.
- The extracted metadata are saved as separate CSV files for each grid cell.

### Step 3: Image Filtering

Combine all CSV files generated in Step 2 into a single dataset and filter out good images based on predefined thresholds for cloud and snow/ice percentages (see `shared/tools.py` in gee_downloader).

### Step 4: Image Download

Download all good Sentinel-2 L1C TOA reflectance images using gee_downloader with the CSV file generated in Step 3 with `mode=download`.

- In Step 4, you can set the backend to `gcld` for level-1 processing of Sentinel-2 data only.

This repository was migrated from arctus_2023 on Dec. 2024
