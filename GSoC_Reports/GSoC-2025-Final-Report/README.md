<p align="left">
  <img src="images/GSoC_Alaska_logo.png" alt="GSoC Alaska Logo" width="1000"/>
</p>

# GSoC 2025 Final Report

| **Organization** | Alaska |
|------------------|:-------|
| **Project Title** | Automated Coastline Extraction for Erosion Modeling in Alaska |
| **Mentors** | Frank Witmer, Rawan Elframawy |
| **Student** | Ritika Kumari |

---

## About Me  
My name is Ritika Kumari, and I am in my final year of B.Tech in Computer Science and Engineering . I had the opportunity to work with Alaska this summer on the Automated Coastline Extraction for Erosion Modeling project, where I contributed to training the models that provide automated coastline extraction for erosion studies.
Through this project, I worked with computer vision, image processing, machine learning, and geospatial data, applying my academic learning to real-world problems.GSoC 2025 gave me the chance to make meaningful contributions to open source.
- 📧 Email: rkjane333@gmail.com  
- 🔗 LinkedIn: www.linkedin.com/in/ritika-kumari-b15b95289  

---

## Project Goal  
The goal of this project is to develop and deliver a highly accurate, automated model that can extract vectorized coastlines directly from PlanetLabs satellite imagery with 3-meter resolution. The work focuses on retraining the [DeepWaterMap](https://github.com/isikdogan/deepwatermap) model, which was originally designed for 30-meter resolution Landsat data, and adapting it to operate effectively on higher-resolution imagery. A custom dataset will be created from PlanetLabs imagery to support this retraining. Ultimately, the project aims to produce a robust model capable of processing large volumes of imagery and accurately predicting coastline changes over time, which is essential for improving coastal erosion forecasts in the **Deering region of Alaska**, particularly under the impacts of climate change.  

| **Project Flow Diagram** |
|---------------------------|
| ![Project Flow Diagram](images/project_Flow_Diagram.png) |

---

## Links to Repositories  
- [CoastlineExtraction](https://github.com/fwitmer/CoastlineExtraction)  
- [CoastlineExtraction (My Fork)](https://github.com/Ritika-K7/CoastlineExtraction)  

---

## What I Did This Summer  
For my GSoC project this year, I focused on preparing a **high-quality dataset using PlanetLabs imagery** so that I could train different models such as [DeepWaterMap](https://github.com/isikdogan/deepwatermap) and also experiment with other models like **U-Net**. A large part of my work involved extensive data preprocessing tasks, which were essential for building a reliable training pipeline. I’ll try to summarize the work that I did in my GSoC project below. Let’s get started!  

---

### 1. ⚙️ Configuration Management System  
I added two files to make the project easier to organize and maintain:  

- **`config_template.json`** – A single configuration file that stores all file paths and folder structures (satellite images, shapefiles, ground truth data, processed outputs, etc.) in one place.  
- **`load_config.py`** – A Python helper module with functions like `get_image_path()`, `get_shapefile_path()`, and `get_raw_data_path()` to quickly access files without hardcoding paths.  

This system avoids hardcoded file paths, so the project is not tied to one person’s computer setup. Anyone can run the code without needing to manually change paths for their system.  

---

### 2. 🛠 Utilities Folder (`utils/`)  
I created a `utils/` folder with five reusable modules to support the coastline extraction workflow:  

| **Module**              | **Description** |
|--------------------------|-----------------|
| `check_crs.py`           | Validates and compares CRS of raster/vector files to prevent mismatch errors. |
| `gis_tools.py`           | Provides spatial analysis functions like `create_transect_points()` (transect–coastline intersections) and `clip_shp()` (shapefile clipping). |
| `rasterize_shapefile.py` | Converts vector coastline shapefiles to raster GeoTIFFs for binary masks used in training/analysis. |
| `shapefile_generator.py` | Creates shapefiles from raster data via morphological active contours, with noise reduction and contour extraction. |
| `spatial_analysis.py`    | Offers spatial checks like `log_spatial_info()` to validate raster–point alignment and overlaps. |

These reusable utilities ensure consistency, enable format conversion, support spatial validation, and make the workflow modular and maintainable.  

---

### 3. 🧭 Fixing the NDWI Labeling Script (`ndwi_labels.py`)  

#### 🔹 Integration of Utility Modules  
- Imported custom utilities: `check_crs`, `crs_match`, `log_spatial_info`, and `save_concatenated_ndwi_with_shapefile`.  
- Integrated the configuration system with `load_config` and path helper functions.  

#### 🔹 Data Validation and Error Prevention  
- Added CRS validation using `check_crs()`.  
- Ensured coordinate system compatibility with `crs_match()`.  
- Used `log_spatial_info()` to validate data alignment and overlap.  

#### 🔹 Robust Spatial Processing  
- Added intersection checking with `buffer.intersects(raster_bounds_geom)` to prevent processing points outside raster bounds.  
- Added proper bounds handling inside the raster context.  
- Imported `box` from shapely for correct geometry operations.  

#### 🔹 Automated Output Generation  
- Integrated `save_concatenated_ndwi_with_shapefile()` to automatically save results and generate coastline shapefiles.  
- Created a `save_ndwi_plots()` function to export visualization outputs instead of only displaying them.  

---

## 4. Data Preprocessing Pipeline  

### Pipeline Overview  
![Data Preprocessing Pipeline](images/DATA%20PREPROCESSING%20PIPELINE.png)  

---

### Step 1: Batch Alignment  
**Purpose:** Align raw satellite images to a common CRS, pixel size, and extent.  
**Tool:** `GDAL warp` (`batch_align.py`)  
**Input:** 32 raw satellite GeoTIFFs from `raw_data/` + NDWI reference image  

**Process:**  
- Reads configuration and alignment parameters from the reference image  
- Sets alignment parameters:  
  - CRS: UTM Zone 3N (EPSG:32603)  
  - Pixel size: **3.125 m**  
  - Spatial extent: `[598355.0, 7326619.0, 605849.5, 7334628.5]`  
- Uses GDAL warp to align CRS, resolution, and extent  

---

### Step 2: Georeferencing  
**Purpose:** Apply pixel-level precision correction for spatial alignment.  
**Tool:** `AROSICS` (`georeference.py`)  
**Input:** Aligned images from Step 1 + NDWI base image + Deering Airstrip coordinates  

**Process:**  
- Coregisters aligned images with AROSICS  
- Applies fine spatial corrections  
- Corrects rotated or tiled images as needed  

**Output:** 32 georeferenced GeoTIFFs (`*_GeoRegistered.tif`)  

---

### Step 3: Mask Generation  
**Purpose:** Create binary water/land masks with scientific validation.  
**Tool:** `create_masks.py`  
**Input:** Georeferenced images from Step 2 + transect points shapefile  

**Process:**  
- Computes NDWI from Green + NIR bands  
- Applies Gaussian blur for noise reduction  
- Performs sliding window analysis around transect points  
- Combines local classification with global NDWI threshold  

**Output:**  
- Binary mask TIFFs (`*_concatenated_ndwi.tif`)  
- Coastline shapefiles (`*_Coast_Contour.shp`)  
- Visualization plots  

 
| **Mask Generation Visualization** |
|-----------------------------------|
| ![Image-Mask Diagram](images/img_mask_diagram.png) |

---

### Step 4: Image Tiling  
**Purpose:** Split images and masks into smaller tiles for training.  
**Tool:** `Rasterio` (`tile_images.py`)  
**Input:** Georeferenced images (Step 2) + Masks (Step 3)  

**Process:**  
- Reads raster data with georeferencing  
- Splits into **512×512** pixel tiles with overlap  
- Applies identical tiling to both images and masks  

**Output:** ~1,100 tiles (~550 images + ~550 masks)  

| **Image Tiling Diagram** |
|---------------------------|
| ![Tile Diagram](images/Tile_Diag.png) |

---

### Step 5: Data Augmentation  
**Purpose:** Expand dataset with rotated and flipped variations.  
**Tools:** `OpenCV + Rasterio` (`augment_tiles.py`)  
**Input:** Image–mask tile pairs from Step 4  

**Process:**  
- Reads each tile pair  
- Generates rotated & flipped variations (**7 augmentations**)  
- Preserves georeferencing metadata  

**Output:** ~8,800 tiles (1,100 original + 7,700 augmented)  


| **Data Augmentation Diagram** |
|-------------------------------|
| ![Augmentation Diagram](images/augment_Diagram.png) |  

---

### Training Dataset Statistics  

| **Step**              | **Details** |
|------------------------|-------------|
| **Batch Alignment**    | 32 images |
| **Georeferencing**     | 32 images |
| **Mask Generation**    | 5 georeferenced images → 5 binary masks |
| **Image Tiling**       | 110 tiles per image × 5 images = 550 image tiles <br> 110 tiles per mask × 5 masks = 550 mask tiles <br> **Total = 1,100 image–mask tiles** |
| **Data Augmentation**  | 1,100 original tiles × 8 versions (1 original + 7 augmented) <br> **= 8,800 tiles (4,400 images + 4,400 masks)** |

---

## 5. 🧩 Models  

### 5.1 UNet  
**Description:**  
U-Net is a convolutional neural network designed for semantic segmentation, widely used for extracting features such as coastlines from satellite images.  

**Dataset Formation:**  
Training data prepared using all preprocessing steps:  
1. Batch Alignment  
2. Georeferencing  
3. Mask Generation  
4. Image Tiling  
5. Data Augmentation  

---

### 5.2 DeepWaterMap  
**Description:**  
DeepWaterMap is a deep learning model specialized for **surface water mapping** using multispectral satellite imagery.  

**Dataset Formation:**  
- Uses preprocessing **Step 1 → Step 4**  
- Instead of separate image & mask files:  
  - Adds **Mask as 5th Band**  
  - Original 4 bands: R, G, B, NIR  
  - New stacked tile:  
    - Band 1 = Red  
    - Band 2 = Green  
    - Band 3 = Blue  
    - Band 4 = NIR  
    - Band 5 = Binary Water Mask (0 = land, 1 = water)  

**Data Format:**  
DeepWaterMap requires **TFRecord format**.  
Each TFRecord contains:  
- **Image tensor:** 5 stacked bands (normalized)  
- **Label tensor:** Water mask band (last channel)  

**Final Dataset Structure:**  
- `train_*.tfrecord` → Training data  
- `test_*.tfrecord` → Validation/Testing data  

| **Bands Visualization** |
|---------------------------------------|
| ![GeoRegistered Bands Visualization 1](images/1483219_0369619_2018-06-05_1034_BGRN_SR_clip_aligned_GeoRegistered_bands_visualization.png) |
| ![GeoRegistered Bands Visualization 2](images/1483240_0369619_2018-06-05_103f_BGRN_SR_clip_aligned_GeoRegistered_bands_visualization.png) |
| ![GeoRegistered Bands Visualization 3](images/1487598_0369619_2018-06-07_1008_BGRN_SR_clip_aligned_GeoRegistered_bands_visualization.png) |
 

---
## Contributions
| PR Link | Description | Status |
|---------|-------------|--------|
| [#42](https://github.com/fwitmer/CoastlineExtraction/pull/42) | Add config_template.json to guide users in setting up image and shape file paths | Merged |
| [#43](https://github.com/fwitmer/CoastlineExtraction/pull/43) | add CRS checker utility for raster and vector geospatial files | Merged |
| [#44](https://github.com/fwitmer/CoastlineExtraction/pull/44) | Integrate load_config for dynamic path handling in ndwi_labels.py and add missing raster file | Merged |
| [#45](https://github.com/fwitmer/CoastlineExtraction/pull/45) | Add CRS validation to ndwi_labels.py for input files. | Merged |
| [#46](https://github.com/fwitmer/CoastlineExtraction/pull/46) | Add spatial_analysis.py utility and integrate with ndwi_labels.py | Merged |
| [#47](https://github.com/fwitmer/CoastlineExtraction/pull/47) | Add intersection check to skip non-overlapping buffers in NDWI processing. | Merged |
| [#48](https://github.com/fwitmer/CoastlineExtraction/pull/48) | Refactor: Extract plotting logic into separate save_ndwi_plots() function | Merged |
| [#49](https://github.com/fwitmer/CoastlineExtraction/pull/49) | feat: integrate shapefile generation for coastline extraction | Merged |
| [#50](https://github.com/fwitmer/CoastlineExtraction/pull/50) | refactor: move GIS utility functions from ndwi_labels.py to utils/gis_tools.py | Merged |
| [#51](https://github.com/fwitmer/CoastlineExtraction/pull/51) | Add utility to rasterize coastline shapefile to GeoTIFF | Merged |
| [#52](https://github.com/fwitmer/CoastlineExtraction/pull/52) | feat : Add batch alignment script for satellite image preprocessing | Merged |
| [#53](https://github.com/fwitmer/CoastlineExtraction/pull/53) | feat: Add fine coregistration script for satellite image fine-tuning | Merged |
| [#54](https://github.com/fwitmer/CoastlineExtraction/pull/54) | feat: Add image tiling script for satellite image preprocessing | Merged |
| [#56](https://github.com/fwitmer/CoastlineExtraction/pull/56) | Enhance batch_align.py with improved resolution handling and documentation | Merged |
| [#57](https://github.com/fwitmer/CoastlineExtraction/pull/57) | Adds three key scripts for preprocessing satellite imagery data| Merged |
| [#58](https://github.com/fwitmer/CoastlineExtraction/pull/58) | Enhanced Data Preprocessing Pipeline with Water Mask Generation | Merged |
| [#59](https://github.com/fwitmer/CoastlineExtraction/pull/59) | Training, Prediction & Mask Band Integration for Coastline Segmentation | Merged |

---
## Future work
Use the dataset created in this project to train the model on the **Chinook** and test it with Planet imagery from **September 4 and September 6, 2016**. Once trained, the model will be applied to extract vectorized coastlines, which will then be compared against **USGS** and **Hi-Res reference coastlines** to evaluate accuracy.

Publish the results in a research paper, highlighting the advancements made in automated coastline extraction and demonstrating the effectiveness of models such as DeepWaterMap and U-Net for improving erosion monitoring in Alaska.


## What challenges did you encounter as part of this project?
One of the most difficult parts of this project was working with **GDAL**. Installation itself was challenging due to dependency issues, and once installed, using GDAL for image alignment required significant effort to understand and apply correctly. Similarly, the **AROSICS** library for fine-tuning alignment (coregistration) demanded careful parameter tuning and repeated troubleshooting to achieve pixel-level accuracy.

Another major challenge was handling large volumes of data. The size of the imagery and intermediate outputs sometimes made it difficult to manage everything on **GitHub**. In cases where I had to delete files or reduce repository size, I needed to rebase the history and start from the beginning, which added extra complexity to the workflow.


## Acknowledgment
I would like to sincerely thank my mentors, Prof. Frank Witmer and Rawan Elframawy, for their continuous guidance and encouragement throughout this project. Their weekly meetings, thoughtful feedback, and patience helped me navigate challenges and stay motivated. Without them, the work never would have been this joyful and rewarding. Both were always available to answer my questions on time, supportive and understanding, and always open to newer ideas.

