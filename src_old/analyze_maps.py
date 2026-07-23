import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import rasterio  # for reading .tif files
from rasterio.plot import show

# Path to your folder
folder_path = r"data/raw/Environment/NZ"

# Find all .tif files
tif_files = glob.glob(os.path.join(folder_path, "*.tif"))

print(f"Found {len(tif_files)} .tif files in {folder_path}")

with rasterio.open("data/raw/Environment/SWI/bcc.tif") as src:
    img = src.read(1)
    fig, ax = plt.subplots(figsize=(8, 6))
    show(src, ax=ax, cmap='viridis')  # preserves geospatial extent
    ax.set_title("GeoTIFF Displayed with Correct Coordinates")
    plt.show()


# # Read and display a few
# for i, tif_path in enumerate(tif_files[:3]):  # limit to first 3 for preview
#     with rasterio.open(tif_path) as src:
#         img = src.read(1)  # read first band (most TIFs are multi-band)
#         print(f"\nFile {i+1}: {os.path.basename(tif_path)}")
#         print(f"  Shape: {img.shape}, dtype: {img.dtype}")
#         print(f"  CRS: {src.crs}")
#         print(f"  Bounds: {src.bounds}")

#         # Plot image
#         plt.figure(figsize=(6, 5))
#         plt.imshow(img)
#         plt.title(os.path.basename(tif_path))
#         plt.colorbar(label='Pixel Value')
#         plt.axis('off')
#         plt.show()

#         # # Plot histogram
#         # plt.figure(figsize=(5, 3))
#         # plt.hist(img.flatten(), bins=50, color='steelblue', edgecolor='black')
#         # plt.title("Pixel Intensity Histogram")
#         # plt.xlabel("Value")
#         # plt.ylabel("Frequency")
#         # plt.show()
