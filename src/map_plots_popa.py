import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
# NOTE: Update these lists with the actual region and group codes available in your data directory.
# Regions should be capitalized as they appear in the data filenames (e.g., AWTtrain_po.csv)
REGIONS = ["AWT", "CAN", "NSW", "SA", "SWI", "NZ"]             # regions to run
GROUPS_BY_REGION = {
        "AWT": ["_plant", "_bird"],
        "CAN": [""],
        "NSW": ["_bat", "_bird", "_plant", "_reptile"], #["_ba", "_db", "_nb", "_ot", "_rt", "_ru", "_sr"],
        "SA" : [""],
        "SWI": [""],
        "NZ": [""]
    }  

# Base directory for data files
DATA_DIR = 'data'
# Directory to save the output plots
OUTPUT_DIR = 'output/map_analysis'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Plots will be saved in the '{OUTPUT_DIR}' directory.")

# --- Plotting Loop ---
for region in REGIONS:
    region_lower = region.lower()
    region_upper = region.upper()

    for group in GROUPS_BY_REGION[region_upper]:
        # Determine the group suffix for file names (e.g., '_bird' or '')
        
        print(f"\nProcessing: Region={region_upper}, Group='{'All' if group == '' else group.capitalize()}'")

        try:
            # 1. Load Geospatial Boundary Data (.gpkg)
            gdf_map_path = os.path.join(DATA_DIR, f'raw/NCEAS/Borders/{region_lower}.gpkg')
            if not os.path.exists(gdf_map_path):
                print(f"   Skipping: Boundary file not found for {region_lower} at {gdf_map_path}")
                continue
            gdf_map = gpd.read_file(gdf_map_path)
            
            # Use the CRS from the boundary file
            crs = str(gdf_map.crs)

            # 2. Load Point Data (assuming both train_po and test_pa files are CSVs)
            
            # Presence-Only (PO) Data
            df_po_path = os.path.join(DATA_DIR, f'processed/NCEAS/Records/train_po/{region_upper}train_po{group}.csv')
            df_po = pd.read_csv(df_po_path)

            # Presence-Absence (PA) Data
            if region == 'NSW':
                pa_folder = 'processed'
            else:
                pa_folder = 'raw'
            df_pa_path = os.path.join(DATA_DIR, f'{pa_folder}/NCEAS/Records/test_pa/{region_upper}test_pa{group}.csv')
            df_pa = pd.read_csv(df_pa_path)
            
            # 3. Data Pre-processing: Ensure coordinates are float and handle potential missing values
            df_pa[['x','y']] = df_pa[['x','y']].astype(float)
            df_po[['x','y']] = df_po[['x','y']].astype(float)
            
            # Drop rows with NaN coordinates if any
            df_pa.dropna(subset=['x', 'y'], inplace=True)
            df_po.dropna(subset=['x', 'y'], inplace=True)


            # 4. Create GeoDataFrames
            
            # Presence-Only GeoDataFrame
            gdf_points_po = gpd.GeoDataFrame(
                df_po,
                geometry=gpd.points_from_xy(df_po.x, df_po.y),
                crs=crs
            )
            
            # Presence-Absence GeoDataFrame
            gdf_points_pa = gpd.GeoDataFrame(
                df_pa,
                geometry=gpd.points_from_xy(df_pa.x, df_pa.y),
                crs=crs
            )

            # 5. Calculate Counts
            count_po = len(gdf_points_po)
            count_pa = len(gdf_points_pa)

            # 6. Plotting
            fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)
            
            # Set a common main title for the figure
            main_title = f"Data Distribution | Region: {region_upper} | Group: {'All Species' if group == '' else group.capitalize()}"
            fig.suptitle(main_title, fontsize=14, fontweight='bold', y=1.02)
            
            # --- Subplot 1: Presence-Absence (PA) ---
            
            # Boundary Map (Base Layer: light gray, dark border)
            gdf_map.plot(ax=ax1, color="#f0f0f0", edgecolor="#333333", linewidth=0.7, alpha=0.8)
            
            # Points (Emerald green)
            gdf_points_pa.plot(
                ax=ax1, 
                color="#10b981", 
                markersize=5, 
                label=f"PA Records ({count_pa})", 
                alpha=0.7
            )
            
            ax1.set_title(f"Presence-Absence Records (Count: {count_pa})", fontsize=12)
            ax1.set_axis_off() # Remove axes for cleaner map view
            ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
            ax1.set_aspect('equal')
            ax1.set_facecolor('#f7f7f7') # Subtle background for the plot area

            # --- Subplot 2: Presence-Only (PO) ---
            
            # Boundary Map (Base Layer: light gray, dark border)
            gdf_map.plot(ax=ax2, color="#f0f0f0", edgecolor="#333333", linewidth=0.7, alpha=0.8)
            
            # Points (Vibrant Red)
            gdf_points_po.plot(
                ax=ax2, 
                color="#ef4444", 
                markersize=5, 
                label=f"PO Records ({count_po})", 
                alpha=0.7
            )

            ax2.set_title(f"Presence-Only Records (Count: {count_po})", fontsize=12)
            ax2.set_axis_off() # Remove axes for cleaner map view
            ax2.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
            ax2.set_aspect('equal')
            ax2.set_facecolor('#f7f7f7') # Subtle background for the plot area

            # Final layout adjustments
            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
            
            # 7. Save the plot
            filename = f"{region_upper}_records{group}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            fig.savefig(save_path, dpi=300)
            plt.close(fig) # Close the figure to free memory
            print(f"   Successfully saved plot to: {save_path}")

        except FileNotFoundError as e:
            # Catch specific errors for missing files
            print(f"   Warning: File not found for {region_upper} and group '{group}'. Please check the data structure.")
            print(f"   Missing file details: {e}")
        except Exception as e:
            # Catch any other plotting or data processing errors
            print(f"   An unexpected error occurred for {region_upper} and group '{group}': {e}")

print("\nPlotting process complete. Check the 'output_plots' directory for your results!")