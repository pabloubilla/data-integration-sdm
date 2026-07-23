import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from src_old.utils import split_pa_train_test_spatially

# --- Configuration ---
# NOTE: Update these lists with the actual region and group codes available in your data directory.
# Regions should be capitalized as they appear in the data filenames (e.g., AWTtrain_po.csv)
# REGIONS = ["AWT", "CAN", "NSW", "SA", "SWI", "NZ"]             # regions to run
REGIONS = ['SWI']
GROUPS_BY_REGION = {
        "AWT": ["_plant", "_bird"],
        "CAN": [""],
        "NSW": ["_bird"],
        # "NSW": ["_bat", "_bird", "_plant", "_reptile"], #["_ba", "_db", "_nb", "_ot", "_rt", "_ru", "_sr"],
        "SA" : [""],
        "SWI": [""],
        "NZ": [""]
    }  

# Base directory for data files
DATA_DIR = 'data'
# Directory to save the output plots
OUTPUT_DIR = 'output/presentation_figures/swi'

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

            # define some colors
            color_background = "#c6cedc"
            edge_color = "#333333"
            
            # Set a common main title for the figure
            # main_title = f"Data Distribution | Region: {region_upper} | Group: {'All Species' if group == '' else group.capitalize()}"
            # fig.suptitle(main_title, fontsize=14, fontweight='bold', y=1.02)
            
            # --- PLOT for PA ONLY ---
            fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
            
            # Boundary Map (Base Layer: light gray, dark border)
            gdf_map.plot(ax=ax1, color=color_background, edgecolor=edge_color, linewidth=1, alpha=0.8)
            
            # Points (Emerald green)
            gdf_points_pa.plot(
                ax=ax1, 
                color="#329A1D", 
                markersize=3, 
                # label=f"PA Records     ({count_pa})", 
                alpha=0.7,
                linewidth=.5
            )
            
            # ax1.set_title(f"Presence-Absence Records ($N={count_pa}$)", fontsize=10)
            ax1.set_axis_off() # Remove axes for cleaner map view
            # ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
            ax1.set_aspect('equal')
            # ax1.set_facecolor('#f7f7f7') # Subtle background for the plot area
            # transparent background
            fig.patch.set_alpha(0.0)
            # tight
            plt.tight_layout()

            # save the plot
            filename = f"{region_upper}_records{group}_PA_only.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            fig.savefig(save_path, dpi=300)


            #### Now split TRAIN and TEST PA points using X,Y blocks
            plt.close(fig) # Close the figure to free memory
            fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
            # Boundary Map (Base Layer: light gray, dark border)
            gdf_map.plot(ax=ax1, color=color_background, edgecolor=edge_color, linewidth=1, alpha=0.8)
            # split PA points into TRAIN and TEST spatially
            X_pa_tr, X_pa_te, Y_pa_tr, Y_pa_te = split_pa_train_test_spatially(
                df_pa[['x','y']], df_pa[df_pa.columns[4:]], test_frac=0.3, seed=42, K = 40
            )
            gdf_points_pa_tr = gpd.GeoDataFrame(
                X_pa_tr,
                geometry=gpd.points_from_xy(X_pa_tr.x, X_pa_tr.y),
                crs=crs
            )
            gdf_points_pa_te = gpd.GeoDataFrame(
                X_pa_te,
                geometry=gpd.points_from_xy(X_pa_te.x, X_pa_te.y),
                crs=crs
            )
            # Points TRAIN (Blue)
            gdf_points_pa_tr.plot(
                ax=ax1,
                color="#1f77b4",
                markersize=3,
                alpha=0.7,
                # label=f"Train",
                linewidth=.5
            )
            # Points TEST (Orange)
            gdf_points_pa_te.plot(
                ax=ax1,
                color="#ff7f0e",
                markersize=3,
                alpha=0.7,
                # label=f"Test",
                linewidth=.5
            )   
            ax1.set_axis_off() # Remove axes for cleaner map view
            ax1.set_aspect('equal')
            # transparent background
            fig.patch.set_alpha(0.0)
            # tight
            plt.tight_layout()
            # save the plot
            filename = f"{region_upper}_records{group}_PA_train_test_split.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            fig.savefig(save_path, dpi=300)
            plt.close(fig) # Close the figure to free memory



            # now make one with split TRAIN and TEST

            # --- PLOT for PO --- 
            
            # Boundary Map (Base Layer: light gray, dark border)
            # gdf_map.plot(ax=ax2, color="#e6e1f6", edgecolor="#333333", linewidth=0.7, alpha=0.8)
            
            # figure
            fig, ax2 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
            
            gdf_map.plot(ax=ax2, color=color_background, 
                         edgecolor=edge_color, linewidth=1, alpha=0.8)
            
            # plot PO, with a different color per species (column spid)
            # for rare species change the name to rare
            gdf_points_po.plot(
                ax=ax2,
                column='spid',
                cmap='tab10',
                markersize=5,
                alpha=0.7,
                linewidth=.5,
                legend=False
            )
            ax2.set_axis_off() # Remove axes for cleaner map view
            ax2.set_aspect('equal')
            # transparent background
            fig.patch.set_alpha(0.0)
            # tight
            plt.tight_layout()
            # save the plot
            filename = f"{region_upper}_records{group}_PO_by_species.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            fig.savefig(save_path, dpi=300)
            plt.close(fig) # Close the figure to free memory


            # same plot but with all PO points in same color
            fig, ax2 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
            
            gdf_map.plot(ax=ax2, color=color_background, 
                         edgecolor=edge_color, linewidth=1, alpha=0.8)
            
            # plot PO, with a different color per species (column spid)
            # for rare species change the name to rare
            gdf_points_po.plot(
                ax=ax2,
                color="#e93131",
                markersize=5,
                alpha=0.7,
                linewidth=.5,
                legend=False
            )
            ax2.set_axis_off() # Remove axes for cleaner map view
            ax2.set_aspect('equal')
            # transparent background
            fig.patch.set_alpha(0.0)
            # tight
            plt.tight_layout()
            # save the plot
            filename = f"{region_upper}_records{group}_PO_all_points.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            fig.savefig(save_path, dpi=300)
            plt.close(fig) # Close the figure to free memory



            # ax2.set_title(f"Presence-Only Records ($N={count_po}$)", fontsize=10)
            # ax2.set_axis_off() # Remove axes for cleaner map view
            # # ax2.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
            # ax2.set_aspect('equal')
            # ax2.set_facecolor('#f7f7f7') # Subtle background for the plot area

            # # Final layout adjustments
            # plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
            
            # # 7. Save the plot
            # filename = f"{region_upper}_records{group}.png"
            # save_path = os.path.join(OUTPUT_DIR, filename)
            # fig.savefig(save_path, dpi=300)
            # plt.close(fig) # Close the figure to free memory
            # print(f"   Successfully saved plot to: {save_path}")

        except FileNotFoundError as e:
            # Catch specific errors for missing files
            print(f"   Warning: File not found for {region_upper} and group '{group}'. Please check the data structure.")
            print(f"   Missing file details: {e}")
        except Exception as e:
            # Catch any other plotting or data processing errors
            print(f"   An unexpected error occurred for {region_upper} and group '{group}': {e}")

            print(f"   Successfully saved plot to: {save_path}")