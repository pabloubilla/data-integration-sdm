import numpy as np
import torch
import pandas as pd
import os
# rasterio
import rasterio
# deepmaxent
from src.models import deepmaxent_model
# tqdm for progress bar
import tqdm

from sklearn.preprocessing import StandardScaler
import joblib
import pickle

add_po_covariate = False
region = 'CAN'
group = ''
env_path = os.path.join('data', 'raw', 'Environment', region)
nodata_guess = [65535, 32767, -9999, -32768, -3.4e38]

# for the environment path load all .tif
env_layers = {}
print("Loading environmental layers...")
border_indexes = None
for file in os.listdir(env_path):
    if file.endswith('.tif'):
        # layer = pd.read_csv(os.path.join(env_path, file), header=None).values
        # # file name without .tif is the key
        # env_layers[file[:-4]] = layer

        with rasterio.open(os.path.join(env_path, file)) as src:
            layer = src.read(1)  # read first band
            env_layers[file[:-4]] = layer

            for nodata in nodata_guess:
                if np.any(layer == nodata):
                    print(f"Layer {file[:-4]} contains nodata value guess: {nodata}")

                # save the indexes
                if border_indexes is None:
                    border_indexes = np.where((layer == nodata))
                # else:
                #     border_indexes = (np.unique(np.concatenate((border_indexes[0], np.where(layer == nodata)[0]))),
                #                       np.unique(np.concatenate((border_indexes[1], np.where(layer == nodata)[1]))))


            # print the first component
            print(f"First 10 values of layer {file[:-4]}: ")
            print(layer.flatten()[:10])
            print('---')


            # print a bit of the layer
            # for i in range(1000):
            #     print(f"Row {i} of layer {file[:-4]}: ")
            #     print(layer[i, :10])
            # exit()

            # if any value is -inf or inf 
            #layer[np.isinf(layer)] = 0
            # print a subset
            print(f"Loaded layer {file[:-4]} with shape {layer.shape}")
print(border_indexes)


if add_po_covariate:
    # create a layer of zeros with the same shape as the other layers
    sample_layer = next(iter(env_layers.values()))
    po_layer = np.zeros_like(sample_layer)
    env_layers['PO'] = po_layer
    print("Added PO layer with zeros.")

# load region model
model_path = os.path.join('output', 'models', f'deepmaxent_{region}{group}_model.pt')
model = torch.load(model_path)

# model = deepmaxent_model(input_size=len(env_layers), hidden_size=250, output_size=10, hidden_nbr=3)

# just to get the right covariate order
test_env_path = os.path.join('data', 'raw', 'Records', 'test_env', f'{region}test_env{group}.csv')
df_test_env = pd.read_csv(test_env_path)
covariates = df_test_env.columns[4:].to_list()  # as the first four columns are group, siteid, x, y   
if add_po_covariate:
    covariates.append('PO') 
species_number = model.fc3_lambda.out_features

# for the envl_layers create a 3D array with shape (nbr_layers, height, width)
env_array = np.array([env_layers[cov] for cov in covariates])
nbr_layers, height, width = env_array.shape

# create an empty array to store the predictions
predictions = np.zeros((height, width, species_number))

# predict for each pixel
# vectorized version (makes sense)
inputs = torch.tensor(env_array.reshape(nbr_layers, -1).T, dtype=torch.float32)  # shape (height*width, nbr_layers)
# transform border_indexes
if border_indexes is not None:
    border_flat_indexes = border_indexes[0] * width + border_indexes[1]
    print(f"Number of border pixels: {len(border_flat_indexes)}")
    # set these indexes to nan
    inputs[border_flat_indexes, :] = torch.nan

# scale inputs
scaler_path = os.path.join('output', 'models', f'scaler_{region}{group}.pkl')
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
# add feature names
inputs_df = pd.DataFrame(inputs.numpy(), columns=covariates)
input_df_no_border = inputs_df.drop(index=border_flat_indexes)
# stats of inputs_df summary
for col in covariates:
    # print stats for non border values
    col_values = input_df_no_border[col].values
    print(f"Feature '{col}': min={np.nanmin(col_values)}, max={np.nanmax(col_values)}, mean={np.nanmean(col_values)}, std={np.nanstd(col_values)}")

inputs_np = scaler.transform(inputs_df[covariates].values)
inputs = torch.tensor(inputs_np, dtype=torch.float32)
# stats after scaling
for i, col in enumerate(covariates):
    # with no borders
    col_values = inputs_np[:, i]
    col_values_no_border = np.delete(col_values, border_flat_indexes)
    print(f"Scaled Feature '{col}': min={np.nanmin(col_values_no_border)}, max={np.nanmax(col_values_no_border)}, mean={np.nanmean(col_values_no_border)}, std={np.nanstd(col_values_no_border)}")

batch_size = 500000 # adjust based on memory capacity
batch_size = min(batch_size, inputs.shape[0]) # in case the image is small
predictions = np.zeros((height * width, species_number))
model.eval()
print("Predicting distribution over the entire region (vectorized)...")
with torch.no_grad():
    for start in tqdm.tqdm(range(0, inputs.shape[0], batch_size), desc="Batches processed"):
        end = min(start + batch_size, inputs.shape[0])
        batch_inputs = inputs[start:end]
        batch_outputs = model(batch_inputs)
        predictions[start:end, :] = batch_outputs.numpy()
# reshape predictions back to (height, width, nbr_species)
raw_predictions = predictions.reshape(height, width, species_number)
# border to nan
if border_indexes is not None:
    for i in range(species_number):
        raw_predictions[border_indexes[0], border_indexes[1], i] = np.nan
# raw_predictions = np.exp(raw_predictions)
# normalize
# raw_predictions = raw_predictions / np.nansum(raw_predictions, axis=(0,1), keepdims=True)
predictions = raw_predictions.copy()

# for each species print the first 10 non zero values
for i in range(species_number):
    species_pred = predictions[:, :, i]
    non_zero_values = species_pred[species_pred > 0]
    print(f"Species {i+1} first 10 non-zero predictions: {non_zero_values.flatten()[:10]}")
print('---\n')

# count how many non nans for each species
predictions = raw_predictions
for i in range(species_number):
    species_pred = predictions[:, :, i]
    non_nan_count = np.count_nonzero(~np.isnan(species_pred))
    print(f"Species {i+1} non-NaN pixel count: {non_nan_count}")
print('---\n')

# mean predictions for non zero values
mean_predictions = []
for i in range(species_number):
    species_pred = predictions[:, :, i]
    valid_pred = species_pred[~np.isnan(species_pred)]
    mean_pred = valid_pred.mean() if valid_pred.size > 0 else np.nan
    mean_predictions.append(mean_pred)
print("Mean predicted probabilities per species:")
for i, mean_pred in enumerate(mean_predictions):
    print(f"Species {i+1}: {mean_pred:.6f}")

# convert zeros back to nans
predictions[predictions == 0] = np.nan

# plot the predictions for each species
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
print("Plotting predictions...")
for species_idx in range(20):
    vmax = np.nanmax(predictions[:, :, species_idx])
    print(f"Species {species_idx+1} vmax: {vmax}")
    vmin = np.nanmin(predictions[:, :, species_idx])
    print(f"Species {species_idx+1} vmin: {vmin}")
    plt.imshow(predictions[:, :, species_idx], cmap='viridis', 
            #norm=mcolors.LogNorm(vmin=vmin, vmax=vmax)#,
            vmax=vmax, vmin=0
               )#, interpolation='nearest')
    plt.title(f'Species {species_idx+1} Prediction')
    plt.colorbar(label='Relative Intensity')
    # max value empirical max
    plt.clim(0, np.nanmax(predictions[:, :, species_idx]))
    plt.show()
    #plt.savefig(os.path.join('output', f'species_{species_idx+1}_prediction.png'))
    plt.clf()
    # sum of predictions
    print(f"Species {species_idx+1} sum of predictions: {np.nansum(predictions[:, :, species_idx])}")