import pandas as pd
import os
import numpy as np
import pickle
from isdm.load_data import load_geoplant_processed

if __name__ == '__main__':
    split_dir = 'outputs/splits/france'
    results_dir = 'outputs/split_sweep'
    data_dir = 'data/processed/GeoPlant/france'
    data = load_geoplant_processed(data_dir)

    y = data.y_pa_train
    n_obs = len(y)
    n_species = len(data.species)
    print(f"n_obs_train: {n_obs}, n_species: {n_species}")

    results_po = pd.read_csv(os.path.join(results_dir, 'summary_po.csv'))
    results_pa = pd.read_csv(os.path.join(results_dir, 'summary_pa.csv'))
    results_popa = pd.read_csv(os.path.join(results_dir, 'summary_popa.csv'))

    split_df = pd.read_csv(os.path.join(split_dir, 'splits.csv'))

    logits_pa_close = np.zeros((n_obs, n_species))
    logits_pa_middle = np.zeros((n_obs, n_species))
    logits_pa_far = np.zeros((n_obs, n_species))

    # iterate split file
    for idx, row in split_df.iterrows():
        split_file = row['split_file']

        # read npz file
        split_data = np.load(os.path.join(split_dir, split_file), allow_pickle=True)

        print(split_data['test_idx'])

        # add the test_idx to the logit of the corresponding split
        exit()

        
