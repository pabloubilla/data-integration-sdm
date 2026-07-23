from isdm.load_data import load_geoplant_processed
import os
import numpy as np

if __name__ == "__main__":


    data_path = "data/processed/GeoPlant/france"


    data = load_geoplant_processed(data_path)

    X_po = data.X_po
    y_po = data.y_po
    X_pa_train = data.X_pa_train
    y_pa_train = data.y_pa_train
    X_pa_test = data.X_pa_test
    y_pa_test = data.y_pa_test
    species = data.species
    covariates = data.covariates

    species_indexes = [i for i in range(len(species))]

    count_species_pa = np.zeros(len(species))
    count_species_po = np.zeros(len(species))

    # PA
    for obs_species in y_pa_train:
        count_species_pa[obs_species] += 1

    # PO
    for obs_species in y_po:
        count_species_po[obs_species] += 1

    # plot the counts
    import matplotlib.pyplot as plt

    # print sorted counts 
    sorted_indexes = np.argsort(count_species_pa)
    sorted_species = [species[i] for i in sorted_indexes]
    sorted_counts_pa = count_species_pa[sorted_indexes]
    sorted_counts_po = count_species_po[sorted_indexes]

    # print the first 10 species and their counts
    print("Top 10 species by PA count:")
    for i in range(200):
        print(f"{sorted_species[i]}: PA count = {sorted_counts_pa[i]}, PO count = {sorted_counts_po[i]}")
