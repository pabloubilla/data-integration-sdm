import random
import numpy as np
import torch


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    # return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_overlapping_species_subset(
    y_train: list[list[int]],
    y_test: list[list[int]],
    num_classes: int,
) -> tuple[list[list[int]], list[list[int]], np.ndarray, int]:
    """
    Restrict train and test to species that appear in both splits.
    
    Returns:
        y_train_local:  y_train remapped to local species indices
        y_test_local:   y_test remapped to local species indices
        global_species: array of original species indices (length = new num_classes)
        local_num_classes: number of overlapping species
    
    Example:
        global species 0..999, but only species [3, 7, 42] appear in both
        train/test → remapped to local indices [0, 1, 2]
        global_species[1] == 7  (local index 1 → global index 7)
    """
    train_species = set(s for obs in y_train for s in obs)
    test_species  = set(s for obs in y_test  for s in obs)
    overlap       = sorted(train_species & test_species)

    if not overlap:
        raise ValueError("Train and test share no species — cannot evaluate this split.")

    global_to_local = {g: l for l, g in enumerate(overlap)}
    global_species  = np.array(overlap, dtype=np.int64)

    def remap(y: list[list[int]]) -> list[list[int]]:
        return [
            [global_to_local[s] for s in obs if s in global_to_local]
            for obs in y
        ]

    return remap(y_train), remap(y_test), global_species, len(overlap)


def filter_and_remap(species_list, y):
    species_set = set(species_list)
    remap = {s: i for i, s in enumerate(species_list)}
    return [[remap[s] for s in obs if s in species_set] for obs in y]