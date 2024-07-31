from deepsets_kmer import RHPs_Dataset, DataLoader, parse_data
import numpy as np
import pandas as pd

def train_on_parents(library:np.ndarray, data:np.ndarray, parents:tuple, threshold = 0.99):
    reps, delta = parse_data(library, data)
    filtered = data[:, parents]
    filtered = np.sum(filtered, axis = 1)
    train_index = (filtered[:] > threshold)
    val_index = ~train_index

    X_train = reps[train_index, :]
    X_val = reps[val_index, :]
    y_train = delta[train_index,]
    y_val = delta[val_index,]
    
    return X_train, X_val, y_train, y_val

def train_on_random_parents(library:np.ndarray, data:np.ndarray, n_parents:int = 16, threshold = 0.90):
    _, N_parents = data.shape
    N_parents = N_parents - 1

    all_indices = np.arange(N_parents)
    np.random.shuffle(all_indices)
    parents = tuple(all_indices[:n_parents])

    return train_on_parents(library, data, parents, threshold)