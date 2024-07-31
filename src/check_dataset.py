from deepsets_kmer import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmer_transform import augment_library


#region load data
path = "../data/20240508_delta.csv"
library_path = "../data/3_1_24.csv"
k = 1 #specify kmer here

data = pd.read_csv(path, header = None).to_numpy()
library = pd.read_csv(library_path, header=None)
library = library.to_numpy()[:48, :]
library = library / np.sum(library, axis = 1, keepdims = True)
library = augment_library(library, k = k)
reps, delta = parse_data(library, data)

reps = reps[:, :, 0:1] * reps[:, :, 1:]
print(np.sum(reps, axis = 1))