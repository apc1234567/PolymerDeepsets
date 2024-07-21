from deepsets_kmer import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmer_transform import augment_library


#region load data
path = "../data/20240528_delta.csv"
library_path = "../data/3_1_24.csv"

data = pd.read_csv(path, header = None).to_numpy()
library = pd.read_csv(library_path, header=None)
library = library.to_numpy()[:48, :]
library = library / np.sum(library, axis = 1, keepdims = True)
reps, delta = parse_data(augment_library(library, k = 2), data) #specify kmer here
#endregion

#region load model
_, blend_capacity, rep_dim = reps.shape
embed_dim = 7

phi = Phi(input_dim=rep_dim,
          hidden_dim=32,
          output_dim=embed_dim,
          n_layers=4)
rho = Rho(input_dim=embed_dim,
          hidden_dim=28,
          n_layers=3)
check_path = "logs/2024-07-20 21:12:38.208586/lightning_logs/version_0/checkpoints/epoch=499-step=6000.ckpt"
deepsets = LitDeepSets.load_from_checkpoint(check_path, phi = phi, rho = rho)
deepsets = deepsets.to('cpu')
#endregion

#region test model
pred = deepsets(torch.from_numpy(reps.astype(np.float32)))
pred = np.squeeze(pred.detach().numpy())

# Determine the range for the diagonal line
min_val = min(min(delta), min(pred))
max_val = max(max(delta), max(pred))

plt.figure(figsize=(6,6))  # Make the figure square
plt.scatter(delta, pred)
plt.plot([min_val, max_val], [min_val, max_val], 'r')  # Diagonal line
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.axis('equal')  # Set the same scale for both axes

plt.show()
plt.savefig(check_path + 'val_20240508_2mer.png')

from sklearn.metrics import r2_score
print(f'**r^2 = {r2_score(delta, pred)}**')
print(f'**MSE = {np.average((delta - pred)**2)}')

#endregion