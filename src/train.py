from deepsets import *
from kmer_transform import transform_2mers
import pandas as pd
import numpy as np

path = "../data/pairwise_delta.csv"
library_path = "../data/3_1_24.csv"
data = pd.read_csv(path, header = None).to_numpy()
library = pd.read_csv(library_path, header=None)
library = library.to_numpy()[:48, :]
library = library / np.sum(library, axis = 1, keepdims = True)
library_2mers = transform_2mers(library)

reps, delta = parse_data(library, data)
delta = -1* delta #fix sign error in pairwise dataset

X_train, X_val, y_train, y_val = train_test_split(
        reps, delta, test_size=0.33, random_state=42
    )

train_dataset = RHPs_Dataset(X_train.astype(np.float32), y_train.astype(np.float32))
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = RHPs_Dataset(X_val.astype(np.float32), y_val.astype(np.float32))
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

_, blend_capacity, rep_dim = reps.shape
embed_dim = 6
lr = 10e-3

phi = Phi(rep_dim,embed_dim)
rho = Rho(embed_dim)

deepsets = LitDeepSets(
    phi,
    rho,
    lr = lr
)

max_epochs = 200

trainer = pl.Trainer(max_epochs = max_epochs, log_every_n_steps = 4, check_val_every_n_epoch=10)
trainer.fit(model=deepsets, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)


import matplotlib.pyplot as plt
pred = deepsets(torch.from_numpy(X_val.astype(np.float32)))
pred = np.squeeze(pred.detach().numpy())

plt.scatter(y_val, pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

from sklearn.metrics import r2_score
print(f'**r^2 = {r2_score(y_val, pred)}**')
print(f'**MSE = {np.average((y_val - pred)**2)}')