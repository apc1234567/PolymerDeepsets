import deepsets
import pandas as pd
import numpy as np

path = "DATA PATH HERE"
data = pd.read_csv(path, header = None).to_numpy()
library = pd.read_csv("LIBRARY PATH HERE", header=None)
library = library.to_numpy()[:48, :]
library = library / np.sum(library, axis = 1, keepdims = True)

reps, delta = parse_data(library, data)

X_train, X_val, y_train, y_val = train_test_split(
        reps, delta, test_size=0.33, random_state=42
    )

train_dataset = RHPs_Dataset(X_train.astype(np.float32), y_train.astype(np.float32))
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = RHPs_Dataset(X_val.astype(np.float32), y_val.astype(np.float32))
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

N_monomers = 3
embed_dim = 6
lr = 10e-3

phi = Phi(N_monomers,embed_dim)
rho = Rho(embed_dim)

deepsets = LitDeepSets(
    phi,
    rho,
    lr = lr
)

max_epochs = 200

trainer = pl.Trainer(max_epochs = max_epochs, log_every_n_steps = 4, check_val_every_n_epoch=10)
trainer.fit(model=deepsets, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)