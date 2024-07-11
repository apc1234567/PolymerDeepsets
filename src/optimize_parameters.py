from deepsets_test import *
from kmer_transform import transform_2mers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna

# fix the random seed
pl.seed_everything(0)

data_path = "../data/pairwise_delta.csv"
data = pd.read_csv(data_path, header = None).to_numpy()

library_path = "../data/3_1_24.csv"
library = pd.read_csv(library_path, header=None)
library = library.to_numpy()[:48, :]
library = library / np.sum(library, axis = 1, keepdims = True)

reps, delta = parse_data(library, data)
delta = -1* delta #fix sign error in pairwise dataset
_, blend_capacity, rep_dim = reps.shape

X_train, X_val, y_train, y_val = train_test_split(
       reps, delta, test_size=0.33, random_state=42
   )
train_dataset = RHPs_Dataset(X_train.astype(np.float32), y_train.astype(np.float32))
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = RHPs_Dataset(X_val.astype(np.float32), y_val.astype(np.float32))
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)


lr = 1e-3
max_epochs = 200

def create_model(trial):
    phi_hidden_dim = trial.suggest_int('phi_hidden_dim', 32, 64)
    phi_layers = trial.suggest_int('phi_layers', 2, 4)
    embed_dim = trial.suggest_int('embed_dim', 3, 10)

    rho_hidden_dim = trial.suggest_int('rho_hidden_dim', 32, 64)
    rho_layers = trial.suggest_int('rho_layers', 2, 4)
    
    phi = Phi(input_dim=rep_dim, hidden_dim=phi_hidden_dim, output_dim=embed_dim, n_layers=phi_layers)
    rho = Rho(input_dim=embed_dim, hidden_dim=rho_hidden_dim, n_layers=rho_layers)

    deepsets = LitDeepSets(
        phi,
        rho,
        lr = lr
    )
    return deepsets

def objective(trial):
    
    deepsets = create_model(trial)
    trainer = pl.Trainer(
        max_epochs = max_epochs,
        log_every_n_steps = 4,
        check_val_every_n_epoch=10,
        #devices = 1 #some bug with using multiple devices
    )
    trainer.fit(model=deepsets, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)

    pred = deepsets(torch.from_numpy(X_val.astype(np.float32)))
    pred = np.squeeze(pred.detach().numpy())

    MSE = np.average((y_val - pred)**2)
    return MSE

sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(
    direction='minimize',
    sampler=sampler
)
study.optimize(objective, n_trials=10)