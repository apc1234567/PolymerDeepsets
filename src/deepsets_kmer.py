import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    BatchNorm1d,
    AvgPool1d,
    Sigmoid,
    Conv1d,
)
from torch.utils.data import DataLoader
import pytorch_lightning as pl # type: ignore
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import math

#region Architecture

class Phi(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 32,
                 output_dim: int = 10,
                 n_layers: int = 2,
                 transform = None):
        super().__init__()
        self.input_dim = input_dim - 1 #remove blending coefficients
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.transform = transform

        layers = [nn.Linear(input_dim - 1, hidden_dim),
            nn.ReLU()]
        for i in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.ReLU())

        self.phi = nn.Sequential(*layers)


    def forward(self, P):
        '''
        takes in B x N x (M + 1) tensor, where B is the batch dimension, 
        N is the number of polymers in the blend, and
        M is the number of monomers

        Outputs B x N x self.output_dim tensor
        '''
        c = P[:, :, 0:1]
        P_ = P[:, :, 1:]
        if self.transform:
            P_ = self.transform(P_)
        P_r = self.phi(P_)
        return P_r * c


class Rho(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 32,
                 n_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers = [nn.Linear(input_dim, hidden_dim),
            nn.ReLU()]
        for i in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1)) #scalar output

        self.rho = nn.Sequential(*layers)

    def forward(self, x):
        x = self.rho(x)
        return x

class Pool(nn.Module):
    def __init__(self, pool_func = (lambda x: x.sum(dim = -2))):
        super().__init__()
        self.pool_func = pool_func

    def forward(self, x):
        return self.pool_func(x)

class LitDeepSets(pl.LightningModule):
    def __init__(self, phi, rho, pool = None, lr = 2e-3):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.pool = pool if pool else lambda x: x.sum(dim = -2) #default is summation
        self.train_loss = []
        self.val_loss = []
        self.lr = lr

    def set_lr(self, lr):
        self.lr = lr

    def forward(self, x):
        x_r = self.pool(self.phi(x))
        y_hat = self.rho(x_r)
        return y_hat

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = torch.squeeze(self.forward(x))
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.train_loss.append(loss.item())
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self.forward(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.squeeze(self.forward(x))
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_loss.append(loss.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
#endregion

#region Data Handling

def convert_vector_to_representation(vector, library, output_shape = None):
    '''
    vector is N_parents dimensional blend vector
    library is N_parents x N_monomers array of parent polymers
    output_shape is the shape of the blend representation
    '''
    N_parents, = vector.shape
    if N_parents != library.shape[0]:
        raise ValueError('library is of incorrect size')

    filter = np.where(vector[:, np.newaxis] > 1e-6, 1, 0)
    all_polymers = library * filter
    all_polymers = np.concatenate((vector[:, np.newaxis], all_polymers), axis = 1)
    rep = all_polymers[~np.all(all_polymers == 0, axis=1)]

    if output_shape:
        n, m = rep.shape
        if n > output_shape[0]:
            raise ValueError('blend is larger than specified output shape')
        padded_rep = np.zeros(output_shape)
        padded_rep[:n,:m] = rep
        return padded_rep
    return rep

def parse_data(library, data):
    """
    library is N_parents x N_monomers array of parent polymers (48 x 3)
    data is N_samples x (N_parents + 1) array of blending data, with turbidities concatenated
    as the last column
    """
    blends = data[:,:-1]
    turbidities = data[:,-1]

    N_parents, N_monomers = library.shape
    where = np.where(blends > 10**-6, 1, 0)
    _ = np.sum(where, axis = 1)
    blend_capacity = np.max(_)
    output_shape = (blend_capacity, N_monomers + 1) #increase output dimension by 1 to fit blend coefficients

    reps = [
        convert_vector_to_representation(blends[i], library, output_shape)
        for i in range(blends.shape[0])
    ]

    return np.array(reps), turbidities

class RHPs_Dataset(Dataset):
    def __init__(self, polymers, activities):
        self.sets = polymers
        self.activity = activities

    def __len__(self):
        return self.sets.shape[0]

    def __getitem__(self, idx):
        return self.sets[idx], self.activity[idx]
#endregion