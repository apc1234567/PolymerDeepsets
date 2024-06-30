from deepsets import *
from kmer_transform import transform_2mers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fix the random seed
pl.seed_everything(0)

path = "../data/pairwise_delta.csv"
library_path = "../data/3_1_24.csv"
data = pd.read_csv(path, header = None).to_numpy()
# remove x_45 != 0 and x_39 !=0  as the validation set



train_index = data[:, 32] != 0
val_index = data[:, 32] == 0 
#train_index = train_index & (data[:, 30] != 0)
#val_index = val_index & (data[:, 30] == 0)


library = pd.read_csv(library_path, header=None)
library = library.to_numpy()[:48, :]
library = library / np.sum(library, axis = 1, keepdims = True)
library_2mers = transform_2mers(library)

reps, delta = parse_data(library, data)
delta = -1* delta #fix sign error in pairwise dataset

#X_train, X_val, y_train, y_val = train_test_split(
#        reps, delta, test_size=0.33, random_state=42
#    )

X_train = reps[train_index]
y_train = delta[train_index]
X_val = reps[val_index]
y_val = delta[val_index]



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


pred = deepsets(torch.from_numpy(X_val.astype(np.float32)))
pred = np.squeeze(pred.detach().numpy())

plt.scatter(y_val, pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
# save the plot
plt.savefig('actual_vs_predicted.png')

# Determine the range for the diagonal line
min_val = min(min(y_val), min(pred))
max_val = max(max(y_val), max(pred))

plt.figure(figsize=(6,6))  # Make the figure square
plt.scatter(y_val, pred)
plt.plot([-0.15, 0.15], [-0.15, 0.15], 'r')  # Diagonal line
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.axis('equal')  # Set the same scale for both axes

# Optionally, set the limits explicitly if you want to ensure they are exactly the same
plt.xlim(-0.15, 0.15)
plt.ylim(-0.15, 0.15)

plt.show()
# Save the plot after displaying it
plt.savefig('actual_vs_predicted.png')

from sklearn.metrics import r2_score
print(f'**r^2 = {r2_score(y_val, pred)}**')
print(f'**MSE = {np.average((y_val - pred)**2)}')
