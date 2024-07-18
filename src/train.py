from deepsets_test import *
from kmer_transform import transform_2mers
from make_split import train_on_parents, train_on_random_parents
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fix the random seed
pl.seed_everything(0)
np.random.seed(0)

#region load data
path = "../data/pairwise_delta.csv"
library_path = "../data/3_1_24.csv"
data = pd.read_csv(path, header = None).to_numpy()

library = pd.read_csv(library_path, header=None)
library = library.to_numpy()[:48, :]
library = library / np.sum(library, axis = 1, keepdims = True)

reps, delta = parse_data(library, data)
#endregion


#region make split

#random split
# X_train, X_val, y_train, y_val = train_test_split(
#        reps, delta, test_size=0.33, random_state=42
#    )

#randomly choose 16 parents
X_train, X_val, y_train, y_val = train_on_random_parents(library, data, n_parents=16)

#endregion


#region make model
train_dataset = RHPs_Dataset(X_train.astype(np.float32), y_train.astype(np.float32))
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = RHPs_Dataset(X_val.astype(np.float32), y_val.astype(np.float32))
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

_, blend_capacity, rep_dim = reps.shape
embed_dim = 7

phi = Phi(input_dim=rep_dim,
          hidden_dim=32,
          output_dim=embed_dim,
          n_layers=4)
rho = Rho(input_dim=embed_dim,
          hidden_dim=28,
          n_layers=3)

deepsets = LitDeepSets(
    phi,
    rho,
)
#endregion
#region train model
max_epochs = 1000
lr = 1e-2
deepsets.set_lr(lr)

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
plt.savefig('actual_vs_predicted2.png')

from sklearn.metrics import r2_score
print(f'**r^2 = {r2_score(y_val, pred)}**')
print(f'**MSE = {np.average((y_val - pred)**2)}')
#endregion