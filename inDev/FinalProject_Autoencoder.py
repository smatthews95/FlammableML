'''
Autoencoder flammability parameter implementation developed by

Huynh Vu
The University of Texas
December 2023
'''

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

FL_url1 = 'https://raw.githubusercontent.com/smatthews95/FlammableML/main/10000_Samples-2023-11-26.csv'
DFLimitsData =pd.read_csv(FL_url1, index_col=0) # Read in limits only dataset

# Drop empty columns
Y = DFLimitsData.drop(['Xf_LFL', 'Xf_UFL', 'Xf_stoich'], axis=1)
#%%

X = Y.copy(deep=True)
#LFL = DFLimitsData['LFL']
#UFL = DFLimitsData['UFL']
#FL = DFLimitsData[['LFL','UFL']]
Su = X[['Su_stoich','LFL','UFL']]

X_train_target, X_val_test_target, Su_train, Su_val_test = train_test_split(X, Su, test_size = 0.2)
X_test_target,X_val_target, Su_test, Su_val = train_test_split(X_val_test_target, Su_val_test, test_size=0.2)

X_train = X_train_target.drop(['Su_stoich','LFL', 'UFL'], axis=1)
X_test = X_test_target.drop(['Su_stoich','LFL', 'UFL'], axis=1)
X_val = X_val_target.drop(['Su_stoich','LFL', 'UFL'], axis=1)
#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Assuming X_train, X_val, X_test are your feature sets

scalers = {}
for i in range(X_train.shape[1]):  # For each column in X
    scalers[i] = StandardScaler()
    X_train.iloc[:, i] = scalers[i].fit_transform(X_train.iloc[:, i].to_numpy().reshape(-1, 1)).ravel()
    X_val.iloc[:, i] = scalers[i].transform(X_val.iloc[:, i].to_numpy().reshape(-1, 1)).ravel()
    X_test.iloc[:, i] = scalers[i].transform(X_test.iloc[:, i].to_numpy().reshape(-1, 1)).ravel()



target_scalers = {}
for i in range(Su_train.shape[1]):  # For each column in y
    target_scalers[i] = StandardScaler()
    Su_train.iloc[:, i] = target_scalers[i].fit_transform(Su_train.iloc[:, i].to_numpy().reshape(-1, 1)).ravel()
    Su_val.iloc[:, i] = target_scalers[i].transform(Su_val.iloc[:, i].to_numpy().reshape(-1, 1)).ravel()
    Su_test.iloc[:, i] = target_scalers[i].transform(Su_test.iloc[:, i].to_numpy().reshape(-1, 1)).ravel()
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Assuming X_train, X_val, X_test, Su_train, Su_val, Su_test are pandas DataFrames

# Convert standardized data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train.values.astype(np.float32))
y_train_tensor = torch.from_numpy(Su_train.values.astype(np.float32))
X_val_tensor = torch.from_numpy(X_val.values.astype(np.float32))
y_val_tensor = torch.from_numpy(Su_val.values.astype(np.float32))

# Rest of the code for defining and training the autoencoder remains the same

#%%
import torch.nn.functional as F

class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim, output_dim):
        super(SupervisedAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),  # Add a hidden layer with 256 units
            nn.ReLU(True),
            nn.Linear(256, 128),  # Add another hidden layer with 128 units
            nn.ReLU(True),
            nn.Linear(128, 64),  # Add another hidden layer with 64 units
            nn.ReLU(True),
            nn.Linear(64, encoding_dim),  # Add another hidden layer with encoding_dim units
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),  # Add a hidden layer with 64 units
            nn.ReLU(True),
            nn.Linear(64, 128),  # Add another hidden layer with 128 units
            nn.ReLU(True),
            nn.Linear(128, 256),  # Add another hidden layer with 256 units
            nn.ReLU(True),
            nn.Linear(256, input_size),  # Add another hidden layer with input_size units
            nn.Sigmoid()
        )
        # Supervised Output
        self.supervised = nn.Sequential(
            nn.Linear(encoding_dim, 32),  # Add a hidden layer with 32 units
            nn.ReLU(True),
            nn.Linear(32, output_dim)  # Output layer
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        supervised_output = self.supervised(encoded)
        return decoded, supervised_output

# Make sure to set the output_dim to 3 (or the number of features in your target variable)


# Instantiate the model
encoding_dim = 32  # This is a hyperparameter you can tune
output_dim = 3  # Adjust this to match the number of features in your target variable
autoencoder = SupervisedAutoencoder(input_size=X_train.shape[1], encoding_dim=encoding_dim, output_dim=output_dim)

# Use the same optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Define loss functions
reconstruction_loss_fn = nn.MSELoss()
supervised_loss_fn = nn.MSELoss()  # Or another appropriate loss function for your task

# Store the losses
train_losses = []
val_losses = []

# Training loop
for epoch in range(300):
    # Training
    autoencoder.train()
    optimizer.zero_grad()
    reconstructed, supervised_output = autoencoder(X_train_tensor)
    reconstruction_loss = reconstruction_loss_fn(reconstructed, X_train_tensor)
    supervised_loss = supervised_loss_fn(supervised_output, y_train_tensor)
    # Combine losses, if necessary you can weigh them differently
    loss = reconstruction_loss + supervised_loss
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Validation
    autoencoder.eval()
    with torch.no_grad():
        reconstructed_val, supervised_output_val = autoencoder(X_val_tensor)
        val_reconstruction_loss = reconstruction_loss_fn(reconstructed_val, X_val_tensor)
        val_supervised_loss = supervised_loss_fn(supervised_output_val, y_val_tensor)
        val_loss = val_reconstruction_loss + val_supervised_loss
        val_losses.append(val_loss.item())
#%%
# Plotting both training and validation losses
plt.figure()
plt.semilogy(train_losses, label='Training Loss')
plt.semilogy(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - Supervised Autoencoder')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()
#%%
X_test_tensor = torch.from_numpy(X_test.values.astype(np.float32))

autoencoder.eval()
with torch.no_grad():
    _, y_pred_test = autoencoder(X_test_tensor)


# Assuming the targets were scaled, reverse the scaling
y_pred_test_np = y_pred_test.numpy()
y_pred_test_unscaled = np.zeros_like(y_pred_test_np)
for i in range(y_pred_test_np.shape[1]):
    y_pred_test_unscaled[:, i] = target_scalers[i].inverse_transform(y_pred_test_np[:, i].reshape(-1, 1)).ravel()

# Also reverse scaling for the true values
y_test_unscaled = np.zeros_like(Su_test.values)
for i in range(Su_test.shape[1]):
    y_test_unscaled[:, i] = target_scalers[i].inverse_transform(Su_test.values[:, i].reshape(-1, 1)).ravel()
#%%

plt.figure(figsize=(15, 5))
for i in range(Su_test.shape[1]):
    plt.subplot(1, 3, i + 1)
    plt.scatter(y_test_unscaled[:, i], y_pred_test_unscaled[:, i], alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Feature {i+1}')
    plt.grid(True)
plt.tight_layout()
plt.show()
