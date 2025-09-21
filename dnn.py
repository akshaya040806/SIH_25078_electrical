import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ===== AE (Autoencoder) Section ===== #
# (Insert your DenseAutoencoder and SparseAutoencoder classes + their train loop from your AE code.)

# Preprocessing and loading data
X_tensor, df, scaler = load_and_preprocess("filled_dates.xlsx")
input_dim = X_tensor.shape[1]
latent_dim = 8

# Train both autoencoders
dense_ae = DenseAutoencoder(input_dim, latent_dim)
train_ae(dense_ae, dataloader, num_epochs=30)
sparse_ae = SparseAutoencoder(input_dim, latent_dim)
train_ae(sparse_ae, dataloader, num_epochs=30, sparse=True)

# Encode data
dense_latents = dense_ae.encoder(X_tensor).detach().numpy()
sparse_latents = sparse_ae.encoder(X_tensor).detach().numpy()

# Build sequences for LSTM input (sliding window)
seq_len = 50
def build_sequences(latents, seq_len=50):
    sequences = []
    for i in range(len(latents) - seq_len + 1):
        sequences.append(latents[i:i+seq_len])
    return np.stack(sequences)

dense_seq = build_sequences(dense_latents, seq_len)
sparse_seq = build_sequences(sparse_latents, seq_len)

# Save sequences for LSTM processing
np.save("dense_latent_seq.npy", dense_seq)
np.save("sparse_latent_seq.npy", sparse_seq)

# ===== LSTM Forecaster Section ===== #
# (Insert your LSTMForecaster code and associated train and scoring logic from your LSTM code.)

# Load sequences and process
X_dense = torch.tensor(np.load("dense_latent_seq.npy"), dtype=torch.float32)
X_sparse = torch.tensor(np.load("sparse_latent_seq.npy"), dtype=torch.float32)
latent_dim = X_dense.shape[2]

# Prepare DataLoader
dense_ds = TensorDataset(X_dense)
dense_dl = DataLoader(dense_ds, batch_size=32, shuffle=True)
sparse_ds = TensorDataset(X_sparse)
sparse_dl = DataLoader(sparse_ds, batch_size=32, shuffle=True)

# Train LSTMs
dense_lstm = LSTMForecaster(input_dim=latent_dim)
train_lstm(dense_lstm, dense_dl)
sparse_lstm = LSTMForecaster(input_dim=latent_dim)
train_lstm(sparse_lstm, sparse_dl)

# Anomaly scoring on all sequences
dense_scores = compute_anomaly_scores(dense_lstm, X_dense)
sparse_scores = compute_anomaly_scores(sparse_lstm, X_sparse)
final_scores = 0.6 * dense_scores + 0.4 * sparse_scores
np.save("lstm_anomaly_scores_fused.npy", final_scores)
