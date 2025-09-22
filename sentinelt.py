import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from prometheus_client import start_http_server, Gauge

# ==== Prometheus metrics ====
ae_loss_gauge = Gauge('ae_loss', 'Average AE loss per epoch', ['epoch'])
lstm_loss_gauge = Gauge('lstm_loss', 'Average LSTM loss per epoch', ['epoch'])

# ==== 1. Data Preprocessing ====
def load_and_preprocess(file_path, time_column=None):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("File type not supported!")

    df = df.dropna(how="all").drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)
    if time_column and time_column in df.columns:
        df = df.sort_values(by=time_column)
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning. Check your file content.")
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor

# ==== 2. AE Model Definitions ====
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sparsity_lambda=1e-3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim), nn.Sigmoid()
        )
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def sparsity_loss(self, z):
        return self.sparsity_lambda * torch.mean(torch.abs(z))

def train_ae(model, dataloader, num_epochs=20, lr=1e-3, sparse=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            x_recon, z = model(x)
            loss = criterion(x_recon, x)
            if sparse and isinstance(model, SparseAutoencoder):
                loss += model.sparsity_loss(z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"AE Epoch {epoch+1}, Loss: {avg_loss:.6f}")
        epoch_losses.append(avg_loss)
    return epoch_losses

def build_sequences(latents, seq_len=50):
    sequences = []
    for i in range(len(latents) - seq_len + 1):
        sequences.append(latents[i:i+seq_len])
    return np.stack(sequences)

# ==== 3. LSTM Model Definition ====
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm(model, dataloader, num_epochs=20, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            seq_batch = batch[0]
            target = seq_batch[:, -1, :]
            pred = model(seq_batch[:, :-1, :])
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"LSTM Epoch {epoch+1}, Loss: {avg_loss:.6f}")
        epoch_losses.append(avg_loss)
    return epoch_losses

# ==== 4. Integrated Pipeline and Prometheus Metrics Exposure ====
def main():
    print("[+] Starting Prometheus exporter on port 8000...")
    start_http_server(8000)

    # --- AE Pipeline ---
    print("[+] Loading and preprocessing data...")
    file_path = r"C:\Users\annie\Downloads\filled_dates.xlsx"
    X_tensor = load_and_preprocess(file_path)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_dim = X_tensor.shape[1]
    latent_dim = 8

    print("[+] Training Dense AE...")
    dense_ae = DenseAutoencoder(input_dim, latent_dim)
    dense_ae_losses = train_ae(dense_ae, dataloader, num_epochs=30)

    print("[+] Training Sparse AE...")
    sparse_ae = SparseAutoencoder(input_dim, latent_dim)
    sparse_ae_losses = train_ae(sparse_ae, dataloader, num_epochs=30, sparse=True)

    # --- AE Latents and Sequence Creation ---
    dense_latents = dense_ae.encoder(X_tensor).detach().numpy()
    sparse_latents = sparse_ae.encoder(X_tensor).detach().numpy()
    seq_len = 50
    dense_seq = build_sequences(dense_latents, seq_len=seq_len)
    sparse_seq = build_sequences(sparse_latents, seq_len=seq_len)
    np.save("dense_latent_seq.npy", dense_seq)
    np.save("sparse_latent_seq.npy", sparse_seq)

    # --- Prometheus AE loss metrics ---
    ae_epochs = min(len(dense_ae_losses), len(sparse_ae_losses))
    for epoch in range(ae_epochs):
        avg_ae = (dense_ae_losses[epoch] + sparse_ae_losses[epoch]) / 2
        ae_loss_gauge.labels(epoch=str(epoch + 1)).set(avg_ae)
        print(f"Prometheus - AE Epoch {epoch+1}, Avg Loss: {avg_ae:.6f}")

    # --- LSTM Pipeline ---
    print("[+] Preparing LSTM training...")
    dense_seq = np.load("dense_latent_seq.npy")
    sparse_seq = np.load("sparse_latent_seq.npy")
    dense_X = torch.tensor(dense_seq, dtype=torch.float32)
    sparse_X = torch.tensor(sparse_seq, dtype=torch.float32)
    dense_dataset = TensorDataset(dense_X)
    sparse_dataset = TensorDataset(sparse_X)
    dense_dl = DataLoader(dense_dataset, batch_size=32, shuffle=True)
    sparse_dl = DataLoader(sparse_dataset, batch_size=32, shuffle=True)
    latent_dim = dense_X.shape[2]

    print("[+] Training Dense LSTM...")
    dense_lstm = LSTMForecaster(input_dim=latent_dim, hidden_dim=64, num_layers=2)
    dense_lstm_losses = train_lstm(dense_lstm, dense_dl, num_epochs=30)

    print("[+] Training Sparse LSTM...")
    sparse_lstm = LSTMForecaster(input_dim=latent_dim, hidden_dim=64, num_layers=2)
    sparse_lstm_losses = train_lstm(sparse_lstm, sparse_dl, num_epochs=30)

    # --- Prometheus LSTM loss metrics ---
    lstm_epochs = min(len(dense_lstm_losses), len(sparse_lstm_losses))
    for epoch in range(lstm_epochs):
        avg_lstm = (dense_lstm_losses[epoch] + sparse_lstm_losses[epoch]) / 2
        lstm_loss_gauge.labels(epoch=str(epoch + 1)).set(avg_lstm)
        print(f"Prometheus - LSTM Epoch {epoch+1}, Avg Loss: {avg_lstm:.6f}")

    print("[+] All pipelines complete. Prometheus metrics ready for Grafana!")

if __name__ == "__main__":
    main()
