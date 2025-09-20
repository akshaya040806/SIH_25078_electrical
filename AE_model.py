import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. Load & Clean SCADA Data
# ===============================

def load_and_preprocess(file_path, time_column=None):

    # Detect file type
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    if ext == ".csv":
        df = pd.read_csv(file_path)

    print(f"Loaded {ext.upper()} file: {df.shape[0]} rows, {df.shape[1]} cols")

    # Drop rows that are completely empty
    before = df.shape[0]
    df = df.dropna(how="all").drop_duplicates()
    after = df.shape[0]
    if before != after:
        print(f"[!+] Dropped {before - after} completely-empty or duplicate rows")

    # Fill NaNs with column mean (fallback = 0 if column is all NaN)
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)

    # Sort by timestamp if provided
    if time_column and time_column in df.columns:
        df = df.sort_values(by=time_column)

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    if df.empty:
        raise ValueError("DataFrame is empty after cleaning. Check your file content.")

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # Convert to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)

    print(f"[+] Preprocessed data: {df.shape[0]} samples, {df.shape[1]} features")

    return X_tensor, df, scaler


# ===============================
# 2. Define Autoencoders
# ===============================
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DenseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sparsity_lambda=1e-3):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def sparsity_loss(self, z):
        return self.sparsity_lambda * torch.mean(torch.abs(z))

# ===============================
# 3. Training Loop
# ===============================
def train_ae(model, dataloader, num_epochs=20, lr=1e-3, sparse=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            x_recon, z = model(x)
            loss = criterion(x_recon, x)

            # Sparsity penalty
            if sparse and isinstance(model, SparseAutoencoder):
                loss += model.sparsity_loss(z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.6f}")


# ===============================
# 4. Sequence Builder for LSTM
# ===============================
def build_sequences(latents, seq_len=50):
    sequences = []
    for i in range(len(latents) - seq_len + 1):
        sequences.append(latents[i:i+seq_len])
    return np.stack(sequences)


# ===============================
# Main Function
# ===============================
if __name__ == "__main__":
    # --- Step 1: Load Excel Sheet ---
    file_path = r"C:\Users\Hafeezur Rahman A\OneDrive\Desktop\filled_dates.xlsx"   
    X_tensor, df, scaler = load_and_preprocess(file_path, time_column=None)

    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    input_dim = X_tensor.shape[1]
    latent_dim = 8

    # --- Step 2: Train Dense AE ---
    print("\nTraining Dense Autoencoder...")
    dense_ae = DenseAutoencoder(input_dim, latent_dim)
    train_ae(dense_ae, dataloader, num_epochs=30)

    # --- Step 3: Train Sparse AE ---
    print("\nTraining Sparse Autoencoder...")
    sparse_ae = SparseAutoencoder(input_dim, latent_dim)
    train_ae(sparse_ae, dataloader, num_epochs=30, sparse=True)

    # --- Step 4: Latents ---
    dense_latents = dense_ae.encoder(X_tensor).detach().numpy()
    sparse_latents = sparse_ae.encoder(X_tensor).detach().numpy()

    print("\nLatent Shapes:")
    print("Dense Latent Shape:", dense_latents.shape)
    print("Sparse Latent Shape:", sparse_latents.shape)

    # --- Step 5: Sequences for LSTM ---
    seq_len = 50
    dense_seq = build_sequences(dense_latents, seq_len=seq_len)
    sparse_seq = build_sequences(sparse_latents, seq_len=seq_len)

    print("\nSequence Shapes:")
    print("Dense Seq Shape:", dense_seq.shape)
    print("Sparse Seq Shape:", sparse_seq.shape)

    # --- Save results ---
    np.save("dense_latent_seq.npy", dense_seq)
    np.save("sparse_latent_seq.npy", sparse_seq)
    print("\n [+] Saved latent sequences for LSTM training as .npy files")
