import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_latent_sequences(file_path):
    seq = np.load(file_path)
    print(f"[+] Loaded latent sequences: {seq.shape}")
    X = torch.tensor(seq, dtype=torch.float32)
    return X

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim) 

    def forward(self, x):
        out, _ = self.lstm(x)          
        out = self.fc(out[:, -1, :])  
        return out

def train_lstm(model, dataloader, num_epochs=20, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

def compute_anomaly_scores(model, X):
    scores = []
    with torch.no_grad():
        for i in range(len(X)):
            seq = X[i:i+1, :-1, :]  
            target = X[i:i+1, -1, :] 
            pred = model(seq)
            error = torch.mean((pred - target)**2).item()
            scores.append(error)
    return np.array(scores)

def run_lstm_pipeline():
    scores_dict = {}

    for file_path, model_name in [("dense_latent_seq.npy", "dense"), ("sparse_latent_seq.npy", "sparse")]:
        X = load_latent_sequences(file_path)
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        latent_dim = X.shape[2]

        print(f"\nTraining LSTM on {model_name.upper()} latents...")
        model = LSTMForecaster(input_dim=latent_dim, hidden_dim=64, num_layers=2)
        train_lstm(model, dataloader, num_epochs=30)

        model_file = f"lstm_forecaster_{model_name}.pth"
        torch.save(model.state_dict(), model_file)
        print(f"[+] LSTM model saved as {model_file}")

        scores = compute_anomaly_scores(model, X)
        scores_dict[model_name] = scores
        #np.save(f"lstm_anomaly_scores_{model_name}.npy", scores)
        #print(f"[+] Saved anomaly scores as lstm_anomaly_scores_{model_name}.npy")

    dense_scores = scores_dict["dense"]
    sparse_scores = scores_dict["sparse"]
    alpha = 0.6
    final_scores = alpha * dense_scores + (1 - alpha) * sparse_scores

    np.save("lstm_anomaly_scores_fused.npy", final_scores)
    print("\n[+] Saved fused anomaly scores as lstm_anomaly_scores_fused.npy")
