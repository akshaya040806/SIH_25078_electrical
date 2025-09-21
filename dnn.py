# dnn.py

from AE_model import run_ae_pipeline
from lstm import run_lstm_pipeline

def main():
    print("[+] Running Autoencoder pipeline (AE_model.py)...")
    run_ae_pipeline()
    print("[+] Autoencoder pipeline complete.")

    print("[+] Running LSTM pipeline (lstm.py)...")
    run_lstm_pipeline()
    print("[+] LSTM pipeline complete.")

    print("[+] SCADA anomaly detection pipeline finished!")

if __name__ == "__main__":
    main()
