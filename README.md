Code explanation for the AE model

a)  First segment - Loading, cleaning and pre-processing the SCADA data we have in excel/CSV data

- First, I asked if we are dealing with Excel/CSV data and once that is sorted, I have taken the DataFrames (rows and columns according to Pandas language)

df[0] - indicates the number of rows
df[1] - indicates the number of columns

def load_and_preprocess(file_path, time_column=None):

```bash
    # Detect file type
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    if ext == ".csv":
        df = pd.read_csv(file_path)

    print(f"Loaded {ext.upper()} file: {df.shape[0]} rows, {df.shape[1]} cols")
```

No what we would do is that we can clean the data that we got using 5 parameters. How?

a) We can first drop all the rows which have purely empty values which are of no use
b) Then what we do is that we are checking if any column is having some parameters as empty, we can fill it with NaN.
c) If there are timestamp (we obviously have), those are sorted and taken care of.
d) If the entire excel sheet is empty, then the error handling should be done.

then, we go for the normalisation and pre-processing

```bash
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # Convert to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)

    print(f"[+] Preprocessed data: {df.shape[0]} samples, {df.shape[1]} features")

    return X_tensor, df, scaler
```

Why are we doing this

Since we have went with Torch, its featuring is in such a way that it can only work with single array based tensor values (cannot work with NumPy based arrays or pandas Data Frames)

So we have used this approach

- We have scaled all the values to a common point (obv normalisation meaning)
- Then we have converted the values to Pytorch tensors for AE
- we have asked to return it for later use as the program progresses.


b)  Second segment - Defining the Auto Encoders 

There are always two blocks of defining auto encoders

 - Encoding Process is the block where you take the input data and make it to 64 neurons (I have given for lessen the drift) and reconstruct it and then make it a latent 

- Decoding Process is the block where you can replicate the data given by basically doing reverse engineering.

Now, for the best efficiency for our project, I have included two types of AE here

- Normal Dense AE (used usually) is the normal AE used when there are usual scenarios of anomaly (like obvious anomalies) can be detected easily by his block of AE.

- Sparse AE (added for sharp or weird anomalies) is used to detect sharp, rare anomalies in SCADA data that dense AE might smooth over. and also produces sparser latent features, which are easier for LSTM to learn sequences from.

I have added the sparse constant as 0.001 for little drift as well for the reconstruction (for LSTM better working)

c)  Third Segment - Training the AE model that we got

First before that, we will see the two aspects important that I have added here for training

- Epochs is the number of times our AE will run for the reconstruction efficiency (the more we run, other better we will get good results)

Generally, using 20-40 epochs are better, I have added 30.

- Loss is the parameter that will assess that tells us how well the AE is reconstructing the input.

Low loss → AE reconstructs well → latent vector captures normal patterns.
High loss → AE cannot reconstruct → input is unusual or AE hasn’t learned enough.

We have got 0.6 and by the end that is in the 30th epoch, we got <0.35, which is very good for us because generally it should be less than 0.35 for a better output.

Then both dense and the sparse AE have worked as how we want it to.

Then we go for the loop, and the  each loop, we have the total loss.
Then we split it as batches for storage and speed.
Then it extracts the actual feature tensor from the batch (PyTorch wraps it in a tuple).

Then the encoder and the decoder is called for reconstruction
Then the loss is also taken and calculated.

Then we do the back propagation and weight update

clears old gradients
computes gradients of loss w.r.t. model parameters
updates AE weights using gradients

This learning step gradually improves AE’s reconstruction with each epochs.

Then the batch epoch is taken and yeah we are done yay.

d)  Fourth Segment - LSTM training 

Here I have just told two things 

- Latent shapes is basically how much features I have compressed for our knowledge.

Latent space is what LSTM will use instead of the raw 13 features.

- Sequence Shape 

I have given a sequence length 50, meaning each LSTM input will look at 50 steps.

8700 - number of sequences generated from 8749 samples
50 - number of time steps per sequence
8 - latent features per time step (that after the compression)

Why 8700 and not 8749?

With seq_len=50, 
the first sequence uses samples [0:50], second [1:51] like that.
Again for efficiency and for speed.

Total sequences = 8749 - 50 + 1 = 8700.

How LSTM sees it is that

Each sequence is a matrix of 50x8: 50 time steps, each with 8 latent features.
LSTM will learn the patterns over these 50 steps.

That is why (8700, 50, 8).

Fifth Segment - Bringing all of them together

```bash
if __name__ == "__main__":
    # --- Step 1: Load Data ---
    file_path = r"C:\Users\Hafeezur Rahman A\OneDrive\Desktop\filled_dates.xlsx"   # or .csv
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

    # --- Step 4: Extract Latents ---
    dense_latents = dense_ae.encoder(X_tensor).detach().numpy()
    sparse_latents = sparse_ae.encoder(X_tensor).detach().numpy()

    print("\nLatent Shapes:")
    print("Dense Latent Shape:", dense_latents.shape)
    print("Sparse Latent Shape:", sparse_latents.shape)

    # --- Step 5: Build Sequences for LSTM ---
    seq_len = 50
    dense_seq = build_sequences(dense_latents, seq_len=seq_len)
    sparse_seq = build_sequences(sparse_latents, seq_len=seq_len)

    print("\nSequence Shapes:")
    print("Dense Seq Shape:", dense_seq.shape)
    print("Sparse Seq Shape:", sparse_seq.shape)

    # --- Save results ---
    np.save("dense_latent_seq.npy", dense_seq)
    np.save("sparse_latent_seq.npy", sparse_seq)
    print("\n [+] Saved latent sequences for LSTM training!")
```








