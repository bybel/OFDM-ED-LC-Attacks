import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration ---
N_FFT = 128
OBSERVATION_RATIO = 0.5
SNR_DB = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30 # Increased for convergence
N_SAMPLES_TRAIN = 8000 # Increased training data
N_SAMPLES_TEST = 2000
MODULATIONS = ['QPSK', '64QAM', '256QAM'] # Testing a range

device = torch.device('cpu') 

# --- Optimized Model with Low-Rank Factorization ---
class OptimizedCNNReceiver(nn.Module):
    def __init__(self, input_len, output_dim, n_filters=32, kernel_size=5, rank=256):
        super().__init__()
        
        # 1. 2-Channel Input (Real, Imag)
        # Input shape: [Batch, 2, Length]
        self.conv_net = nn.Sequential(
            nn.Conv1d(2, n_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters),
            
            nn.Conv1d(n_filters, n_filters*2, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters*2),
            
            nn.Flatten()
        )
        
        # Flatten Dim = Length * (Filters*2)
        # e.g., 140 * 64 = 8960
        self.flatten_dim = input_len * (n_filters*2)
        
        # 2. Low-Rank Factorization Head
        # Instead of Huge Linear(8960 -> 32768), we do:
        # Linear(8960 -> Rank) -> ReLU -> Linear(Rank -> 32768)
        self.fc_bottleneck = nn.Sequential(
            nn.Linear(self.flatten_dim, rank),
            nn.ReLU(),
            nn.BatchNorm1d(rank),
            nn.Linear(rank, output_dim)
        )

    def forward(self, x):
        # x input is [Batch, 2, L] directly
        feat = self.conv_net(x)
        return self.fc_bottleneck(feat)

# --- Data Generation (Updated for 2-channel) ---
def get_mod_config(mod_name):
    if mod_name == 'BPSK':
        return np.array([-1, 1], dtype=complex), np.array([0, 1])
    elif mod_name == 'QPSK':
        return np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex), np.array([0, 1, 3, 2])
    elif 'QAM' in mod_name:
        M = int(mod_name.replace('QAM', ''))
        k = int(np.log2(M))
        sqrt_M = int(np.sqrt(M))
        def gray_code(n):
            if n==1: return ['0','1']
            p = gray_code(n-1)
            return ['0'+g for g in p] + ['1'+g for g in reversed(p)]
        gs = gray_code(k//2)
        pts = []; bmp = []
        levels = 2 * np.arange(1, sqrt_M+1) - (sqrt_M+1)
        for r_i, r in enumerate(levels):
            for i_i, im in enumerate(levels):
                pts.append(r + 1j*im)
                val = (int(gs[r_i],2) << (k//2)) | int(gs[i_i],2)
                bmp.append(val)
        return np.array(pts, dtype=complex), np.array(bmp)
    return None, None

def generate_dataset_2ch(n_samples, points):
    n_sub = N_FFT
    n_cls = len(points)
    idxs = np.random.randint(0, n_cls, (n_samples, n_sub))
    syms = points[idxs]
    tx = np.fft.ifft(syms, n=N_FFT, axis=1) * np.sqrt(N_FFT)
    obs_len = int(N_FFT * OBSERVATION_RATIO)
    rx = tx[:, :obs_len]
    
    pwr = np.mean(np.abs(rx)**2)
    noise_pwr = pwr / (10**(SNR_DB/10))
    noise = (np.random.randn(*rx.shape) + 1j*np.random.randn(*rx.shape)) * np.sqrt(noise_pwr/2)
    rx = rx + noise
    
    # [Batch, 2, L]
    X_real = np.real(rx)
    X_imag = np.imag(rx)
    # Stack [Batch, 2, L]
    X = np.stack([X_real, X_imag], axis=1).astype(np.float32)
    
    # NORMALIZATION: Robust scaler per batch to unit variance
    # This is critical for high-order QAM where values can be large
    std = np.std(X, axis=(0, 2), keepdims=True)
    X = X / (std + 1e-8)
    
    Y = idxs.astype(np.int64)
    return X, Y

def main():
    print("--- Optimized Model Training Benchmark ---")
    
    results = {}
    
    for mod in MODULATIONS:
        print(f"\nProcessing {mod}...")
        pts, bmp = get_mod_config(mod)
        n_cls = len(pts)
        bits = int(np.log2(n_cls))
        out_dim = N_FFT * n_cls
        
        Xt, Yt = generate_dataset_2ch(N_SAMPLES_TRAIN, pts)
        Xv, Yv = generate_dataset_2ch(N_SAMPLES_TEST, pts)
        
        # Scale rank with complexity
        # Base rank 512 + extra for higher order
        eff_rank = 512 if 'QPSK' in mod else 1024
        
        # Instantiate Optimized Model
        # Input Len = Xt.shape[2]
        model = OptimizedCNNReceiver(Xt.shape[2], out_dim, n_filters=64, rank=eff_rank).to(device)
        
        # Count Parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,} (Original ~{73000000 if '64' in mod else 293000000 if '256' in mod else 4000000})")
        
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        crit = nn.CrossEntropyLoss()
        
        ds_t = TensorDataset(torch.from_numpy(Xt), torch.from_numpy(Yt))
        dl_t = DataLoader(ds_t, batch_size=BATCH_SIZE, shuffle=True)
        
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            loss_accum = 0
            for xb, yb in dl_t:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = model(xb)
                loss = crit(out.view(-1, n_cls), yb.view(-1))
                loss.backward()
                opt.step()
                loss_accum += loss.item()
            # print(f"    Ep {ep}: loss={loss_accum/len(dl_t):.4f}")
            
        print(f"  Training Time: {time.time()-t0:.2f}s")
        
        # Eval
        model.eval()
        bers = []
        corrs = []
        
        # Pre-calculate reconstruction constants
        # IFFT scaling factor to match time-domain energy if needed, but for corr it cancels out.
        # However, to be precise: ifft(syms) * sqrt(N) is what we used in generation.
        
        with torch.no_grad():
            ds_v = TensorDataset(torch.from_numpy(Xv), torch.from_numpy(Yv))
            dl_v = DataLoader(ds_v, batch_size=BATCH_SIZE)
            for xb, yb in dl_v:
                xb, yb = xb.to(device), yb.to(device)
                
                # 1. Prediction
                # out: [Batch, N_FFT, n_cls]
                out = model(xb).view(-1, N_FFT, n_cls)
                preds = torch.argmax(out, dim=2).cpu().numpy()
                true = yb.cpu().numpy()
                
                # 2. BER Calculation
                p_bits = bmp[preds]; t_bits = bmp[true]
                xor = p_bits ^ t_bits
                pop = np.zeros_like(xor)
                for b in range(bits): pop += (xor >> b) & 1
                
                # BER per symbol (averaged over subcarriers) or per sequence.
                # Let's collect BER per sequence (OFDM symbol) to plot CDF
                # Shape: [Batch, N_FFT] -> avg over N_FFT -> [Batch]
                batch_bers = np.mean(pop, axis=1) / bits
                bers.extend(batch_bers)

                # 3. Correlation Calculation (Time Domain)
                # Reconstruct Time Domain Signals from symbols
                # preds: [Batch, N_FFT] (indices) -> symbols
                sym_pred = pts[preds] # [Batch, N_FFT] (complex)
                sym_true = pts[true]  # [Batch, N_FFT] (complex)
                
                # IFFT to get time domain
                # generation used: np.fft.ifft(syms, n=N_FFT, axis=1) * np.sqrt(N_FFT)
                tx_pred = np.fft.ifft(sym_pred, axis=1) * np.sqrt(N_FFT)
                tx_true = np.fft.ifft(sym_true, axis=1) * np.sqrt(N_FFT)
                
                # Calculate Correlation on the UNOBSERVED part
                obs_len = int(N_FFT * OBSERVATION_RATIO)
                
                # Slice: [Batch, obs_len:]
                rem_pred = tx_pred[:, obs_len:]
                rem_true = tx_true[:, obs_len:]
                
                # Correlation Coefficient per sample in batch
                # |<x, y>| / (||x|| * ||y||)
                # dot product along time axis (axis 1)
                num = np.abs(np.sum(rem_pred * np.conj(rem_true), axis=1))
                den = np.sqrt(np.sum(np.abs(rem_pred)**2, axis=1)) * np.sqrt(np.sum(np.abs(rem_true)**2, axis=1))
                
                # Avoid division by zero
                batch_corrs = num / (den + 1e-9)
                corrs.extend(batch_corrs)
                
        avg_ber = np.mean(bers)
        avg_corr = np.mean(corrs)
        print(f"  Final BER: {avg_ber:.4f}, Mean Corr: {avg_corr:.4f}")
        results[mod] = {'ber': bers, 'corr': corrs, 'params': params} # Store full arrays for CDF

    print("\n--- Summary ---")
    for mod, res in results.items():
        print(f"{mod}: Mean BER={np.mean(res['ber']):.4f}, Mean Corr={np.mean(res['corr']):.4f}")

    # --- Plotting CDFs ---
    print("\nGenerating CDF Plots...")
    
    # 1. BER CDF
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, mod in enumerate(MODULATIONS):
        data = np.sort(results[mod]['ber'])
        y = np.arange(1, len(data)+1) / len(data)
        plt.plot(data, y, label=f'{mod} (Mean={np.mean(data):.4f})', color=colors[i%len(colors)])
    
    plt.title(f'CDF of BER over Observation Period (Ratio={OBSERVATION_RATIO}, SNR={SNR_DB}dB)')
    plt.xlabel('Bit Error Rate (BER)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('ai_optimized_ber_cdf.png')
    print("Saved ai_optimized_ber_cdf.png")

    # 2. Correlation CDF
    plt.figure(figsize=(10, 6))
    for i, mod in enumerate(MODULATIONS):
        data = np.sort(results[mod]['corr'])
        y = np.arange(1, len(data)+1) / len(data)
        plt.plot(data, y, label=f'{mod} (Mean={np.mean(data):.4f})', color=colors[i%len(colors)])
        
    plt.title(f'CDF of Absolute Correlation Coefficient (Attack Period) (Ratio={OBSERVATION_RATIO})')
    plt.xlabel('Absolute Correlation Coefficient')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(loc='upper left')
    plt.savefig('ai_optimized_corr_cdf.png')
    print("Saved ai_optimized_corr_cdf.png")

if __name__ == "__main__":
    main()
