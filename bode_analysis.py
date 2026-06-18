import sys
import numpy as np
import matplotlib.pyplot as plt

def compute_bode(v_signal, d_signal, fs=80000, use_hann=False):
    """Computes Frequency Response Functions using chosen window structures."""
    N = len(v_signal)
    v_detrend = v_signal - np.mean(v_signal)
    d_detrend = d_signal - np.mean(d_signal)
    
    if use_hann:
        window = np.hanning(N)
        # Scale window back up to keep power balanced
        window /= np.mean(window)
    else:
        window = np.ones(N) # Rectangular Window (Correct window for pure linear chirps)
        
    V_f = np.fft.rfft(v_detrend * window)
    D_f = np.fft.rfft(d_detrend * window)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    
    H = V_f / (D_f + 1e-12)
    gain_db = 20 * np.log10(np.abs(H) + 1e-12)
    phase_deg = np.unwrap(np.angle(H)) * (180.0 / np.pi) # Fixed unwrapping sequence error
    
    return freqs, gain_db, phase_deg

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bode_analysis.py <chirp_capture_file.npz>")
        sys.exit(1)
        
    data = np.load(sys.argv[1])
    # Extract raw data streams and auto-convert
    v_real = (data['raw'] / 4095.0) * 3.3 if 'raw' in data else data['v']
    duty = data['duty']
    
    f_rect, g_rect, p_rect = compute_bode(v_real, duty, use_hann=False)
    f_hann, g_hann, p_hann = compute_bode(v_real, duty, use_hann=True)
    
    # Plotting comparison overlay windows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.semilogx(f_rect, g_rect, label='Correct (Rectangular Window)', color='tab:blue', alpha=0.9)
    ax1.semilogx(f_hann, g_hann, label='Incorrect (Hann Edge Attenuation)', color='tab:red', linestyle='--', alpha=0.8)
    ax1.set_ylabel('Gain (dB)'); ax1.grid(True, which='both'); ax1.legend()
    ax1.set_title("Bode Windows Structural Comparison")
    ax1.set_xlim(100, 10000)
    
    ax2.semilogx(f_rect, p_rect, color='tab:blue')
    ax2.semilogx(f_hann, p_hann, color='tab:red', linestyle='--')
    ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('Phase (deg)'); ax2.grid(True, which='both')
    ax2.set_xlim(100, 10000)
    
    plt.tight_layout()
    plt.savefig("bode_window_comparison.png", dpi=150)
    print("Saved dual-window analysis diagnostic comparison as 'bode_window_comparison.png'")