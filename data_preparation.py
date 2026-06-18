import os
import re
import sys
import csv
import shutil
import numpy as np
from scipy.interpolate import PchipInterpolator

# ==================== 1. CALIBRATION SUB ENGINE ====================
_CALIB_TABLE = np.array([
    [14, 0.35], [38, 0.56], [84, 0.96], [124, 1.32], [161, 1.66],
    [201, 1.98], [240, 2.31], [277, 2.63], [309, 2.95], [343, 3.27],
    [377, 3.59], [412, 3.91], [450, 4.23], [483, 4.55], [516, 4.87],
    [553, 5.19], [590, 5.52], [623, 5.84], [659, 6.16], [2658, 24.5], 
    [2699, 24.8], [2734, 25.1], [2810, 25.7], [2851, 26.1], [3130, 28.7]
]) # Non-monotonic typos explicitly dropped here

_INTERPOLATOR = PchipInterpolator(_CALIB_TABLE[:, 0], _CALIB_TABLE[:, 1], extrapolate=True)

def adc_raw_to_real_v(adc_raw):
    """Converts raw ADC values (0-4095) directly to calibrated Volts."""
    adc_mv = (adc_raw / 4095.0) * 3300.0
    return _INTERPOLATOR(adc_mv)

# ==================== 2. VERIFICATION & COMBINATION ENGINE ====================
def verify_and_copy(folders, out_dir="new_data"):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    
    for folder in folders:
        if not os.path.isdir(folder): continue
        for fname in os.listdir(folder):
            if not fname.endswith('.npz'): continue
            fpath = os.path.join(folder, fname)
            
            try:
                data = np.load(fpath, allow_pickle=True)
                raw, duty = data['raw'], data['duty']
            except Exception:
                print(f"[FAIL] {fname} - Corrupted NPZ container")
                continue
                
            # Integrity verification checks
            checks = [
                (np.all((raw >= 0) & (raw <= 4095)), "ADC range limits"),
                (np.all((duty >= 0) & (duty <= 100)), "Duty cycle range bounds"),
                (not (np.isnan(raw).any() or np.isinf(raw).any()), "Inf/NaN evaluation")
            ]
            
            if not all(c[0] for c in checks):
                print(f"[SKIP] {fname} failed integrity verification.")
                continue
                
            # Standardized descriptive naming rules
            meta = data.get('meta', {}).item() if 'meta' in data.files else {}
            mode = meta.get('mode', 'original')
            note = re.sub(r'[^a-z0-9_]', '_', meta.get('note', '').lower())
            dest_name = f"{mode}_{note}_{fname}" if note else fname
            
            shutil.copy2(fpath, os.path.join(out_dir, dest_name))
            manifest.append({'new_name': dest_name, 'source': fpath, 'mode': mode})
            
    # Save the deployment tracking manifest index
    with open(os.path.join(out_dir, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['new_name', 'source', 'mode'])
        w.writeheader()
        w.writerows(manifest)
    print(f"\nSuccessfully verified and compiled {len(manifest)} files into '{out_dir}/'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_preparation.py <raw_folder1> <raw_folder2> ...")
    else:
        verify_and_copy(sys.argv[1:])