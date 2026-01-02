# acw
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# SETTINGS
# -----------------------------
TR = 2.3  # seconds
fs = 1 / TR  # sampling frequency

# Folders containing voxelwise timeseries
base_dir = "/home/stubanadean/voxelwise_timeseries_nilearn"
rois = ["Interoception", "Exteroception", "Cognition"]

# Output CSV
output_csv = os.path.join(base_dir, "ACW_results.csv")

# -----------------------------
# FUNCTION TO COMPUTE ACW
# -----------------------------
def compute_acw(x, fs, threshold0=0, threshold50=0.5, isplot=False):
    x = x - np.mean(x)
    acf = np.correlate(x, x, mode="full") / np.sum(x**2)
    acf = acf[len(x)-1:]  # keep only non-negative lags
    lags = np.arange(len(acf)) / fs

    # ACW-0: first lag where ACF <= threshold
    idx0 = np.where(acf <= threshold0)[0]
    acw_0 = lags[idx0[0]] if len(idx0) > 0 else np.nan  # return NaN if not found

    # ACW-50: first lag where ACF <= threshold
    idx50 = np.where(acf <= threshold50)[0]
    acw_50 = lags[idx50[0]] if len(idx50) > 0 else np.nan

    if isplot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        plt.plot(lags, acf, 'k', label='ACF')
        if not np.isnan(acw_0):
            plt.axvline(acw_0, color='m', linestyle='--', label='ACW-0')
        if not np.isnan(acw_50):
            plt.axvline(acw_50, color='r', linestyle='--', label='ACW-50')
        plt.xlabel('Lag (s)')
        plt.ylabel('Autocorrelation')
        plt.legend()
        plt.show()

    return acw_0, acw_50

# -----------------------------
# SANITY CHECK
# -----------------------------
def sanity_check():
    print("=== ACW SANITY CHECK ===")
    n_samples = 158  # same length as your fMRI
    t = np.arange(n_samples) * TR

    # Sine 0.1 Hz
    sine = np.sin(2 * np.pi * 0.1 * t)
    acw_0, acw_50 = compute_acw(sine, fs)
    print(f"Sine 0.1Hz -> ACW-0: {acw_0:.3f} s, ACW-50: {acw_50:.3f} s")

    # White noise
    white = np.random.randn(n_samples)
    acw_0, acw_50 = compute_acw(white, fs)
    print(f"White noise -> ACW-0: {acw_0:.3f} s, ACW-50: {acw_50:.3f} s")

    # Pink noise approx 1/f
    freqs = np.fft.rfftfreq(n_samples, d=TR)
    pink_fft = (np.random.randn(len(freqs)) + 1j*np.random.randn(len(freqs))) / np.maximum(freqs, 1e-6)
    pink = np.fft.irfft(pink_fft, n=n_samples)
    acw_0, acw_50 = compute_acw(pink, fs)
    print(f"Pink noise -> ACW-0: {acw_0:.3f} s, ACW-50: {acw_50:.3f} s")

    # Random fMRI-like signals
    fMRI_sim = np.random.randn(n_samples, 5)
    mean_acw0 = np.mean([compute_acw(fMRI_sim[:,i], fs)[0] for i in range(fMRI_sim.shape[1])])
    print(f"Voxelwise random fMRI-like signals (5 voxels) -> mean ACW-0: {mean_acw0:.3f} s")
    print("==========================\n")

# -----------------------------
# RUN SANITY CHECK
# -----------------------------
sanity_check()

# -----------------------------
# PROCESS YOUR ROI FILES
# -----------------------------
results = []
for roi in rois:
    roi_folder = os.path.join(base_dir, roi)
    # Automatically find the .npy file
    npy_files = [f for f in os.listdir(roi_folder) if f.endswith(".npy")]
    if len(npy_files) == 0:
        print(f"No .npy file found in {roi_folder}")
        continue
    file_path = os.path.join(roi_folder, npy_files[0])
    data = np.load(file_path)  # shape: n_timepoints x n_voxels

    acw0_list, acw50_list = [], []
    for v in range(data.shape[1]):
        acw_0, acw_50 = compute_acw(data[:,v], fs)
        acw0_list.append(acw_0)
        acw50_list.append(acw_50)

    acw0_array = np.array(acw0_list)
    acw50_array = np.array(acw50_list)

    print(f"{roi} | {npy_files[0]}")
    print(f"  Voxels used : {data.shape[1]}")
    print(f"  Mean ACW-0  : {np.nanmean(acw0_array):.3f} s")
    print(f"  Mean ACW-50 : {np.nanmean(acw50_array):.3f} s")
    print("--------------------------------------------------")

    results.append({
        "Layer": roi,
        "ROI_file": npy_files[0],
        "Timepoints": data.shape[0],
        "Num_voxels_used": data.shape[1],
        "Mean_ACW_0_s": np.nanmean(acw0_array),
        "Median_ACW_0_s": np.nanmedian(acw0_array),
        "Std_ACW_0_s": np.nanstd(acw0_array),
        "Min_ACW_0_s": np.nanmin(acw0_array),
        "Max_ACW_0_s": np.nanmax(acw0_array),
        "Mean_ACW_50_s": np.nanmean(acw50_array),
        "Median_ACW_50_s": np.nanmedian(acw50_array),
        "Std_ACW_50_s": np.nanstd(acw50_array),
        "Min_ACW_50_s": np.nanmin(acw50_array),
        "Max_ACW_50_s": np.nanmax(acw50_array)
    })

# -----------------------------
# SAVE RESULTS
# -----------------------------
output_csv = os.path.join(base_dir, "ACW_summary_results.csv")
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\nResults saved to {output_csv}")

