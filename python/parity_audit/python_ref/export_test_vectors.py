import os, glob, pickle, numpy as np, pandas as pd
from pathlib import Path

WIN, HOP = 128, 64
FS, F0, DELTA = 2750.0, 95.0, 25.0
DATA_DIR = "D:/Karim/TUHH/Semester4/Thesis/pipeline_check/data/new_test_dataset"
OUT_H    = "cpp_ref/test_vectors.h"
MAX_WIN  = 300  # limit for convenience
PER_CLASS = 60  # e.g., 5 classes * 60 = 300 total

def read_xyz_df(path):
    # robust delimiter + column selection (ax,ay,az or first 3 numeric)
    with open(path, "r", newline="") as fh:
        sample = fh.read(2048)
    delim = "," if sample.count(",") >= sample.count(";") else ";"
    df = pd.read_csv(path, delimiter=delim, engine="python")
    if {"ax","ay","az"}.issubset(df.columns):
        arr = df[["ax","ay","az"]].astype("float32").to_numpy()
    else:
        num = df.select_dtypes(include=["number"])
        if num.shape[1] < 3:
            raise ValueError(f"{path}: need ≥3 numeric columns, got {num.shape[1]}. Columns={list(df.columns)}")
        arr = num.iloc[:, :3].astype("float32").to_numpy()
    return arr

def spectral_entropy(a):
    P = np.abs(np.fft.rfft(a - a.mean()))**2
    P = P / (P.sum() + 1e-12)
    H = -(P * np.log(P + 1e-12)).sum()
    return float(H / np.log(len(P)))

def band_energy_ratio_pow(a, f_lo, f_hi):
    freqs = np.fft.rfftfreq(len(a), 1/FS)
    X = np.abs(np.fft.rfft(a - a.mean()))**2
    band = (freqs>=f_lo) & (freqs<=f_hi)
    return float(X[band].sum() / (X.sum() + 1e-12))

def spectral_stats_basic(a):
    freqs = np.fft.rfftfreq(len(a), 1/FS)
    S = np.abs(np.fft.rfft(a - a.mean())) + 1e-9     # magnitude
    cen = float((freqs*S).sum() / S.sum())
    cen_n = cen/(FS/2.0)
    high = freqs>=2.5*F0
    band = (freqs>=F0) & (freqs<=3*F0)
    hr = float((S[high].sum()+1e-9) / (S[band].sum()+1e-9))
    return cen_n, hr

def crest_factor(a):
    rms = np.sqrt((a*a).mean() + 1e-12)
    return float(np.max(np.abs(a)) / (rms + 1e-12))

def compute_feats(seg):
    ax, ay, az = seg[:,0], seg[:,1], seg[:,2]
    a = np.sqrt(ax*ax + ay*ay + az*az).astype(np.float32)
    cen_n, hr = spectral_stats_basic(a)
    jerk = np.diff(a, prepend=a[0]) * FS
    jerk_rms = float(np.sqrt((jerk*jerk).mean()))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(a - a.mean())))))
    ber_f0 = band_energy_ratio_pow(a, F0-10, F0+10)
    Xpow_all = np.abs(np.fft.rfft(a - a.mean()))**2
    band_pow = band_energy_ratio_pow(a, F0-2*DELTA, F0+2*DELTA) * (Xpow_all.sum() + 1e-12)
    Hn = spectral_entropy(a)
    bpoe = band_pow / (Hn + 1e-12)
    cf = crest_factor(a)
    return np.array([jerk_rms, cen_n, zcr, cf, ber_f0, hr, bpoe], dtype=np.float32)

def as_c_array(name, arr, dims):
    flat = arr.flatten()
    vals = ", ".join(f"{x:.8f}f" for x in flat)
    shape = ", ".join(str(d) for d in dims)
    return (
        f"static const int {name}_shape[{len(dims)}] = {{{shape}}};\n"
        f"static const float {name}[] = {{{vals}}};\n"
    )

# --- Load scaler
with open("./scalers/scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# Resolve DATA_DIR relative to THIS script, not the current shell folder
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ABS = (SCRIPT_DIR / DATA_DIR).resolve()

# --- Collect windows (unchanged), but store (file, start, label) too
windows, feats, meta = [], [], []
for cdir in sorted([p for p in DATA_ABS.iterdir() if p.is_dir()]):
    taken_class = 0
    for fp in sorted(cdir.glob("*.csv")):
        xyz = read_xyz_df(str(fp))
        for s in range(0, xyz.shape[0] - WIN + 1, HOP):
            seg = xyz[s:s+WIN, :].astype(np.float32)
            windows.append(seg)
            feats.append(compute_feats(seg))
            meta.append((fp.name, s, cdir.name))
            taken_class += 1
            if taken_class >= PER_CLASS:
                break
        if taken_class >= PER_CLASS:
            break
        
if len(windows) == 0:
    raise RuntimeError("No windows produced. Check DATA_DIR / CSV columns.")

windows = np.asarray(windows, dtype=np.float32)       # (N,128,3)
feats   = np.asarray(feats,   dtype=np.float32)       # (N,7)
labels  = np.array([m[2] for m in meta], dtype=object)
print("Shapes:", windows.shape, feats.shape)

feats_s = scaler.transform(feats).astype(np.float32)

# --- write header (C header for ESP32 stays as before) ---
os.makedirs(os.path.dirname(OUT_H), exist_ok=True)
with open(OUT_H, "w") as f:
    f.write("#pragma once\n#include <cstdint>\n\n")
    f.write(f"static const int TV_N = {windows.shape[0]};\n")
    f.write(as_c_array("TV_IMU", windows, windows.shape))
    f.write(as_c_array("TV_FEAT_S", feats_s, feats_s.shape))
print(f"✅ Exported {windows.shape[0]} windows to {OUT_H}")

# --- ALSO save numpy + index for Python-side comparison
np.save("../logs/tv_windows.npy", windows)
np.save("../logs/tv_feats_scaled.npy", feats_s)
np.save("../logs/tv_labels.npy", labels)

import csv
with open("../logs/tv_index.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["n","file","start","label"])
    for i,(fn,s,lab) in enumerate(meta):
        w.writerow([i,fn,s,lab])

print("Saved: logs/tv_windows.npy, tv_feats_scaled.npy, tv_labels.npy, tv_index.csv")