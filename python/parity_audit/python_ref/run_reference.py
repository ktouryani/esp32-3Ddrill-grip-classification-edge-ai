import os, csv, pickle
import numpy as np
import pandas as pd, glob

WIN = 128
HOP = 64
FS  = 2750.0
FEATURES_KEEP = ["jerk_rms","centroid_norm","zcr","crest_factor_amp","ber_f0","high_ratio","bandpower_over_entropy_amp"]

DATA_DIR = "../data/new_test_dataset"
LOG_DIR  = "../logs"
os.makedirs(LOG_DIR, exist_ok=True)

df = pd.read_csv(glob.glob("../data/new_test_dataset/*/*.csv")[0])
print(df.columns.tolist()[:10])

# ---------- helpers ----------
def spectral_entropy(x, fs):
    P = np.abs(np.fft.rfft(x - np.mean(x)))**2
    P = P / (np.sum(P) + 1e-12)
    H = -np.sum(P * np.log(P + 1e-12))
    return float(H / np.log(len(P)))

def zero_crossing_rate(x):
    xm = x - np.mean(x)
    return float(np.mean(np.abs(np.diff(np.signbit(xm)))))

def band_energy_ratio(x, fs, f_lo, f_hi):
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    X = np.abs(np.fft.rfft(x - np.mean(x)))**2
    total = np.sum(X) + 1e-12
    band = (freqs>=f_lo) & (freqs<=f_hi)
    return float(np.sum(X[band]) / total)

def spectral_stats_basic(x, fs, f0=95):
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    S = np.abs(np.fft.rfft(x - x.mean())) + 1e-9
    peak_ratio   = float(np.max(S[1:]) / (np.sum(S) + 1e-9))
    centroid     = float(np.sum(freqs*S) / np.sum(S))
    centroid_norm= centroid/(fs/2.0)
    geo          = np.exp(np.mean(np.log(S)))
    flatness     = float(geo/np.mean(S))
    band = (freqs>=f0) & (freqs<=3*f0)
    high = freqs>=2.5*f0
    high_ratio   = float((np.sum(S[high])+1e-9) / (np.sum(S[band])+1e-9))
    return peak_ratio, centroid_norm, flatness, high_ratio

def crest_factor(x):
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    return float(np.max(np.abs(x)) / (rms + 1e-12))

def compute_selected_features(ax, ay, az, fs=FS):
    a = np.sqrt(ax*ax + ay*ay + az*az).astype(np.float32)
    pr, cen_n, flat, hr = spectral_stats_basic(a, fs)
    jerk = np.diff(a, prepend=a[0]) * fs
    jerk_rms = float(np.sqrt(np.mean(jerk*jerk)))
    zcr = zero_crossing_rate(a)
    ber_f0 = band_energy_ratio(a, fs, 85, 105)
    Xpow_all = np.abs(np.fft.rfft(a - np.mean(a)))**2
    band_pow = band_energy_ratio(a, fs, 45, 145) * (np.sum(Xpow_all)+1e-12)
    Hn = spectral_entropy(a, fs)
    bandpower_over_entropy_amp = band_pow / (Hn + 1e-12)
    cf = crest_factor(a)
    return [jerk_rms, cen_n, zcr, cf, ber_f0, hr, bandpower_over_entropy_amp]

def sliding_starts(N, win, hop):
    return range(0, max(0, N - win + 1), hop)

def read_xyz_csv(path):
    import pandas as pd, csv as _csv
    # 1) auto-detect delimiter
    with open(path, "r", newline="") as fh:
        sample = fh.read(2048)
        try:
            dialect = _csv.Sniffer().sniff(sample)
            delim = dialect.delimiter
        except Exception:
            delim = ","  # fallback
    # 2) read via pandas (handles headers & mixed types)
    df = pd.read_csv(path, delimiter=delim, engine="python")
    # 3) prefer named columns if present
    possible_axes = ["ax","ay","az","Accel_X","Accel_Y","Accel_Z","x","y","z"]
    have = [c for c in possible_axes if c in df.columns]
    if len(have) >= 3:
        out = df[have[:3]].astype("float32").to_numpy()
        return out
    # 4) otherwise take first 3 numeric columns
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < 3:
        raise ValueError(f"{path}: found {num.shape[1]} numeric cols; need ≥3. "
                         f"Columns seen: {list(df.columns)}")
    out = num.iloc[:, :3].astype("float32").to_numpy()
    return out

# ---------- load scaler ----------
with open("../python_ref/scalers/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------- enumerate data ----------
class_dirs = sorted([d for d in glob.glob(os.path.join(DATA_DIR, "*")) if os.path.isdir(d)])
print(f"Found {len(class_dirs)} class folders under {DATA_DIR}")
csv_files = []
for c in class_dirs:
    files = sorted(glob.glob(os.path.join(c, "*.csv")))
    print(f"  {os.path.basename(c)}: {len(files)} files")
    csv_files.extend(files)

if not csv_files:
    raise FileNotFoundError(f"No CSVs found under {DATA_DIR}. Check the relative path from python_ref/.")

# ---------- process ----------
features = []
total_windows = 0
for fp in csv_files:
    xyz = read_xyz_csv(fp)
    if len(xyz) < WIN:
        print(f"  SKIP (too short): {fp}  len={len(xyz)}")
        continue
    wins_here = 0
    for s in sliding_starts(len(xyz), WIN, HOP):
        seg = xyz[s:s+WIN, :]
        feats = compute_selected_features(seg[:,0], seg[:,1], seg[:,2])
        features.append(feats)
        wins_here += 1
    total_windows += wins_here
    print(f"  {os.path.basename(fp)} -> windows: {wins_here}")

features = np.asarray(features, dtype=np.float32)
print(f"\nTOTAL windows: {total_windows}, features shape: {features.shape}")

if features.size == 0:
    raise RuntimeError("No windows produced. Check CSV lengths, delimiter, and DATA_DIR path.")

# scale only if we have rows
features_s = scaler.transform(features)

# ---------- save ----------
import csv as _csv
with open(os.path.join(LOG_DIR, "features_python.csv"), "w", newline="") as f:
    w = _csv.writer(f); w.writerow(FEATURES_KEEP); w.writerows(features)

with open(os.path.join(LOG_DIR, "scaled_features_python.csv"), "w", newline="") as f:
    w = _csv.writer(f); w.writerow(FEATURES_KEEP); w.writerows(features_s)

print("\n✅ Saved logs/features_python.csv and logs/scaled_features_python.csv")
