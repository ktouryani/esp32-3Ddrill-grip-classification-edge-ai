import os, glob, csv, pickle
import numpy as np
import tensorflow as tf
import pandas as pd

WIN, HOP, FS = 128, 64, 2750.0
DATA_DIR = "data/new_test_dataset"
LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------- same robust CSV reader as before ----------
def read_xyz_df(path):
    with open(path, "r", newline="") as fh:
        sample = fh.read(2048)
    delim = "," if sample.count(",")>=sample.count(";") else ";"
    df = pd.read_csv(path, delimiter=delim, engine="python")
    if {"ax","ay","az"}.issubset(df.columns):
        arr = df[["ax","ay","az"]].astype("float32").to_numpy()
    else:
        arr = df.select_dtypes(include=["number"]).iloc[:, :3].astype("float32").to_numpy()
    return arr

def sliding_starts(N, win, hop):
    return range(0, max(0, N-win+1), hop)

# ---------- feature fns (same as training) ----------
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
    centroid     = float(np.sum(freqs*S) / np.sum(S))
    centroid_norm= centroid/(fs/2.0)
    high = freqs>=2.5*f0
    band = (freqs>=f0) & (freqs<=3*f0)
    high_ratio   = float((np.sum(S[high])+1e-9) / (np.sum(S[band])+1e-9))
    return centroid_norm, high_ratio

def crest_factor(x):
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    return float(np.max(np.abs(x)) / (rms + 1e-12))

def compute_features(ax, ay, az, fs=FS):
    a = np.sqrt(ax*ax + ay*ay + az*az).astype(np.float32)
    cen_n, hr = spectral_stats_basic(a, fs)
    jerk = np.diff(a, prepend=a[0]) * fs
    jerk_rms = float(np.sqrt(np.mean(jerk*jerk)))
    zcr = zero_crossing_rate(a)
    ber_f0 = band_energy_ratio(a, fs, 85, 105)
    Xpow_all = np.abs(np.fft.rfft(a - np.mean(a)))**2
    band_pow = band_energy_ratio(a, fs, 95-2*25, 95+2*25) * (np.sum(Xpow_all)+1e-12)
    Hn = spectral_entropy(a, fs)
    bpoe = band_pow / (Hn + 1e-12)
    cf = crest_factor(a)
    return [jerk_rms, cen_n, zcr, cf, ber_f0, hr, bpoe]

# ---------- load scaler + models ----------
with open("../python_ref/scalers/scaler.pkl","rb") as f: scaler = pickle.load(f)
with open("../python_ref/scalers/label_encoder.pkl","rb") as f: le = pickle.load(f)
classes = list(le.classes_)

keras_model = tf.keras.models.load_model("../python_ref/model/best_hybrid_model.keras")

tflite_path = "../python_ref/model/model_int8.tflite"
tflite = tf.lite.Interpreter(model_path=tflite_path)
tflite.allocate_tensors()
# we’ll read input tensor indices (two inputs: imu + feats)
input_details = tflite.get_input_details()
output_details = tflite.get_output_details()
# helpful print
print("TFLite inputs:", [(d['index'], d['shape'], d['dtype'], d['quantization']) for d in input_details])
print("TFLite output:", [(d['index'], d['shape'], d['dtype'], d['quantization']) for d in output_details])

# ---------- build windows + features ----------
X_time = []
X_feat = []
file_ids = []

for cdir in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
    label = os.path.basename(cdir)
    for fp in sorted(glob.glob(os.path.join(cdir, "*.csv"))):
        xyz = read_xyz_df(fp)
        for s in sliding_starts(len(xyz), WIN, HOP):
            seg = xyz[s:s+WIN,:]
            X_time.append(seg.astype(np.float32))               # (128,3)
            feats = compute_features(seg[:,0],seg[:,1],seg[:,2])
            X_feat.append(feats)
            file_ids.append((label, os.path.basename(fp), s))

X_time = np.asarray(X_time, dtype=np.float32)
X_feat = np.asarray(X_feat, dtype=np.float32)
X_feat_s = scaler.transform(X_feat)

print("Shapes -> time:", X_time.shape, " feat:", X_feat_s.shape)

# ---------- KERAS PRED ----------
p_keras = keras_model.predict([X_time, X_feat_s], verbose=0)
top1_k = p_keras.argmax(1)

# ---------- TFLITE PRED (INT8) ----------
# Quantize each input using its scale/zp
def quantize(arr_f32, scale, zp, dtype=np.int8):
    q = np.round(arr_f32/scale + zp).astype(np.int32)
    return np.clip(q, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)

imu_in = input_details[0]
feat_in= input_details[1]
out    = output_details[0]

imu_scale, imu_zp = imu_in['quantization']
feat_scale,feat_zp= feat_in['quantization']

p_tfl = []
for i in range(X_time.shape[0]):
    imu_q  = quantize(X_time[i], imu_scale, imu_zp)
    feat_q = quantize(X_feat_s[i], feat_scale, feat_zp)
    tflite.set_tensor(imu_in['index'], imu_q.reshape(imu_in['shape']))
    tflite.set_tensor(feat_in['index'], feat_q.reshape(feat_in['shape']))
    tflite.invoke()
    logits_q = tflite.get_tensor(out['index'])  # int8 or dequantized?
    # if int8, dequantize:
    if out['dtype'] == np.int8:
        out_scale, out_zp = out['quantization']
        probs = (logits_q.astype(np.float32) - out_zp) * out_scale
    else:
        probs = logits_q.astype(np.float32)
    p_tfl.append(probs.squeeze())

p_tfl = np.vstack(p_tfl)
top1_t = p_tfl.argmax(1)

# ---------- save logs ----------
with open(os.path.join(LOG_DIR,"predictions_python_keras.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["file","start","true","pred","probs"])
    for (lab,fn,s),pred,probs in zip(file_ids, top1_k, p_keras):
        w.writerow([fn,s,lab,classes[pred], ";".join(f"{p:.6f}" for p in probs)])

with open(os.path.join(LOG_DIR,"predictions_python_tflite.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["file","start","true","pred","probs"])
    for (lab,fn,s),pred,probs in zip(file_ids, top1_t, p_tfl):
        w.writerow([fn,s,lab,classes[pred], ";".join(f"{p:.6f}" for p in probs)])

print("✅ Wrote predictions: logs/predictions_python_keras.csv & ..._tflite.csv")
