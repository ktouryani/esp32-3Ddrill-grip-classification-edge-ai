import os, glob, csv, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt

WIN, HOP, FS = 128, 64, 2750.0
DATA_DIR = "../data/new_test_dataset"
LOG_DIR  = "../logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------- robust CSV loader ----------
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

def sliding(N, win, hop):
    return range(0, max(0, N - win + 1), hop)

# ---------- features (identical to training) ----------
def spectral_entropy(x):
    P = np.abs(np.fft.rfft(x - np.mean(x)))**2
    P = P / (np.sum(P) + 1e-12)
    H = -np.sum(P * np.log(P + 1e-12))
    return float(H / np.log(len(P)))

def zero_crossing_rate(x):
    xm = x - np.mean(x)
    return float(np.mean(np.abs(np.diff(np.signbit(xm)))))

def band_energy_ratio(x, f_lo, f_hi):
    freqs = np.fft.rfftfreq(len(x), 1/FS)
    X = np.abs(np.fft.rfft(x - np.mean(x)))**2
    total = np.sum(X) + 1e-12
    band = (freqs>=f_lo) & (freqs<=f_hi)
    return float(np.sum(X[band]) / total)

def spectral_stats_basic(x, f0=95.0):
    freqs = np.fft.rfftfreq(len(x), 1/FS)
    S = np.abs(np.fft.rfft(x - x.mean())) + 1e-9  # magnitude
    centroid     = float(np.sum(freqs*S) / np.sum(S))
    centroid_norm= centroid/(FS/2.0)
    high = freqs>=2.5*f0
    band = (freqs>=f0) & (freqs<=3*f0)
    high_ratio   = float((np.sum(S[high])+1e-9) / (np.sum(S[band])+1e-9))
    return centroid_norm, high_ratio

def crest_factor(x):
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    return float(np.max(np.abs(x)) / (rms + 1e-12))

def compute_feats(ax, ay, az):
    a = np.sqrt(ax*ax + ay*ay + az*az).astype(np.float32)
    cen_n, hr = spectral_stats_basic(a)
    jerk = np.diff(a, prepend=a[0]) * FS
    jerk_rms = float(np.sqrt(np.mean(jerk*jerk)))
    zcr = zero_crossing_rate(a)
    ber_f0 = band_energy_ratio(a, 95-10, 95+10)
    Xpow_all = np.abs(np.fft.rfft(a - np.mean(a)))**2
    band_pow = band_energy_ratio(a, 95-2*25, 95+2*25) * (np.sum(Xpow_all)+1e-12)
    Hn = spectral_entropy(a)
    bpoe = band_pow / (Hn + 1e-12)
    cf = crest_factor(a)
    return [jerk_rms, cen_n, zcr, cf, ber_f0, hr, bpoe]

# ---------- load scaler/labels/models ----------
with open("../python_ref/scalers/scaler.pkl","rb") as f: scaler = pickle.load(f)
with open("../python_ref/scalers/label_encoder.pkl","rb") as f: le = pickle.load(f)
class_names = list(le.classes_)

keras_model = tf.keras.models.load_model("../python_ref/model/best_hybrid_model.keras")
tflite = tf.lite.Interpreter(model_path="../python_ref/model/model_int8.tflite")
tflite.allocate_tensors()
ins, outs = tflite.get_input_details(), tflite.get_output_details()
in_imu, in_feat, out = ins[0], ins[1], outs[0]
imu_scale, imu_zp = in_imu['quantization']
feat_scale, feat_zp = in_feat['quantization']
out_scale, out_zp = out['quantization']

def quantize(x, s, zp):
    q = np.round(x/s + zp).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)

# ---------- build dataset ----------
X_time, X_feat, y_true, meta = [], [], [], []
for cdir in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
    label = os.path.basename(cdir)
    for fp in sorted(glob.glob(os.path.join(cdir, "*.csv"))):
        xyz = read_xyz_df(fp)
        for s in sliding(len(xyz), WIN, HOP):
            seg = xyz[s:s+WIN,:]
            X_time.append(seg.astype(np.float32))
            X_feat.append(compute_feats(seg[:,0], seg[:,1], seg[:,2]))
            y_true.append(label)
            meta.append((os.path.basename(fp), s, label))

X_time  = np.asarray(X_time, dtype=np.float32)
X_feat  = np.asarray(X_feat, dtype=np.float32)
y_true  = le.transform(np.array(y_true, dtype=object))
X_featS = scaler.transform(X_feat).astype(np.float32)

# ---------- Keras predictions ----------
p_keras = keras_model.predict([X_time, X_featS], verbose=0)
y_hat_k = np.argmax(p_keras, axis=1)

# ---------- TFLite predictions ----------
preds_t = []
for i in range(X_time.shape[0]):
    imu_q  = quantize(X_time[i],  imu_scale,  imu_zp)
    feat_q = quantize(X_featS[i], feat_scale, feat_zp)
    tflite.set_tensor(in_imu['index'],  imu_q.reshape(in_imu['shape']))
    tflite.set_tensor(in_feat['index'], feat_q.reshape(in_feat['shape']))
    tflite.invoke()
    y = tflite.get_tensor(out['index'])
    if out['dtype'] == np.int8:
        probs = (y.astype(np.float32) - out_zp) * out_scale
    else:
        probs = y.astype(np.float32)
    preds_t.append(probs.reshape(-1))
preds_t = np.vstack(preds_t)
y_hat_t = np.argmax(preds_t, axis=1)

# ---------- metrics ----------
def make_cm_report(y_true_idx, y_pred_idx, tag):
    acc = accuracy_score(y_true_idx, y_pred_idx)
    cm  = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(class_names)))
    rpt = classification_report(y_true_idx, y_pred_idx, target_names=class_names, digits=3)
    print(f"\n[{tag}] Accuracy: {acc*100:.2f}%")
    print(f"[{tag}] Classification report:\n{rpt}")

    # save confusion matrix as image
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap="Blues", interpolation='nearest')
    ax.set_title(f"Confusion Matrix - {tag}")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(LOG_DIR, f"confusion_{tag}.png"), dpi=150)
    with open(os.path.join(LOG_DIR, f"report_{tag}.txt"), "w") as f:
        f.write(f"Accuracy: {acc*100:.2f}%\n\n{rpt}\n")
    return acc, cm

acc_k, cm_k = make_cm_report(y_true, y_hat_k, "keras")
acc_t, cm_t = make_cm_report(y_true, y_hat_t, "tflite")

# save per-window predictions
with open(os.path.join(LOG_DIR, "test_predictions_python.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file","start","true_idx","pred_keras_idx","pred_tflite_idx"])
    for (fn,s,_), tk, tt, yt in zip(meta, y_hat_k, y_hat_t, y_true):
        w.writerow([fn, s, yt, tk, tt])

print("\nSaved:")
print("  logs/confusion_keras.png, logs/report_keras.txt")
print("  logs/confusion_tflite.png, logs/report_tflite.txt")
print("  logs/test_predictions_python.csv")
