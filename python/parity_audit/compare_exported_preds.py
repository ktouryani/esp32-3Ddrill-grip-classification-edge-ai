import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# Load predictions
# -----------------------------
py = pd.read_csv(os.path.join(LOG_DIR, "predictions_python_exported.csv"))   # n,true_label,pred_py,probs_py
es = pd.read_csv(os.path.join(LOG_DIR, "predictions_cpp_esp32.csv"))         # n,probs  OR n,p0,p1,...

def parse_probs(x):
    if isinstance(x, str):
        return np.array([float(t) for t in x.split(";")], dtype=np.float32)
    # already numeric columns (n,p0,...)
    return np.array(x, dtype=np.float32)

# Normalize ESP columns into vec_es
if es.shape[1] == 2 and "probs" in es.columns:
    es["vec_es"] = es["probs"].apply(parse_probs)
    es = es[["n","vec_es"]]
else:
    cols = ["n"] + [f"p{i}" for i in range(es.shape[1]-1)]
    es.columns = cols
    es["vec_es"] = es[[c for c in cols if c!="n"]].astype(float).apply(lambda r: r.to_numpy(np.float32), axis=1)
    es = es[["n","vec_es"]]

# Merge and compute top-1 indices
m = py.merge(es, on="n", how="inner")    # has: n, true_label, probs_py, vec_es
py_mat = np.vstack(m["probs_py"].apply(parse_probs).to_numpy())
es_mat = np.vstack(m["vec_es"].to_numpy())
py_idx = py_mat.argmax(axis=1)
es_idx = es_mat.argmax(axis=1)

# Get class names from label_encoder if available
try:
    with open("python_ref/scalers/label_encoder.pkl","rb") as f:
        le = pickle.load(f)
    class_names = list(le.classes_)
    # map true labels (strings) -> indices with the encoder
    true_idx = le.transform(m["true_label"].astype(object).to_numpy())
except Exception:
    # Fallback: infer class order from probs length and alphabetical label order
    K = es_mat.shape[1]
    class_names = [f"C{i}" for i in range(K)]
    labs = sorted(m["true_label"].unique())
    lut = {lab:i for i,lab in enumerate(labs)}
    true_idx = m["true_label"].map(lut).to_numpy()

K = len(class_names)

# -----------------------------
# Confusion helpers
# -----------------------------
def cm_counts(rows, cols, K):
    cm = np.zeros((K, K), dtype=int)
    for r, c in zip(rows, cols):
        cm[r, c] += 1
    return cm

def save_cm(cm, labels, title, fname, normalize=False, xlabel="", ylabel=""):
    M = cm.astype(np.float32)
    if normalize:
        rs = M.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        M = (M / rs) * 100.0
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="Blues", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    vmax = M.max() if M.size else 1
    fmt = ".0f" if not normalize else ".1f"
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, format(M[i, j], fmt) + ("" if not normalize else "%"),
                    ha="center", va="center",
                    color=("white" if M[i, j] > vmax*0.6 else "black"), fontsize=8)
    fig.tight_layout()
    path = os.path.join(LOG_DIR, fname)
    plt.savefig(path, dpi=150); plt.close(fig)
    return path

def per_class_table(cm, labels):
    """Row-wise: true label vs predicted (counts). Returns a DataFrame with
       support, correct, accuracy% per true class."""
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    acc_pct = np.where(support>0, correct / support * 100.0, 0.0)
    df = pd.DataFrame({
        "true_label": labels,
        "support": support,
        "correct": correct,
        "accuracy_%": acc_pct
    })
    return df

def save_row_normalized_csv(cm, labels, fname_csv):
    M = cm.astype(np.float32)
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    P = (M / rs) * 100.0
    df = pd.DataFrame(P, index=labels, columns=labels)
    df.to_csv(os.path.join(LOG_DIR, fname_csv), float_format="%.2f")

# -----------------------------
# Build matrices
# -----------------------------
# A) Agreement: Python pred vs ESP32 pred
cm_agree = cm_counts(py_idx, es_idx, K)
p1 = save_cm(cm_agree, class_names,
             "Confusion (Python pred vs ESP32 pred) — counts",
             "confusion_py_vs_esp32_counts.png",
             normalize=False, xlabel="ESP32 predicted", ylabel="Python predicted")
p2 = save_cm(cm_agree, class_names,
             "Confusion (Python pred vs ESP32 pred) — row-norm %",
             "confusion_py_vs_esp32_norm.png",
             normalize=True, xlabel="ESP32 predicted", ylabel="Python predicted")

# B) True vs Python
cm_true_py = cm_counts(true_idx, py_idx, K)
p3 = save_cm(cm_true_py, class_names,
             "True vs Python — counts",
             "confusion_true_vs_python_counts.png",
             normalize=False, xlabel="Python predicted", ylabel="True label")
p4 = save_cm(cm_true_py, class_names,
             "True vs Python — row-norm %",
             "confusion_true_vs_python_norm.png",
             normalize=True, xlabel="Python predicted", ylabel="True label")

# C) True vs ESP32
cm_true_es = cm_counts(true_idx, es_idx, K)
p5 = save_cm(cm_true_es, class_names,
             "True vs ESP32 — counts",
             "confusion_true_vs_esp32_counts.png",
             normalize=False, xlabel="ESP32 predicted", ylabel="True label")
p6 = save_cm(cm_true_es, class_names,
             "True vs ESP32 — row-norm %",
             "confusion_true_vs_esp32_norm.png",
             normalize=True, xlabel="ESP32 predicted", ylabel="True label")

# -----------------------------
# Per-class tables (% by true label)
# -----------------------------
tbl_py = per_class_table(cm_true_py, class_names)
tbl_es = per_class_table(cm_true_es, class_names)
tbl_py.to_csv(os.path.join(LOG_DIR, "per_class_python.csv"), index=False)
tbl_es.to_csv(os.path.join(LOG_DIR, "per_class_esp32.csv"), index=False)

# Also save row-normalized confusion (percent tables) as CSVs
save_row_normalized_csv(cm_true_py, class_names, "confusion_true_vs_python_rownorm.csv")
save_row_normalized_csv(cm_true_es, class_names, "confusion_true_vs_esp32_rownorm.csv")

# Summary
match_rate = (py_idx == es_idx).mean()*100.0
print(f"Rows compared: {len(py_idx)}")
print(f"Top-1 match rate (Py vs ESP32): {match_rate:.2f}%")
print("Saved:")
for p in [p1,p2,p3,p4,p5,p6]:
    print("  ", p)
print("  logs/per_class_python.csv")
print("  logs/per_class_esp32.csv")
print("  logs/confusion_true_vs_python_rownorm.csv")
print("  logs/confusion_true_vs_esp32_rownorm.csv")
