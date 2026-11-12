import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle, os
import matplotlib.pyplot as plt

LOG_DIR = "logs"

# labels & index from export
labels = np.load(os.path.join(LOG_DIR, "tv_labels.npy"), allow_pickle=True)
idx_df = pd.read_csv(os.path.join(LOG_DIR, "tv_index.csv"))  # n,file,start,label

# ESP32 predictions (n, probs)
es = pd.read_csv(os.path.join(LOG_DIR, "predictions_cpp_esp32.csv"))
def parse_probs(s):
    if isinstance(s, str): return np.array([float(x) for x in s.split(";")], dtype=np.float32)
    return np.array(s, dtype=np.float32)

# normalize columns
if es.shape[1] == 2 and "probs" in es.columns:
    es["vec"] = es["probs"].apply(parse_probs)
    es = es[["n","vec"]]
else:
    cols = ["n"] + [f"p{i}" for i in range(es.shape[1]-1)]
    es.columns = cols
    es["vec"] = es[[c for c in cols if c!="n"]].astype(float).apply(lambda r: r.to_numpy(np.float32), axis=1)
    es = es[["n","vec"]]

# align & compute preds
m = idx_df.merge(es, on="n", how="inner")
probs = np.vstack(m["vec"].to_numpy())
pred_es = probs.argmax(axis=1)

# map labels to indices via your label encoder
with open("python_ref/scalers/label_encoder.pkl","rb") as f:
    le = pickle.load(f)
true_idx = le.transform(labels[m["n"].to_numpy()].astype(object))

# metrics
acc = accuracy_score(true_idx, pred_es)
cm  = confusion_matrix(true_idx, pred_es, labels=range(len(le.classes_)))
rpt = classification_report(true_idx, pred_es, target_names=list(le.classes_), digits=3)

print(f"[ESP32] Accuracy: {acc*100:.2f}%")
print("[ESP32] Classification report:\n", rpt)

# plot/save confusion matrix
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(cm, cmap="Blues", interpolation='nearest')
ax.set_title("Confusion Matrix - ESP32")
ax.set_xticks(range(len(le.classes_))); ax.set_xticklabels(le.classes_, rotation=45, ha='right')
ax.set_yticks(range(len(le.classes_))); ax.set_yticklabels(le.classes_)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i,j]>cm.max()/2 else "black")
fig.tight_layout()
fig.savefig(os.path.join(LOG_DIR, "confusion_esp32.png"), dpi=150)

with open(os.path.join(LOG_DIR, "report_esp32.txt"), "w") as f:
    f.write(f"Accuracy: {acc*100:.2f}%\n\n{rpt}\n")

print("\nSaved: logs/confusion_esp32.png, logs/report_esp32.txt")
