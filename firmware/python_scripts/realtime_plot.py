import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# --- FILES (edit these paths) ---
PRED_CSV = "pred_new.csv"                 # your ESP32 log
GT_CSV   = "ground_truth.csv"           # ground-truth: start_ms,end_ms,label

# --- class order used on-device (must match the prob columns!) ---
LABELS = ["held_loose_no_contact",
          "held_tight_no_contact",
          "held_with_hard_contact",
          "held_with_light_contact",
          "standing_freely"]          # <- change to your actual order

# ---------- Load predictions ----------
dfp = pd.read_csv(PRED_CSV, header=0, names=["t_ms","pred"] + [f"p{i}" for i in range(len(LABELS))])
# Top-1 from probs (robust even if 'pred' text is absent or you're bypassing smoothing)
probs = dfp[[f"p{i}" for i in range(len(LABELS))]].to_numpy()
top1_idx = np.argmax(probs, axis=1)
dfp["pred_top1"] = [LABELS[i] for i in top1_idx]

# ---------- Load GT segments ----------
gt = pd.read_csv(GT_CSV)  # columns: start_ms,end_ms,label
gt = gt.sort_values("start_ms").reset_index(drop=True)

# Map each prediction timestamp to a GT label
# Efficient interval join: assign GT row index to each t_ms
# (assumes segments are non-overlapping and cover the times of interest)
starts = gt["start_ms"].to_numpy()
ends   = gt["end_ms"].to_numpy()
labels = gt["label"].to_numpy()

def lookup_label(t):
    # binary search for segment where start <= t <= end
    # (linear scan is fine too if logs are short)
    lo, hi = 0, len(starts)-1
    ans = None
    while lo <= hi:
        mid = (lo+hi)//2
        if t < starts[mid]:
            hi = mid-1
        elif t > ends[mid]:
            lo = mid+1
        else:
            ans = labels[mid]
            break
    return ans

dfp["gt"] = [lookup_label(t) for t in dfp["t_ms"].to_numpy()]
dfp = dfp.dropna(subset=["gt"])  # keep rows that fall inside annotated segments

# ---------- Metrics ----------
y_true = dfp["gt"].to_numpy()
y_pred = dfp["pred_top1"].to_numpy()

acc = (y_true == y_pred).mean()
print(f"Real-time top-1 accuracy: {acc*100:.2f}%  (N={len(dfp)})\n")

print("Per-class precision/recall/F1 (real-time):")
print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

# ---------- Confusion matrix ----------
cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize=None)
cm_norm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize='true')  # row-normalized

fig, ax = plt.subplots(1,2, figsize=(12,4))
ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Confusion (counts)")
for axis in ax:
    axis.tick_params(axis='x', labelrotation=30)
    plt.setp(axis.get_xticklabels(), ha='right')

ConfusionMatrixDisplay(cm_norm, display_labels=LABELS).plot(ax=ax[1], cmap="Blues", colorbar=False)
ax[1].set_title("Confusion (row-normalized)")
for axis in ax:
    axis.tick_params(axis='x', labelrotation=30)
    plt.setp(axis.get_xticklabels(), ha='right')

plt.tight_layout(); plt.show()

# ---------- Compact accuracy bar ----------
per_class_acc = (pd.crosstab(dfp["gt"], dfp["pred_top1"], dropna=False)
                 .reindex(index=LABELS, columns=LABELS, fill_value=0))
per_class_acc = (per_class_acc.values.diagonal() / per_class_acc.sum(axis=1))
plt.figure(figsize=(6,3))
plt.bar(LABELS, per_class_acc)
plt.xticks(rotation=30, ha="right"); plt.ylim(0,1.0); plt.ylabel("Accuracy")
plt.title("Real-time accuracy per class")
plt.tight_layout(); plt.show()

# (Optional) save confusion to CSV for the thesis
pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv("confusion_counts.csv")
pd.DataFrame(cm_norm, index=LABELS, columns=LABELS).to_csv("confusion_row_normalized.csv")
