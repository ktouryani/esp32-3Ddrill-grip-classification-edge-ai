import pandas as pd
import numpy as np

# Load logs
f_py  = pd.read_csv("logs/features_python.csv")
fs_py = pd.read_csv("logs/scaled_features_python.csv")
f_cpp = pd.read_csv("logs/features_cpp.csv")
fs_cpp= pd.read_csv("logs/scaled_features_cpp.csv")

# Sanity check lengths
N = min(len(f_py), len(f_cpp))
if len(f_py) != len(f_cpp):
    print(f"⚠️ Length mismatch: python={len(f_py)} cpp={len(f_cpp)}. Using first {N} rows.")

# Compute diffs
diff_raw   = np.abs(f_py.iloc[:N].to_numpy() - f_cpp.iloc[:N].to_numpy())
diff_scaled= np.abs(fs_py.iloc[:N].to_numpy() - fs_cpp.iloc[:N].to_numpy())

print("=== Feature diff summary ===")
print("Raw   max abs diff per feature:", diff_raw.max(axis=0))
print("Raw   mean abs diff per feature:", diff_raw.mean(axis=0))
print("Scaled max abs diff per feature:", diff_scaled.max(axis=0))
print("Scaled mean abs diff per feature:", diff_scaled.mean(axis=0))

# Pick a few random rows to display side by side
sample_idx = np.random.choice(N, size=5, replace=False)
cols = f_py.columns

print("\n=== Random sample of raw features ===")
for i in sample_idx:
    print(f"\nWindow {i}:")
    for j,c in enumerate(cols):
        print(f"{c:25s} Py={f_py.iloc[i,j]:.6f} | Cpp={f_cpp.iloc[i,j]:.6f} | Δ={diff_raw[i,j]:.2e}")

print("\n=== Random sample of scaled features ===")
for i in sample_idx:
    print(f"\nWindow {i}:")
    for j,c in enumerate(cols):
        print(f"{c:25s} Py={fs_py.iloc[i,j]:.6f} | Cpp={fs_cpp.iloc[i,j]:.6f} | Δ={diff_scaled[i,j]:.2e}")
