import pandas as pd
import numpy as np

py_raw = pd.read_csv("logs/features_python.csv")
cpp_raw = pd.read_csv("logs/features_cpp.csv")

py_scaled = pd.read_csv("logs/scaled_features_python.csv")
cpp_scaled = pd.read_csv("logs/scaled_features_cpp.csv")

print("=== RAW FEATURE COMPARISON ===")
for c in py_raw.columns:
    diff = np.abs(py_raw[c]-cpp_raw[c])
    print(f"{c}: max={diff.max():.4e}, mean={diff.mean():.4e}")

print("\n=== SCALED FEATURE COMPARISON ===")
for c in py_scaled.columns:
    diff = np.abs(py_scaled[c]-cpp_scaled[c])
    print(f"{c}: max={diff.max():.4e}, mean={diff.mean():.4e}")
