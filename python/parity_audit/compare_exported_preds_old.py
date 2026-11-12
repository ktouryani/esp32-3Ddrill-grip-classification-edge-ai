import pandas as pd, numpy as np 

py = pd.read_csv("logs/predictions_python_exported.csv") # n,true_label,pred_py,probs_py 

es = pd.read_csv("logs/predictions_cpp_esp32.csv") # should be: n,probs (or n,p0,p1,...) 

# Parse probs flexibly 
def parse_probs(s): 
    if isinstance(s, str): # "p0;p1;...;p4" 
        return np.array([float(x) for x in s.split(";")], dtype=np.float32) 
    # Numeric columns? (n,p0,...) 
    return np.array(s, dtype=np.float32) 

# Normalize ESP columns 
if es.shape[1] == 2 and "probs" in es.columns: 
    es["vec_es"] = es["probs"].apply(parse_probs) 
    es = es[["n","vec_es"]] 
else:
    cols = ["n"] + [f"p{i}" for i in range(es.shape[1]-1)] 
    es.columns = cols 
    es["vec_es"] = es[[c for c in cols if c!="n"]].astype(float).apply(lambda r: r.to_numpy(np.float32), axis=1) 
    es = es[["n","vec_es"]] 
    
# Merge 
m = py.merge(es, on="n", how="inner") 
py_mat = np.vstack(m["probs_py"].apply(parse_probs).to_numpy()) 
es_mat = np.vstack(m["vec_es"].to_numpy()) 

py_top1 = py_mat.argmax(axis=1) 
es_top1 = es_mat.argmax(axis=1) 

# Side-by-side table 
comp = pd.DataFrame({ 
    "n": m["n"], 
    "true_label": m["true_label"], 
    "pred_py": py_top1, 
    "pred_es": es_top1 
    }) 

# Save mismatches 
mism = comp[comp["pred_py"] != comp["pred_es"]].copy() 
mism.to_csv("logs/mismatches.csv", index=False) 

# Confusion matrix (Python vs ESP32) 
num_classes = es_mat.shape[1] 
cm = np.zeros((num_classes, num_classes), dtype=int) 
for p, e in zip(py_top1, es_top1): 
    cm[p, e] += 1 
    
print(f"Rows compared: {len(comp)}") 
print("Top-1 match rate: {:.2f}%".format((py_top1==es_top1).mean()*100)) 
print("\nConfusion (rows=Python pred, cols=ESP32 pred):\n", cm) 
print("\nSaved per-window table of mismatches to logs/mismatches.csv") 

# Also, how often each true label appears & is matched 
by_true = comp.groupby("true_label").apply(lambda df: (df["pred_py"]==df["pred_es"]).mean()*100) 
print("\nMatch rate by true label (%):\n", by_true)