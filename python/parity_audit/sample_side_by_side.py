import pandas as pd, numpy as np, pickle

# ---- load artifacts
idx  = pd.read_csv("logs/tv_index.csv")                # n,file,start,label
py   = pd.read_csv("logs/predictions_python_exported.csv")  # n,true_label,pred_py,probs_py
es   = pd.read_csv("logs/predictions_cpp_esp32.csv")        # n,probs  OR n,p0,p1,...

# label names for indices
with open("python_ref/scalers/label_encoder.pkl","rb") as f:
    le = pickle.load(f)
class_names = list(le.classes_)

def parse_probs(s):
    if isinstance(s, str):
        return np.array([float(x) for x in s.split(";")], dtype=np.float32)
    return np.array(s, dtype=np.float32)

# normalize ESP columns
if es.shape[1] == 2 and "probs" in es.columns:
    es["vec_es"] = es["probs"].apply(parse_probs)
    es = es[["n","vec_es"]]
else:
    cols = ["n"] + [f"p{i}" for i in range(es.shape[1]-1)]
    es.columns = cols
    es["vec_es"] = es[[c for c in cols if c!="n"]].astype(float).apply(lambda r: r.to_numpy(np.float32), axis=1)
    es = es[["n","vec_es"]]

# merge & compute predicted classes
m = idx.merge(py[["n","pred_py","probs_py"]], on="n").merge(es, on="n")

m["py_vec"] = m["probs_py"].apply(parse_probs)
m["es_vec"] = m["vec_es"]
m["py_top1"] = m["py_vec"].apply(lambda v: int(np.argmax(v)))
m["es_top1"] = m["es_vec"].apply(lambda v: int(np.argmax(v)))
m["py_label"] = m["py_top1"].apply(lambda k: class_names[k])
m["esp32_label"] = m["es_top1"].apply(lambda k: class_names[k])
m["agree"]    = (m["py_top1"] == m["es_top1"])

# sample 10 rows
# replace the last lines in sample_side_by_side.py
sample = m.sample(n=10)   # no random_state -> different every run
print(sample[["n","file","start","label","py_label","esp32_label","agree"]].to_string(index=False))

