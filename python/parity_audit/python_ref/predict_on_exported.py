import numpy as np, csv, tensorflow as tf

X_time  = np.load("../logs/tv_windows.npy")       # (N,128,3)
X_featS = np.load("../logs/tv_feats_scaled.npy")  # (N,7)
labels  = np.load("../logs/tv_labels.npy", allow_pickle=True)  # (N,)

inter = tf.lite.Interpreter(model_path="../python_ref/model/model_int8.tflite")
inter.allocate_tensors()
ins  = inter.get_input_details(); outs = inter.get_output_details()
in_imu, in_feat = ins[0], ins[1]; out = outs[0]
imu_scale, imu_zp = in_imu['quantization']; feat_scale, feat_zp = in_feat['quantization']
out_scale, out_zp = out['quantization']

def quantize(x, s, zp):
    q = np.round(x/s + zp).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)

rows=[]
for i in range(X_time.shape[0]):
    imu_q  = quantize(X_time[i],  imu_scale,  imu_zp)
    feat_q = quantize(X_featS[i], feat_scale, feat_zp)
    inter.set_tensor(in_imu['index'], imu_q.reshape(in_imu['shape']))
    inter.set_tensor(in_feat['index'], feat_q.reshape(in_feat['shape']))
    inter.invoke()
    y = inter.get_tensor(out['index'])
    if out['dtype'] == np.int8:
        probs = (y.astype(np.float32) - out_zp) * out_scale
    else:
        probs = y.astype(np.float32)
    probs = probs.reshape(-1)
    pred  = int(np.argmax(probs))
    rows.append((i, labels[i], pred, ";".join(f"{p:.6f}" for p in probs)))

with open("../logs/predictions_python_exported.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["n","true_label","pred_py","probs_py"])
    w.writerows(rows)

print("Wrote logs/predictions_python_exported.csv")
