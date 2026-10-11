# -*- coding: utf-8 -*-
"""batch29 reviewer: CONCRETE demonstration of the native-length contract violation.
Loads the saved E2 checkpoint (no fitting) and runs FORWARD-ONLY inference on CPU
two ways: (A) exactly as b29_train/predict_all does (full width 2000, pad code 4),
(B) native length L (slice codes[:, :L]). Reports how much the val output changes.
"""
import json, csv, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("E:/promoter_atlas")
sys.path.insert(0, str(ROOT/"runs/breakthrough_20261002/code/batch02"))
import trackB_grouped_cv as T

DB = ROOT/"data/breakthrough_20261002/batch29"
ATT = ROOT/"runs/breakthrough_20261002/batch29/attempts/E2_cnn_true_seed20261101"

z = np.load(DB/"encoded.npz"); codes, lens = z["codes"], z["lens"]
print("encoded.codes shape:", codes.shape, "dtype", codes.dtype, "pad code count:", int((codes==4).sum()))
print("lens range:", int(lens.min()), int(lens.max()), "maxlen:", codes.shape[1])
frac_pad = float((codes==4).sum())/codes.size
print("fraction of ALL code cells that are pad code 4: %.4f" % frac_pad)

rows = list(csv.DictReader(open(DB/"rows.tsv", encoding="utf-8"), delimiter="\t"))
sp = np.array([r["split"] for r in rows], dtype=object)
va = np.where(sp=="val")[0]

ck = torch.load(ATT/"ckpt.pt", map_location="cpu", weights_only=False)
model = T.build_cnn("legnet"); model.load_state_dict(ck["model"]); model.eval()

def fwd(ids):
    with torch.no_grad():
        x = torch.from_numpy(codes[ids].astype(np.int64))
        return model(x)[:,0].numpy()

def fwd_native(ids):
    outs = np.zeros(len(ids))
    for k,i in enumerate(ids):
        L = int(lens[i])
        with torch.no_grad():
            x = torch.from_numpy(codes[i:i+1,:L].astype(np.int64))
            outs[k] = model(x)[:,0].item()
    return outs

sel = va[:120]  # bounded, CPU forward only
pa = fwd(sel)            # width-2000 (as worker code)
pn = fwd_native(sel)     # native length L
mu, sd = 0.26694700089558726, 0.608848616327574
pa_y, pn_y = pa*sd+mu, pn*sd+mu
print("mean|pad - native| in y-units: %.4f  max: %.4f" % (np.mean(np.abs(pa_y-pn_y)), np.max(np.abs(pa_y-pn_y))))
print("corr(pad, native) pearson: %.4f" % np.corrcoef(pa, pn)[0,1])
print("example L=200 row: pad_pred_z=%.4f native_pred_z=%.4f" % (pa[0], pn[0]))
# length dependence of the padding penalty
Lsel = lens[sel]
for lo,hi in [(200,499),(500,999),(1000,1499),(1500,2000)]:
    m = (Lsel>=lo)&(Lsel<=hi)
    if m.sum()>=3:
        print("  L in [%d,%d] n=%d  mean|diff|_y=%.4f" % (lo,hi,int(m.sum()), float(np.mean(np.abs(pa_y[m]-pn_y[m])))))
json.dump({"width_from_code": int(codes.shape[1]), "native_len_max": int(lens.max()),
           "frac_pad_cells": frac_pad, "mean_abs_diff_y_120val": float(np.mean(np.abs(pa_y-pn_y))),
           "max_abs_diff_y_120val": float(np.max(np.abs(pa_y-pn_y))),
           "pearson_pad_vs_native": float(np.corrcoef(pa,pn)[0,1])},
          open(ROOT/"reports/breakthrough_20261002/batch29_review/padding_demo.json","w",encoding="utf-8"), indent=2)
