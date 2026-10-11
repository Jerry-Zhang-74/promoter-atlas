# -*- coding: utf-8 -*-
"""BATCH29 three figures (no refit): metrics comparison, true-vs-pred, depth/length/sensitivity."""
import json, numpy as np, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("E:/promoter_atlas")
REP = ROOT / "reports/breakthrough_20261002/batch29"
FIG = ROOT / "figures/batch29"; FIG.mkdir(parents=True, exist_ok=True)
tm = json.load(open(REP / "test_metrics.json", encoding="utf-8"))
st = json.load(open(REP / "strata.json", encoding="utf-8"))
z = np.load(REP / "test_predictions.npz")
models = ["baseline_train_mean", "E1_ridge", "E2_cnn_true_seed20261101", "E3_cnn_true_seed20261102", "E4_cnn_shuffle_seed20261101"]
labels = [m.replace("_seed", "\ns").replace("cnn_true", "cnn").replace("_", " ") for m in models]

# Fig 1 metrics
def val(m, k):
    v = tm["models"][m]["all_test"][k]
    return float("nan") if v is None else v
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for a, (k, ttl) in zip(ax, [("spearman", "Spearman rho"), ("mae", "MAE"), ("group_balanced_mae", "group-balanced MAE"), ("r2", "R2")]):
    vals = [val(m, k) for m in models]
    a.bar(range(len(models)), [0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in vals])
    a.set_xticks(range(len(models))); a.set_xticklabels(labels, fontsize=7, rotation=20)
    a.set_title(ttl); a.axhline(0, color="k", lw=0.6)
fig.suptitle("batch29 v6.20.1-sureproxy.rc1 : test metrics on ALL test rows incl. zero-RNA (K562 per-fragment proxy)")
fig.tight_layout(); fig.savefig(FIG / "b29_metrics.png", dpi=130); plt.close(fig)

# Fig 2 true vs pred (selected)
sel = tm["selected_primary"]
yt = z["y"]; yp = z[sel]
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(yt, yp, s=4, alpha=0.3)
lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
ax[0].plot(lim, lim, "r--", lw=1)
ax[0].set_xlabel("observed y (proxy)"); ax[0].set_ylabel("predicted y (%s)" % sel); ax[0].set_title("true vs pred (test)")
frac0 = float(np.mean(yt == 0))
ax[1].hist(yt, bins=60); ax[1].set_title("test y distribution (zero frac=%.3f)" % frac0); ax[1].set_xlabel("y")
fig.tight_layout(); fig.savefig(FIG / "b29_true_vs_pred.png", dpi=130); plt.close(fig)

# Fig 3 depth / length / sensitivity
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
for m in ["E1_ridge", "E2_cnn_true_seed20261101", "E3_cnn_true_seed20261102"]:
    d = st[m]["depth"]; labs = list(d.keys())
    ax[0].plot(range(len(labs)), [d[l]["spearman"] for l in labs], marker="o", label=m)
    ax[1].plot(range(len(labs)), [d[l]["spearman"] for l in labs], marker="o", label=m)
    break
for m in ["E1_ridge", "E2_cnn_true_seed20261101", "E3_cnn_true_seed20261102"]:
    L = st[m]["length"]; labs = list(L.keys())
    ax[0].plot(range(len(labs)), [L[l]["spearman"] for l in labs], marker="s", ls="--", label=m + " (len)")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(list(st["E1_ridge"]["depth"].keys()))
ax[0].set_ylabel("spearman"); ax[0].legend(fontsize=6); ax[0].set_title("by depth bin (fixed xticks=depth)")
# length panel
for m in ["E1_ridge", "E2_cnn_true_seed20261101", "E3_cnn_true_seed20261102"]:
    L = st[m]["length"]; labs = list(L.keys())
    ax[1].plot(range(len(labs)), [L[l]["spearman"] for l in labs], marker="o", label=m)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(list(st["E1_ridge"]["length"].keys()))
ax[1].set_title("spearman by length bin"); ax[1].legend(fontsize=7)
sens = tm["target_sensitivity"]; ms = list(sens.keys())
w = 0.25
ax[2].bar(np.arange(len(ms)) - w, [sens[m]["rho_vs_primary_y"] for m in ms], w, label="D=1+P (primary)")
ax[2].bar(np.arange(len(ms)), [sens[m]["rho_vs_D2plus_y"] for m in ms], w, label="D=2+P")
ax[2].bar(np.arange(len(ms)) + w, [sens[m]["rho_vs_D05plus_y"] for m in ms], w, label="D=0.5+P")
ax[2].set_xticks(range(len(ms))); ax[2].set_xticklabels([m.split("_")[0] for m in ms]); ax[2].legend(fontsize=7)
ax[2].set_title("label sensitivity (no refit)")
fig.tight_layout(); fig.savefig(FIG / "b29_domain_sensitivity.png", dpi=130); plt.close(fig)
print("figures written")
