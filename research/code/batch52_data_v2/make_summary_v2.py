# -*- coding: utf-8 -*-
"""Assemble the batch52 delivery: figures + Chinese short result + prediction command.
Reads the frozen DEV histories (Nabs_ready.json/Njoint_ready.json) and the final TEST JSON."""
from __future__ import annotations
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:/promoter_atlas"
MODELS = os.path.join(ROOT, "data/breakthrough_20261002/batch52/corrected/models")
OUTDIR = os.path.join(ROOT, "reports/breakthrough_20261002/batch52")
FIG = os.path.join(OUTDIR, "figures")
os.makedirs(FIG, exist_ok=True)


def _load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fig_training(nabs, njoint):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, r, c in (("Nabs", nabs, "#1f77b4"), ("Njoint", njoint, "#d62728")):
        if not r:
            continue
        h = r["history"]
        ep = [x["epoch"] for x in h]
        ax[0].plot(ep, [x["loss"] for x in h], marker="o", color=c, label=name)
        ax[1].plot(ep, [x["dev"]["rho"] for x in h], marker="o", color=c, label=name)
        ax[2].plot(ep, [x["dev_pair_mae"] for x in h], marker="o", color=c, label=name)
    ax[0].set_title("TRAIN loss (z-units)"); ax[1].set_title("DEV activity Spearman")
    ax[2].set_title("DEV group-mean pair MAE")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend()
    fig.suptitle("batch52 Nabs vs Njoint training (6 full TRAIN epochs, DEV selection)")
    fig.tight_layout()
    p = os.path.join(FIG, "training_curves.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_activity(final):
    arms = final["arms"]
    names = [a for a in ("R0", "Nabs", "Njoint") if a in arms]
    rho = [arms[a]["activity"]["rho"] for a in names]
    mae = [arms[a]["activity"]["group_mae"] for a in names]
    t10 = [arms[a]["activity"]["top10_precision"] for a in names]
    x = range(len(names))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].bar(x, rho, color="#4c72b0"); ax[0].set_title("TEST activity Spearman")
    ax[1].bar(x, mae, color="#dd8452"); ax[1].set_title("TEST group-equal MAE")
    ax[2].bar(x, t10, color="#55a868"); ax[2].set_title("TEST top-10%% precision")
    for a in ax:
        a.set_xticks(list(x)); a.set_xticklabels(names); a.grid(alpha=0.3, axis="y")
    fig.suptitle("batch52 held-out (TEST) activity metrics (all arms)")
    fig.tight_layout()
    p = os.path.join(FIG, "test_activity.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_pair(final):
    arms = final["arms"]; base = final["fixed_baselines"]
    labels, mae, sign, cols = [], [], [], []
    for a in ("Nabs", "Njoint"):
        labels.append(a); mae.append(arms[a]["pair"]["group_mean_mae"])
        sign.append(arms[a]["pair"]["macro_sign_accuracy"]); cols.append("#4c72b0" if a == "Nabs" else "#c44e52")
    for b, v in base.items():
        labels.append(b); mae.append(v["group_mean_mae"]); sign.append(v["macro_sign_accuracy"]); cols.append("#999999")
    x = range(len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].bar(x, mae, color=cols); ax[0].set_title("TEST pair group-mean MAE(delta)")
    ax[1].bar(x, sign, color=cols); ax[1].set_title("TEST macro strict sign accuracy")
    bt = final.get("bootstrap_Nabs_vs_Njoint", {})
    if bt.get("ok"):
        d = bt["mae_A_minus_B"]
        ax[1].text(0.02, 0.05, "MAE(A-B) 95%% CI [%.3f, %.3f]" % (d["lo"], d["hi"]), transform=ax[1].transAxes)
    for a in ax:
        a.set_xticks(list(x)); a.set_xticklabels(labels, rotation=25, ha="right"); a.grid(alpha=0.3, axis="y")
    fig.suptitle("batch52 held-out (TEST) pair-delta metrics vs fixed baselines")
    fig.tight_layout()
    p = os.path.join(FIG, "test_pair.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    nabs = _load(os.path.join(MODELS, "Nabs_ready.json"))
    njoint = _load(os.path.join(MODELS, "Njoint_ready.json"))
    final = _load(os.path.join(OUTDIR, "final_test_metrics.json"))
    figs = []
    if nabs or njoint:
        figs.append(fig_training(nabs, njoint))
    if final:
        figs.append(fig_activity(final)); figs.append(fig_pair(final))
    cmd = (".venv/Scripts/python.exe -X utf8 code/batch52_data_v2/predict_v2.py "
           "--checkpoint data/breakthrough_20261002/batch52/corrected/models/Nabs/Nabs_selected.ckpt --example")
    lines = ["batch52 v6.38.0-full-hek-nested-multitask.rc1 -- 结果摘要", "=" * 40]
    if final:
        for a in ("R0", "Nabs", "Njoint"):
            if a in final["arms"]:
                ac = final["arms"][a]["activity"]; pr = final["arms"][a].get("pair", {})
                lines.append("%s: activity rho=%.3f groupMAE=%.3f top10=%.3f | pair groupMAE=%.3f macroSign=%.3f"
                             % (a, ac["rho"], ac["group_mae"], ac["top10_precision"],
                                pr.get("group_mean_mae", float("nan")), pr.get("macro_sign_accuracy", float("nan"))))
        lines.append("investigation_target: %s" % json.dumps(final["investigation_target"], ensure_ascii=False))
        lines.append("verdict: %s" % final["verdict"])
    lines += ["", "预测命令 (usable prediction command):", cmd,
              "模型: Nabs/Njoint 序列CNN + R0 组成ridge；目标 Log2RPM_HEK293；参考=Ensembl GRCh37(hg19)代理，非克隆序列/非293FT验证"]
    open(os.path.join(OUTDIR, "RESULT_CN.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(json.dumps({"figures": figs, "final": bool(final)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
