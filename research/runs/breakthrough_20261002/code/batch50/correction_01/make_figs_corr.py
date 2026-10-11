# -*- coding: utf-8 -*-
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["svg.fonttype"]="none"
ROOT=Path("E:/promoter_atlas")
SRC=ROOT/"reports/breakthrough_20261002/batch50"                    # frozen numbers (read-only)
FIG=ROOT/"reports/breakthrough_20261002/batch50/correction_01/figures"
FIG.mkdir(parents=True,exist_ok=True)
fi=json.loads((SRC/"figure_inputs.json").read_text(encoding="utf-8"))
g1=fi["fig1_local293ft_applicability"]; g2=fi["fig2_public_transfer_sameassay"]; g3=fi["fig3_route_capability"]
def val(d): return d["value"]
print("figs script start")

# ---------------- FIGURE 1 ----------------
fam=["ACTB","GAPDH","CMV","宏平均"]
famr=["ACTB","GAPDH","CMV","合计（3/21）"]
parm=[val(g1["ACTB_PARM_direction_accuracy"]),val(g1["GAPDH_PARM_direction_accuracy"]),val(g1["CMV_PARM_direction_accuracy"]),val(g1["macro_family_accuracy"])]
inc =[val(g1["ACTB_baseline_increasing_len"]),val(g1["GAPDH_baseline_increasing_len"]),val(g1["CMV_baseline_increasing_len"]),val(fi["fig1_local293ft_applicability"]["macro_family_accuracy"])]
dec =[val(g1["ACTB_baseline_decreasing_len"]),val(g1["GAPDH_baseline_decreasing_len"]),val(g1["CMV_baseline_decreasing_len"]),None]
inc[3]=0.56666666; dec[3]=0.43333333
corr=[0,2,1,3]; tot=[10,10,1,21]
fig,axs=plt.subplots(1,2,figsize=(13.2,6.6))
ax=axs[0]; import numpy as np
x=np.arange(4); w=0.26
ax.bar(x-w,parm,w,label="PARM 方向准确率",color="#c0392b")
ax.bar(x,inc,w,label="固定基线：+长度",color="#2980b9")
ax.bar(x+w,dec,w,label="固定基线：-长度",color="#7fb3d5")
for i,v in enumerate(parm): ax.text(x[i]-w,v+0.02,f"{v:.2f}",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(fam); ax.set_ylim(0,1.05)
ax.set_ylabel("族内有序对方向准确率"); ax.set_title("A 各族方向准确率 vs 固定长度基线")
ax.legend(fontsize=9,loc="upper left"); ax.grid(axis="y",alpha=0.3)
print("fig1 A done")

ax=axs[1]; y=np.arange(4)
ax.barh(y,[c/ t for c,t in zip(corr,tot)],color="#27ae60",label="预测方向正确")
ax.barh(y,[(t-c)/t for c,t in zip(corr,tot)],left=[c/t for c,t in zip(corr,tot)],color="#e67e22",label="方向错误")
for i,(c,t) in enumerate(zip(corr,tot)): ax.text(1.02,y[i],f"{c}/{t}",va="center",fontsize=10)
ax.set_yticks(y); ax.set_yticklabels(famr); ax.set_xlim(0,1.13); ax.set_xlabel("族内有序对比例")
ax.set_title("B 实际正确/总数（D28 终点）"); ax.legend(fontsize=9,loc="lower right"); ax.grid(axis="x",alpha=0.3)
fig.suptitle("图1 局部293FT适用性：12个受支持构建；软件上限600bp（D28 终点）",fontsize=13)
foot=("数据源 b31 metrics.json /local/families（解析正确/总数；SHA 见 figure_inputs.json）。"
 "要点：D28 终点=每个构建 3 个技术管门的均值 / 同批 WT阴性参考(log10)——无生物学独立 n。"
 "3 族（ACTB/GAPDH/CMV）不等于 12 个受支持构建、21 个相互依赖的有序比较；不提供泛化置信区间；"
 "A20/G20 2000bp 超出软件 L_max=600 → UNSUPPORTED（仅观测）；样本→质粒条码未验证。"
 "右列“合计（3/21）”为池化总数 14.29%（非宏平均）；左列宏平均=0.40 为三族准确率均值。")
fig.text(0.5,0.015,foot,ha="center",va="bottom",fontsize=8.2,wrap=True,
         bbox=dict(boxstyle="round",fc="#fdf6e3",ec="#b58900"))
fig.tight_layout(rect=[0,0.10,1,0.95])
fig.savefig(FIG/"fig1_local293ft_applicability.png",dpi=170)
fig.savefig(FIG/"fig1_local293ft_applicability.svg")
plt.close(fig)
print("fig1 saved")

# ---------------- FIGURE 2 ----------------
fig,axs=plt.subplots(2,2,figsize=(13.6,9.4))
lbl=["PARM","H1","GC"]
rho41=[val(g2["b41_pooled_rho_PARAM"]),val(g2["b41_pooled_rho_H1"]),val(g2["b41_pooled_rho_GC"])]
top41=[val(g2["b41_top10_precision_PARAM"]),val(g2["b41_top10_precision_H1"]),val(g2["b41_top10_precision_GC"])]
ax=axs[0,0]; ax.bar(lbl,rho41,color=["#c0392b","#2980b9","#27ae60"])
for i,v in enumerate(rho41): ax.text(i,v+0.004,f"{v:.4f}",ha="center",fontsize=9)
ax.axhline(0.30,ls="--",color="k",lw=1); ax.text(2.3,0.305,"预设门槛 0.30",ha="right",fontsize=8.5)
ax.set_ylim(0,0.36); ax.set_ylabel("pooled Spearman ρ"); ax.grid(axis="y",alpha=0.3)
ax.set_title("A b41 公开HEK293FT冻结迁移：池化ρ（n=7537生物行, 270bp插入）")
ax=axs[0,1]; ax.bar(lbl,top41,color=["#c0392b","#2980b9","#27ae60"])
for i,v in enumerate(top41): ax.text(i,v+0.003,f"{v:.4f}",ha="center",fontsize=9)
ax.axhline(0.10,ls="--",color="k",lw=1); ax.text(2.3,0.103,"基线比例 0.10",ha="right",fontsize=8.5)
ax.set_ylim(0,0.20); ax.set_ylabel("前10%精确率"); ax.grid(axis="y",alpha=0.3)
ax.set_title("B b41 前10%精确率（PARM<H1<GC）")
cu=["常数基线","R0 组成岭","R1 全序列岭","Rnull 置换标签"]
mae=[val(g2["b42_const_groupMAE"]),val(g2["b42_R0_groupMAE"]),val(g2["b42_R1_groupMAE"]),val(g2["b42_Rnull_groupMAE"])]
ax=axs[1,0]; ax.bar(cu,mae,color=["#95a5a6","#27ae60","#c0392b","#8e44ad"])
for i,v in enumerate(mae): ax.text(i,v+0.003,f"{v:.4f}",ha="center",fontsize=9)
ax.set_ylim(0,0.21); ax.set_ylabel("组平衡 MAE（alpha 单位，越低越好）"); ax.grid(axis="y",alpha=0.3)
ax.set_title("C b42 同单位组平衡MAE（R1未优于R0，常数最低）")
ax=axs[1,1]
rho42=[val(g2["b42_R0_rho"]),val(g2["b42_R1_rho"]),val(g2["b42_Rnull_rho"])]
ax.bar(["R0","R1","Rnull"],rho42,color=["#27ae60","#c0392b","#8e44ad"])
for i,v in enumerate(rho42): ax.text(i,v+(0.004 if v>=0 else -0.012),f"{v:.4f}",ha="center",fontsize=9)
ax.axhline(0,color="k",lw=0.8)
ax.text(0.02,0.02,"常数基线：Spearman 未定义（常数列，非0）",transform=ax.transAxes,fontsize=9,
        bbox=dict(boxstyle="round",fc="#f0f0f0",ec="#999"))
ax.set_ylim(-0.12,0.20); ax.set_ylabel("pooled Spearman ρ"); ax.grid(axis="y",alpha=0.3)
ax.set_title("D b42 池化ρ（R1<R0，Rnull为负）")
print("fig2 panels done")

fig.suptitle("图2 公开迁移与同实验学习：b41 冻结迁移 与 b42 固定三次拟合（均含负/常数/空结果）",fontsize=13)
foot2=("b41：公开 HEK293FT OligoMPRA getAlpha 活性，270bp 插入，7537 生物行；PARM-H1 差值 %.4f，条件组分自举 CI [%.4f, %.4f]；PARM 未过 0.30 门槛 → LIMITED。\n"
 "b42：冻结并集划分 TRAIN 7219行/92组，TEST 318行/21组；同 alpha 单位的 R0/R1/Rnull 与常数基线；R1 未胜 R0（Δρ=%.4f）；TEST 组 21<30 → 无 CI/显著性（未定义，非0）。\n"
 "b43 纠正：batch42 中缓存的 PARM/H1 分-对-alpha MAE（.466390/.353606）为跨单位无效，本图不显示；TEST 误差以组内项为主（R0 %.3f, R1 %.3f, Rnull %.3f, 常数 %.3f）。\n"
 "b31 HEK673 仅作分离的语境陈述：PARM ρ=%.4f（回顾性，训练暴露 UNKNOWN；非本地293FT验证，不与上图共用“改进”坐标）。"
 ) % (val(g2["b41_parm_minus_h1"]),val(g2["b41_parm_minus_h1_CI_lo"]),val(g2["b41_parm_minus_h1_CI_hi"]),
      val(g2["b42_R1_minus_R0_rho"]),val(g2["b43_within_frac"]["R0"]),val(g2["b43_within_frac"]["R1"]),
      val(g2["b43_within_frac"]["Rnull"]),val(g2["b43_within_frac"]["const"]),val(g2["b31_HEK673_context"]))
fig.text(0.5,0.008,foot2,ha="center",va="bottom",fontsize=8.0,
         bbox=dict(boxstyle="round",fc="#fdf6e3",ec="#b58900"))
fig.tight_layout(rect=[0,0.115,1,0.955])
fig.savefig(FIG/"fig2_public_transfer_sameassay.png",dpi=170)
fig.savefig(FIG/"fig2_public_transfer_sameassay.svg")
plt.close(fig)
print("fig2 saved")

# ---------------- FIGURE 3 ----------------
fig,ax=plt.subplots(figsize=(13.6,9.6)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
def box(x0,y0,x1,y1,title,body,fc,ec="#333",ts=10.5,bs=9.0):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0.06",
        fc=fc,ec=ec,lw=1.4))
    ax.text((x0+x1)/2,y1-0.28,title,ha="center",va="top",fontsize=ts,fontweight="bold")
    ax.text(x0+0.14,y1-0.72,body,ha="left",va="top",fontsize=bs,wrap=True)
def arrow(x0,y0,x1,y1,c="#555"):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=16,lw=1.6,color=c))
ax.text(5,9.88,"图3 证据链接的路线/能力图",ha="center",fontsize=13.5,fontweight="bold")
box(0.2,8.0,9.8,9.5,"输入软件能力（已交付，batch49 CLI；本报告无新拟合/无新下载）",
    "fragment：原始FASTA → 冻结的公开 HEK293 PARM 分数（推理用 CUDA 前向；无新拟合，权重冻结）。\n"
    "软件支持 1..600；>600（含2000bp）→ 分数 null / UNSUPPORTED_LENGTH；目标 293FT 仅元数据=UNVALIDATED。\n"
    "gene-observed/gene-rank：CPU 目录查询/选择；E_raw×S_raw Pareto，B_raw 仅门控；默认前沿 12/18947；跨组织，非293FT/GFP。",
    "#eaf2fb")
box(0.2,5.0,3.5,7.42,"局部293FT：未验证（D28 FAIL）",
    "族内方向 3/21\n(ACTB 0/10, GAPDH 2/10, CMV 1/1)；\n"
    "宏平均 0.40 < 固定长度基线。\n"
    "仅 3 技术管，非生物学 n；\n条码 UNKNOWN；A20/G20 2000bp 不支持。\n"
    "无证据表明对新293FT/截短序列有改善。","#fdecea","#c0392b",ts=9.6,bs=8.4)
box(3.7,5.0,6.8,7.42,"公开冻结迁移：LIMITED（非改进）",
    "b41（非本批监督学习）：公开\nHEK293FT OligoMPRA 上冻结\nPARM 池化ρ .108379 < H1 .109938\n≈ GC .121350；未过 0.30 门槛。\n"
    "两项已测过程未见可部署改进。","#fdecea","#c0392b",ts=9.6,bs=8.4)
box(7.0,5.0,9.8,7.42,"同实验监督试点：LIMITED",
    "b42（同实验内监督，非跨细胞）：\nR1 全序列岭 ρ .091055 <\nR0 组成岭 .158409；常数基线\n组平衡MAE 最低；21 组无CI。\n失败机制未定。","#fdecea","#c0392b",ts=9.6,bs=8.4)
print("fig3 boxes1 done")

box(0.2,1.2,4.8,4.15,"已有数据（present）",
    "公开 HEK293FT OligoMPRA 7537 生物行\n（270bp 同插入文库；b41 中已全部暴露）。\n"
    "本地 14 构建（A1..A5,G1..G5,CMVcore/full,\nA20/G20）为技术重复。\n"
    "→ 这些不等于“新的独立证据”。","#eafaf1","#27ae60",ts=10.0,bs=8.6)
box(5.2,1.2,9.8,4.15,"相关的对外需求（非因果证明）",
    "评估上下文匹配的新序列：同实验/同亲本/共同3'端 的变长序列，配权威实际DNA/上下文与分组。\n"
    "≥30 留出组只是此前协议用于条件CI的选定门槛，非生物学定律、也非独立性的充分证明；\n"
    "更小的真实试点可为描述性、无CI/无操作 PASS。\n"
    "未取得前：不自动拟合、不在已暴露TEST上继续调参、不重开已关闭 GSE301246v2。","#fff7e6","#b58900",ts=10.0,bs=8.2)
arrow(1.85,8.0,1.85,7.44)
arrow(5.25,8.0,5.25,7.44)
arrow(8.4,8.0,8.4,7.44)
ax.text(0.2,0.72,"本报告=证据综合（report revision），非性能增益、非项目完成；不宣称已发布模型整体无用或数据不存在。",
        ha="left",fontsize=8.4,color="#555")
fig.tight_layout()
fig.savefig(FIG/"fig3_route_capability.png",dpi=170)
fig.savefig(FIG/"fig3_route_capability.svg")
plt.close(fig)
print("fig3 saved")
