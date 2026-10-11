# -*- coding: utf-8 -*-
"""BATCH27 figure 3: long-fragment (SuRE K562) cohort coverage. Descriptive only."""
import json, gzip, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "reports/breakthrough_20261002/batch27/"
p1 = json.load(open(R + "sure_pass1.json", encoding="utf-8"))
g1 = json.load(open(R + "sure_group_graph.json", encoding="utf-8"))
g2 = json.load(open(R + "sure_tier2_group_graph.json", encoding="utf-8"))

lh = {int(k): v for k, v in p1["len_hist_100bp"].items()}
ks = sorted(lh)
tot = sum(lh.values())

per_chrom = collections.Counter()
with gzip.open("data/breakthrough_20261002/batch27/sure_tier2_cohort.tsv.gz", "rt", encoding="utf-8") as f:
    f.readline()
    for line in f:
        per_chrom[line.split("\t", 1)[0]] += 1

order = ["chr%d" % i for i in range(1, 23)] + ["chrX", "chrY"]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
a = ax[0]
a.bar([k + 50 for k in ks], [lh[k] for k in ks], width=95, color="#4477aa")
a.axvspan(200, 2000, color="#ffcc66", alpha=0.25, label="tier-1 window 200-2000 bp")
a.set_yscale("log")
a.set_xlabel("measured insert length (bp)")
a.set_ylabel("rows (log10)")
a.set_title("all %d SuRE K562 fragment rows\n(139,978,177 rows with len>=200)" % tot)
a.legend(fontsize=7)

a = ax[1]
vals = [per_chrom.get(c, 0) for c in order]
a.bar(range(len(order)), vals, color="#66c2a5")
a.set_xticks(range(len(order)))
a.set_xticklabels([c.replace("chr", "") for c in order], fontsize=7)
a.set_xlabel("chromosome")
a.set_ylabel("inserts")
a.set_title("tier-2 analysed cohort per chromosome\n(n=%d, both bioreps cDNA>=1)" % g2["tier2_cohort_size"])

a = ax[2]
h = {int(k): v for k, v in g2["component_size_hist_top"].items()}
sizes = []
for k, v in h.items():
    sizes += [k] * v
sizes = np.array(sizes)
if len(sizes):
    smax = int(sizes.max())
    bins = np.arange(0.5, smax + 1.5)
    a.hist(sizes, bins=bins, color="#cc6677")
    a.set_yscale("log")
a.set_xlabel("overlap-component size (inserts)")
a.set_ylabel("components (log10)")
a.set_title("genome-wide overlap components\n%d total, %d singletons, max %d" % (
    g2["n_overlap_components"], g2["n_singleton_components"], g2["component_size_max"]))
fig.suptitle("BATCH27 SuRE K562 long-fragment cohort coverage (label-free structure; no model)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(R + "figures/long_fragment_cohort.png", dpi=130)
print("saved")
