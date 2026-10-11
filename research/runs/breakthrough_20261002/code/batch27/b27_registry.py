# -*- coding: utf-8 -*-
"""BATCH27: append-only registry + changelog entry. Preserves every prior entry; no pointer changes."""
import json, os, hashlib, time, shutil

REG = "reports/breakthrough_20261002/version_registry.json"
CL = "reports/breakthrough_20261002/CHANGELOG.txt"
R = "reports/breakthrough_20261002/batch27/"
now = time.strftime("%Y-%m-%dT%H:%M:%S+08:00")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

old_sha = sha(REG)
d = json.load(open(REG, encoding="utf-8"))
keys_before = list(d["data_versions"])
assert "d_sure_k562_batch27" not in keys_before
d["data_versions"]["d_sure_k562_batch27"] = {
  "path": "data/breakthrough_20261002/batch27/source/GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz",
  "sha256": "8868fa641c2c89bef0c8aa18b6b553030cc7199ea4be4f12c2cd1f3ecd458b85",
  "role": "PRIMARY measured per-fragment SuRE K562 counts (GSE78709); tier-1 eligible 8,033,394; tier-2 253,629",
  "note": "batch27 append; read-only source; NOT a training freeze - no model was trained on it",
  "secondary_table": {"path": "data/breakthrough_20261002/batch27/source/GSE78709_SuRE34.counts.JvA170901.txt.gz",
                      "sha256": "240c0ed2e9e31599401b35826a769a7f2d6fe0d59e1c33c9c93432b85efe424b",
                      "used": False, "reason": "different SuRE library; header/row field counts inconsistent"},
  "derived": {"tier2_cohort": "data/breakthrough_20261002/batch27/sure_tier2_cohort.tsv.gz",
              "manifest": R + "autonomous_fragment_data_ready.json",
              "manifest_sha256": sha(R + "autonomous_fragment_data_ready.json")},
  "appended_at": now, "registry_sha256_before_append": old_sha
}
tmp = REG + ".tmp_batch27"
json.dump(d, open(tmp, "w", encoding="utf-8"), indent=1)
os.replace(tmp, REG)
assert list(json.load(open(REG, encoding="utf-8"))["data_versions"])[:len(keys_before)] == keys_before
print("registry appended; prior keys preserved:", keys_before)

lines = []
lines.append("[%s] batch27 DATA/EVIDENCE REVISION (v6.19.1-promoter-readiness.rc1) - NO model, NO estimator fit, NO inference. " % now)
lines.append("A1: batch26 fit-budget FAIL preserved (5 executions vs declared 4); corrected manifest addendum with b26_train.py prereg vs executed + save-fix, b26_figs.py hash and malformed resume_check key; first test read 17:56:51 AFTER selection 17:56:46 (18:07:53 is a later regeneration). ")
lines.append("A2: label-free contamination list built BEFORE subset scores from group representatives (3 same-length Hamming<=16 edges + 1 variable-length cov90/ident90 edge = 3 cross-split edges -> 2 excluded TEST groups). Retained 2838 groups/8793 rows. A1 rho 0.471871->0.471464, group-balanced MAE 0.355892->0.356009; A0 0.418832->0.418425. Bootstrap1000 delta-rho A1-A0 +0.0524 [0.0341,0.0721]. Ranking gain PERSISTS; post-hoc sensitivity, not a new independent test; unknown near-homology unresolved. ")
lines.append("A3: descriptive variant-delta binning (motif-length and |delta| bins, exact zero separate); sign agreement rises with |observed delta|; shuffled control Spearman stays UNDEFINED (not converted to zero). ")
lines.append("B: GSE78709 SuRE K562 autonomous-promoter source resolved. File GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz sha256 8868fa64...458b85, 1,187,342,132 B, Content-Length matched, gzip CRC/EOF OK. 153,562,178 measured insert rows; tier-1 eligible (canonical, 200-2000bp, input>=10) 8,033,394; tier-2 (cDNA detected in BOTH bioreps) 253,629 inserts / 186,969 independent overlap components. Hash-ordered label-blind selection of 25,000 frozen before labels were read. Sequence: Ensembl GRCh37, 2500/2500 exact length match, 0 N, 0 exact-dup pairs, 0 RC pairs. DATA-READY GATE MET (>=1000 inserts AND >=500 groups). No training performed; tier-2 eligibility needs parent approval. ")
lines.append("C: reusable execution gate (atomic append-only ledger, reservation before callback, 5th call of max-4 rejected, failed retry consumes 2, crash consumes reserved slot, resume only with recorded model/optimizer/RNG/epoch identity); 9/9 mock tests, zero numerical fits. ")
lines.append("Production pointer unchanged (v6.3 main / v6.4 cellline pilot). All prior entries preserved.")
txt = "\n".join(lines) + "\n"
with open(CL, "a", encoding="utf-8") as f:
    f.write(txt)
print("changelog appended", len(txt), "bytes")
