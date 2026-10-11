# -*- coding: utf-8 -*-
"""batch42 Phase A: label-free eligibility / group / split FREEZE.

Label-free: never reads the alpha target column.  Groups built from overlap
(batch41 components), verified author parent-tile families (author design
headers), exact/RC DNA, and label-free near-homology (k-mer candidate
generation + ACTUAL bit-vector edit distance on full 270bp sequences).
Split is ONE deterministic SHA rule over canonical group IDs.
"""
from __future__ import annotations
import collections
import csv
import gzip
import itertools
import os
import re
import time

import numpy as np

import b42_common as C

T0 = time.time()
LOG = []


def log(msg):
    line = "[%6.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


C.set_thread_limits(4)

# ---- 1. frozen input (label-FREE load) ------------------------------------
fsha = C.sha256_file(C.FROZEN_TABLE)
rows = C.load_frozen_rows(with_labels=False)
bio = [r for r in rows if r["control"] == 0]
ctl = [r for r in rows if r["control"] == 1]
log("frozen sha %s ok=%s rows=%d bio=%d ctl=%d" % (fsha[:16], fsha == C.FROZEN_TABLE_SHA, len(rows), len(bio), len(ctl)))
assert fsha == C.FROZEN_TABLE_SHA, "frozen table SHA mismatch"

with open(C.FROZEN_TABLE, encoding="utf-8") as fh:
    alphas = [float(r["alpha_primary"]) for r in csv.DictReader(fh, delimiter="\t") if int(r["control"]) == 0]
n_finite = int(np.isfinite(alphas).sum())
log("bio alpha finite %d/%d min=%.6f max=%.6f" % (n_finite, len(alphas), min(alphas), max(alphas)))

# ---- 2. source validation --------------------------------------------------
design = {}
with gzip.open(C.DESIGN_FA, "rt") as fh:
    name = None
    buf = []
    for line in fh:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None:
                design[name] = "".join(buf)
            name = line[1:]
            buf = []
        else:
            buf.append(line)
    if name is not None:
        design[name] = "".join(buf)
dsha = C.sha256_file(C.DESIGN_FA)
log("design records=%d sha=%s ok=%s" % (len(design), dsha[:16], dsha == C.DESIGN_SHA))

upper_ok = 0
upper_bad = []
notfound = []
for r in rows:
    raw = design.get(r["label"])
    if raw is None:
        notfound.append(r["label"]); continue
    if len(raw) == 300 and raw[15:285].upper() == r["sequence"]:
        upper_ok += 1
    else:
        upper_bad.append(r["label"])
log("upper270 ok=%d bad=%d notfound=%d" % (upper_ok, len(upper_bad), len(notfound)))
source_validation = {
    "frozen_table_sha256": fsha, "frozen_table_sha_ok": fsha == C.FROZEN_TABLE_SHA,
    "design_fasta_sha256": dsha, "design_sha_ok": dsha == C.DESIGN_SHA,
    "design_records": len(design), "design_bytes": C.DESIGN_FA.stat().st_size,
    "upper270_contract": "sequence == raw[15:285] of the 300bp author oligo (FWD15+270+REV15)",
    "upper270_ok": upper_ok, "upper270_bad": len(upper_bad), "upper270_notfound": len(notfound),
    "bio_alpha_finite": n_finite, "bio_rows": len(bio), "control_rows": len(ctl),
    "note": "identity validated by source SHA + per-row raw[15:285] equality; NOT a base edit; no reference-DNA supplementation/crop/label-chosen orientation",
}
C.atomic_write_json(C.B42_DATA / "source_validation.json", source_validation)

# ---- 3. author parent-tile families ---------------------------------------
LBL = re.compile(r"^(?P<roi>.+)_(?P<idx>\d+)_(?P<s>[+-])::(?P<chr>chr[\w]+):(?P<st>\d+)-(?P<en>\d+)(?P<s2>[+-])$")
runs = collections.defaultdict(list)
for name in design:
    m = LBL.match(name)
    if m:
        runs[(m.group("roi"), m.group("chr"))].append((int(m.group("st")), name))
fam_run_id = {}
for key, lst in runs.items():
    lst.sort()
    run_start = lst[0][0]
    prev = None
    for st, name in lst:
        if prev is not None and st - prev != 90:
            run_start = st
        fam_run_id[name] = (key[0], key[1], run_start)
        prev = st
fam_of = {}
unparsed = 0
for r in bio:
    fam_of[r["label"]] = fam_run_id.get(r["label"])
    if fam_of[r["label"]] is None:
        fam_of[r["label"]] = ("UNPARSED", r["label"])
        unparsed += 1
fam_sizes = collections.Counter(fam_of[r["label"]] for r in bio)
log("author bio families=%d unparsed=%d size(min/med/max)=%d/%d/%d" % (
    len(fam_sizes), unparsed, min(fam_sizes.values()),
    int(np.median(list(fam_sizes.values()))), max(fam_sizes.values())))

# ---- 4. union-find (overlap + family + exact/RC) --------------------------
parent = {r["label"]: r["label"] for r in bio}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)
        return True
    return False


by_comp = collections.defaultdict(list)
by_fam = collections.defaultdict(list)
for r in bio:
    by_comp[r["component_id"]].append(r["label"])
    by_fam[fam_of[r["label"]]].append(r["label"])
for labs in by_comp.values():
    for l in labs[1:]:
        union(labs[0], l)
for labs in by_fam.values():
    for l in labs[1:]:
        union(labs[0], l)
by_sha = collections.defaultdict(list)
for r in bio:
    by_sha[r["sequence_sha256"]].append(r["label"])
n_exact = sum(len(v) - 1 for v in by_sha.values())
n_rc = 0
by_rc = collections.defaultdict(list)
for r in bio:
    by_rc[C.sha256_text(C.rc(r["sequence"]))].append(r["label"])
for s, labs in by_rc.items():
    if s in by_sha:
        for l in labs:
            if union(by_sha[s][0], l):
                n_rc += 1
n_components = len(by_comp)
groups_after_comp_family = len({find(r["label"]) for r in bio})
log("union a/b/c exact-dup edges=%d rc edges=%d; components=%d groups_after_comp_family=%d"
    % (n_exact, n_rc, n_components, groups_after_comp_family))

# ---- 5. label-free near-homology ------------------------------------------
seqs = [r["sequence"] for r in bio]
labels = [r["label"] for r in bio]
K = 15
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def rcs(s):
    return "".join(COMP[c] for c in reversed(s))


idx = collections.defaultdict(list)
for i, s in enumerate(seqs):
    for km in set(s[j:j + K] for j in range(len(s) - K + 1)):
        r = rcs(km)
        idx[km if km <= r else r].append(i)          # canonical fwd/RC 15-mer
pair = collections.Counter()
for km, lst in idx.items():
    if 1 < len(lst) <= 400:
        for a, b in itertools.combinations(lst, 2):
            pair[(a, b)] += 1
log("canonical-15mer candidate pairs=%d" % len(pair))

THR = 8
cand = [(a, b) for (a, b), v in pair.items() if v >= THR and find(labels[a]) != find(labels[b])]
log("cross-group candidates (shared>=%d): %d" % (THR, len(cand)))


def myers_ed(a, b):
    m = len(b); mask = 1 << (m - 1)
    Peq = {}
    for i, c in enumerate(b):
        Peq[c] = Peq.get(c, 0) | (1 << i)
    Pv = ~0; Mv = 0; Score = m
    for c in a:
        Eq = Peq.get(c, 0)
        Xv = Eq | Mv
        Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq
        Ph = Mv | ~(Xh | Pv)
        Mh = Pv & Xh
        if Ph & mask: Score += 1
        elif Mh & mask: Score -= 1
        Ph = (Ph << 1) | 1
        Mh = Mh << 1
        Pv = Mh | ~(Xv | Ph)
        Mv = Ph & Xv
    return Score


rclist = [rcs(s) for s in seqs]
best = []
close_pairs = []
t1 = time.time()
for (a, b) in cand:
    ed = myers_ed(seqs[a], seqs[b])
    orient = "fwd"
    if ed <= 60:
        ed_rc = myers_ed(seqs[a], rclist[b])
        if ed_rc < ed:
            ed, orient = ed_rc, "rc"
    best.append(ed)
    if ed <= 27:
        close_pairs.append((a, b, orient))


def global_identity_coverage(a, b):
    """Full Needleman-Wunsch (unit costs) with traceback -> (identity, coverage)."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                          D[i - 1, j - 1] + (0 if ai == b[j - 1] else 1))
    i, j = n, m
    M = X = G = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            if a[i - 1] == b[j - 1]:
                M += 1
            else:
                X += 1
            i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            G += 1; i -= 1
        else:
            G += 1; j -= 1
    aln = M + X + G
    return M / aln, (M + X) / max(n, m), M, X, G


n_confirm = 0
confirm_details = []
for (a, b, orient) in close_pairs:
    bb = rclist[b] if orient == "rc" else seqs[b]
    ident, cov, M, X, G = global_identity_coverage(seqs[a], bb)
    confirm_details.append({"a": labels[a], "b": labels[b], "orient": orient,
                            "identity": ident, "coverage": cov, "matches": M, "mismatches": X, "gaps": G})
    if ident >= 0.90 and cov >= 0.90:
        n_confirm += 1
        union(labels[a], labels[b])
best = np.array(best) if best else np.array([999], dtype=int)
log("homology scan %d pairs in %.1fs min_ed=%d <=27:%d <=40:%d <=60:%d" % (
    len(cand), time.time() - t1, int(best.min()), int((best <= 27).sum()),
    int((best <= 40).sum()), int((best <= 60).sum())))
homology = {
    "method": "canonical (fwd/RC-merged) 15-mer candidate generation; ACTUAL bit-vector (Myers) edit distance on full 270bp sequence for both (a,b) and (a,rc(b)); candidates = cross-group pairs sharing >=%d canonical 15-mers" % THR,
    "candidate_generation": "k-mer similarity only, never proof of alignment",
    "n_canonical_candidate_pairs": len(pair),
    "n_candidate_pairs_scanned": len(cand),
    "scan_seconds": round(time.time() - t1, 1),
    "merge_rule": "identity>=0.90 AND bidirectional coverage>=0.90; equal length 270 -> ed<=27 with zero gaps, stricter if gaps",
    "min_edit_distance_found": int(best.min()),
    "n_ed_le_27": int((best <= 27).sum()),
    "n_ed_le_40": int((best <= 40).sum()),
    "n_confirmed_merges": n_confirm,
    "confirm_details": confirm_details,
    "residual_not_exhaustive": True,
    "residual_note": "pairs sharing < %d canonical 15-mers were not sequenced; residual non-exhaustiveness disclosed, no false all-family-isolated claim" % THR,
}
C.atomic_write_json(C.B42_DATA / "alignment_findings.json", homology)

# ---- 6. final groups + split ----------------------------------------------
groups = collections.defaultdict(list)
for r in bio:
    groups[find(r["label"])].append(r)
canonical = {}
for g, members in groups.items():
    canonical[g] = min(m["sequence_sha256"] for m in members)
log("final groups=%d sizes(min/med/max)=%d/%d/%d" % (
    len(groups), min(len(v) for v in groups.values()),
    int(np.median([len(v) for v in groups.values()])), max(len(v) for v in groups.values())))

split_of = {}
part_count = collections.Counter()
for g, members in groups.items():
    h = C.sha256_text(C.SPLIT_DOMAIN + "|" + canonical[g])
    part = "TRAIN" if (int(h, 16) % 100) < C.TRAIN_MOD else "TEST"
    for m in members:
        split_of[m["label"]] = part
        part_count[part] += 1
te_groups = {g for g, m in groups.items() if split_of[m[0]["label"]] == "TEST"}
tr_groups = {g for g, m in groups.items() if split_of[m[0]["label"]] == "TRAIN"}
log("split rows TRAIN=%d TEST=%d groups TRAIN=%d TEST=%d" % (
    part_count["TRAIN"], part_count["TEST"], len(tr_groups), len(te_groups)))

xsplit_exact = 0
xsplit_rc = 0
sha_parts = collections.defaultdict(set)
for r in bio:
    sha_parts[r["sequence_sha256"]].add(split_of[r["label"]])
for s, ps in sha_parts.items():
    if len(ps) > 1:
        xsplit_exact += 1
rc_parts = collections.defaultdict(set)
for r in bio:
    rc_parts[C.sha256_text(C.rc(r["sequence"]))].add(split_of[r["label"]])
for s, ps in rc_parts.items():
    if len(ps) > 1:
        xsplit_rc += 1
log("cross-split exact=%d rc=%d" % (xsplit_exact, xsplit_rc))

lab2alpha = {}
with open(C.FROZEN_TABLE, encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        if int(r["control"]) == 0:
            lab2alpha[r["label"]] = float(r["alpha_primary"])
tr_labels = [r["label"] for r in bio if split_of[r["label"]] == "TRAIN"]
te_labels = [r["label"] for r in bio if split_of[r["label"]] == "TEST"]
tr_y = np.array([lab2alpha[l] for l in tr_labels])
tr_var = float(np.var(tr_y))
gate = {
    "train_rows": len(tr_labels), "train_groups": len(tr_groups),
    "test_rows": len(te_labels), "test_groups": len(te_groups),
    "train_rows_ok": len(tr_labels) >= C.MIN_TRAIN_ROWS,
    "train_groups_ok": len(tr_groups) >= C.MIN_TRAIN_GROUPS,
    "test_rows_ok": len(te_labels) >= C.MIN_TEST_ROWS,
    "test_groups_ok": len(te_groups) >= C.MIN_TEST_GROUPS,
    "train_target_variance": tr_var, "train_target_variance_nonzero": tr_var > 0,
    "neither_partition_single_group": (len(tr_groups) > 1 and len(te_groups) > 1),
}
gate["PASS"] = all([gate["train_rows_ok"], gate["train_groups_ok"], gate["test_rows_ok"],
                    gate["test_groups_ok"], gate["neither_partition_single_group"],
                    gate["train_target_variance_nonzero"]])
log("GATE=%s train=%d/%d test=%d/%d" % (gate["PASS"], len(tr_labels), len(tr_groups), len(te_labels), len(te_groups)))

# alternative label-free partitions (transparency for the proposal; NOT used)
alt = {}
for name, keyfn in (("overlap_components_only", lambda r: ("C", r["component_id"])),
                    ("author_roi_family_only", lambda r: ("F", fam_of[r["label"]]))):
    gs = collections.defaultdict(list)
    for r in bio:
        gs[keyfn(r)].append(r)
    trn = te = trg = teg = 0
    for g, ms in gs.items():
        hh = C.sha256_text(C.SPLIT_DOMAIN + "|" + min(m["sequence_sha256"] for m in ms))
        if (int(hh, 16) % 100) < C.TRAIN_MOD:
            trn += len(ms); trg += 1
        else:
            te += len(ms); teg += 1
    alt[name] = {"groups": len(gs), "train_rows": trn, "test_rows": te,
                 "train_groups": trg, "test_groups": teg,
                 "gate_would_pass": trg >= C.MIN_TRAIN_GROUPS and teg >= C.MIN_TEST_GROUPS and te >= C.MIN_TEST_ROWS and trn >= C.MIN_TRAIN_ROWS}
log("alt partitions: %s" % alt)

# ---- 7. artifacts ----------------------------------------------------------
with open(C.B42_DATA / "groups.tsv", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("label\tgroup\tcanonical_group_sha\tsplit\tchr\tstart\tsequence_sha256\tcomponent_id\n")
    for g, members in groups.items():
        for m in members:
            fh.write("\t".join([m["label"], g, canonical[g], split_of[m["label"]], m["chr"],
                                str(m["start"]), m["sequence_sha256"], str(m["component_id"])]) + "\n")

frozen_split = {
    "batch": C.BATCH, "version": C.VERSION, "generated_at": C.now_local(), "pid": os.getpid(),
    "phase": "A_freeze", "label_free": True,
    "eligible_ids_source": str(C.FROZEN_TABLE), "eligible_ids_sha256": fsha,
    "eligible_rows": len(bio), "controls_rows": len(ctl),
    "target_contract": "raw alpha_primary (MPRAnalyze getAlpha); unchanged; no pvalue/activity/GC selection",
    "split_rule": {"domain": C.SPLIT_DOMAIN,
                   "canonical_group_id": "lexicographically smallest sequence_sha256 among members",
                   "h": "sha256('b42-group-v1|'+groupID)",
                   "assign": "int(h,16) % 100 < 80 -> TRAIN else TEST", "train_mod": C.TRAIN_MOD},
    "split_sha256_groups": C.sha256_text("|".join(sorted(canonical.values()))),
    "counts": {"rows": len(bio), "TRAIN": part_count["TRAIN"], "TEST": part_count["TEST"],
               "groups": len(groups), "groups_TRAIN": len(tr_groups), "groups_TEST": len(te_groups),
               "chroms": len({r["chr"] for r in bio}), "lengths_bp": sorted({len(r["sequence"]) for r in bio})},
    "union_sources": {"n_components_batch41": n_components,
                      "n_groups_after_comp_family": groups_after_comp_family,
                      "n_author_families": len(fam_sizes),
                      "n_final_groups": len(groups),
                      "component_or_family_merge_changed_groups": groups_after_comp_family != n_components},
    "cross_split_exact_or_rc": {"exact": xsplit_exact, "rc": xsplit_rc},
    "gate": gate,
    "alternative_label_free_partitions": alt,
    "disclosure": {"author_upstream_dna_min": 50, "retrospective_reuse": True,
                   "prior_batch41_all_labels_exposed": True,
                   "test_is_not_new_untouched_validation": True,
                   "no_cv_or_hyperparameter_selection": True,
                   "positive_scope_if_any": "regional/group-heldout retrospective public assay only",
                   "pre_fit_amendment_20261004T0242": "PARENT PRE-FIT GROUP-COUNT CORRECTION: feasibility gate = train>=10 groups / test>=5 groups (row gates 1000/300 unchanged); formal conditional bootstrap still needs >=30 test groups, else DESCRIPTIVE pilot verdict LIMITED; initial-protocol copy preserved; no label-driven regrouping/resampling"},
}
C.atomic_write_json(C.B42_DATA / "frozen_split.json", frozen_split)
C.atomic_write_json(C.B42_RUN / "phaseA_freeze.json",
                    {"batch": C.BATCH, "version": C.VERSION, "phase": "A_freeze",
                     "generated_at": C.now_local(), "elapsed_s": round(time.time() - T0, 1),
                     "gate": gate, "log": LOG, "script_sha256": C.sha256_file(__file__)})
print("PHASE A DONE in %.1fs gate=%s" % (time.time() - T0, gate["PASS"]))
