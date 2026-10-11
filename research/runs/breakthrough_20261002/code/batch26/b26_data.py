# -*- coding: utf-8 -*-
"""batch26 step 1: full-row reconstruction audit, label-free grouping, split freeze."""
from __future__ import annotations

import collections
import difflib
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b26_common as C

RC = str.maketrans("ACGTN", "TGCAN")


def revcomp(s):
    return s.translate(RC)[::-1]


def load_variants():
    rows = []
    with open(C.VARIANTS_TSV, "r", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) != len(hdr):
                raise SystemExit("bad column count")
            rows.append(dict(zip(hdr, c)))
    return hdr, rows


def load_design():
    key2probe = {}
    key2nrep = {}
    loci = set()
    with open(C.ARRAY_PROBES, "r", encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < len(h):
                continue
            r = dict(zip(h, c))
            if r["organism"] != "hg38":
                continue
            k = r["id"] + "_" + r["motif"] + "_" + r["allele"]
            key2probe[k] = r["probe_seq"]
            try:
                key2nrep[(r["id"], r["motif"])] = float(r["nrep_ref"])
            except ValueError:
                pass
            loci.add(r["id"])
    return key2probe, key2nrep, loci


def load_coords():
    coords = {}
    with open(C.TSS_STR_PAIRS, "r", encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < len(h):
                continue
            r = dict(zip(h, c))
            if r["organism"] != "hg38":
                continue
            try:
                coords[r["str_id"]] = (r["str_chr"], int(r["str_pos"]), int(r["str_end"]))
            except ValueError:
                continue
    return coords


def build_design_groups():
    """(locus, motif) -> {allele: vreg} from ALL author design alleles."""
    groups = collections.defaultdict(dict)
    with open(C.ARRAY_PROBES, "r", encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < len(h):
                continue
            r = dict(zip(h, c))
            if r["organism"] != "hg38":
                continue
            p = r["probe_seq"]
            if not p.startswith(C.FIVE):
                continue
            i = p.find(C.ASISI)
            if i <= 0:
                continue
            groups[(r["id"], r["motif"])][r["allele"]] = p[len(C.FIVE):i]
    return groups


def solve_split(motif, alleles, nrep_ref):
    """Find (left_len, right_len) consistent with motif*copies for every allele."""
    def copies(al):
        d = C.COPY_VAR.get(al)
        if d is None or nrep_ref is None:
            return None
        return max(0, int(round(nrep_ref)) + d)
    items = [(al, s, copies(al)) for al, s in alleles.items()]
    items = [it for it in items if it[2] is not None]
    if not items:
        return None
    anchor = max(items, key=lambda it: (it[2], len(it[1])))
    if anchor[2] <= 0:
        return None
    aseq = anchor[1]
    block = motif * anchor[2]
    good = []
    start = 0
    while True:
        j = aseq.find(block, start)
        if j < 0:
            break
        left, right = aseq[:j], aseq[j + len(block):]
        if all(s == left + motif * cp + right for _al, s, cp in items):
            good.append(j)
        start = j + 1
    if not good:
        return None
    cen = len(aseq) / 2.0
    j = min(good, key=lambda x: abs(x - cen))
    return {"left_len": j, "right_len": len(aseq) - j - len(block),
            "n_candidates": len(good)}


def reconstruct(rows, key2probe, key2nrep):
    """Verify every row's vreg against the author design, ALL rows."""
    mism = {"missing_probe": 0, "no_adapter": 0, "no_asisi": 0,
            "seq_mismatch": 0, "len_mismatch": 0, "motif_copy_mismatch": 0}
    examples = []
    dgroups = build_design_groups()
    struct = {}
    for (locus, motif), alleles in dgroups.items():
        st = solve_split(motif, alleles, key2nrep.get((locus, motif)))
        struct[(locus, motif)] = st
    fails = []
    n_struct_ok = 0
    for r in rows:
        var = r["variant"]
        seq = r["vreg_seq"]
        probe = key2probe.get(var)
        if probe is None:
            mism["missing_probe"] += 1
            if len(examples) < 8:
                examples.append(["missing_probe", var])
            continue
        if not probe.startswith(C.FIVE):
            mism["no_adapter"] += 1
            continue
        i = probe.find(C.ASISI)
        if i <= 0:
            mism["no_asisi"] += 1
            continue
        rec = probe[len(C.FIVE):i]
        if rec != seq:
            mism["seq_mismatch"] += 1
            if len(examples) < 8:
                examples.append(["seq_mismatch", var, rec[:40], seq[:40]])
        if len(rec) != int(r["vreg_len"]):
            mism["len_mismatch"] += 1
        parts = var.split("_")
        locus, motif, allele = "Human_STR_" + parts[2], parts[3], parts[4]
        st = struct.get((locus, motif))
        nr = key2nrep.get((locus, motif))
        if st is None or nr is None:
            mism["motif_copy_mismatch"] += 1
            if len(fails) < 40:
                fails.append([locus, motif, "no_consistent_design_split"])
            continue
        dl = C.COPY_VAR.get(allele)
        if dl is None:
            mism["motif_copy_mismatch"] += 1
            continue
        cp = max(0, int(round(nr)) + dl)
        left = rec[:st["left_len"]]
        right = rec[st["left_len"] + len(motif) * cp:]
        if left + motif * cp + right != rec or left + motif * cp != rec[:st["left_len"] + len(motif) * cp]:
            mism["motif_copy_mismatch"] += 1
            if len(fails) < 40:
                fails.append([locus, motif, allele, cp])
            continue
        n_struct_ok += 1
    locus_struct = {}
    for (locus, motif), st in struct.items():
        if st is not None:
            locus_struct.setdefault(locus, st)
    mism["motif_struct_fail"] = sum(1 for v in struct.values() if v is None)
    return mism, examples, locus_struct, fails, n_struct_ok, len(struct)


def replicate_audit(rows, n_per_stratum=60):
    assoc = C.B25_SRC / "GSE306816_hSTR1_association_merged_BC_STRs.tsv.gz"
    bc2var = {}
    with gzip.open(assoc, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                bc2var[p[0]] = p[1]
    meas = collections.defaultdict(dict)
    for rp in (1, 2, 3):
        fp = C.B25_SRC / ("GSE306816_hSTR1_HEK293_BC_cDNA_gDNA_ratio_rep%d.csv.gz" % rp)
        with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split(",")
                if len(p) < 5:
                    continue
                try:
                    ratio = float(p[4]); rc = float(p[2]); rg = float(p[3])
                except ValueError:
                    continue
                meas[p[0]][rp] = (ratio, rc, rg)
    bc_info = {}
    for bc, d in meas.items():
        rr = [v[0] for v in d.values() if v[0] > 0]
        bc_info[bc] = {"nrep": len(d),
                       "log2": [math.log2(x) for x in rr],
                       "minreads": min(min(v[1], v[2]) for v in d.values()),
                       "var": bc2var.get(bc)}
    by_allele = collections.defaultdict(list)
    for r in rows:
        by_allele[r["variant"].rsplit("_", 1)[1]].append(r["variant"])
    chosen = []
    for al in ("ref", "m5", "p3", "p5"):
        lst = sorted(by_allele.get(al, []), key=lambda v: C.sha256_bytes(v.encode()))
        chosen += lst[:n_per_stratum]
    var2bcs = collections.defaultdict(list)
    for bc, info in bc_info.items():
        if info["var"] is not None:
            var2bcs[info["var"]].append(bc)
    lookup = {r["variant"]: r for r in rows}
    n_checked = n_exact = n_nbc = 0
    maxdiff = 0.0
    rep_counts = collections.Counter()
    n_fail = 0
    for v in chosen:
        r = lookup.get(v)
        if r is None:
            continue
        bcs = [bc for bc in var2bcs.get(v, [])
               if bc_info[bc]["minreads"] >= 10 and bc_info[bc]["nrep"] >= 2]
        # frozen aggregation order (verified): mean log2 within a barcode across its
        # present biological replicates, THEN mean across eligible barcodes.
        bc_means = [sum(bc_info[bc]["log2"]) / len(bc_info[bc]["log2"])
                    for bc in bcs if bc_info[bc]["log2"]]
        if len(bcs) < 3 or not bc_means:
            continue
        n_checked += 1
        d = abs(sum(bc_means) / len(bc_means) - float(r["mean_log2_ratio"]))
        maxdiff = max(maxdiff, d)
        if d < 1e-6:
            n_exact += 1
        else:
            n_fail += 1
        if int(r["n_barcodes"]) == len(bcs):
            n_nbc += 1
        for bc in bcs:
            rep_counts[bc_info[bc]["nrep"]] += 1
    return {
        "association_barcodes": len(bc2var),
        "ratio_barcodes": len(meas),
        "sample_variants_by_allele": {k: len(v) for k, v in sorted(by_allele.items())},
        "recomputed_variants": n_checked,
        "exact_match_variants": n_exact,
        "nonzero_diff_variants": n_fail,
        "max_abs_diff_log2": maxdiff,
        "n_barcodes_column_match": n_nbc,
        "technical_barcodes_per_sampled_variant": {str(k): v for k, v in sorted(rep_counts.items())},
        "note": ("barcodes are TECHNICAL constructs within one assay; the study has 3 BIOLOGICAL "
                 "replicates.  The freeze averages log2(cDNA/gDNA) over eligible technical barcodes "
                 "present in >=2 of 3 biological replicates with reads>=10 in both channels."),
    }


class UF:
    def __init__(self):
        self.p = {}

    def add(self, x):
        self.p.setdefault(x, x)

    def find(self, x):
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_groups(rows, struct, coords):
    uf = UF()
    seqs = collections.defaultdict(list)
    for r in rows:
        locus = "Human_STR_" + r["variant"].split("_")[2]
        uf.add(locus)
        seqs[locus].append(r["vreg_seq"])
    by_seq = collections.defaultdict(list)
    by_rc = collections.defaultdict(list)
    for locus, ss in seqs.items():
        for s in set(ss):
            by_seq[s].append(locus)
            by_rc[revcomp(s)].append(locus)
    n_exact_edges = 0
    for s, ls in by_seq.items():
        for i in range(1, len(ls)):
            uf.union(ls[0], ls[i]); n_exact_edges += 1
    for s, ls in by_rc.items():
        base = by_seq.get(s)
        if base:
            for x in ls:
                for y in base:
                    if x != y:
                        uf.union(x, y); n_exact_edges += 1
    for locus in struct:
        uf.add(locus)
    iv = []
    no_coord = 0
    for locus, st in struct.items():
        c = coords.get(locus)
        if c is None or st is None:
            no_coord += 1
            continue
        chrom, pos, end = c
        iv.append((chrom, pos - st["left_len"], end + st["right_len"], locus))
    iv.sort()
    n_overlap_edges = 0
    merged_iv = 0
    cur = None
    for chrom, a, b, locus in iv:
        if cur is not None and cur[0] == chrom and a <= cur[3]:
            n_overlap_edges += 1
            if uf.find(cur[1]) != uf.find(locus):
                uf.union(cur[1], locus)
                merged_iv += 1
            if b > cur[3]:
                cur = (chrom, cur[1], min(cur[2], a), b)
        else:
            cur = (chrom, locus, a, b)
    # components over ALL design loci, then project onto the frozen loci
    members = collections.defaultdict(list)
    frozen_roots = {}
    for l in uf.p:
        frozen_roots.setdefault(uf.find(l), []).append(l)
    n_comps_all = len(frozen_roots)
    for root, ls in frozen_roots.items():
        fl = [l for l in ls if l in seqs]
        if fl:
            members[root] = fl
    locus2group = {}
    n_groups = len(members)
    for root, ls in members.items():
        shas = []
        for l in ls:
            for s in set(seqs[l]):
                shas.append(C.sha256_bytes(s.encode()))
        gid = C.groupid_of(ls, shas)
        for l in ls:
            locus2group[l] = gid
    sizes = collections.Counter(len(v) for v in members.values())
    gene_of = {}
    with open(C.TSS_STR_PAIRS, "r", encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < len(h):
                continue
            r = dict(zip(h, c))
            if r["organism"] == "hg38":
                gene_of[r["str_id"]] = r["gene_id"]
    multi_gene = 0
    for root, ls in members.items():
        gs = set(gene_of.get(l) for l in ls if gene_of.get(l))
        if len(gs) > 1:
            multi_gene += 1
    return {
        "n_design_loci_in_uf": len(uf.p), "n_frozen_loci": sum(len(v) for v in members.values()),
        "n_groups": n_groups,
        "n_exact_rc_edges": n_exact_edges, "n_overlap_edges": n_overlap_edges,
        "n_overlap_merges": merged_iv, "loci_without_coords": no_coord,
        "group_size_hist": {str(k): v for k, v in sorted(sizes.items())},
        "max_group_size": max(sizes) if sizes else 0,
        "groups_spanning_multiple_genes": multi_gene,
    }, locus2group


def near_homology_audit(rows, locus2group, cap_pairs=120000, time_cap=420):
    """Bounded near-homology screen over one representative per group."""
    t0 = time.time()
    K = 20
    by_group = collections.defaultdict(set)
    for r in rows:
        locus = "Human_STR_" + r["variant"].split("_")[2]
        by_group[locus2group[locus]].add(r["vreg_seq"])
    reps = {g: min(ss, key=lambda s: C.sha256_bytes(s.encode())) for g, ss in by_group.items()}
    kmers = {}
    for g, s in reps.items():
        kmers[g] = set(s[i:i + K] for i in range(len(s) - K + 1))
    glist = sorted(reps)
    idx = collections.defaultdict(list)
    for g in glist:
        for km in kmers[g]:
            idx[km].append(g)
    pairs = set()
    capped = False
    # informative kmers first (small group-support = more specific); deterministic order
    keys = sorted(idx, key=lambda k: (len(idx[k]), k))
    for km in keys:
        if time.time() - t0 > time_cap:
            capped = True; break
        gs = idx[km]
        if len(gs) < 2 or len(gs) > 8:
            continue
        u = sorted(gs)
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                pairs.add((u[i], u[j]))
        if len(pairs) >= cap_pairs:
            capped = True; break
    same_len_h16 = 0
    align_hits = 0
    cross = 0
    sm_used = 0
    for a, b in sorted(pairs):
        if time.time() - t0 > time_cap:
            capped = True; break
        cross += 1
        sa, sb = reps[a], reps[b]
        if len(sa) == len(sb):
            if sum(1 for x, y in zip(sa, sb) if x != y) <= 16:
                same_len_h16 += 1
        ja = len(kmers[a] & kmers[b]); jb = len(kmers[a] | kmers[b])
        if jb and ja / jb >= 0.7 and sm_used < 20000:
            sm_used += 1
            sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
            matched = sum(bl.size for bl in sm.get_matching_blocks())
            if matched / min(len(sa), len(sb)) >= 0.9 and matched / max(len(sa), len(sb)) >= 0.9:
                align_hits += 1
    return {
        "k": K, "n_group_representatives": len(glist),
        "candidate_pairs": len(pairs), "candidate_pairs_cap": cap_pairs,
        "candidate_pairs_checked": cross, "capped": bool(capped),
        "same_length_hamming_le16": same_len_h16,
        "jaccard_prefiltered_pairs": sm_used,
        "alignment_cov90_ident90": align_hits,
        "elapsed_s": round(time.time() - t0, 1),
        "note": ("20-mer-seeded CANDIDATE screen over one representative per group; bounded, "
                 "NOT exhaustive all-pairs.  Candidate pairs sharing a 20-mer with a same-length "
                 "Hamming<=16 or a cover>=0.90/identity>=0.90 alignment are counted.  Residual "
                 "near-homology below the seed threshold is not excluded, so the split is "
                 "locus/group-held-out with residual-homology uncertainty, not strict isolation."),
    }


def encode(seqs):
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    maxlen = max(len(s) for s in seqs)
    codes = np.full((len(seqs), maxlen), 4, dtype=np.int8)
    lens = np.zeros(len(seqs), dtype=np.int16)
    bad = []
    for i, s in enumerate(seqs):
        lens[i] = len(s)
        for j, ch in enumerate(s):
            v = m.get(ch)
            if v is None:
                bad.append((i, j, ch))
                v = 4
            codes[i, j] = v
    return codes, lens, bad


def main():
    t0 = time.time()
    got = C.sha256_file(C.VARIANTS_TSV)
    if got != C.VARIANTS_SHA:
        raise SystemExit("variants SHA mismatch: %s" % got)
    hdr, rows = load_variants()
    key2probe, key2nrep, design_loci = load_design()
    coords = load_coords()
    mism, examples, struct, fails, n_struct_ok, n_design_groups = reconstruct(rows, key2probe, key2nrep)
    n_bad = sum(v for k, v in mism.items() if k != "motif_struct_fail")
    recon = {
        "input_table": str(C.VARIANTS_TSV.relative_to(C.ROOT)),
        "input_sha256": got, "n_rows": len(rows),
        "design_probe_keys": len(key2probe), "design_hg38_loci": len(design_loci),
        "design_locus_motif_groups": n_design_groups,
        "groups_with_consistent_split": sum(1 for v in struct.values() if v is not None),
        "mismatches": mism, "examples": examples,
        "rows_passing_motif_copy_check": n_struct_ok,
        "row_motif_copy_failures": fails[:40],
        "n_row_motif_copy_failures": len(fails),
        "all_rows_reconstructed": (n_bad == 0),
        "note": ("EVERY row vreg_seq re-derived from the author array_probes probe_seq between the "
                 "5' adapter ACTGGCCGCTTGACG and Gibson_AsiSI CACTGCGGCTCCTGCGATCGC and compared "
                 "byte-for-byte; allele copy counts checked against the author design's own "
                 "left/right flank split, consistent with motif*copies for the full allele set."),
    }
    C.atomic_write_json(C.REPORT26 / "reconstruction_audit.json", recon)
    print("[recon] rows=%d bad=%d struct_fail=%d %.1fs" % (len(rows), n_bad, len(fails), time.time() - t0), flush=True)
    rep = replicate_audit(rows)
    C.atomic_write_json(C.REPORT26 / "replicate_audit.json", rep)
    print("[rep] checked=%d exact=%d maxdiff=%.2e" % (rep["recomputed_variants"], rep["exact_match_variants"], rep["max_abs_diff_log2"]), flush=True)
    grp, locus2group = build_groups(rows, struct, coords)
    C.atomic_write_json(C.REPORT26 / "groups.json", grp)
    print("[groups] loci=%d groups=%d overlap=%d exact=%d maxsize=%d" % (grp["n_frozen_loci"], grp["n_groups"], grp["n_overlap_edges"], grp["n_exact_rc_edges"], grp["max_group_size"]), flush=True)
    nh = near_homology_audit(rows, locus2group)
    C.atomic_write_json(C.REPORT26 / "near_homology_audit.json", nh)
    print("[nh] cand=%d checked=%d h16=%d align=%d capped=%s %.1fs" % (nh["candidate_pairs"], nh["candidate_pairs_checked"], nh["same_length_hamming_le16"], nh["alignment_cov90_ident90"], nh["capped"], nh["elapsed_s"]), flush=True)
    freeze_split(rows, locus2group, grp, t0)


def freeze_split(rows, locus2group, grp, t0):
    split_of_group = {}
    msgs = C.SPLIT_MSG
    for g in set(locus2group.values()):
        u = C.stable_hash_u01(msgs + "|" + g)
        split_of_group[g] = 0 if u < 0.70 else (1 if u < 0.85 else 2)
    counts = collections.Counter()
    for r in rows:
        locus = "Human_STR_" + r["variant"].split("_")[2]
        g = locus2group[locus]
        r["_group"] = g
        r["_split"] = split_of_group[g]
        counts[r["_split"]] += 1
    gcounts = collections.Counter(split_of_group.values())
    if gcounts[0] < 1000 or gcounts[1] < 100 or gcounts[2] < 100:
        raise SystemExit("split group counts too small: %r" % (dict(gcounts),))
    gmembers = collections.defaultdict(list)
    for locus, g in locus2group.items():
        gmembers[g].append(locus)
    lines = ["group_id\tsplit\tn_loci\tloci"]
    for g in sorted(split_of_group):
        lines.append("%s\t%d\t%d\t%s" % (g, split_of_group[g], len(gmembers[g]), ",".join(sorted(gmembers[g]))))
    split_manifest = C.DATA26 / "split_manifest.tsv"
    C.atomic_write_text(split_manifest, "\n".join(lines) + "\n")
    out = ["variant\tlocus_id\tmotif\tallele\tvreg_len\tn_barcodes\tmean_log2_ratio\tgroup_id\tsplit\tvreg_seq"]
    for r in rows:
        out.append("\t".join([r["variant"], r["locus_id"], r["motif"], r["variant"].rsplit("_", 1)[1],
                              str(r["vreg_len"]), str(r["n_barcodes"]), r["mean_log2_ratio"],
                              r["_group"], str(r["_split"]), r["vreg_seq"]]))
    rowtab = C.DATA26 / "str_variant_rows.tsv"
    C.atomic_write_text(rowtab, "\n".join(out) + "\n")
    seqs = [r["vreg_seq"] for r in rows]
    codes, lens, bad = encode(seqs)
    npz = C.DATA26 / "encoded.npz"
    tmp = str(npz) + ".tmp.npz"
    np.savez_compressed(tmp, codes=codes, lens=lens)
    os.replace(tmp, npz)
    freeze = {
        "batch": "batch26", "task": "STR variant-effect prediction (GSE306816 hSTR1, HEK293T)",
        "input_table": str(C.VARIANTS_TSV.relative_to(C.ROOT)), "input_sha256": C.VARIANTS_SHA,
        "n_rows": len(rows), "n_loci": grp["n_frozen_loci"], "n_groups": grp["n_groups"],
        "vreg_len_min": int(min(lens)), "vreg_len_max": int(max(lens)), "unknown_bases": len(bad),
        "motifs": {k: v for k, v in sorted(collections.Counter(r["motif"] for r in rows).items())},
        "alleles": {k: v for k, v in sorted(collections.Counter(r["variant"].rsplit("_", 1)[1] for r in rows).items())},
        "split": {"seed": C.SPLIT_SEED, "msg": msgs,
                  "rule": "u=sha256(msg|group_id)[:16]/2^64; u<0.70 train, <0.85 validation, else test",
                  "train_groups": gcounts[0], "val_groups": gcounts[1], "test_groups": gcounts[2],
                  "train_rows": counts[0], "val_rows": counts[1], "test_rows": counts[2]},
        "grouping": grp, "target_units": "log2(cDNA/gDNA) already; not re-logged",
        "status": "FROZEN_BEFORE_ANY_BATCH26_FIT",
        "files": {"split_manifest.tsv": C.sha256_file(split_manifest),
                  "str_variant_rows.tsv": C.sha256_file(rowtab),
                  "encoded.npz": C.sha256_file(npz)},
    }
    C.atomic_write_json(C.REPORT26 / "data_freeze.json", freeze)
    C.atomic_write_json(C.REPORT26 / "settings.json", {
        "split_seed": C.SPLIT_SEED, "shuffle_seed": C.SHUFFLE_SEED, "ridge_alpha": 100,
        "cnn_arch": "batch02 trackB_grouped_cv LegNetInspired(1, ch=64) read-only import",
        "lr": 0.001, "weight_decay": 0.0001, "batch": 32, "max_epochs": 25,
        "patience": 5, "grad_clip": 1.0, "optimizer": "AdamW",
        "scheduler": "constant LambdaLR(1.0)", "seeds": {"true": [20261026, 20261027], "shuffle": 20261026},
        "features": "mono/di/tri/tetra freq (4+16+64+256=340) + GC + log1p(len) = 342",
        "fit_budget": {"baseline": 1, "cnn_true": 2, "cnn_shuffle": 1, "total": 4},
    })
    print("[freeze] train_g=%d val_g=%d test_g=%d rows=%d/%d/%d unknown=%d %.1fs" %
          (gcounts[0], gcounts[1], gcounts[2], counts[0], counts[1], counts[2], len(bad), time.time() - t0), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "audit-rep":
        _h, _rows = load_variants()
        _rep = replicate_audit(_rows)
        C.atomic_write_json(C.REPORT26 / "replicate_audit.json", _rep)
        print("[rep] checked=%d exact=%d maxdiff=%.2e" % (
            _rep["recomputed_variants"], _rep["exact_match_variants"], _rep["max_abs_diff_log2"]))
    else:
        main()
