# -*- coding: utf-8 -*-
"""BATCH15 rc-retrieval-audit-20261003-r1 production cross-orientation audit.

Bounded question: does an explicit FORWARD-index vs REVERSE-COMPLEMENT-query
retrieval find eligible near-RC promoter pairs connecting DIFFERENT batch13
components? No training, no metric optimization, 0 fits. CPU<=4, no GPU.
Read-only over all prior data; writes only new batch15 artifacts.
"""
import os, sys, csv, json, time, hashlib, argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b15_rc_common as C
from b15_rc_common import (rc_bytes, rc_batch, compute_sketch, band_matches,
                           hash_params, verify_pairs_batch, sha256_file, sha256_bytes,
                           atomic_write_json, atomic_write_text, now_local, worker_identity,
                           MEMMAP, DATA13, DATA05, DATA03, DATA15, REPORT15)

C.set_thread_limits(4)

def load_eligible():
    gids, comps, nproms = [], [], []
    with open(DATA13 / "eligible_genes.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            gids.append(row[0])
            comps.append(int(row[2]))
            nproms.append(int(row[3]))
    return gids, comps, nproms

def load_folds():
    folds = {}
    with open(DATA13 / "fold_assignment.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            folds[row[0]] = int(row[2])
    return folds

def load_mapping():
    z = np.load(DATA03 / "trackA_gene_promoters.npz", allow_pickle=True)
    gid_arr = z["gene_id"].astype(str)
    r400 = z["r400"].astype(np.int64)
    gene2rows = defaultdict(list)
    for i in range(len(gid_arr)):
        gene2rows[gid_arr[i]].append(int(r400[i]))
    for g in gene2rows:
        gene2rows[g].sort()
    return gene2rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ident = worker_identity()
    DATA15.mkdir(parents=True, exist_ok=True)
    REPORT15.mkdir(parents=True, exist_ok=True)

    cfg_path = DATA15 / "config.json"
    if not cfg_path.exists():
        raise SystemExit("REFUSE: config.json missing; run b15_rc_config.py first")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    src_ok = {}
    for k, v in cfg["sources_sha256"].items():
        src_ok[k] = sha256_file(v["path"]) == v["sha256"]
    if not all(src_ok.values()):
        raise SystemExit("SOURCE SHA MISMATCH: %s" % json.dumps(src_ok))

    gids, comps, nproms = load_eligible()
    folds = load_folds()
    gene2rows = load_mapping()
    assert len(gids) == 18769, "eligible gene count mismatch"

    promoter_rows = []
    promoter_gene = []
    for gi, gid in enumerate(gids):
        rows = gene2rows[gid]
        assert len(rows) == nproms[gi], "n_promoters mismatch for %s" % gid
        for r in rows:
            promoter_rows.append(r)
            promoter_gene.append(gi)
    promoter_rows = np.asarray(promoter_rows, dtype=np.int64)
    promoter_gene = np.asarray(promoter_gene, dtype=np.int64)
    n_prom = len(promoter_rows)
    n_genes = len(gids)

    comp_of_gene = np.asarray(comps, dtype=np.int64)
    fold_of_gene = np.asarray([folds[g] for g in gids], dtype=np.int64)

    idx_path = DATA15 / "promoter_index.npz"
    if not idx_path.exists() or args.resume:
        np.savez_compressed(idx_path, promoter_rows=promoter_rows, promoter_gene=promoter_gene,
                            gene_ids=np.asarray(gids, dtype=object),
                            comp_of_gene=comp_of_gene, fold_of_gene=fold_of_gene)
    print("n_eligible_genes", n_genes, "n_eligible_promoters", n_prom, flush=True)

    mm = np.memmap(str(MEMMAP), dtype="uint8", mode="r", shape=(C.N_HUMAN, C.L400))
    S = np.asarray(mm[promoter_rows], dtype=np.uint8)

    fwd_keys = set()
    can_keys = set()
    for i in range(0, n_prom, 5000):
        blk = S[i:i+5000]
        for row in blk:
            fwd_keys.add(row.tobytes())
            rkey = rc_bytes(row).tobytes()
            c = row.tobytes() if row.tobytes() <= rkey else rkey
            can_keys.add(c)
    n_unique_fwd = len(fwd_keys)
    n_unique_canon = len(can_keys)
    del fwd_keys, can_keys
    print("n_unique_forward_sequences", n_unique_fwd, "n_unique_canonical_sequences", n_unique_canon, flush=True)

    sk_path = DATA15 / "sketches.npz"
    if args.resume and sk_path.exists():
        z = np.load(sk_path)
        F = z["F"]; R = z["R"]
        print("resumed sketches", F.shape, flush=True)
        del S
    else:
        a, b = hash_params(C.SEED)
        F = np.zeros((n_prom, C.N_HASH), dtype=np.uint64)
        R = np.zeros((n_prom, C.N_HASH), dtype=np.uint64)
        CHUNK = 20000
        for st in range(0, n_prom, CHUNK):
            en = min(st + CHUNK, n_prom)
            s_chunk = S[st:en]
            F[st:en] = compute_sketch(s_chunk, a, b)
            R[st:en] = compute_sketch(rc_batch(s_chunk), a, b)
            print("sketch", en, "/", n_prom, round(time.time()-t0, 1), flush=True)
        del S
        np.savez_compressed(sk_path, F=F, R=R)

    cand_path = DATA15 / "candidates.npz"
    cand_ledger_path = DATA15 / "candidates_ledger.json"
    if args.resume and cand_path.exists():
        z = np.load(cand_path)
        keys = z["keys"]; cand_bands = z["n_bands"]
        cand_ledger = json.loads(cand_ledger_path.read_text(encoding="utf-8")) if cand_ledger_path.exists() else {"resumed": True}
        n_candidates = len(keys)
    else:
        cand, cand_ledger = band_matches(F, R)
        n_candidates = len(cand)
        keys = np.asarray(sorted(cand.keys()), dtype=np.int64)
        cand_bands = np.asarray([cand[tuple(k)] for k in keys], dtype=np.int64)
        np.savez_compressed(cand_path, keys=keys, n_bands=cand_bands)
        atomic_write_json(cand_ledger_path, cand_ledger)
    print("cross-orientation candidates (>=%d tuple-verified bands)" % C.MIN_BANDS, n_candidates, flush=True)
    print("candidate ledger", json.dumps(cand_ledger), flush=True)

    verified_promoter = []
    if n_candidates:
        BATCH = 20000
        for st in range(0, len(keys), BATCH):
            c = keys[st:st+BATCH]
            Arows = np.asarray(mm[promoter_rows[c[:, 0]]], dtype=np.uint8)
            Brows = np.asarray(mm[promoter_rows[c[:, 1]]], dtype=np.uint8)
            vid, vcov, vshift, vov, vmatches = verify_pairs_batch(Arows, Brows)
            hit = (vid >= C.IDENTITY_MIN) & (vcov >= C.COVERAGE_MIN)
            for k in np.flatnonzero(hit):
                i = int(c[k, 0]); j = int(c[k, 1])
                gi = int(promoter_gene[i]); gj = int(promoter_gene[j])
                if gi == gj:
                    continue
                verified_promoter.append({
                    "promoter_pos_i": i, "promoter_pos_j": j,
                    "promoter_row_i": int(promoter_rows[i]), "promoter_row_j": int(promoter_rows[j]),
                    "gene_a": gids[gi], "gene_b": gids[gj],
                    "component_a": int(comp_of_gene[gi]), "component_b": int(comp_of_gene[gj]),
                    "fold_a": int(fold_of_gene[gi]), "fold_b": int(fold_of_gene[gj]),
                    "orientation": "rc",
                    "shift": int(vshift[k]), "overlap": int(vov[k]), "matches": int(vmatches[k]),
                    "identity": float(vid[k]), "coverage": float(vcov[k]),
                    "n_bands": int(cand_bands[st+k]),
                })
    n_verified_promoter = len(verified_promoter)
    print("verified promoter-level cross-gene edges", n_verified_promoter, flush=True)

    seen_prom = set()
    promoter_edges = []
    for e in verified_promoter:
        pkey = (min(e["promoter_row_i"], e["promoter_row_j"]), max(e["promoter_row_i"], e["promoter_row_j"]))
        if pkey in seen_prom:
            continue
        seen_prom.add(pkey)
        e2 = dict(e)
        e2["gene_pair_key"] = "|".join(sorted([e["gene_a"], e["gene_b"]]))
        promoter_edges.append(e2)

    gene_edges = {}
    for e in promoter_edges:
        gk = e["gene_pair_key"]
        if gk not in gene_edges:
            gene_edges[gk] = {
                "gene_a": min(e["gene_a"], e["gene_b"]), "gene_b": max(e["gene_a"], e["gene_b"]),
                "component_a": e["component_a"] if e["gene_a"] < e["gene_b"] else e["component_b"],
                "component_b": e["component_b"] if e["gene_a"] < e["gene_b"] else e["component_a"],
                "fold_a": e["fold_a"] if e["gene_a"] < e["gene_b"] else e["fold_b"],
                "fold_b": e["fold_b"] if e["gene_a"] < e["gene_b"] else e["fold_a"],
                "n_promoter_evidence": 0,
                "max_identity": 0.0, "min_identity": 1.0,
                "promoter_evidence": [],
            }
        ge = gene_edges[gk]
        ge["n_promoter_evidence"] += 1
        ge["max_identity"] = max(ge["max_identity"], e["identity"])
        ge["min_identity"] = min(ge["min_identity"], e["identity"])
        ge["promoter_evidence"].append({
            "promoter_row_i": e["promoter_row_i"], "promoter_row_j": e["promoter_row_j"],
            "identity": e["identity"], "coverage": e["coverage"], "shift": e["shift"],
            "overlap": e["overlap"], "matches": e["matches"], "n_bands": e["n_bands"],
        })

    n_gene_edges = len(gene_edges)
    n_cross_component = sum(1 for e in gene_edges.values() if e["component_a"] != e["component_b"])
    n_within_component = n_gene_edges - n_cross_component
    n_cross_fold = sum(1 for e in gene_edges.values() if e["fold_a"] != e["fold_b"])
    n_same_fold_cross_component = sum(1 for e in gene_edges.values()
                                      if e["component_a"] != e["component_b"] and e["fold_a"] == e["fold_b"])

    if promoter_edges:
        with open(DATA15 / "verified_promoter_edges.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["gene_a", "gene_b", "component_a", "component_b", "fold_a", "fold_b",
                        "cross_component", "cross_fold", "promoter_row_i", "promoter_row_j",
                        "orientation", "shift", "overlap", "matches", "identity", "coverage", "n_bands"])
            for e in sorted(promoter_edges, key=lambda x: (x["gene_pair_key"], x["promoter_row_i"], x["promoter_row_j"])):
                w.writerow([e["gene_a"], e["gene_b"], e["component_a"], e["component_b"],
                            e["fold_a"], e["fold_b"], int(e["component_a"] != e["component_b"]),
                            int(e["fold_a"] != e["fold_b"]), e["promoter_row_i"], e["promoter_row_j"],
                            "rc", e["shift"], e["overlap"], e["matches"],
                            "%.6f" % e["identity"], "%.6f" % e["coverage"], e["n_bands"]])
    if gene_edges:
        with open(DATA15 / "verified_gene_edges.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["gene_a", "gene_b", "component_a", "component_b", "fold_a", "fold_b",
                        "cross_component", "cross_fold", "n_promoter_evidence",
                        "min_identity", "max_identity"])
            for gk in sorted(gene_edges):
                e = gene_edges[gk]
                w.writerow([e["gene_a"], e["gene_b"], e["component_a"], e["component_b"],
                            e["fold_a"], e["fold_b"], int(e["component_a"] != e["component_b"]),
                            int(e["fold_a"] != e["fold_b"]), e["n_promoter_evidence"],
                            "%.6f" % e["min_identity"], "%.6f" % e["max_identity"]])

    metrics = {
        "report_version": "rc-retrieval-audit-20261003-r1",
        "question": ("does an explicit cross-orientation (forward-index vs reverse-complement-query) "
                     "retrieval find eligible near-RC promoter pairs connecting different batch13 components?"),
        "populations": {
            "n_eligible_genes": n_genes,
            "n_eligible_promoters": n_prom,
            "n_unique_forward_sequences": n_unique_fwd,
            "n_unique_canonical_sequences": n_unique_canon,
            "dedup_policy": ("promoter pairs deduplicated as unordered (row_i, row_j); gene pairs deduplicated "
                             "as unordered (gene_a, gene_b); multiple promoter pairs per gene pair retained as "
                             "evidence; cross-gene only (gene_a != gene_b)"),
        },
        "retrieval": {
            "orientation": "FORWARD-index vs REVERSE-COMPLEMENT-query",
            "k": C.K, "n_hash": C.N_HASH, "bands": C.N_BANDS, "rows": C.ROWS,
            "seed": C.SEED, "min_bands": C.MIN_BANDS, "band_cap": C.BAND_CAP, "pair_cap": C.PAIR_CAP,
            "band_shared_definition": (">=2 distinct bands whose actual 4-value minhash tuples are equal "
                                       "(not XOR-only)"),
            "verification": ("exact deterministic gapless RC alignment, shifts [-40,40]; identity=matches/overlap_len "
                             "(N never a match); coverage=overlap_len/400; confirmed iff identity>=0.90 AND "
                             "coverage>=0.90 (overlap>=360)"),
            "n_cross_orientation_candidates": n_candidates,
            "candidate_ledger": cand_ledger,
        },
        "results": {
            "n_verified_promoter_edges": len(promoter_edges),
            "n_verified_gene_edges": n_gene_edges,
            "n_within_component_gene_edges": n_within_component,
            "n_cross_component_gene_edges": n_cross_component,
            "n_cross_fold_gene_edges": n_cross_fold,
            "n_same_fold_cross_component_gene_edges": n_same_fold_cross_component,
            "cross_component_gene_pairs": sorted([gk for gk, e in gene_edges.items() if e["component_a"] != e["component_b"]]),
            "cross_fold_gene_pairs": sorted([gk for gk, e in gene_edges.items() if e["fold_a"] != e["fold_b"]]),
        },
        "comparison_to_saved_forward": cfg.get("forward_reference", {
            "note": "descriptive; forward channel is a DIFFERENT orientation (fwd vs rc) and candidate step; not a causal comparison",
        }),
        "interpretation_limits": [
            "zero detected cross-component pairs means ONLY: no pairs detected under THIS retrieval/verification/coverage and caps; never 'RC channel does not threaten the gain'",
            "positive pairs do not quantify score inflation; a later sensitivity fit needs actual graph change + changed fold membership",
            "retrieval is approximate and bounded (band_cap, pair_cap, max_shift, min_bands, Jaccard band threshold); not exhaustive proof",
        ],
        "source_sha_verified": src_ok,
        "worker_identity_start": ident,
        "elapsed_s": round(time.time() - t0, 1),
    }
    atomic_write_json(REPORT15 / "audit_metrics.json", metrics)
    atomic_write_json(REPORT15 / "coverage_cap_ledger.json", {
        "report_version": "rc-retrieval-audit-20261003-r1",
        "candidate_ledger": cand_ledger,
        "n_candidates": n_candidates,
        "n_verified_promoter_edges": len(promoter_edges),
        "n_verified_gene_edges": n_gene_edges,
        "caps": {"band_cap": C.BAND_CAP, "pair_cap": C.PAIR_CAP, "max_shift": C.MAX_SHIFT,
                 "identity_min": C.IDENTITY_MIN, "coverage_min": C.COVERAGE_MIN,
                 "min_bands": C.MIN_BANDS},
        "caps_note": "caps/timeouts mean this audit is INCOMPLETE, not zero leakage or completion",
        "sketched_promoters": n_prom,
    })
    print("n_cross_component_gene_edges", n_cross_component, "n_cross_fold", n_cross_fold, flush=True)
    print("WROTE audit_metrics.json + coverage_cap_ledger.json", flush=True)

if __name__ == "__main__":
    main()
