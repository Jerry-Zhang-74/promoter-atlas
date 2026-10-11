#!/usr/bin/env python3
"""Batch05 frozen sequence-component split builder (deterministic, target-independent).

Graph over GENES (batch03 human-only sym400 400bp promoters):
  edge A: canonical exact / reverse-complement duplicate promoter sequences
          (min(seq,rc(seq)) as entity) shared between two genes.
  edge B: detected near-identity promoter pair (>=90% identity over >=90% of the
          shorter 400bp sequence, either orientation) between two genes.

All promoters of a gene always stay together (genes are indivisible nodes).
Connected components of the gene graph are indivisible and deterministically
assigned to train/val/test (target 80/10/10 by gene count, seed 20261011) using
SEQUENCE/SIZE information only (gene_id + promoter count); target labels are never
read by the split builder.

Near-identity candidate retrieval (frozen, bounded, NOT exhaustive):
  * k=15 k-mers (2-bit ACGT); k-mers with <3 distinct bases or containing N are
    excluded from the sketch (low-complexity / ambiguity masking).
  * 48 MinHash values per promoter, hash h_i(x)=(a_i*x+b_i) mod (2^61-1),
    a_i,b_i from numpy default_rng(20261011).
  * 12 bands x 4 rows; band key = bitwise XOR of the 4 minhash values.
  * candidate pairs = promoter pairs sharing >=2 band keys (bucket cap 2000,
    pair-count cap 5e6).
  * candidates CONFIRMED by gapless diagonal alignment (forward + RC) over
    shifts [-40,40]: identity=matches/overlap_len, coverage=overlap_len/400;
    confirmed iff identity>=0.90 and coverage>=0.90 in either orientation.

Because candidate retrieval is bounded, the split is named "exact/RC +
detected-near-identity blocked" (NOT homology-free); residual risk reported.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_common import atomic_write_json, sha256_file  # noqa: E402

BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
sys.path.insert(0, str(BATCH03))

DATA03 = ROOT / "data" / "breakthrough_20261002" / "batch03"
OUT = ROOT / "data" / "breakthrough_20261002" / "batch05"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch05"

N_HUMAN = 507365
L400 = 400
K = 15
N_HASH = 48
N_BANDS = 12
ROWS = 4
BAND_CAP = 2000
PAIR_CAP = 5_000_000
MIN_EQUAL = 30
MAX_SHIFT = 40
IDENTITY_MIN = 0.90
COVERAGE_MIN = 0.90
SEED = 20261011
TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}

P = 2 ** 61 - 1


def rc_bytes(b):
    comp = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
    return comp[b[::-1]]


def rc_batch(B):
    """Per-row reverse complement of a (batch, L400) uint8 array."""
    comp = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
    return comp[B[:, ::-1]]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def load_data():
    z = np.load(DATA03 / "trackA_gene_promoters.npz", allow_pickle=True)
    gene_id = z["gene_id"].astype(str)
    r400 = z["r400"].astype(np.int64)
    score_e = z["Score_E"].astype(np.float64)
    starts = z["starts"].astype(np.int64)
    n_genes = len(starts) - 1
    return gene_id, r400, score_e, starts, n_genes


def load_memmap():
    return np.memmap(ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap",
                     dtype="uint8", mode="r", shape=(N_HUMAN, L400))


def kmer_codes(seq2, k):
    N, L = seq2.shape
    nwin = L - k + 1
    codes = np.zeros((N, nwin), dtype=np.uint32)
    c = np.zeros(N, dtype=np.uint32)
    for j in range(k):
        c = (c << np.uint32(2)) | seq2[:, j].astype(np.uint32)
    codes[:, 0] = c
    mask = (np.uint32(1) << np.uint32(2 * k)) - np.uint32(1)
    for p in range(1, nwin):
        c = ((c << np.uint32(2)) & mask) | seq2[:, p + k - 1].astype(np.uint32)
        codes[:, p] = c
    return codes


def window_distinct(seq, k):
    N, L = seq.shape
    nwin = L - k + 1
    cnt = np.zeros((N, nwin), dtype=np.uint8)
    for c in range(4):
        hc = (seq == c).astype(np.int32)
        cs = np.cumsum(hc, axis=1)
        end = cs[:, k - 1:]
        start = np.zeros_like(end)
        start[:, 1:] = cs[:, :nwin - 1]
        cnt += (end - start > 0).astype(np.uint8)
    return cnt


def minhash_sketch(seq, codes, valid, distinct_ok, a, b):
    N = seq.shape[0]
    codes64 = codes.astype(np.uint64)
    keep = valid & distinct_ok
    codes64[~keep] = 0
    sketch = np.zeros((N, N_HASH), dtype=np.uint64)
    for i in range(N_HASH):
        hv = (a[i] * codes64 + b[i]) % np.uint64(P)
        hv[~keep] = np.uint64(P - 1)
        sketch[:, i] = hv.min(axis=1)
    return sketch


def gapless_batch(A, B, max_shift=MAX_SHIFT):
    P = A.shape[0]
    best_ident = np.zeros(P, dtype=np.float64)
    best_cov = np.zeros(P, dtype=np.float64)
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            L = L400 - s
            matches = (A[:, :L] == B[:, s:]).sum(axis=1)
        else:
            s2 = -s
            L = L400 - s2
            matches = (A[:, s2:] == B[:, :L]).sum(axis=1)
        ident = matches / L
        cov = L / float(L400)
        better = (cov >= COVERAGE_MIN) & (ident > best_ident)
        best_ident[better] = ident[better]
        best_cov[better] = cov
    return best_ident, best_cov


def exact_rc_edges(mm, r400, gene_of, n_genes):
    canon = {}
    for i in range(len(r400)):
        key = mm[r400[i]].tobytes()
        rkey = rc_bytes(np.frombuffer(key, dtype=np.uint8)).tobytes()
        c = key if key <= rkey else rkey
        canon.setdefault(c, set()).add(int(gene_of[i]))
    n_multi = 0
    edges = []
    for key, genes in canon.items():
        if len(genes) > 1:
            n_multi += 1
            gl = sorted(genes)
            for j in range(1, len(gl)):
                edges.append((gl[0], gl[j]))
    return edges, {"n_canonical_keys": len(canon),
                   "n_keys_multi_gene": n_multi,
                   "n_promoter_rows": int(len(r400))}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    gene_id, r400, score_e, starts, n_genes = load_data()
    gene_gid = gene_id[starts[:-1]]  # gene-level IDs (one per gene)
    gene_of = np.repeat(np.arange(n_genes, dtype=np.int64), np.diff(starts))
    n_prom = np.diff(starts).astype(np.int64)
    mm = load_memmap()

    print("n_genes", n_genes, "n_promoters", len(r400), flush=True)

    edges, exact_stats = exact_rc_edges(mm, r400, gene_of, n_genes)
    print("exact/rc multi-gene canonical keys", exact_stats["n_keys_multi_gene"],
          "edges", len(edges), flush=True)

    s = np.asarray(mm[r400], dtype=np.uint8)
    seq2 = np.where(s == 4, 0, s).astype(np.uint8)
    codes = kmer_codes(seq2, K)
    hasN = (s == 4).astype(np.int32)
    csN = np.cumsum(hasN, axis=1)
    nwin = L400 - K + 1
    endN = csN[:, K - 1:]
    startN = np.zeros_like(endN)
    startN[:, 1:] = csN[:, :nwin - 1]
    win_valid = (endN - startN) == 0
    distinct = window_distinct(s, K)
    distinct_ok = distinct >= 3
    print("informative k-mer fraction", float((win_valid & distinct_ok).mean()), flush=True)

    rng = np.random.default_rng(SEED)
    a = rng.integers(1, 2 ** 31, size=N_HASH, dtype=np.uint64)
    b = rng.integers(0, P, size=N_HASH, dtype=np.uint64)
    sketch = minhash_sketch(s, codes, win_valid, distinct_ok, a, b)
    del codes, s, seq2, hasN, csN, endN, startN, win_valid, distinct, distinct_ok
    print("sketch done", round(time.time() - t0, 1), flush=True)

    band_keys = np.zeros((len(r400), N_BANDS), dtype=np.uint64)
    for j in range(N_BANDS):
        band_keys[:, j] = np.bitwise_xor.reduce(sketch[:, j * ROWS:(j + 1) * ROWS], axis=1)

    pair_count = {}
    n_skipped_buckets = 0
    n_truncated = 0
    for j in range(N_BANDS):
        keys = band_keys[:, j]
        order = np.argsort(keys, kind="stable")
        keys_s = keys[order]
        bnd = np.flatnonzero(np.diff(keys_s)) + 1
        starts_b = np.concatenate([[0], bnd])
        ends_b = np.concatenate([bnd, [len(keys_s)]])
        for si, ei in zip(starts_b, ends_b):
            bsize = int(ei - si)
            if bsize < 2:
                continue
            if bsize > BAND_CAP:
                n_skipped_buckets += 1
                continue
            pids = order[si:ei]
            pids_sorted = np.sort(pids)
            ii = np.repeat(pids_sorted, bsize)
            jj = np.tile(pids_sorted, bsize)
            mask = ii < jj
            ii, jj = ii[mask], jj[mask]
            for x, y in zip(ii.tolist(), jj.tolist()):
                keyp = (x, y)
                if keyp in pair_count:
                    pair_count[keyp] += 1
                elif len(pair_count) < PAIR_CAP:
                    pair_count[keyp] = 1
                else:
                    n_truncated += 1

    print("candidate pairs tracked", len(pair_count),
          "skipped_buckets", n_skipped_buckets, "truncated", n_truncated,
          "time", round(time.time() - t0, 1), flush=True)

    cand = [k for k, v in pair_count.items() if v >= 2]
    n_tracked = len(pair_count)
    n_share_ge2 = len(cand)
    del pair_count
    print("candidates share>=2 bands", n_share_ge2, flush=True)

    if cand:
        cand = np.asarray(cand, dtype=np.int64)
        pa = cand[:, 0]
        pb = cand[:, 1]
        neq = np.zeros(len(cand), dtype=np.int32)
        for h in range(N_HASH):
            neq += (sketch[pa, h] == sketch[pb, h]).astype(np.int32)
        keep = neq >= MIN_EQUAL
        cand = cand[keep]
    else:
        cand = np.zeros((0, 2), dtype=np.int64)
    print("candidates after minhash-equality prefilter", len(cand), flush=True)

    confirmed = []
    BATCH = 100000
    for st in range(0, len(cand), BATCH):
        c = cand[st:st + BATCH]
        A = np.asarray(mm[r400[c[:, 0]]], dtype=np.uint8)
        B = np.asarray(mm[r400[c[:, 1]]], dtype=np.uint8)
        idf, cf = gapless_batch(A, B)
        rcB = rc_batch(B)
        idr, cr = gapless_batch(A, rcB)
        hit = (idf >= IDENTITY_MIN) | (idr >= IDENTITY_MIN)
        for k in np.flatnonzero(hit):
            pa, pb = int(c[k, 0]), int(c[k, 1])
            ga, gb = int(gene_of[pa]), int(gene_of[pb])
            orient = "rc" if (idr[k] >= IDENTITY_MIN and idr[k] > idf[k]) else "fwd"
            confirmed.append((pa, pb, ga, gb,
                              float(max(idf[k], idr[k])),
                              float(max(cf[k], cr[k])), orient))
        print("verify", min(st + BATCH, len(cand)), "/", len(cand), flush=True)

    print("confirmed near-identity pairs", len(confirmed), flush=True)

    # ---- union genes: exact/RC edges + cross-gene confirmed near-identity ----
    uf = UnionFind(n_genes)
    for ga, gb in edges:
        uf.union(int(ga), int(gb))
    n_edges_cross_gene_near = 0
    for (pa, pb, ga, gb, ident, cov, orient) in confirmed:
        if ga != gb:
            uf.union(int(ga), int(gb))
            n_edges_cross_gene_near += 1
    print("near-identity cross-gene edges", n_edges_cross_gene_near, flush=True)

    comp_of = np.array([uf.find(g) for g in range(n_genes)], dtype=np.int64)
    root_to_id = {}
    comp_dense = np.empty(n_genes, dtype=np.int64)
    for g in range(n_genes):
        r = int(comp_of[g])
        if r not in root_to_id:
            root_to_id[r] = len(root_to_id)
        comp_dense[g] = root_to_id[r]
    n_components = len(root_to_id)

    comp_genes = {}
    comp_prom = {}
    for g in range(n_genes):
        c = int(comp_dense[g])
        comp_genes.setdefault(c, []).append(g)
        comp_prom[c] = comp_prom.get(c, 0) + int(n_prom[g])
    comp_sizes = {c: len(v) for c, v in comp_genes.items()}
    sizes = np.array([comp_sizes[c] for c in range(n_components)], dtype=np.int64)
    largest = int(sizes.max()) if n_components else 0
    print("n_components", n_components, "largest component genes", largest, flush=True)

    min_gene = np.array([min(comp_genes[c]) for c in range(n_components)], dtype=np.int64)

    def seeded_tiebreak(g):
        import hashlib
        h = hashlib.sha256(("%d:%s" % (SEED, gene_gid[g])).encode("utf-8")).hexdigest()
        return int(h[:15], 16)  # 60-bit, fits int64

    tie = np.array([seeded_tiebreak(int(min_gene[c])) for c in range(n_components)],
                   dtype=np.int64)
    comp_order = sorted(range(n_components),
                        key=lambda c: (-int(sizes[c]), int(tie[c])))

    bin_counts = {"train": 0, "val": 0, "test": 0}
    comp_split = np.empty(n_components, dtype=object)
    for c in comp_order:
        sz = int(sizes[c])
        best_bin = None
        best_ratio = None
        for name in ["train", "val", "test"]:
            ratio = (bin_counts[name] + sz) / TARGET[name]
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_bin = name
        comp_split[c] = best_bin
        bin_counts[best_bin] += sz

    gene_split = np.array([comp_split[int(comp_dense[g])] for g in range(n_genes)])
    realized = {k: int((gene_split == k).sum()) for k in ["train", "val", "test"]}
    print("realized gene split", realized, flush=True)

    # ---- audit exact/RC intersplit overlap (must be zero) ----
    canon_split = {}
    overlap = {"train_val": set(), "train_test": set(), "val_test": set()}
    for i in range(len(r400)):
        key = mm[r400[i]].tobytes()
        rkey = rc_bytes(np.frombuffer(key, dtype=np.uint8)).tobytes()
        c = key if key <= rkey else rkey
        sp = gene_split[int(gene_of[i])]
        canon_split.setdefault(c, set()).add(sp)
    for key, sps in canon_split.items():
        s = sps
        if "train" in s and "val" in s:
            overlap["train_val"].add(key)
        if "train" in s and "test" in s:
            overlap["train_test"].add(key)
        if "val" in s and "test" in s:
            overlap["val_test"].add(key)
    exact_rc_overlap = {k: len(v) for k, v in overlap.items()}
    print("exact/rc cross-split overlap unique keys", exact_rc_overlap, flush=True)

    # ---- bounded near-identity residual sampling (prespecified) ----
    del sketch
    rng2 = np.random.default_rng(SEED + 777)
    n_sample = 200000
    pairs = []
    while len(pairs) < n_sample:
        i = int(rng2.integers(0, len(r400)))
        j = int(rng2.integers(0, len(r400)))
        if gene_split[int(gene_of[i])] != gene_split[int(gene_of[j])]:
            pairs.append((i, j))
    pairs = np.asarray(pairs, dtype=np.int64)
    residual_hits = 0
    for st in range(0, len(pairs), BATCH):
        c = pairs[st:st + BATCH]
        A = np.asarray(mm[r400[c[:, 0]]], dtype=np.uint8)
        B = np.asarray(mm[r400[c[:, 1]]], dtype=np.uint8)
        idf, _ = gapless_batch(A, B)
        rcB = rc_batch(B)
        idr, _ = gapless_batch(A, rcB)
        residual_hits += int(((idf >= IDENTITY_MIN) | (idr >= IDENTITY_MIN)).sum())
    print("residual sample hits", residual_hits, "/", n_sample, flush=True)

    # ---- write artifacts ----
    import pandas as pd
    gid_list = [str(x) for x in gene_gid]
    assignment_rows = [(gid_list[g], int(comp_dense[g]), gene_split[g], int(n_prom[g]))
                       for g in range(n_genes)]
    pd.DataFrame(assignment_rows, columns=["gene_id", "component_id", "split", "n_promoters"]).to_csv(
        OUT / "split_assignment.tsv", sep="\t", index=False)

    comp_rows = [(int(c), comp_split[c], int(comp_sizes[c]), int(comp_prom[c]),
                  gid_list[int(min_gene[c])]) for c in range(n_components)]
    pd.DataFrame(comp_rows, columns=["component_id", "split", "n_genes", "n_promoters", "min_gene_id"]).to_csv(
        OUT / "components.tsv", sep="\t", index=False)

    near_rows = [(int(pa), int(pb), gid_list[ga], gid_list[gb],
                  float(ident), float(cov), orient)
                 for (pa, pb, ga, gb, ident, cov, orient) in confirmed]
    pd.DataFrame(near_rows, columns=["promoter_i", "promoter_j", "gene_a", "gene_b",
                                     "identity", "coverage", "orientation"]).to_csv(
        OUT / "near_identity_pairs.tsv", sep="\t", index=False)

    plan = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "runs/breakthrough_20261002/batch05/PROTOCOL.txt",
        "source_data": {
            "npz": str(DATA03 / "trackA_gene_promoters.npz"),
            "npz_sha256": sha256_file(DATA03 / "trackA_gene_promoters.npz"),
            "memmap_sym400": str(ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap"),
            "n_promoters": int(len(r400)),
            "n_genes": int(n_genes),
        },
        "E_label_definition": {
            "field": "Score_E",
            "source": "labels/build/gene_expression_label_table.tsv (gene-level)",
            "definition": "rank-scaled [0,1] gene-level expression strength: E_raw(g)=mean over tissues of log2(1+TPM_g,t); Score_E=rank(E_raw)/(n-1). NOT raw tissue mean; gene-level (all promoters share it).",
            "verified": "trackA_rows.tsv Score_E equals gene_expression_label_table.tsv Score_E for sampled genes; unique per gene",
        },
        "graph": {
            "nodes": "genes (all promoters of a gene stay together)",
            "edge_exact_rc": "canonical exact/RC duplicate sym400 promoter sequence shared between genes",
            "edge_near_identity": "detected near-identity promoter pair (>=90% identity over >=90% of 400bp, either orientation) between genes",
        },
        "near_identity_method": {
            "k": K, "n_hash": N_HASH, "bands": N_BANDS, "rows": ROWS,
            "hash": "h_i(x)=(a_i*x+b_i) mod (2^61-1); a_i,b_i from numpy default_rng(20261011)",
            "low_complexity_filter": "k-mers with <3 distinct ACGT bases or containing N excluded from sketch",
            "band_key": "bitwise XOR of 4 minhash values",
            "candidate_rule": "promoter pairs sharing >=2 band keys; band bucket cap %d; pair-count cap %d" % (BAND_CAP, PAIR_CAP),
            "minhash_equality_prefilter": ">=%d of %d minhash values equal before alignment" % (MIN_EQUAL, N_HASH),
            "verification": "gapless diagonal alignment (forward + RC, no indels), shifts [-40,40]; identity=matches/overlap_len; coverage=overlap_len/400; confirmed iff identity>=0.90 and coverage>=0.90 in either orientation",
            "exhaustive": False,
            "split_name": "exact/RC + detected-near-identity blocked (NOT homology-free)",
        },
        "retrieval_report": {
            "n_candidate_pairs_tracked": int(n_tracked),
            "n_candidates_share_ge2_bands": int(n_share_ge2),
            "n_candidates_after_minhash_prefilter": int(len(cand)),
            "n_confirmed_near_identity_pairs": int(len(confirmed)),
            "n_confirmed_cross_gene_edges": int(n_edges_cross_gene_near),
            "n_skipped_band_buckets_over_cap": int(n_skipped_buckets),
            "n_truncated_pair_increments": int(n_truncated),
        },
        "assignment": {
            "seed": SEED,
            "target": TARGET,
            "algorithm": "components sorted by gene_count desc, tie-break seeded hash of min gene_id; greedy assign to bin minimizing (count+size)/target_fraction",
            "info_used": "gene_id + promoter count (sequence/size only); target labels never read",
            "realized_gene_counts": realized,
            "n_components": int(n_components),
            "largest_component_genes": int(largest),
        },
        "audit": {
            "exact_rc_cross_split_overlap_unique_keys": exact_rc_overlap,
            "near_identity_residual_sampling": {
                "n_sampled_cross_split_pairs": int(n_sample),
                "n_hits": int(residual_hits),
                "procedure": "uniform random promoter pairs from different new-split partitions; gapless fwd+rc alignment at the same threshold",
                "note": "sampling is a bounded residual check, not a comprehensive homology guarantee",
            },
        },
        "target_labels_not_used": True,
        "elapsed_s": round(time.time() - t0, 1),
    }
    atomic_write_json(OUT / "split_plan.json", plan)
    atomic_write_json(OUT / "exact_rc_summary.json", {
        "stats": exact_stats,
        "cross_split_overlap_unique_keys": exact_rc_overlap,
    })
    split_code_map = {"train": 0, "val": 1, "test": 2}
    split_code = np.array([split_code_map[s] for s in gene_split], dtype=np.int8)
    np.savez_compressed(OUT / "split_arrays.npz",
                        gene_split=gene_split,
                        split_code=split_code,
                        split_code_map=np.array(list(split_code_map.items()), dtype=object),
                        comp_dense=comp_dense,
                        n_promoters=n_prom)
    print("WROTE split artifacts", round(time.time() - t0, 1), "s")
    print(json.dumps({"realized": realized, "n_components": n_components,
                      "largest": largest,
                      "confirmed_near": len(confirmed),
                      "exact_rc_overlap": exact_rc_overlap,
                      "residual_hits": residual_hits}, indent=2))


if __name__ == "__main__":
    main()
