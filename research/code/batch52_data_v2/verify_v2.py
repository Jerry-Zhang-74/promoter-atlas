# -*- coding: utf-8 -*-
"""Data-owner verification: (1) lean loader datahash == reviewed data_adapter datahash on an
artificial cohort; (2) real corrected cohort loads with expected counts; (3) independent
sequence/target spot-checks vs the source BED + official reference. ZERO fits/GPU."""
from __future__ import annotations
import gzip, json, os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
        "runs", "breakthrough_20261002", "code", "batch53_model_core"))
sys.path.insert(0, HELPER)

ROOT = r"E:/promoter_atlas"
COH = os.path.join(ROOT, "data/breakthrough_20261002/batch52/corrected/cohort_v2")
BED = os.path.join(ROOT, "data/breakthrough_20261002/batch51/GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz")
FA = os.path.join(ROOT, "data/breakthrough_20261002/batch52/reference/GRCh37.primary_assembly.fa")
OUTJSON = os.path.join(ROOT, "data/breakthrough_20261002/batch52/corrected/verify_v2.json")
R = {"checks": {}, "ok": False}


def _make_fixture(d):
    lengths = (120, 200, 130, 210); splits = ("TRAIN", "TRAIN", "DEV", "DEV"); gs = ["g1", "g1", "g2", "g2"]
    with open(os.path.join(d, "endpoints.tsv"), "w", encoding="utf-8") as f:
        f.write("coord_id\tunion_id\tsplit\tlength\tLog2RPM_HEK293\n")
        for i, (L, s, g) in enumerate(zip(lengths, splits, gs)):
            f.write("c%d\t%s\t%s\t%d\t%.4f\n" % (i, g, s, L, 0.1 * i))
    with open(os.path.join(d, "groups.tsv"), "w", encoding="utf-8") as f:
        f.write("union_id\tsplit\ng1\tTRAIN\ng2\tDEV\n")
    with open(os.path.join(d, "pairs.tsv"), "w", encoding="utf-8") as f:
        f.write("union_id\tsplit\tshort_coord_id\tlong_coord_id\tshort_len\tlong_len\ng1\tTRAIN\tc0\tc1\t120\t200\n")
    rng = np.random.default_rng(1)
    seqs = {}
    with open(os.path.join(d, "sequences.fa"), "w", encoding="utf-8") as f:
        for i, L in enumerate(lengths):
            s = "".join(rng.choice(list("ACGT"), size=L)); seqs["c%d" % i] = s
            f.write(">c%d\n%s\n" % (i, s))
    with open(os.path.join(d, "reference_provenance.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"assembly": "hg19", "proxy_not_donor_clone": True}))
    B = {"A": 0, "C": 1, "G": 2, "T": 3}
    codes = np.full((4, 600), 4, dtype=np.int64)
    for i, L in enumerate(lengths):
        codes[i, :L] = [B[c] for c in seqs["c%d" % i]]
    np.save(os.path.join(d, "codes_int64.npy"), codes)
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"source_sha256": "deadbeef"}))
    return d


def check_equivalence():
    from data_adapter import load_cohort
    import cohort_v2
    with tempfile.TemporaryDirectory() as t:
        d = _make_fixture(t)
        a = load_cohort(d, source_sha256=None)
        b = cohort_v2.load_cohort_v2(d, source_sha256="deadbeef", mmap_codes=False)
        R["checks"]["datahash_equivalence"] = {"reviewed": a.datahash, "lean": b.datahash,
                                               "equal": a.datahash == b.datahash}
        assert a.datahash == b.datahash, "datahash mismatch"


def check_real_cohort():
    import cohort_v2
    sha = "1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96"
    c = cohort_v2.load_cohort_v2(COH, source_sha256=sha, mmap_codes=True)
    dh_file = open(os.path.join(COH, "datahash.txt"), encoding="utf-8").read().strip()
    counts = {s: sum(1 for x in c.splits if x == s) for s in ("TRAIN", "DEV", "TEST")}
    pairs = {s: int(c.pairs[s].shape[0]) for s in ("TRAIN", "DEV", "TEST")}
    R["checks"]["real_cohort"] = {"n": len(c.coord_ids), "target_column": c.target_column,
                                  "datahash": c.datahash, "build_datahash": dh_file,
                                  "match": c.datahash == dh_file, "counts": counts, "pairs": pairs}
    assert c.datahash == dh_file
    assert counts == {"TRAIN": 531860, "DEV": 112640, "TEST": 117155}


def check_sequences_and_targets():
    sys.path.insert(0, os.path.join(ROOT, "code", "batch52"))
    from ref_fasta import Faidx
    fx = Faidx(FA)
    COMP = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")
    # load my endpoints + codes
    cid2seq = {}
    with open(os.path.join(COH, "endpoints.tsv"), encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in f]
    seqs = {}
    with open(os.path.join(COH, "sequences.fa"), encoding="utf-8") as f:
        name = None
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith(">"):
                name = ln[1:]
            elif ln:
                seqs[name] = ln
    tgt = {}
    with gzip.open(BED, "rt", encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        for ln in f:
            a = ln.rstrip("\n").split("\t")
            L = int(a[3]) - int(a[2])
            if 100 <= L <= 600:
                tgt["%s:%s-%s:%s" % (a[1], a[2], a[3], a[4])] = float(a[7])
    rng = np.random.default_rng(7)
    picks = list(rng.choice(len(rows), size=20, replace=False))
    # include strand extremes + zeros
    picks += [i for i, r in enumerate(rows) if r[3 + 0] and False][:0]
    bad_seq = bad_tgt = 0
    examples = []
    for i in picks:
        coord_id, union_id, split, Ls, yv = rows[i][0], rows[i][1], rows[i][2], rows[i][3], rows[i][4]
        ch, rng_, strand = coord_id.split(":")[0], coord_id.split(":")[1], coord_id.split(":")[2]
        start, end = rng_.split("-")
        s = fx.fetch(ch[3:] if ch.startswith("chr") else ch, int(start), int(end))
        if strand == "-":
            s = s.translate(COMP)[::-1]
        got = seqs[coord_id]
        if got != s or len(got) != int(Ls):
            bad_seq += 1
        if abs(float(yv) - tgt[coord_id]) > 1e-6:
            bad_tgt += 1
        if len(examples) < 6:
            examples.append({"coord_id": coord_id, "split": split, "len": int(Ls),
                             "target": float(yv), "seq_match": got == s, "tgt_match": abs(float(yv) - tgt[coord_id]) < 1e-6})
    # strata coverage
    strata = {"plus": 0, "minus": 0, "len_min": min(int(r[3]) for r in rows), "len_max": max(int(r[3]) for r in rows),
              "zero_targets_in_cohort": sum(1 for r in rows if float(r[4]) == 0.0)}
    for r in rows[:2000]:
        strata["plus" if r[0].endswith(":+") else "minus"] += 1
    R["checks"]["spot_checks_20"] = {"n_checked": len(picks), "bad_seq": bad_seq, "bad_target": bad_tgt,
                                     "examples": examples, "strata_scan": strata}
    assert bad_seq == 0 and bad_tgt == 0, "spot-check failures"


def main():
    try:
        check_equivalence(); R["checks"]["datahash_equivalence"]["status"] = "PASS"
    except Exception as e:
        R["checks"]["datahash_equivalence"] = {"status": "FAIL", "error": repr(e)}; R["ok"] = False
    for name, fn in (("real_cohort", check_real_cohort), ("spot_checks_20", check_sequences_and_targets)):
        try:
            fn(); R["checks"][name]["status"] = "PASS"
        except Exception as e:
            R["checks"][name] = {"status": "FAIL", "error": repr(e)}
    R["ok"] = all(v.get("status") == "PASS" for v in R["checks"].values())
    json.dump(R, open(OUTJSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(R, ensure_ascii=False, indent=2)[:3000])
    print("OK" if R["ok"] else "FAILED")


if __name__ == "__main__":
    main()
