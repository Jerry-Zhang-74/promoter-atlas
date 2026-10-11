#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen SMALL numerical-test manifest for batch34, generated BEFORE any inference.

Deterministic; re-running reproduces byte-identical inputs. New artificial sequences are
software tests only and carry NO biological-validation claim.
"""
import hashlib, json, os, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "breakthrough_20261002" / "batch34" / "testdata"
OUT.mkdir(parents=True, exist_ok=True)
B31 = REPO / "data" / "breakthrough_20261002" / "batch31"
B31LOCAL = B31 / "inputs" / "local_supported.fasta"
B31SCORES = B31 / "results" / "parm_pred_local_supported.tsv"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_fa(p):
    d = {}
    name = None
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            name = line[1:].strip(); d[name] = ""
        elif name:
            d[name] += line.strip()
    return d


local = read_fa(B31LOCAL)
saved = {}
for line in Path(B31SCORES).read_text(encoding="utf-8").splitlines()[1:]:
    parts = line.split("\t")
    saved[parts[1]] = float(parts[2])

CTRL = {"A1|ACTB|100": local["A1|ACTB|100"], "CMVfull|CMV|584": local["CMVfull|CMV|584"]}


def one_base(seq):
    i = len(seq) // 2
    old = seq[i]
    new = "A" if old != "A" else "C"
    return seq[:i] + new + seq[i + 1:], i, old, new


m1, i1, o1, n1 = one_base(CTRL["A1|ACTB|100"])
m2, i2, o2, n2 = one_base(CTRL["CMVfull|CMV|584"])

rng = random.Random(34)
novel133 = "".join(rng.choice("ACGT") for _ in range(133))
seq600 = "TATAAA" * 100
seq601 = seq600 + "A"
seq2000 = "ACGTGCAATC" * 200  # exactly 2000 bp
valid_det = "GACATTGATTATTGACTAGTTATTAATAGTAATCAATTACGGGGTC"  # 46 bp deterministic
ambig = valid_det[:20] + "R" + valid_det[21:]
digit = valid_det[:20] + "1" + valid_det[21:]
withN = "ACGTACGT" + "N" * 8 + "ACGTTGCA"

files = {}
def add(fname, records, expect, desc):
    p = OUT / fname
    with open(p, "w", encoding="utf-8", newline="") as f:
        for rid, seq in records:
            f.write(">" + rid + "\n" + seq + "\n")
    files[fname] = {"path": str(p.relative_to(REPO)).replace("\\", "/"), "sha256": sha(p),
                    "description": desc, "expect": expect}

add("test_controls.fasta", list(CTRL.items()),
    [{"id": k, "input_validity": "VALID", "software_length_support": "SUPPORTED",
      "batch31_saved_score": saved[k], "tolerance": 1e-4} for k in CTRL],
    "2 already-scored local controls re-inferred; compare to batch31 saved fivefold score (tolerance 1e-4, no bit-identity claim)")
add("test_mutants.fasta", [("ctrl_A1_mut_%d_%s2%s" % (i1, o1, n1), m1), ("ctrl_CMVfull_mut_%d_%s2%s" % (i2, o2, n2), m2)],
    [{"id": "ctrl_A1_mut_%d_%s2%s" % (i1, o1, n1), "input_validity": "VALID", "software_length_support": "SUPPORTED", "note": "1bp changed from A1 at index %d (%s->%s)" % (i1, o1, n1)},
     {"id": "ctrl_CMVfull_mut_%d_%s2%s" % (i2, o2, n2), "input_validity": "VALID", "software_length_support": "SUPPORTED", "note": "1bp changed from CMVfull at index %d (%s->%s)" % (i2, o2, n2)}],
    "2 deterministic one-base-changed local fragments (software tests, NOT biological validation)")
add("test_valid_novel.fasta", [("synthetic600_TATA6x100", seq600), ("novel133_seed34", novel133)],
    [{"id": "synthetic600_TATA6x100", "input_validity": "VALID", "software_length_support": "SUPPORTED"},
     {"id": "novel133_seed34", "input_validity": "VALID", "software_length_support": "SUPPORTED"}],
    "one valid 600bp deterministic synthetic seq; one 133bp novel deterministic seq (software tests only)")
add("test_unsupported.fasta", [("unsupported601", seq601), ("unsupported2000", seq2000)],
    [{"id": "unsupported601", "input_validity": "VALID", "software_length_support": "UNSUPPORTED", "score": None},
     {"id": "unsupported2000", "input_validity": "VALID", "software_length_support": "UNSUPPORTED", "score": None}],
    "601bp and 2000bp: >L_max=600 -> UNSUPPORTED, null score, no truncation/tiling")
add("test_mixed.fasta", [("mixed_valid_det", valid_det), ("mixed_ambig_R", ambig), ("mixed_digit_1", digit)],
    [{"id": "mixed_valid_det", "input_validity": "VALID", "software_length_support": "SUPPORTED"},
     {"id": "mixed_ambig_R", "input_validity": "INVALID", "software_length_support": "NA", "score": None, "invalid_reason": "INVALID_CHARS:R"},
     {"id": "mixed_digit_1", "input_validity": "INVALID", "software_length_support": "NA", "score": None, "invalid_reason": "INVALID_CHARS:1"}],
    "mixed FASTA: 1 valid (still scored) + 1 IUPAC-ambiguous (R) + 1 digit -> deterministic per-record handling")
add("test_N_allowed.fasta", [("valid_with_N", withN)],
    [{"id": "valid_with_N", "input_validity": "VALID", "software_length_support": "SUPPORTED", "contains_N": True}],
    "N is officially supported (all-zero one-hot): allowed and flagged, NOT removed")
add("test_empty.fasta", [], [{"global": "EMPTY_INPUT_NO_RECORDS"}],
    "empty FASTA -> fatal input error before inference (exit 2)")
add("test_duplicates.fasta", [("dup1", valid_det), ("dup1", valid_det)],
    [{"global": "DUPLICATE_RECORD_IDS", "duplicate_ids": ["dup1"]}],
    "duplicate record IDs -> fatal input error before inference (exit 2)")

man = {
    "batch": "batch34", "version": "v6.23.0-scoped-inference.rc1",
    "frozen_before_inference": True,
    "generated_at_note": "deterministic generator code/batch34/make_test_manifest.py (Random seed 34)",
    "control_source": "data/breakthrough_20261002/batch31/inputs/local_supported.fasta + saved scores parm_pred_local_supported.tsv",
    "accepted_alphabet_policy": {"canonical": "ACGT", "official_ambiguous": "N (allowed, flagged)",
                                 "rejected": "all other chars -> record INVALID"},
    "length_policy": {"supported": "1..600", "unsupported": ">600 null score"},
    "files": files,
}
manp = REPO / "reports" / "breakthrough_20261002" / "batch34" / "test_manifest.json"
manp.parent.mkdir(parents=True, exist_ok=True)
manp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"wrote": str(manp), "n_files": len(files), "files": list(files)}, ensure_ascii=False))
