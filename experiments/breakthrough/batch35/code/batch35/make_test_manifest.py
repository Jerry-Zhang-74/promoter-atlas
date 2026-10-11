# -*- coding: utf-8 -*-
"""Batch35 frozen test manifest for the NEW fixed wrapper. Deterministic; generated BEFORE inference.

Adds the two usability-defect regression cases:
  D#1 fresh nested output path (json parent must not pre-exist) with SEPARATE json/tsv parents;
  D#2 invalid-only / failure paths must run with NO GPU available.
Crosscheck target = frozen batch31/batch34 fivefold score for A1|ACTB|100 (documented tolerance 1e-4).
"""
import hashlib, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TD = REPO / "data/breakthrough_20261002/batch35/testdata"
TD.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


A1 = "GGCGCGGGGCCAATCAGCGTGCGCCGTTCCGAAAGTTGCCTTTTATGGCTCGAGCGGCCGCGGCGGCGCCCTATAAAACCCAGCGGCGCGACGCGCCACC"
assert len(A1) == 100 and set(A1) <= set("ACGT")
valid_det = "GACATTGATTATTGACTAGTTATTAATAGTAATCAATTACGGGGTC"
ambig = valid_det[:20] + "R" + valid_det[21:]
digit = valid_det[:20] + "1" + valid_det[21:]

files = {}


def write(name, records, expect, desc):
    p = TD / name
    with open(p, "w", encoding="utf-8", newline="") as f:
        for rid, seq in records:
            f.write(">" + rid + "\n" + seq + "\n")
    files[name] = {"path": str(p.relative_to(REPO)).replace("\\", "/"), "sha256": sha(p),
                   "description": desc, "expect": expect}


write("known_control.fasta", [("A1|ACTB|100", A1)],
      [{"id": "A1|ACTB|100", "input_validity": "VALID", "software_length_support": "SUPPORTED",
        "frozen_batch31_fivefold_score": 3.1036716, "tolerance": 1e-4,
        "crosscheck_note": "frozen batch31 saved score, independently reproduced by batch34 at abs_diff 0.0"}],
      "ONE valid known control (A1|ACTB|100) for the fresh-nested-path GPU regression")
write("invalid_only.fasta", [("inv_R", ambig), ("inv_digit", digit)],
      [{"id": "inv_R", "input_validity": "INVALID", "invalid_reason": "INVALID_CHARS:R", "score": None},
       {"id": "inv_digit", "input_validity": "INVALID", "invalid_reason": "INVALID_CHARS:1", "score": None}],
      "invalid-only FASTA: no record is scored; must run with NO GPU")
write("empty.fasta", [], [{"global": "EMPTY_INPUT_NO_RECORDS"}],
      "empty FASTA -> fatal input error exit 2 (no GPU)")
write("duplicates.fasta", [("dup1", valid_det), ("dup1", valid_det)],
      [{"global": "DUPLICATE_RECORD_IDS", "duplicate_ids": ["dup1"]}],
      "duplicate record IDs -> fatal input error exit 2 (no GPU)")

man = {"batch": "batch35", "version": "v6.24.0-293ft-cohort-readiness.rc1",
       "frozen_before_inference": True,
       "generated_at_note": "deterministic generator code/batch35/make_test_manifest.py",
       "crosscheck_source": "data/breakthrough_20261002/batch31/results/parm_pred_local_supported.tsv",
       "tolerance": 1e-4, "files": files}
(REPO / "reports/breakthrough_20261002/batch35").mkdir(parents=True, exist_ok=True)
(REPO / "reports/breakthrough_20261002/batch35/test_manifest.json").write_text(
    json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"n_files": len(files), "files": list(files)}, ensure_ascii=False))
