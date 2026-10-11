# -*- coding: utf-8 -*-
"""batch43 v6.30.0-group-error-diagnostic.rc1 shared paths + fixed contracts.

FINITE posthoc diagnostic reuse of the ALREADY FROZEN batch42 group-heldout
pilot.  ZERO estimator fits/solves/regression/calibration/gradients/optimizer
steps, ZERO GPU/neural forward, ZERO web/download/source search.  Old batch42
artifacts are READ-ONLY.  Feature functions are imported byte-identically from
the frozen batch42 code module (feature_code_sha256 pinned in the saved models).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

BATCH = "batch43"
VERSION = "v6.30.0-group-error-diagnostic.rc1"

B42_DATA = ROOT / "data" / "breakthrough_20261002" / "batch42"
B42_REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch42"
B42_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch42"
B43_DATA = ROOT / "data" / "breakthrough_20261002" / "batch43"
B43_RUN = ROOT / "runs" / "breakthrough_20261002" / "batch43"
B43_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch43"
B43_REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch43"
B43_FIG = B43_REPORT / "figures"

# --- frozen batch42 inputs (read-only) --------------------------------------
FROZEN_TABLE = ROOT / "data" / "breakthrough_20261002" / "batch41" / "frozen_input_table.tsv"
DESIGN_FA = ROOT / "data" / "breakthrough_20261002" / "batch40" / "source" / "GSM9580234_design_rmIllegalChars.fa.gz"
H1_CACHE = ROOT / "data" / "breakthrough_20261002" / "batch41" / "cache" / "h1" / "preds.json"
PARM_CACHE = ROOT / "data" / "breakthrough_20261002" / "batch41" / "cache" / "parm" / "preds.json"
GROUPS_TSV = B42_DATA / "groups.tsv"
FROZEN_SPLIT = B42_DATA / "frozen_split.json"
PRED_TEST = B42_DATA / "predictions_test.tsv"
B42_METRICS = B42_REPORT / "metrics.json"
B42_REVIEW = B42_REPORT / "independent_review.json"

# expected SHAs (frozen, from batch42 artifacts)
FROZEN_TABLE_SHA = "984093604759d7f7e0a968765d278177941c9658c37f356149afb6ee78a9fbf1"
DESIGN_SHA = "a61f3d7624bdc1cd6066cb0cfb015d482e2994885cf321da9e0bdb7e753aa82b"
FEATURE_CODE_SHA = "732d270e5ca198cf9c6d8e87633f67612399b20278b34acd26698760dd1339fa"
B42_METRICS_SHA = "48390a69b82a0cd8277ff6e32073b7b66a776997078627dad45f6690882309ec"

# --- import frozen batch42 feature/tooling contract (read-only, byte-identical)
sys.path.insert(0, str(B42_CODE))
import b42_common as B42  # noqa: E402  (frozen feature contract)

assert B42.sha256_file(B42_CODE / "b42_common.py") == FEATURE_CODE_SHA, "b42 feature-code SHA mismatch"

composition_features = B42.composition_features
h1_features = B42.h1_features
spearman = B42.spearman
mae = B42.mae
r2 = B42.r2
rc = B42.rc
sha256_file = B42.sha256_file
sha256_text = B42.sha256_text
now_local = B42.now_local

CONST_BASELINE = None  # loaded from metrics.json (train weighted mean)


def set_thread_limits(n=4):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[k] = str(n)


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_groups():
    """label -> dict(group, canonical_group_sha, split, chr, start, sha, component_id)"""
    gmap = {}
    with open(GROUPS_TSV, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            lab, grp, canon, sp, chrom, start, seqsha, comp = line.rstrip("\n").split("\t")
            gmap[lab] = {"group": grp, "canonical_group_sha": canon, "split": sp,
                         "chr": chrom, "start": int(start), "sequence_sha256": seqsha,
                         "component_id": int(comp)}
    return gmap


def load_models():
    return {rid: jload(B42_DATA / "models" / (rid + ".json")) for rid in ("R0", "R1", "Rnull")}


def predict(model, Z):
    """Frozen additive replay: normalized projection + weighted-train-label-mean intercept."""
    import numpy as np
    mu = np.asarray(model["feature_mean"], float)
    sc = np.asarray(model["feature_scale"], float)
    b = np.asarray(model["coeff"], float)
    return model["intercept_weighted_train_label_mean"] + ((Z - mu) / sc) @ b


def author_parent_families(labels):
    """Reproduce batch42 Phase-A author parent-tile run families WITHOUT any
    homology work (label metadata only).  Returns label -> family-key."""
    import collections
    import gzip
    import re
    LBL = re.compile(r"^(?P<roi>.+)_(?P<idx>\d+)_(?P<s>[+-])::(?P<chr>chr[\w]+):(?P<st>\d+)-(?P<en>\d+)(?P<s2>[+-])$")
    design = {}
    with gzip.open(DESIGN_FA, "rt") as fh:
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
    runs = collections.defaultdict(list)
    for nm in design:
        m = LBL.match(nm)
        if m:
            runs[(m.group("roi"), m.group("chr"))].append((int(m.group("st")), nm))
    fam_run_id = {}
    for key, lst in runs.items():
        lst.sort()
        run_start = lst[0][0]
        prev = None
        for st, nm in lst:
            if prev is not None and st - prev != 90:
                run_start = st
            fam_run_id[nm] = (key[0], key[1], run_start)
            prev = st
    return {lab: fam_run_id.get(lab, ("UNPARSED", lab)) for lab in labels}
