# -*- coding: utf-8 -*-
"""Batch35: independently recompute SHA256 of the pinned official PARM source + 5 HEK293 fold
weights (read-only, referenced in place) and write reports/batch35/frozen_assets.json.

Nothing licensed is copied; only hashes are recorded. The asset tree itself is untouched.
"""
import hashlib, json, os, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPO / "data/breakthrough_20261002/batch31/_external"
OUT = REPO / "reports/breakthrough_20261002/batch35/frozen_assets.json"
FOLDS = [f"HEK293/HEK293_fold{i}.parm" for i in range(5)]
SRC = ["parm_pkg/PARM/PARM_predict.py", "parm_pkg/PARM/PARM_utils_load_model.py",
       "parm_pkg/PARM/PARM_misc.py", "parm_pkg/PARM/version.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


weights = {rel: sha(ASSET_ROOT / "src" / rel) for rel in FOLDS}
source = {rel: sha(ASSET_ROOT / rel) for rel in SRC}
man = {
    "batch": "batch35", "version": "v6.24.0-293ft-cohort-readiness.rc1",
    "purpose": "Frozen SHA manifest of the pinned unmodified official PARM source + 5 HEK293 fold weights, reused read-only from batch31 _external. batch35 recomputes these hashes independently (not copied from batch34).",
    "provenance": "https://github.com/vansteensellab/PARM commit 260a27082942060364c1c4e2393018a0fa128844",
    "license": "PARM repo LICENSE (academic local evaluation). Weights/source NOT redistributed/copied into any deliverable as ours; referenced in place.",
    "asset_root": "data/breakthrough_20261002/batch31/_external",
    "independently_recomputed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "weights_sha256": weights, "source_sha256": source,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"wrote": str(OUT.relative_to(REPO)).replace("\\", "/"),
                  "n_weights": len(weights), "n_source": len(source)}, ensure_ascii=False))
