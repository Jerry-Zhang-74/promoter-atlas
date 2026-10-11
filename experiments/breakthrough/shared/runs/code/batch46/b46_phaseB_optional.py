# -*- coding: utf-8 -*-
"""batch46 optional descriptive: within-triplet 3-point Spearman(pred_K562, meas_K562).
Descriptive only; NOT a substitute for the fixed selection utility; unnecessary if redundant."""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C
import b46_tables as T
import b46_phaseB_analysis as A


def main():
    frags = T.build_fragments()
    triplets = T.build_triplets(frags, T.load_freezer())
    recs = A.per_triplet_records(triplets)
    out = {}
    for lib in ("short", "long"):
        rhos = []
        for t in [x for x in triplets if x["library"] == lib]:
            c = t["cands"]
            rho, n = C.spearman([c[s]["pred_K562"] for s in C.SELS],
                                [c[s]["K562_meas"] for s in C.SELS])
            if rho is not None:
                rhos.append(rho)
        out[lib] = {"n_triplets": sum(1 for t in triplets if t["library"] == lib),
                    "n_finite_rho": len(rhos),
                    "mean_within_triplet_rho": (sum(rhos) / len(rhos)) if rhos else None}
    C.write_json(C.OUT / "optional_within_triplet_rho.json", {
        "batch": C.BATCH, "version": C.VERSION, "phase": "B_optional",
        "definition": "3-point Spearman between published K562 prediction and K562 measured summary of the "
                      "three candidates in each triplet; undefined triplets (ties) excluded.",
        "status": "descriptive only; the fixed selection utility is the primary evidence; NOT a broad pooled rho.",
        "by_library": out})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
