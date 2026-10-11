#!/usr/bin/env python3
"""Batch04 version registry (append-only, atomic) + CHANGELOG append."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_common import atomic_write_json, sha256_file  # noqa: E402

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch04"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch04"
TOP = ROOT / "reports" / "breakthrough_20261002"
CANDS = {"repeat_bag": "v6.6.0-scratchgene.rc4", "diverse_bag": "v6.6.0-scratchgene.rc5"}
SEEDS = [20261011, 20261012, 20261013]


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


def main():
    cons = json.loads((REPORT / "consolidated_summary.json").read_text(encoding="utf-8"))
    candidates = []
    for arm, cand in CANDS.items():
        ckpts = [str(MODEL_DIR / ("%s_s%d" % (arm, s)) / "final.pt") for s in SEEDS]
        am = cons["arms"][arm]["mean"]
        candidates.append({
            "candidate_version": cand,
            "track": "gene-label E (sequence-only from scratch), batch04 gene-promoter diversity contrast",
            "internal_id": arm,
            "status": "reviewed candidate (scoped, non-causal); NOT promoted",
            "checkpoints": ckpts,
            "checkpoint_sha256": {str(s): sha256_file(MODEL_DIR / ("%s_s%d" % (arm, s)) / "final.pt")
                                  for s in SEEDS},
            "lr": 0.001,
            "epochs": 12,
            "seeds": SEEDS,
            "metrics": {
                "test_4slot_full_E_sp_mean": am["test_4slot_full_sp"],
                "test_4slot_unexp_E_sp_mean": am["test_4slot_unexp_sp"],
                "test_4slot_full_E_r2_mean": am["test_4slot_full_r2"],
                "val_4slot_full_E_sp_mean": am["val_4slot_full_sp"],
                "test_allprom_full_E_sp_mean": am["test_allprom_full_sp"],
                "test_allprom_unexp_E_sp_mean": am["test_allprom_unexp_sp"],
                "per_seed_test_4slot_sp": {str(s): cons["arms"][arm]["seeds"][str(s)]["test_4slot_full_sp"]
                                           for s in SEEDS},
            },
            "bag_policy": ("repeat same anchor in 4 slots" if arm == "repeat_bag"
                           else "up to 4 distinct promoters, deterministic cycling"),
            "training_scope": "human original-train genes only (46618), random init, no old weights/teacher/transferred labels",
            "exposure": "none inherited (scratch); retrospective val/test already historically seen; residual cross-gene homology/exact-dup remains",
            "decision": "scoped controlled-contrast evidence; NOT a claim that arm similarity proves capacity causality; no production promotion",
        })
    # annotation ridge entry
    candidates.append({
        "candidate_version": "v6.6.0-annridge.rc1",
        "track": "gene-label E annotation diagnostics (ridge, 3 feature variants)",
        "internal_id": "ann_ridge",
        "status": "reviewed diagnostic baseline (descriptive, non-causal)",
        "checkpoints": [str(MODEL_DIR / "ann_diagnostics_npz.npz")] if (MODEL_DIR / "ann_diagnostics_npz.npz").exists() else [],
        "lr": "n/a (ridge alpha selected on train-inner)",
        "epochs": "n/a",
        "seeds": [20261011],
        "decision": "diagnostic baseline only; NOT a new NN arm; additive count info reported descriptively",
    })

    registry = {
        "batch": "batch04",
        "generated_at": now_iso(),
        "policy": "VERSION_POLICY.txt",
        "production_pointer": "promoter_atlas_v6.3 (retained); v6.4 cellline pilot (retained); NO production pointer change",
        "candidates": candidates,
        "review_status": "pending independent review",
        "key_observation": "B2 (diverse promoter bag) exceeds B1 (repeat anchor) by ~0.11 test E Spearman across 3 seeds under matched input/head/update count; however the diverse NN (test sp ~0.734) is essentially tied by a simple count+composition ridge (test sp ~0.731) and the ridge R2 (~0.548) exceeds the NN R2 (~0.514), so the NN's marginal ranking value over cheap annotation features is limited",
        "manifest_sha256": json.loads((ROOT / "runs/breakthrough_20261002/batch04/selection_run_manifest.json").read_text(encoding="utf-8")).get("manifest_sha256"),
    }
    atomic_write_json(REPORT / "version_registry.json", registry)

    # append CHANGELOG
    lines = []
    for c in candidates:
        lines.append("[%s] %s (%s) -> %s. %s" % (
            now_iso(), c["candidate_version"], c["internal_id"], c["status"],
            json.dumps(c.get("metrics", {}), ensure_ascii=False)))
    lines.append("[%s] production pointer unchanged (v6.3 main / v6.4 cellline pilot)." % now_iso())
    with open(TOP / "CHANGELOG.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(registry, indent=2, ensure_ascii=False))
    print("DONE")


if __name__ == "__main__":
    main()
