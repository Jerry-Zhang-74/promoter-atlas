#!/usr/bin/env python3
"""Batch05 version registry (append-only, atomic) + CHANGELOG append."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_common import atomic_write_json, sha256_file  # noqa: E402

REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch05"
MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch05"
TOP = ROOT / "reports" / "breakthrough_20261002"
RUNS = ROOT / "runs" / "breakthrough_20261002" / "batch05"
CANDS = {"repeat_bag": "v6.6.0-scratchgene.rc6", "diverse_bag": "v6.6.0-scratchgene.rc7"}
SEED = 20261011


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


def main():
    cons = json.loads((REPORT / "consolidated_summary_v5.json").read_text(encoding="utf-8"))
    candidates = []
    for arm, cand in CANDS.items():
        ckpt = str(MODEL_DIR / ("%s_s%d" % (arm, SEED)) / "final.pt")
        t = cons["arms"][arm]["test"]["full"]
        v = cons["arms"][arm]["val"]["full"]
        candidates.append({
            "candidate_version": cand,
            "track": "gene-label E (sequence-only from scratch), batch05 frozen sequence-component split",
            "internal_id": arm,
            "status": "reviewed candidate (scoped, non-causal); NOT promoted",
            "checkpoints": [ckpt],
            "checkpoint_sha256": {str(SEED): sha256_file(MODEL_DIR / ("%s_s%d" % (arm, SEED)) / "final.pt")},
            "lr": 0.001, "epochs": 12, "seeds": [SEED],
            "metrics": {
                "test_4slot_full_E_sp": t["spearman"],
                "test_4slot_full_E_r2": t["r2"],
                "test_4slot_full_E_mae": t["mae"],
                "val_4slot_full_E_sp": v["spearman"],
                "val_4slot_full_E_r2": v["r2"],
            },
            "bag_policy": ("repeat same anchor in 4 slots" if arm == "repeat_bag"
                           else "up to 4 distinct promoters, deterministic cycling"),
            "training_scope": "human genes on frozen split (57827 genes), random init, no old weights/teacher/transferred labels",
            "split": "exact/RC + detected-near-identity blocked (NOT homology-free)",
            "exposure": "none inherited (scratch); retrospective; single seed",
            "decision": "scoped controlled-contrast evidence under frozen split; NOT a claim that arm similarity proves capacity causality; no production promotion",
        })

    candidates.append({
        "candidate_version": "v6.6.0-annridge.rc2",
        "track": "gene-label E annotation diagnostics (ridge, 3 feature variants, group-preserving inner fold)",
        "internal_id": "ann_ridge",
        "status": "reviewed diagnostic baseline (descriptive, non-causal)",
        "checkpoints": [str(MODEL_DIR / "ann_diagnostics_v5.npz")] if (MODEL_DIR / "ann_diagnostics_v5.npz").exists() else [],
        "lr": "n/a (ridge alpha selected on train-inner group-preserving fold)",
        "epochs": "n/a", "seeds": [SEED],
        "decision": "diagnostic baseline only; NOT a new NN arm; additive count info reported descriptively",
    })

    registry = {
        "batch": "batch05",
        "generated_at": now_iso(),
        "policy": "VERSION_POLICY.txt",
        "production_pointer": "promoter_atlas_v6.3 (retained); v6.4 cellline pilot (retained); NO production pointer change",
        "candidates": candidates,
        "review_status": "pending independent review",
        "key_observation": json.loads((REPORT / "consolidated_summary_v5.json").read_text(encoding="utf-8"))["evidence"],
        "frozen_manifest_sha256": json.loads((RUNS / "frozen_manifest.json").read_text(encoding="utf-8")).get("manifest_sha256"),
    }
    atomic_write_json(REPORT / "version_registry_v5.json", registry)

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
