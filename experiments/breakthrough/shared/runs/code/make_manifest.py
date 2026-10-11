#!/usr/bin/env python3
"""Compute SHA-256 manifests for code + models + key data/outputs of batch01."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "breakthrough_20261002"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(pattern_root, glob_pat, rel_to):
    out = {}
    for p in sorted(pattern_root.glob(glob_pat)):
        if p.is_file():
            out[str(p.relative_to(rel_to))] = sha256_file(p)
    return out


def main():
    code_dir = ROOT / "runs" / "breakthrough_20261002" / "code"
    model_dir = ROOT / "models" / "breakthrough_20261002"
    data_dir = ROOT / "data" / "breakthrough_20261002"
    rep_dir = ROOT / "reports" / "breakthrough_20261002"

    manifest = {
        "batch": "breakthrough_20261002_batch01",
        "code": {k: v for k, v in sorted(sha256_dir(code_dir, "*.py", code_dir).items())},
        "code_protocol": sha256_file(code_dir / "protocol.json"),
        "models": sha256_dir(model_dir, "*.pt", model_dir),
        "data_key": {
            "trackA_rows.tsv": sha256_file(data_dir / "trackA_rows.tsv"),
            "trackA_embeddings.f32": sha256_file(data_dir / "trackA_embeddings.f32"),
            "trackA_seqfeats.f32": sha256_file(data_dir / "trackA_seqfeats.f32"),
            "trackB_grouped.tsv": sha256_file(data_dir / "trackB_grouped.tsv"),
            "trackB_split_manifest.json": sha256_file(data_dir / "trackB_split_manifest.json"),
            "trackB_components.json": sha256_file(data_dir / "trackB_components.json"),
        },
        "reports_key": {
            "protocol.json": sha256_file(rep_dir / "protocol.json"),
            "report.html": sha256_file(rep_dir / "report.html"),
            "self_review.json": sha256_file(rep_dir / "self_review.json"),
            "next_batch_proposal.json": sha256_file(rep_dir / "next_batch_proposal.json"),
            "status.json": sha256_file(rep_dir / "status.json"),
            "trackA_selection_manifest.json": sha256_file(rep_dir / "trackA_selection_manifest.json"),
            "trackA_test_results.json": sha256_file(rep_dir / "trackA_test_results.json"),
            "trackA_val_predictions.tsv": sha256_file(rep_dir / "trackA_val_predictions.tsv"),
            "trackA_test_predictions.tsv": sha256_file(rep_dir / "trackA_test_predictions.tsv"),
            "trackB_results.json": sha256_file(rep_dir / "trackB_results.json"),
            "trackB_sequence_predictions.tsv": sha256_file(rep_dir / "trackB_sequence_predictions.tsv"),
            "trackB_reproduction.json": sha256_file(rep_dir / "trackB_reproduction.json"),
            "encoding_audit.json": sha256_file(rep_dir / "encoding_audit.json"),
        },
        "figures": sha256_dir(rep_dir, "*.png", rep_dir),
    }
    (rep_dir / "sha_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
