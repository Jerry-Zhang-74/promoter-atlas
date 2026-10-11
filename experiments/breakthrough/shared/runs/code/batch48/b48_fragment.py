# -*- coding: utf-8 -*-
"""batch48 fragment: explicit raw-FASTA -> frozen PARM HEK293 model score routing.

NEW version of the batch20 raw-FASTA routing. It REUSES, read-only and unmodified, the batch35
PARM forward engine (code/batch35/parm_fasta_predict.py): naked FASTA, official central zero-padding
to L_max=600, no RC, no DNA change, no hidden windows, equal mean of the 5 official HEK293 folds,
SHA-verified external assets. >600 bp (incl. 2000) -> UNSUPPORTED, null score, no truncation.

The routing layer here ONLY changes bookkeeping so that one bad record cannot abort the whole file:
every record is retained with an explicit status. It never invents a score.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from b48_common import (B35_ENGINE, B35_MANIFEST, B48InputError, DEFAULT_ASSET_ROOT, L_MAX,
                        SCORE_LABEL, MODEL_SOURCE_CELL, BIOLOGICAL_STATUS, sha256_file, sha256_text)

SCOPE_SOFTWARE = ("Software input support: naked FASTA, official central zero-padding to L_max=600, "
                  "no RC / no DNA change / no hidden windows / no tiling / no truncation. "
                  "Supported length 1..600 inclusive; >600 (e.g. 2000) -> UNSUPPORTED null score.")
SCOPE_BIOLOGICAL = ("Biological scope: the model is a PARM HEK293 promoter-activity model. Its score is a "
                    "cross-cell ranking descriptor, NOT 293FT activity, NOT a GFP measurement, NOT a "
                    "probability, NOT an E/S/B value and NOT a reliability CI. The caller-selected target "
                    "cell is METADATA ONLY and never selects model weights.")
ELIG_OK = ("Exploratory cross-cell ranking descriptor only: PARM HEK293 model score, NOT 293FT activity / "
           "NOT D28 GFP / NOT a probability. UNVALIDATED for new 293FT families.")
ELIG_UNSUP = ("No numerical score produced: sequence length exceeds the PARM software input limit "
              "(L_max=600). No truncation, windowing or tiling was applied.")
ELIG_INVALID = "No numerical score produced: record failed input validation (see invalid_reason)."


def _load_b35():
    """Import the read-only b35 engine module for its UNCHANGED verify_assets/classify/hash helpers."""
    sys.path.insert(0, str(B35_ENGINE.parent))
    import parm_fasta_predict as P35  # noqa: E402
    return P35


def fragment_route(fasta, out_json, out_tsv, asset_root=None, target_cell="293FT",
                   frozen_manifest=None, engine_override=None):
    from b48_common import write_json, write_tsv
    roottag = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    asset_root = str(asset_root or DEFAULT_ASSET_ROOT)
    manifest = str(frozen_manifest or B35_MANIFEST)
    engine = str(engine_override or B35_ENGINE)

    # ---- global input checks (before any GPU) ----
    if not Path(fasta).exists():
        raise B48InputError("INPUT_FILE_NOT_FOUND: %s" % fasta)
    try:
        recs = __import__("b48_common").read_fasta(fasta)
    except B48InputError as e:
        raise B48InputError("EMPTY_INPUT_NO_RECORDS: %s" % e)
    input_sha = sha256_file(fasta)

    P35 = _load_b35()
    if not Path(asset_root).exists():
        raise B48InputError("ASSET_ROOT_NOT_FOUND: %s" % asset_root)
    if not Path(engine).exists():
        raise B48InputError("ENGINE_NOT_FOUND: %s" % engine)
    man, problems = P35.verify_assets(asset_root, manifest)
    if problems:
        raise B48InputError("ASSET_SHA_VERIFICATION_FAILED: %s" % "; ".join(problems))
    engine_sha = sha256_file(engine)

    # ---- per-record classification (UNCHANGED b35 semantics) ----
    keys = [ (n.split()[0] if n.split() else n) for (n, _s) in recs ]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1

    rows = []
    to_score = []          # (index, engine_key, seq)
    seen = set()
    for i, (name, seq) in enumerate(recs):
        s = seq.strip().upper()
        key = keys[i]
        validity, reason, support, flags = P35.classify(s)
        row = {"record_id": name, "engine_key": key, "length": len(s),
               "input_validity": validity, "invalid_reason": reason,
               "software_length_support": support, "contains_N": flags["contains_N"],
               "duplicate_of": "", "score": None, "eligibility_note": ""}
        if counts.get(key, 0) > 1:
            row["route_status"] = "DUPLICATE_RECORD_ID"
            row["input_validity"] = "INVALID"
            row["invalid_reason"] = "DUPLICATE_RECORD_ID:%d" % counts[key]
            row["duplicate_of"] = key
            row["eligibility_note"] = ELIG_INVALID
        elif validity == "INVALID":
            row["route_status"] = ("EMPTY_SEQUENCE" if reason == "EMPTY_SEQUENCE" else "INVALID_CHARS")
            row["eligibility_note"] = ELIG_INVALID
        elif support == "UNSUPPORTED":
            row["route_status"] = "UNSUPPORTED_LENGTH"
            row["eligibility_note"] = ELIG_UNSUP
        else:
            row["route_status"] = "SCORED"
            row["eligibility_note"] = ELIG_OK
            to_score.append((i, key, s))
        rows.append(row)

    # ---- ONE true-CUDA invocation for the valid+supported+unique records ----
    runtime = {"engine_invoked": False}
    engine_payload = None
    if to_score:
        tmp = tempfile.mkdtemp(prefix="b47_frag_")
        tf = os.path.join(tmp, "valid_unique.fasta")
        oj = os.path.join(tmp, "engine_out.json")
        ot = os.path.join(tmp, "engine_out.tsv")
        with open(tf, "w", encoding="utf-8") as fh:
            for (_i, key, s) in to_score:
                fh.write(">" + key + "\n" + s + "\n")
        cmd = [sys.executable, "-X", "utf8", engine, "--input", tf, "--out-json", oj, "--out-tsv", ot,
               "--asset-root", asset_root, "--frozen-asset-manifest", manifest, "--target-cell", target_cell]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        runtime["engine_invoked"] = True
        runtime["engine_returncode"] = p.returncode
        runtime["engine_elapsed_child"] = None
        if Path(oj).exists():
            import json as _json
            engine_payload = _json.loads(Path(oj).read_text(encoding="utf-8"))
        if p.returncode != 0:
            # CUDA-unavailable / device-assertion / asset failure inside the engine is FATAL: no CPU fallback.
            detail = (engine_payload or {}).get("error") or p.stderr[-800:]
            try:
                os.remove(tf); os.remove(oj); os.remove(ot); os.rmdir(tmp)
            except OSError:
                pass
            raise B48InputError("BACKEND_FORWARD_FAILED: returncode=%d error=%s" % (p.returncode, detail))
        rt = (engine_payload or {}).get("runtime", {})
        runtime.update({k: rt.get(k) for k in ("pid", "pid_creation_winfiletime", "device",
                        "model_params_on_cuda", "model_param_device", "input_tensor_on_cuda",
                        "official_forward_wall_s", "elapsed_s", "n_inferred", "parm_version",
                        "nvidia_smi_start", "nvidia_smi_during", "nvidia_smi_after")})
        score_by_key = {r["id"]: r.get("score") for r in (engine_payload or {}).get("records", [])}
        for (i, key, s) in to_score:
            rows[i]["score"] = score_by_key.get(key)
        try:
            os.remove(tf); os.remove(oj); os.remove(ot); os.rmdir(tmp)
        except OSError:
            pass

    weights_sha = man.get("weights_sha256", {})
    n_scored = sum(1 for r in rows if r["route_status"] == "SCORED" and r["score"] is not None)
    n_unsup = sum(1 for r in rows if r["route_status"] == "UNSUPPORTED_LENGTH")
    n_invalid = sum(1 for r in rows if r["route_status"] in ("INVALID_CHARS", "EMPTY_SEQUENCE"))
    n_dup = sum(1 for r in rows if r["route_status"] == "DUPLICATE_RECORD_ID")

    out_records = []
    for r in rows:
        out_records.append({
            "record_id": r["record_id"], "length": r["length"], "route_status": r["route_status"],
            "input_validity": r["input_validity"], "invalid_reason": r["invalid_reason"],
            "software_length_support": r["software_length_support"], "contains_N": r["contains_N"],
            "duplicate_of": r["duplicate_of"],
            "score_label": SCORE_LABEL,
            "PARM_HEK293_model_score": r["score"],
            "model_source_cell": MODEL_SOURCE_CELL,
            "target_requested_cell": target_cell,
            "target_cell_selects_weights": False,
            "biological_validation_status": BIOLOGICAL_STATUS,
            "eligibility_note": r["eligibility_note"],
            "input_sha256": input_sha,
        })

    warnings = [
        {"level": "warn", "text": "Score is a PARM HEK293 cross-cell ranking descriptor; NOT 293FT "
                                  "activity, NOT GFP, NOT a probability, NOT E/S/B, NO reliability CI."},
        {"level": "warn", "text": "The caller target cell (%s) is METADATA only and does NOT select model "
                                  "weights (model source cell is fixed HEK293)." % target_cell},
    ]
    if n_dup:
        warnings.append({"level": "warn", "text": "%d record(s) dropped as duplicate IDs (b35 policy: duplicate "
                         "record IDs are invalid input); retained with status, no score invented." % n_dup})
    if any(r["contains_N"] for r in out_records):
        warnings.append({"level": "warn", "text": "N bases encoded via official all-zero one-hot (allowed, flagged)."})
    result = {
        "status": "OK", "batch": __import__("b48_common").BATCH,
        "version": __import__("b48_common").VERSION, "command": "fragment",
        "task": "fragment_activity_ranking", "generated_at": roottag,
        "score_label": SCORE_LABEL,
        "scope": {"software_support": SCOPE_SOFTWARE, "biological_scope": SCOPE_BIOLOGICAL,
                  "model_cell": MODEL_SOURCE_CELL, "target_requested_cell": target_cell,
                  "validated_for_target_cell": False},
        "backend": {
            "name": "PARM", "engine_script": engine, "engine_sha256": engine_sha,
            "parm_version": runtime.get("parm_version"),
            "source_repo": "https://github.com/vansteensellab/PARM",
            "commit": "260a27082942060364c1c4e2393018a0fa128844",
            "folds": 5, "aggregation": "equal mean of 5 official HEK293 folds",
            "model_source_cell": MODEL_SOURCE_CELL,
            "asset_root": os.path.abspath(asset_root),
            "frozen_asset_manifest": os.path.abspath(manifest),
            "weights_sha256": weights_sha},
        "input": {"path": os.path.abspath(fasta), "sha256": input_sha, "n_records": len(recs)},
        "routing_summary": {"n_records": len(recs), "n_scored": n_scored,
                            "n_unsupported_length": n_unsup, "n_invalid": n_invalid,
                            "n_duplicate_id": n_dup},
        "model_disagreement_5fold_spread": {"value": None, "reason": "NOT_CARRIED_BY_B35_ENGINE"},
        "runtime": runtime,
        "records": out_records, "warnings": warnings,
        "definitions": {
            "PARM_HEK293_model_score": "frozen PARM HEK293 promoter-activity model score, equal mean of 5 "
                                       "official HEK293 folds, official central zero-padding L_max=600",
            "route_status": "SCORED / UNSUPPORTED_LENGTH / INVALID_CHARS / EMPTY_SEQUENCE / DUPLICATE_RECORD_ID",
        },
    }
    write_json(out_json, result)
    cols = ["record_id", "length", "route_status", "input_validity", "invalid_reason",
            "software_length_support", "contains_N", "duplicate_of", "score_label",
            "PARM_HEK293_model_score", "model_source_cell", "target_requested_cell",
            "biological_validation_status", "eligibility_note", "input_sha256"]
    write_tsv(out_tsv, out_records, cols)
    return result
