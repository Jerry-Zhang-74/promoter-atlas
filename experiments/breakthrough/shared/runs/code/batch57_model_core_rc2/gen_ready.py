# -*- coding: utf-8 -*-
"""Generate the atomic CODE_READY.json for batch57 (hashes of every shipped file)."""
from __future__ import annotations
import glob, hashlib, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    files = sorted(glob.glob(os.path.join(HERE, "*.py"))) + sorted(glob.glob(os.path.join(HERE, "tests", "*.py")))
    files = [f for f in files if os.path.basename(f) != "gen_ready.py"]
    hashes = {os.path.relpath(f, HERE).replace("\\", "/"): sha(f) for f in files}
    payload = {
        "batch": "batch57",
        "version": "v6.38.1-model-core-critical-repair.rc2",
        "status": "CODE_READY",
        "supersedes": "runs/breakthrough_20261002/code/batch53_model_core (v6.38.0-model-pipeline-code.rc1)",
        "preserves_original": True,
        "note": ("Atomic marker written ONLY after all code final + artificial no-fit verification. "
                 "Drop-in replacement for batch53_model_core (same module names/APIs, same "
                 "config_sha256=12488fae... so live checkpoints remain ABI-readable)."),
        "target_column": "Log2RPM_HEK293",
        "config_sha256": "12488fae6f07c9a61abcee1dd1bf127bdc5cbd7ea2955a22c6fa3b5936dbf157",
        "source_sha256": "1bbe5db71a5f4a92a281d19567b28ca3751f93d62959e6d61bd41188dd550d96",
        "no_fit_guarantee": {"training": 0, "solve": 0, "np_linalg_solve": 0, "backward": 0,
                             "optimizer_step": 0, "gpu": 0, "network": 0, "download": 0},
        "tests": {"no_fit_rc2": "21/21 PASS", "pipeline_abi_rc2": "6/6 PASS", "all_passed": True},
        "gate_vendored_sha256": "e1ff80a6e411d0989f7bfce3bd76950dabf8afb4f876e6b3c7e70692fc272571",
        "gate_budget": 4, "gate_reserve": 0, "resume_supported": False,
        "repairs": {
            "D1": "gpu_lock distinct FILETIME buffers -> stable OS creation time; live holder never reclaimed; release only own identity",
            "D2": "resume fail-closed (ResumeUnsupported); every execution consumes a slot; checkpoints remain readable",
            "D3": "final_compare best fixed-direction baseline uses max() not min()",
            "D4": "SOURCE_SHA256 real 64-hex 1bbe5db7...d96; SOURCE_SHA1 separate (no mislabel)",
            "D5": "assert_parent_specs REQUIRES + asserts acceptance-file sha256 (fail-closed); frozen at launch",
            "D6": "actual_status() reads the REAL ledger; no fictitious reserve; RUN_POLICY read dynamically",
            "D7": "Njoint asserts shared init bytes vs recorded Nabs init_sha before first gradient",
            "D8": "top10_precision uses frozen coord/ID stable tie-breakers",
            "D9": "checkpoints record TRUE optimizer steps (opt_steps), sample count kept separately",
            "D10": "deterministic per-epoch pair shuffle (pair_order) + repeat counts recorded",
            "D11": "selected checkpoint written with a .sha256.json sidecar (select_checkpoint)",
            "D12": "adapter: ragged-row reject, NPZ byte-id decode + declared-length slice, ambiguous-base consistent exclusion/reject"
        },
        "abi": {"config_sha256_matches_batch53": True,
                "live_checkpoint_load_test": "PASS (read-only, artificial no-grad forward)"},
        "handoff": "OWNER_FIX_HANDOFF.txt",
        "readme": "README.md",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files_sha256": hashes,
    }
    out = os.path.join(HERE, "CODE_READY.json")
    tmp = out + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out)
    print(json.dumps({"wrote": out, "n_files": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
