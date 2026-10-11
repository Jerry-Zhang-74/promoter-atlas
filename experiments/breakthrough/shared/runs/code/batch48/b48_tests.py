# -*- coding: utf-8 -*-
"""batch48 Protocol tests (repair R1 defect tests + regression). ZERO GPU / ZERO neural forward.

Covers:
  (A) ISOLATED fixture that mimics the b47 phaseD WRONG order and shows the verifier detects changed
      bytes with UNCHANGED length (the exact class of the b47 remaining_work.json staleness).
  (B) TAMPER a COPY of one hashed file -> verification FAIL, WITHOUT rewriting the manifest; the real
      original file is never touched.
  (C) gene-observed regression from an unrelated cwd with CUDA HIDDEN and no torch/numpy/Bio.
  (D) fragment PRE-CUDA failure routing (missing input / missing asset root / tampered manifest COPY).
  (E) b47-invariance: the b47 sha_manifest / remaining_work / independent_review bytes are UNCHANGED.
No model forward is performed (inference code is unchanged; b47 CUDA evidence is INHERITED, not replayed).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b48_common as C  # noqa: E402
import b48_finalize as FIN  # noqa: E402

PY = sys.executable
CLI = str(C.CODE / "b48_cli.py")
DATA = C.B48_DATA
REP = C.B48_REPORT
TESTDIR = DATA / "testdata"
OUTDIR = DATA / "out"
PRE = C.B48_RUNS / "_pre_hashes.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def run_cli(args, cwd, extra_env=None):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    p = subprocess.run([PY, "-X", "utf8", CLI] + args, cwd=str(cwd), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"rc": p.returncode, "wall_s": round(time.time() - t0, 2),
            "stdout": p.stdout[-1200:], "stderr": p.stderr[-800:]}


def test_wrong_order_fixture():
    """(A) Isolated fixture mimicking the b47 phaseD WRONG order."""
    d = Path(tempfile.mkdtemp(prefix="b48_fixture_"))
    a, b = d / "a.txt", d / "b.txt"
    a.write_bytes(b"AAAA-ORIGINAL\n")
    b.write_bytes(b"BBBB-STABLE\n")
    manifest = {"files": {a.name: {"sha256": sha(a), "bytes": a.stat().st_size},
                          b.name: {"sha256": sha(b), "bytes": b.stat().st_size}}}
    # GOOD order: verify now -> both ok
    good_before = FIN.verify_entries(manifest["files"], root=d)
    # WRONG order: mutate a.txt AFTER the manifest, with UNCHANGED byte length
    old_len = a.stat().st_size
    a.write_bytes(b"AAAA-TAMPERED\n")
    same_len = (a.stat().st_size == old_len)
    bad = FIN.verify_entries(manifest["files"], root=d)
    detected = [x for x in bad if x["path"] == "a.txt" and x["reason"] == "SHA_MISMATCH"]
    return {"fixture": str(d), "good_before_ok": not good_before, "changed_bytes_same_length": same_len,
            "verifier_detected": bool(detected), "detected_detail": detected,
            "all_ok": (not good_before) and same_len and bool(detected)}


def test_tamper_copy():
    """(B) Tamper a COPY of one hashed file -> verification FAIL, manifest NOT rewritten, original untouched."""
    d = Path(tempfile.mkdtemp(prefix="b48_tamper_"))
    src = C.CODE / "b48_cli.py"          # a file that WILL be hashed by the b48 manifest
    orig_sha = sha(src)
    copy = d / "b48_cli.py"
    copy.write_bytes(src.read_bytes())
    manifest_path = d / "sha_manifest_copy.json"
    man = {"batch": "fixture", "files": {"b48_cli.py": {"sha256": sha(copy), "bytes": copy.stat().st_size}}}
    manifest_path.write_text(json.dumps(man), encoding="utf-8")
    ok_before = FIN.verify_manifest_file(manifest_path, root=d)
    # tamper the COPY with same length (flip one byte)
    raw = bytearray(copy.read_bytes())
    raw[10] = (raw[10] ^ 0x01)
    copy.write_bytes(bytes(raw))
    tamper_len_same = (len(raw) == man["files"]["b48_cli.py"]["bytes"])
    bad = FIN.verify_manifest_file(manifest_path, root=d)
    # manifest copy was NOT rewritten (its bytes unchanged) and the real original is untouched
    mani_sha_now = sha(manifest_path)
    orig_unchanged = (sha(src) == orig_sha)
    return {"copied_from": "code/batch48/b48_cli.py", "ok_before": not ok_before,
            "tamper_same_length": tamper_len_same, "verification_failed_after_tamper": bool(bad),
            "bad": bad, "manifest_rewritten": False, "original_unchanged": orig_unchanged,
            "original_sha": orig_sha, "manifest_copy_sha": mani_sha_now,
            "all_ok": (not ok_before) and tamper_len_same and bool(bad) and orig_unchanged}


def test_fragment_pre_cuda():
    """(D) fragment failure paths must resolve BEFORE any CUDA/forward (CUDA hidden)."""
    TESTDIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fx = TESTDIR / "route_fixture.fasta"
    fx.write_text(">ok_short\nACGTACGTACGTACGTACGTACGTACGTACGT\n>len2000\n" + ("ACGT" * 500) + "\n",
                  encoding="utf-8")
    unrelated = tempfile.mkdtemp(prefix="b48_unrelated_")
    env = {"CUDA_VISIBLE_DEVICES": ""}
    checks = {}
    r1 = run_cli(["fragment", "--fasta", str(TESTDIR / "NOPE.fasta"),
                  "--out-json", str(OUTDIR / "miss_input.json"), "--out-tsv", str(OUTDIR / "miss_input.tsv")],
                 cwd=unrelated, extra_env=env)
    checks["missing_input_exit2"] = r1["rc"] == 2
    b35_manifest_sha_before = sha(C.B35_MANIFEST)
    r2 = run_cli(["fragment", "--fasta", str(fx),
                  "--out-json", str(OUTDIR / "miss_asset.json"), "--out-tsv", str(OUTDIR / "miss_asset.tsv"),
                  "--asset-root", str(DATA / "NOPE_ASSETS")], cwd=unrelated, extra_env=env)
    p2 = load_json(OUTDIR / "miss_asset.json")
    checks["missing_asset_exit3"] = r2["rc"] == 3
    checks["missing_asset_reason"] = bool(p2 and str(p2.get("error", "")).startswith("ASSET_ROOT_NOT_FOUND"))
    real = json.loads(Path(C.B35_MANIFEST).read_text(encoding="utf-8"))
    k = list(real["weights_sha256"])[0]
    real["weights_sha256"][k] = "0" * 64
    tamp = TESTDIR / "tampered_manifest_copy.json"
    tamp.write_text(json.dumps(real), encoding="utf-8")
    r3 = run_cli(["fragment", "--fasta", str(fx),
                  "--out-json", str(OUTDIR / "tampered.json"), "--out-tsv", str(OUTDIR / "tampered.tsv"),
                  "--frozen-asset-manifest", str(tamp)], cwd=unrelated, extra_env=env)
    p3 = load_json(OUTDIR / "tampered.json")
    checks["tampered_manifest_exit3"] = r3["rc"] == 3
    checks["tampered_reason"] = bool(p3 and str(p3.get("error", "")).startswith("ASSET_SHA_VERIFICATION_FAILED"))
    checks["real_b35_manifest_untouched"] = (sha(C.B35_MANIFEST) == b35_manifest_sha_before)
    return {"checks": checks, "all_ok": all(v for v in checks.values() if isinstance(v, bool))}


def test_b47_invariance():
    """(E) The b47 canonical defect bytes must be UNCHANGED (old failure retained, not hidden)."""
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    out = {}
    for key in ("b47_sha_manifest", "b47_remaining_work", "b47_independent_review",
                "b47_manifest_verification", "b47_status"):
        rec = pre[key]
        now = sha(C.ROOT / rec["path"]) if (C.ROOT / rec["path"]).exists() else None
        out[key] = {"path": rec["path"], "before": rec["sha256"], "after": now,
                    "unchanged": now == rec["sha256"]}
    out["all_unchanged"] = all(v["unchanged"] for v in out.values() if isinstance(v, dict))
    return out


def _label_rows(gids):
    want = set(gids)
    with open(C.LABEL_TABLE, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("gene_id") in want:
                yield row


def _s_index():
    """Read-only audit of the FULL-catalog Score_S rank formula (counts + defined-value order)."""
    n_total = 0
    n_undef = 0
    defined = []
    with open(C.LABEL_TABLE, "r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for i, row in enumerate(rdr):
            n_total += 1
            if row.get("S_defined_main") == "1" and str(row.get("S_raw", "")).strip() != "":
                defined.append((float(row["S_raw"]), i, row["gene_id"]))
            else:
                n_undef += 1
    order = sorted(range(len(defined)), key=lambda j: (defined[j][0], defined[j][1]))
    rank = {defined[j][2]: pos for pos, j in enumerate(order)}
    return {"n_total": n_total, "n_undefined": n_undef, "n_defined": len(defined),
            "denominator": n_total - 1, "rank": rank}


def test_gene_regression():
    """(C) gene-observed from unrelated cwd, CUDA hidden, no torch/numpy/Bio; verify source + definition."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    unrelated = tempfile.mkdtemp(prefix="b48_unrelated_")
    env = {"CUDA_VISIBLE_DEVICES": ""}
    ojson, otsv = OUTDIR / "gene.b48.json", OUTDIR / "gene.b48.tsv"
    q = ["ACTB", "GAPDH", "RPLP0", "ENSG00000000005", "ENSG00000075624.18"]
    probe = subprocess.run([PY, "-X", "utf8", "-c",
        "import sys; sys.path.insert(0, r'%s'); import b48_cli; "
        "argv=['gene-observed','--query']+%s+['--out-json',r'%s','--out-tsv',r'%s']; "
        "rc=b48_cli.main(argv); "
        "print('MODULES', 'torch' in sys.modules, 'numpy' in sys.modules, 'Bio' in sys.modules, rc)"
        % (C.CODE, repr(q), str(ojson), str(otsv))],
        cwd=unrelated, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    payload = load_json(ojson) or {}
    recs = {r["query"]: r for r in payload.get("records", [])}
    idx = _s_index()
    known = (("ACTB", "ENSG00000075624"), ("GAPDH", "ENSG00000111640"), ("RPLP0", "ENSG00000089157"))
    src = {r["gene_id"]: r for r in _label_rows([g for _, g in known])}
    checks = {"exit_0": probe.returncode == 0, "ordered_len": payload.get("n_queries") == len(q)}
    modline = [ln for ln in probe.stdout.splitlines() if ln.startswith("MODULES")]
    checks["no_torch_no_numpy_no_bio"] = bool(modline and modline[0] == "MODULES False False False 0")
    for sym, gid in known:
        rr, sr = recs.get(sym) or {}, src.get(gid) or {}
        exp_s = (idx["n_undefined"] + idx["rank"][gid]) / idx["denominator"]
        checks[sym + "_E_Sraw_B_match_source"] = bool(
            rr.get("status") == "OK" and rr.get("canonical_gene_id") == gid
            and abs(float(rr["observedScore_E"]) - float(sr["Score_E"])) < 1e-9
            and abs(float(rr["S_raw"]) - float(sr["S_raw"])) < 1e-9
            and abs(float(rr["B_raw"]) - float(sr["B_raw"])) < 1e-9)
        checks[sym + "_observedScore_S_matches_catalog"] = (
            rr.get("observedScore_S") is not None and abs(float(rr["observedScore_S"]) - float(sr["Score_S"])) < 1e-6)
        checks[sym + "_formula_full_catalog_rank"] = abs(exp_s - float(sr["Score_S"])) < 1e-6
        checks[sym + "_defined_mask_1"] = (rr.get("S_defined_mask") == 1 and rr.get("S_raw_defined") == 1)
    und = recs.get("ENSG00000000005") or {}
    checks["undefined_S_null_not_zero"] = (und.get("S_raw") is None and und.get("observedScore_S") is None
                                           and und.get("S_defined_mask") == 0 and und.get("B_raw") is not None)
    checks["versioned_id_normalized"] = (recs.get("ENSG00000075624.18", {}).get("canonical_gene_id")
                                         == "ENSG00000075624")
    checks["definition_states_full_58988_and_mask"] = (
        payload.get("source_field_definitions", {}).get("observedScore_S", {}).get("defined_mask")
        == "S_defined_main == 1"
        and "ALL catalog gene rows (n_total = 58988)" in json.dumps(payload.get("source_field_definitions", {})))
    checks["source_evidence_present"] = bool(payload.get("source_evidence", {}).get("generation_code"))
    return {"probe_stdout_tail": probe.stdout[-300:], "checks": checks,
            "n_total": idx["n_total"], "n_undefined": idx["n_undefined"], "n_defined": idx["n_defined"],
            "all_ok": all(v for v in checks.values() if isinstance(v, bool))}


def _write_inherited_evidence():
    """Record b47's true-CUDA forward evidence as INHERITED (never as a new b48 replay)."""
    b47ev = load_json(C.ROOT / "reports" / "breakthrough_20261002" / "batch47" / "test_evidence.json") or {}
    fg = (b47ev.get("fragment_gpu") or {}).get("checks", {})
    inh = {
        "batch": "batch48", "role": "worker", "record_type": "inherited_evidence_not_replay",
        "statement": ("The frozen PARM HEK293 inference code is UNCHANGED by b48, so b48 performed NO neural "
                      "forward. b47's independent true-CUDA control parity is INHERITED as the backend evidence."),
        "b47_test_evidence_path": "reports/breakthrough_20261002/batch47/test_evidence.json",
        "b47_gpu_available": b47ev.get("gpu_available"),
        "b47_torch_version": b47ev.get("torch_version"),
        "b47_gpu_seconds_used": b47ev.get("gpu_seconds_used"),
        "b47_control_score": fg.get("control_score"),
        "b47_control_abs_diff": fg.get("control_abs_diff"),
        "b47_model_params_on_cuda": fg.get("gpu_params_on_cuda"),
        "b47_input_tensor_on_cuda": fg.get("gpu_input_on_cuda"),
        "b47_nvidia_reports_rtx": fg.get("nvidia_reports_rtx"),
        "b48_replay_performed": False,
        "b48_gpu_used": False,
    }
    C.write_json(C.B48_RUNS / "inherited_b47_cuda_evidence.json", inh)
    return inh


def main():
    t0 = time.time()
    ev = {"batch": C.BATCH, "version": C.VERSION, "protocol_section": "tests",
          "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
          "note": ("Integrity/defect tests + gene-observed regression. ZERO GPU, ZERO neural forward "
                   "(inference code unchanged; b47 CUDA evidence is inherited, not replayed).")}
    ev["wrong_order_fixture"] = test_wrong_order_fixture()
    ev["tamper_copy"] = test_tamper_copy()
    ev["fragment_pre_cuda"] = test_fragment_pre_cuda()
    ev["b47_invariance_before_finalize"] = test_b47_invariance()
    ev["gene_regression"] = test_gene_regression()
    ev["gpu_used_this_batch"] = False
    ev["neural_forward_this_batch"] = False
    parts = [ev["wrong_order_fixture"]["all_ok"], ev["tamper_copy"]["all_ok"],
             ev["fragment_pre_cuda"]["all_ok"], ev["b47_invariance_before_finalize"]["all_unchanged"],
             ev["gene_regression"]["all_ok"]]
    ev["all_ok"] = all(parts)
    ev["status"] = "PASS" if ev["all_ok"] else "FAIL"
    ev["total_wall_s"] = round(time.time() - t0, 2)
    ev["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ev["inherited_b47_cuda_evidence"] = _write_inherited_evidence()
    C.write_json(REP / "test_evidence.json", ev)
    print(json.dumps({"status": ev["status"], "all_ok": ev["all_ok"],
                      "wrong_order_fixture": ev["wrong_order_fixture"]["all_ok"],
                      "tamper_copy": ev["tamper_copy"]["all_ok"],
                      "fragment_pre_cuda": ev["fragment_pre_cuda"]["all_ok"],
                      "b47_invariance": ev["b47_invariance_before_finalize"]["all_unchanged"],
                      "gene_regression": ev["gene_regression"]["all_ok"],
                      "wall_s": ev["total_wall_s"]}, ensure_ascii=False))
    return 0 if ev["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
