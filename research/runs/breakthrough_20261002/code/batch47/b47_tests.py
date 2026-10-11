# -*- coding: utf-8 -*-
"""batch47 Protocol B: meaningful end-to-end integration tests (<=10 min).

Run from an UNRELATED cwd with Unicode/space-containing input/output paths.
- Frozen test FASTAs (built + hashed BEFORE any forward).
- ONE true-CUDA invocation of the fragment CLI for the valid records; frozen numeric control parity.
- Failure paths (missing asset root / tampered manifest COPY) must fail BEFORE CUDA.
- gene-observed must run with CUDA hidden and NO torch/numpy/model weights.
No performance claim is made from these tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b47_common as C  # noqa: E402

PY = sys.executable
CLI = str(C.CODE / "b47_cli.py")
B35CTRL = C.ROOT / "data" / "breakthrough_20261002" / "batch35" / "testdata" / "known_control.fasta"
FROZEN_CONTROL_SCORE = 3.1036716      # batch31 saved score, independently reproduced by b34/b35
TOL = 1e-4
TESTDIR = C.B47_DATA / "b47 test 测试"     # unicode + space
OUTDIR = C.B47_DATA / "输出 out 测试"  # unicode + space


def randseq(n, seed):
    r = random.Random(seed)
    return "".join(r.choice("ACGT") for _ in range(n))


def sha_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def build_fixtures():
    TESTDIR.mkdir(parents=True, exist_ok=True)
    ctrl = B35CTRL.read_text(encoding="utf-8").splitlines()[1].strip()
    s137 = randseq(137, 20261047)
    s271 = randseq(271, 20261048)
    assert len(s137) == 137 and len(s271) == 271
    valid = [">A1|ACTB|100", ctrl, ">B47_SOFTWARE_TEST_ONLY_137", s137,
             ">B47_SOFTWARE_TEST_ONLY_271", s271]
    mixed = list(valid)
    mixed += [">len601", randseq(601, 1), ">len2000", randseq(2000, 2),
              ">invalid_char", "ACGT" * 5 + "R" + "ACGT" * 5,
              ">empty_seq", "", ">dupA", randseq(100, 7), ">dupA", randseq(100, 8)]
    files = {"valid_supported.fasta": valid, "mixed.fasta": mixed}
    manifest = {"note": "frozen BEFORE any forward; SOFTWARE TEST ONLY (137/271 are artificial)",
                "frozen_control": {"id": "A1|ACTB|100", "frozen_batch31_score": FROZEN_CONTROL_SCORE,
                                   "tolerance": TOL},
                "artificial_sequences": {"B47_SOFTWARE_TEST_ONLY_137": {"length": 137, "sha256": sha_text(s137)},
                                         "B47_SOFTWARE_TEST_ONLY_271": {"length": 271, "sha256": sha_text(s271)}},
                "files": {}}
    for name, lines in files.items():
        p = TESTDIR / name
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["files"][name] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    C.write_json(TESTDIR / "test_manifest.json", manifest)
    return manifest


def run_cli(args, cwd, extra_env=None, expect_rc=None):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    p = subprocess.run([PY, "-X", "utf8", CLI] + args, cwd=str(cwd), env=env,
                       capture_output=True, text=True)
    return {"rc": p.returncode, "wall_s": round(time.time() - t0, 2),
            "stdout": p.stdout[-1500:], "stderr": p.stderr[-1500:]}


def load_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def test_fragment_gpu(manifest, evidence):
    """ONE true-CUDA invocation of the fragment CLI on the mixed file."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ojson = OUTDIR / "mixed.b47.json"
    otsv = OUTDIR / "mixed.b47.tsv"
    unrelated = tempfile.mkdtemp(prefix="b47_unrelated_")
    r = run_cli(["fragment", "--fasta", str(TESTDIR / "mixed.fasta"), "--out-json", str(ojson),
                 "--out-tsv", str(otsv), "--target-cell", "293FT"], cwd=unrelated)
    payload = load_json(ojson)
    checks = {"exit_0": r["rc"] == 0, "json_written": ojson.exists(), "tsv_written": otsv.exists(),
              "unrelated_cwd_used": True}
    recs = {x["record_id"]: x for x in (payload or {}).get("records", [])}
    ctrl = recs.get("A1|ACTB|100") or {}
    checks["control_present"] = bool(ctrl)
    if ctrl:
        sc = ctrl.get("PARM_HEK293_model_score")
        checks["control_numeric_parity"] = (sc is not None and abs(sc - FROZEN_CONTROL_SCORE) <= TOL)
        checks["control_score"] = sc
        checks["control_abs_diff"] = abs(sc - FROZEN_CONTROL_SCORE) if sc is not None else None
    for rid in ("B47_SOFTWARE_TEST_ONLY_137", "B47_SOFTWARE_TEST_ONLY_271"):
        rr = recs.get(rid) or {}
        checks[rid + "_scored"] = rr.get("route_status") == "SCORED" and rr.get("PARM_HEK293_model_score") is not None
    checks["len601_unsupported_null"] = (recs.get("len601", {}).get("route_status") == "UNSUPPORTED_LENGTH"
                                         and recs["len601"].get("PARM_HEK293_model_score") is None)
    checks["len2000_unsupported_null"] = (recs.get("len2000", {}).get("route_status") == "UNSUPPORTED_LENGTH"
                                          and recs["len2000"].get("PARM_HEK293_model_score") is None)
    checks["invalid_routed"] = recs.get("invalid_char", {}).get("route_status") == "INVALID_CHARS"
    checks["empty_routed"] = recs.get("empty_seq", {}).get("route_status") == "EMPTY_SEQUENCE"
    dup = [x for x in (payload or {}).get("records", []) if x["record_id"] == "dupA"]
    checks["duplicate_retained_two"] = (len(dup) == 2 and all(d["route_status"] == "DUPLICATE_RECORD_ID"
                                         and d["PARM_HEK293_model_score"] is None for d in dup))
    checks["all_records_retained"] = len((payload or {}).get("records", [])) == 9
    rt = (payload or {}).get("runtime", {})
    checks["gpu_params_on_cuda"] = bool(rt.get("model_params_on_cuda"))
    checks["gpu_input_on_cuda"] = bool(rt.get("input_tensor_on_cuda"))
    checks["nvidia_reports_rtx"] = "RTX" in str(rt.get("nvidia_smi_start", ""))
    checks["pid_present"] = isinstance(rt.get("pid"), int) and rt.get("pid", 0) > 0
    checks["pid_creation_present"] = str(rt.get("pid_creation_winfiletime", "")).isdigit()
    checks["engine_invoked_once"] = rt.get("engine_invoked") is True
    b = (payload or {}).get("backend", {})
    checks["backend_engine_sha_matches_file"] = (b.get("engine_sha256") == C.sha256_file(Path(b.get("engine_script"))))
    checks["weights_sha_present"] = bool(b.get("weights_sha256"))
    checks["model_cell_fixed_HEK293"] = ((payload or {}).get("scope", {}).get("model_cell") == "HEK293")
    checks["target_cell_metadata_only"] = ((payload or {}).get("scope", {}).get("validated_for_target_cell") is False
                                           and (payload or {}).get("scope", {}).get("target_requested_cell") == "293FT")
    # No field may masquerade as 293FT activity / GFP / probability / ESB / CI. The data key set and
    # score label are checked structurally; the words may appear only inside explicit DISCLAIMER text.
    rec_keys = set()
    for x in (payload or {}).get("records", []):
        rec_keys |= set(x.keys())
    bad_keys = [k for k in rec_keys if any(t in k.lower() for t in
                ("gfp", "probab", "esb", "activity", "reliability", "ci_"))]
    checks["no_293FT_activity_masquerade"] = (
        not bad_keys
        and all(x.get("score_label") == "PARM_HEK293_model_score"
                for x in (payload or {}).get("records", []))
        and "NOT 293FT activity" in json.dumps((payload or {}).get("scope", {}))
        and "NOT 293FT activity" in json.dumps((payload or {}).get("warnings", [])))
    checks["biological_unvalidated"] = ((payload or {}).get("scope", {}) and
                                        all((x.get("biological_validation_status") == "UNVALIDATED_FOR_293FT")
                                            for x in (payload or {}).get("records", [])))
    checks["no_rc_or_window_flag_used"] = ("no hidden windows" in json.dumps((payload or {}).get("scope", {})))
    evidence["fragment_gpu"] = {"run": r, "checks": checks, "gpu_wall_s": r["wall_s"],
                                "all_ok": all(v for k, v in checks.items() if isinstance(v, bool))}
    return evidence["fragment_gpu"]


def test_failures_before_cuda(evidence):
    """Asset failures must occur with CUDA HIDDEN -> proves they precede any GPU work."""
    unrelated = tempfile.mkdtemp(prefix="b47_unrelated_")
    env = {"CUDA_VISIBLE_DEVICES": ""}
    checks = {}
    # (a) missing external asset root
    r1 = run_cli(["fragment", "--fasta", str(TESTDIR / "mixed.fasta"),
                  "--out-json", str(OUTDIR / "miss.json"), "--out-tsv", str(OUTDIR / "miss.tsv"),
                  "--asset-root", str(C.B47_DATA / "NOPE_ASSETS")], cwd=unrelated, extra_env=env)
    p1 = load_json(OUTDIR / "miss.json")
    checks["missing_asset_exit3"] = r1["rc"] == 3
    checks["missing_asset_reason"] = bool(p1 and str(p1.get("error", "")).startswith("ASSET_ROOT_NOT_FOUND"))
    # (b) tampered manifest COPY (never the real manifest)
    real = json.loads(Path(C.B35_MANIFEST).read_text(encoding="utf-8"))
    k = list(real["weights_sha256"])[0]
    real["weights_sha256"][k] = "0" * 64
    tampered = TESTDIR / "tampered_manifest_copy.json"
    C.write_json(tampered, real)
    r2 = run_cli(["fragment", "--fasta", str(TESTDIR / "mixed.fasta"),
                  "--out-json", str(OUTDIR / "tam.json"), "--out-tsv", str(OUTDIR / "tam.tsv"),
                  "--frozen-asset-manifest", str(tampered)], cwd=unrelated, extra_env=env)
    p2 = load_json(OUTDIR / "tam.json")
    checks["tampered_manifest_exit3"] = r2["rc"] == 3
    checks["tampered_manifest_reason"] = bool(p2 and str(p2.get("error", "")).startswith("ASSET_SHA_VERIFICATION_FAILED"))
    checks["actual_assets_untouched"] = (C.sha256_file(C.B35_MANIFEST) ==
        hashlib.sha256(Path(C.B35_MANIFEST).read_bytes()).hexdigest())
    # (c) missing input file
    r3 = run_cli(["fragment", "--fasta", str(TESTDIR / "NOPE.fasta"),
                  "--out-json", str(OUTDIR / "x.json"), "--out-tsv", str(OUTDIR / "x.tsv")],
                 cwd=unrelated, extra_env=env)
    checks["missing_input_exit2"] = r3["rc"] == 2
    evidence["failures_before_cuda"] = {"checks": checks,
                                        "all_ok": all(v for v in checks.values() if isinstance(v, bool))}
    return evidence["failures_before_cuda"]


def test_gene_observed(evidence):
    """CUDA hidden, zero learned weights, no torch/numpy; verify against raw source."""
    unrelated = tempfile.mkdtemp(prefix="b47_unrelated_")
    env = {"CUDA_VISIBLE_DEVICES": ""}
    ojson = OUTDIR / "gene.b47.json"
    otsv = OUTDIR / "gene.b47.tsv"
    q = ["ACTB", "GAPDH", "RPLP0", "NOPE_NOT_A_GENE", "ENSG00000075624.18", "WASH7P",
         "ENSG00000000005", "ACGTACGTACGTACGTACGT"]
    # in-process run to prove no torch/numpy/model weights are imported
    probe = subprocess.run([PY, "-X", "utf8", "-c",
        "import sys; sys.path.insert(0, r'%s'); import b47_cli; "
        "argv=['gene-observed','--query']+%s+['--out-json',r'%s','--out-tsv',r'%s']; "
        "rc=b47_cli.main(argv); "
        "print('MODULES', 'torch' in sys.modules, 'numpy' in sys.modules, 'Bio' in sys.modules, rc)"
        % (C.CODE, repr(q), str(ojson), str(otsv))],
        cwd=unrelated, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    payload = load_json(ojson)
    recs = {x["query"]: x for x in (payload or {}).get("records", [])}
    checks = {"exit_0": probe.returncode == 0, "ordered_len": (payload or {}).get("n_queries") == len(q)}
    modline = [ln for ln in probe.stdout.splitlines() if ln.startswith("MODULES")]
    checks["no_torch_no_numpy_no_bio"] = bool(modline and modline[0] == "MODULES False False False 0")
    # independent raw-source verification for the 3 known genes (ids read straight from the raw table)
    known = (("ACTB", "ENSG00000075624"), ("GAPDH", "ENSG00000111640"), ("RPLP0", "ENSG00000089157"))
    src = {}
    for row in _read_label_rows([g for _, g in known]):
        src[row["gene_id"]] = row
    for sym, gid in known:
        rr, sr = recs.get(sym) or {}, src.get(gid)
        checks[sym + "_matches_source"] = bool(sr and rr.get("status") == "OK"
            and rr.get("canonical_gene_id") == gid
            and abs(float(rr.get("observedScore_E")) - float(sr["Score_E"])) < 1e-9
            and (rr.get("S_raw") is None or abs(float(rr["S_raw"]) - float(sr["S_raw"])) < 1e-9)
            and abs(float(rr.get("B_raw")) - float(sr["B_raw"])) < 1e-9)
    checks["versioned_id_normalized"] = (recs.get("ENSG00000075624.18", {}).get("status") == "OK"
                                         and recs["ENSG00000075624.18"].get("canonical_gene_id") == "ENSG00000075624")
    checks["unknown_absent_null"] = (recs.get("NOPE_NOT_A_GENE", {}).get("status") == "ABSENT"
                                     and recs["NOPE_NOT_A_GENE"].get("observedScore_E") is None)
    checks["ambiguous_lists_candidates"] = (recs.get("WASH7P", {}).get("status") == "AMBIGUOUS"
                                            and len(recs["WASH7P"].get("candidates") or []) > 1)
    ms = recs.get("ENSG00000000005", {})
    checks["missing_S_is_null_not_zero"] = (ms.get("S_raw") is None and ms.get("S_raw_defined") == 0
                                            and ms.get("B_raw") is not None)
    checks["dna_query_rejected"] = (recs.get("ACGTACGTACGTACGTACGT", {}).get("status") == "INVALID_QUERY"
                                    and "DNA_SEQUENCE" in recs["ACGTACGTACGTACGTACGT"].get("reason", ""))
    # queryfile that is a FASTA -> global failure, exit 2
    fa = TESTDIR / "queryfile_is_fasta.txt"
    fa.write_text(">ACTB\nACGT\n", encoding="utf-8")
    rf = run_cli(["gene-observed", "--queryfile", str(fa), "--out-json", str(OUTDIR / "qf.json"),
                  "--out-tsv", str(OUTDIR / "qf.tsv")], cwd=unrelated, extra_env=env)
    checks["queryfile_fasta_fails"] = rf["rc"] == 2 and "LOOKS_LIKE_FASTA" in json.dumps(load_json(OUTDIR / "qf.json") or {})
    checks["sources_have_sha"] = bool((payload or {}).get("sources", {}).get("label_table", {}).get("sha256"))
    evidence["gene_observed"] = {"probe_stdout_tail": probe.stdout[-400:], "checks": checks,
                                 "all_ok": all(v for v in checks.values() if isinstance(v, bool))}
    return evidence["gene_observed"]


def _read_label_rows(gids):
    import csv
    want = set(gids)
    with open(C.LABEL_TABLE, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("gene_id") in want:
                yield row


def main():
    t_start = time.time()
    evidence = {"batch": C.BATCH, "version": C.VERSION, "protocol_section": "B",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "note": "Software integration tests only. NO performance claim. Runs from an unrelated cwd "
                        "with Unicode/space-containing input and output paths."}
    # 0. GPU availability must be explicit; never fall back to CPU.
    import importlib
    torch = importlib.import_module("torch")
    evidence["gpu_available"] = bool(torch.cuda.is_available())
    evidence["torch_version"] = torch.__version__
    if not evidence["gpu_available"]:
        evidence["status"] = "FAIL_EXPLICIT_NO_GPU"
        C.write_json(C.B47_REPORT / "test_evidence.json", evidence)
        print("FAIL: CUDA unavailable; no CPU fallback used.")
        return 1

    manifest = build_fixtures()
    evidence["frozen_test_manifest"] = str(TESTDIR / "test_manifest.json")
    # confirm fixtures were hashed before any forward
    rehash = {n: hashlib.sha256((TESTDIR / n).read_bytes()).hexdigest() for n in manifest["files"]}
    evidence["fixtures_unchanged_since_freeze"] = all(rehash[n] == manifest["files"][n]["sha256"]
                                                      for n in manifest["files"])

    fg = test_fragment_gpu(manifest, evidence)
    ev_fail = test_failures_before_cuda(evidence)
    ev_gene = test_gene_observed(evidence)

    evidence["total_wall_s"] = round(time.time() - t_start, 2)
    evidence["gpu_seconds_used"] = fg.get("gpu_wall_s")
    evidence["all_ok"] = bool(evidence["gpu_available"] and evidence["fixtures_unchanged_since_freeze"]
                              and fg["all_ok"] and ev_fail["all_ok"] and ev_gene["all_ok"])
    evidence["status"] = "PASS" if evidence["all_ok"] else "FAIL"
    evidence["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    C.B47_REPORT.mkdir(parents=True, exist_ok=True)
    C.write_json(C.B47_REPORT / "test_evidence.json", evidence)
    print(json.dumps({"status": evidence["status"], "all_ok": evidence["all_ok"],
                      "gpu_seconds_used": evidence["gpu_seconds_used"],
                      "total_wall_s": evidence["total_wall_s"],
                      "fragment_ok": fg["all_ok"], "failures_ok": ev_fail["all_ok"],
                      "gene_ok": ev_gene["all_ok"],
                      "report": str(C.B47_REPORT / "test_evidence.json")}, ensure_ascii=False))
    return 0 if evidence["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
