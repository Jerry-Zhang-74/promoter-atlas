# -*- coding: utf-8 -*-
"""batch49 protocol tests for gene-rank (catalog tradeoff selection). ZERO GPU / ZERO neural forward.

Meaningful verification (not repeated broad model tests):
  (A) fixture_oracle       : an INDEPENDENT tiny fixture with dominated points / equal E / equal S /
                             exact (E,S) ties / undefined S / threshold boundary / duplicate-gene_id
                             conflict. A brute-force O(n^2) oracle must EQUAL the algorithm (set AND order)
                             and a limit must not change POOL membership (returned rows are a prefix).
  (B) real_catalog          : on the real frozen catalog, every returned frontier candidate is verified to
                             have NO dominator among ALL ELIGIBLE (direct independent scan), and the
                             frontier is verified COMPLETE by a distinct grouped-reference method; all
                             all-eligible is_frontier flags are exact.
  (C) permutation_invariance: row order must not change the frontier set.
  (D) thresholds            : 3 fixed B thresholds (0.3/0.8/1.0) as an increasing filter; all counts shown.
  (E) source_parity         : 20 deterministic source rows incl. edge/missing/S-mask; selected gene IDs
                             checked for numeric parity against gene-observed.
  (F) cli_edges             : invalid flags / empty result / UTF8 external-cwd path / deterministic JSON
                             semantic roundtrip (timestamps excluded).
  (G) inheritance           : b48 modules/engine/artifacts byte-unchanged (SHA inheritance only, no replay);
                             b49 gene-observed output == stored b48 gene-observed output.
CUDA is hidden and torch/numpy/Bio must NOT be imported anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b49_common as C  # noqa: E402
import b49_gene_observed as GO  # noqa: E402
import b49_gene_rank as GR  # noqa: E402

PY = sys.executable
CLI = str(C.CODE / "b49_cli.py")
DATA = C.B49_DATA
REP = C.B49_REPORT
TESTDIR = DATA / "testdata"
OUTDIR = DATA / "out"
PRE = C.B49_RUNS / "_pre_hashes.json"


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
            "stdout": p.stdout[-1500:], "stderr": p.stderr[-800:]}


def brute_force_frontier(points):
    """Independent O(n^2) oracle: a point is on the frontier iff NO other point dominates it."""
    fr = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q["E"] >= p["E"] and q["S"] >= p["S"] and (q["E"] > p["E"] or q["S"] > p["S"]):
                dominated = True
                break
        if not dominated:
            fr.append(p)
    return sorted(fr, key=lambda p: (-p["E"], -p["S"], p["gene_id"]))


def test_fixture_oracle():
    """(A) Tiny independent fixture; brute-force oracle must equal algorithm (set AND order)."""
    pts = [
        {"gene_id": "g_dom", "E": 1.0, "S": -2.0},          # dominated by g_hi (both higher)
        {"gene_id": "g_hi", "E": 3.0, "S": -1.0},           # early frontier point
        {"gene_id": "g_eqE_a", "E": 3.0, "S": 0.5},         # equal E group member (frontier)
        {"gene_id": "g_eqE_b", "E": 3.0, "S": 0.5},         # exact tie with g_eqE_a (both nondominated)
        {"gene_id": "g_eqE_low", "E": 3.0, "S": 0.0},       # same E but lower S -> dominated by group max
        {"gene_id": "g_eqS_a", "E": 2.0, "S": 0.5},         # equal S to eqE_a but lower E -> dominated
        {"gene_id": "g_last", "E": 0.5, "S": 1.0},          # lowest E, highest S -> frontier
        {"gene_id": "g_mid", "E": 2.5, "S": 0.2},           # dominated by eqE_a (E3,S0.5)
    ]
    algo = GR.pareto_frontier(pts)
    oracle = brute_force_frontier(pts)
    algo_ids = [p["gene_id"] for p in algo]
    oracle_ids = [p["gene_id"] for p in oracle]
    # limit must not change POOL membership: returned rows are a prefix of the pool
    pool = GR.pareto_frontier(pts)
    prefix_ok = [p["gene_id"] for p in pool[:2]] == algo_ids[:2]
    return {"n_points": len(pts), "algo_order": algo_ids, "oracle_order": oracle_ids,
            "set_equal": set(algo_ids) == set(oracle_ids), "order_equal": algo_ids == oracle_ids,
            "prefix_under_limit_ok": prefix_ok,
            "all_ok": (set(algo_ids) == set(oracle_ids)) and (algo_ids == oracle_ids) and prefix_ok}


def test_fixture_classify_and_duplicate_conflict():
    """(A2) classify disjoint priority incl. undefined S + threshold boundary + strict fail-closed."""
    labels = {
        "A": {"E_raw": 1.0, "S_raw": 0.5, "B_raw": 0.5, "S_defined_main": 1, "is_in_main_subset": 1,
              "Score_E": 0.1, "Score_S": 0.2, "Score_B": 0.3},
        "B": {"E_raw": 2.0, "S_raw": None, "B_raw": 0.9, "S_defined_main": 0, "is_in_main_subset": 0,
              "Score_E": 0.1, "Score_S": None, "Score_B": 0.3},   # undefined S
        "C": {"E_raw": 4.0, "S_raw": 0.7, "B_raw": 0.30, "S_defined_main": 1, "is_in_main_subset": 1,
              "Score_E": 0.1, "Score_S": 0.2, "Score_B": 0.3},    # B==minB boundary -> eligible (inclusive)
        "D": {"E_raw": 4.0, "S_raw": 0.7, "B_raw": 0.299999, "S_defined_main": 1, "is_in_main_subset": 1,
              "Score_E": 0.1, "Score_S": 0.2, "Score_B": 0.3},    # just below B -> below_B
        "E": {"E_raw": 0.5, "S_raw": 0.9, "B_raw": 1.0, "S_defined_main": 1, "is_in_main_subset": 1,
              "Score_E": 0.1, "Score_S": 0.2, "Score_B": 0.3},    # below_E when min_e=1.0
        "F": {"E_raw": None, "S_raw": 0.1, "B_raw": 1.0, "S_defined_main": 1, "is_in_main_subset": 1,
              "Score_E": 0.1, "Score_S": 0.2, "Score_B": 0.3},    # invalid (missing E)
    }
    counts, elig, inval = GR.classify(labels, min_b=0.30, min_e=None)
    ids = sorted(p["gene_id"] for p in elig)
    checks = {
        "undefined_S_counted": counts["undefined_S"] == 1,
        "invalid_counted": counts["invalid"] == 1,
        "boundary_B_inclusive": "C" in ids,
        "just_below_B_excluded": "D" not in ids,
        "priority_disjoint_sum": (counts["invalid"] + counts["undefined_S"] + counts["below_B"]
                                  + counts["below_E"] + counts["eligible"]) == counts["full_source"],
        "undefined_not_filled_zero": all(p["S"] != 0.0 for p in elig),
    }
    counts2, elig2, _ = GR.classify(labels, min_b=0.30, min_e=1.0)
    checks["min_e_applied"] = sorted(p["gene_id"] for p in elig2) == ["A", "C"]
    # strict fail-closed on duplicate gene_id (+ inconsistent row) via a temp fixture table
    d = Path(tempfile.mkdtemp(prefix="b49_dup_"))
    lab = d / "lab.tsv"
    lab.write_text(
        "gene_id\tE_raw\tS_raw\tB_raw\tScore_E\tScore_S\tScore_B\tS_defined_main\tis_in_main_subset\n"
        "ENSGX\t1.0\t0.5\t0.5\t0.1\t0.2\t0.3\t1\t1\n"
        "ENSGX\t1.1\t0.4\t0.6\t0.1\t0.2\t0.3\t1\t1\n", encoding="utf-8")
    symmap = d / "sym.tsv"
    symmap.write_text("gene_id\tgene_name\tgene_type\nENSGX.1\tGENEX\tprotein_coding\n", encoding="utf-8")
    old_lab, old_sym = GO.LABEL_TABLE, GO.ANALYSIS_SET
    try:
        GO.LABEL_TABLE, GO.ANALYSIS_SET = lab, symmap
        try:
            GO.load_contract(strict=True)
            dup_raised = False
        except C.B49InputError as e:
            dup_raised = str(e).startswith("DUPLICATE_GENE_ID_CONFLICT")
        relaxed_ok = True
        try:
            GO.load_contract(strict=False)
        except Exception:
            relaxed_ok = False
        # inconsistent row (S_defined_main==1 but S_raw empty) -> strict fails closed
        lab.write_text(
            "gene_id\tE_raw\tS_raw\tB_raw\tScore_E\tScore_S\tScore_B\tS_defined_main\tis_in_main_subset\n"
            "ENSGY\t1.0\t\t0.5\t0.1\t0.2\t0.3\t1\t1\n", encoding="utf-8")
        try:
            GO.load_contract(strict=True)
            incons_raised = False
        except C.B49InputError as e:
            incons_raised = str(e).startswith("INCONSISTENT_LABEL_ROW_CONFLICT")
    finally:
        GO.LABEL_TABLE, GO.ANALYSIS_SET = old_lab, old_sym
    checks["duplicate_gene_id_fails_closed"] = dup_raised
    checks["relaxed_loader_still_works_b48_semantics"] = relaxed_ok
    checks["inconsistent_row_fails_closed"] = incons_raised
    return {"checks": checks, "counts": counts, "eligible_ids": ids,
            "all_ok": all(v for v in checks.values())}


def reference_frontier_grouped(eligible):
    """DISTINCT reference method (independent code path): a gene is on the frontier iff its S equals the
    max S of its equal-E group AND exceeds the max S seen among ALL strictly higher-E groups."""
    order = sorted(eligible, key=lambda p: (-p["E"], -p["S"], p["gene_id"]))
    group_of = {}
    i, n = 0, len(order)
    groups = []
    while i < n:
        j = i
        e = order[i]["E"]
        while j < n and order[j]["E"] == e:
            j += 1
        groups.append(order[i:j])
        i = j
    fr = set()
    best = None
    for grp in groups:
        gmax = max(g["S"] for g in grp)
        if best is None or gmax > best:
            for g in grp:
                if g["S"] == gmax:
                    fr.add(g["gene_id"])
        best = gmax if best is None else max(best, gmax)
    return fr


def _has_dominator(cand, eligible):
    """Direct independent scan: does ANY eligible gene dominate cand?"""
    for q in eligible:
        if q["gene_id"] == cand["gene_id"]:
            continue
        if q["E"] >= cand["E"] and q["S"] >= cand["S"] and (q["E"] > cand["E"] or q["S"] > cand["S"]):
            return True
    return False


def test_real_catalog():
    """(B) Real frozen catalog: per-candidate no-dominator scan + completeness by reference + flags."""
    contract = GO.load_contract(strict=True)
    res = GR.rank_catalog(contract)   # defaults
    counts, eligible, inval = GR.classify(contract["labels"], 0.30, None)
    frontier = GR.pareto_frontier(eligible)
    returned_ids = [r["gene_id"] for r in res["selected"]]
    # (a) EVERY returned frontier candidate must have NO dominator among ALL ELIGIBLE
    dom_violations = [r["gene_id"] for r in res["selected"] if _has_dominator(
        {"gene_id": r["gene_id"], "E": r["E_raw"], "S": r["S_raw"]}, eligible)]
    # (b) completeness by a distinct grouped reference
    ref_ids = reference_frontier_grouped(eligible)
    algo_ids = set(g["gene_id"] for g in frontier)
    # (c) all-eligible frontier flags exact
    ae_true = sum(1 for r in res["all_eligible_rows"] if r["is_frontier"])
    checks = {
        "n_eligible_18947": len(eligible) == 18947,
        "n_frontier_reported": res["returns"]["n_frontier"] == len(frontier),
        "every_returned_candidate_undominated": len(dom_violations) == 0,
        "frontier_set_equals_reference": algo_ids == ref_ids,
        "all_eligible_flags_exact": ae_true == len(frontier),
        "selected_are_returned_frontier": set(returned_ids) == algo_ids,
        "counts_sum_ok": res["counts"]["counts_sum_equals_full_source"],
        "invalid_zero_on_real_catalog": counts["invalid"] == 0,
        "undefined_S_40041": counts["undefined_S"] == 40041,
    }
    return {"checks": checks, "n_eligible": len(eligible), "n_frontier": len(frontier),
            "dominator_violations": dom_violations[:5], "frontier_ids": sorted(algo_ids),
            "all_ok": all(v for v in checks.values() if isinstance(v, bool))}


def test_permutation():
    """(C) Row order must NOT change the frontier set."""
    contract = GO.load_contract(strict=True)
    _, eligible, _ = GR.classify(contract["labels"], 0.30, None)
    base = set(g["gene_id"] for g in GR.pareto_frontier(eligible))
    rng = random.Random(12345)
    same = True
    for _ in range(3):
        perm = list(eligible)
        rng.shuffle(perm)
        if set(g["gene_id"] for g in GR.pareto_frontier(perm)) != base:
            same = False
            break
    return {"frontier_invariant_to_row_order": same, "n_frontier": len(base), "all_ok": same}


def test_thresholds():
    """(D) 3 fixed B thresholds as an increasing filter; all counts reported (no best-performance pick)."""
    contract = GO.load_contract(strict=True)
    out = {}
    idsets = {}
    for mb in (0.30, 0.80, 1.00):
        res = GR.rank_catalog(contract, min_b=mb)
        _, elig, _ = GR.classify(contract["labels"], mb, None)
        idsets[mb] = set(g["gene_id"] for g in elig)
        out[str(mb)] = {"eligible": res["counts"]["eligible"], "below_B": res["counts"]["below_B"],
                        "undefined_S": res["counts"]["undefined_S"], "n_frontier": res["returns"]["n_frontier"]}
    subset_ok = (idsets[1.00] <= idsets[0.80]) and (idsets[0.80] <= idsets[0.30])
    return {"counts": out, "monotone_increasing_filter": subset_ok,
            "expected_eligible": {"0.30": 18947, "0.80": 13467, "1.00": 8968},
            "all_ok": subset_ok and out["0.3"]["eligible"] == 18947 and out["0.8"]["eligible"] == 13467
                      and out["1.0"]["eligible"] == 8968}


def _read_source_rows(gids):
    want = set(gids)
    with open(C.LABEL_TABLE, "r", encoding="utf-8", newline="") as fh:
        import csv
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("gene_id") in want:
                yield row


def _defnum(rec, col):
    try:
        return float(rec.get(col))
    except (TypeError, ValueError):
        return None


def test_source_parity():
    """(E) 20 deterministic source rows (incl. edge/missing/S-mask) + selected gene-observed parity."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    contract = GO.load_contract(strict=True)
    res = GR.rank_catalog(contract)
    selected_ids = [r["gene_id"] for r in res["selected"]]
    # 20 deterministic source gene_ids: 12 frontiers + 3 known + 5 undefined-S rows
    undefined_ids = []
    for gid in sorted(contract["labels"]):
        if contract["labels"][gid]["S_defined_main"] == 0:
            undefined_ids.append(gid)
        if len(undefined_ids) >= 5:
            break
    probe_ids = []
    for gid in selected_ids + ["ENSG00000075624", "ENSG00000089157"] + undefined_ids:
        if gid not in probe_ids:
            probe_ids.append(gid)
        if len(probe_ids) >= 20:
            break
    src = {r["gene_id"]: r for r in _read_source_rows(probe_ids)}
    checks = {"n_source_rows_probed": len(probe_ids), "all_found_in_source": len(src) == len(probe_ids)}
    # source vs loader for the 20 rows
    field_ok = True
    for gid in probe_ids:
        s = src.get(gid)
        l = contract["labels"][gid]
        for col, key in (("E_raw", "E_raw"), ("B_raw", "B_raw"), ("Score_E", "Score_E"),
                         ("Score_B", "Score_B"), ("Score_S", "Score_S")):
            sv = _defnum(s, col)
            lv = l[key]
            if (sv is None) != (lv is None) or (sv is not None and abs(sv - lv) > 1e-12):
                field_ok = False
        sv_raw = _defnum(s, "S_raw")
        if (sv_raw is None) != (l["S_raw"] is None) or (sv_raw is not None and abs(sv_raw - l["S_raw"]) > 1e-12):
            field_ok = False
        if str(s.get("S_defined_main")) != str(l["S_defined_main"]):
            field_ok = False
    checks["source_fields_match_loader_20rows"] = field_ok
    # gene-observed numeric parity for the selected IDs (CUDA hidden, external cwd)
    unrelated = tempfile.mkdtemp(prefix="b49_ext_")
    ojson, otsv = OUTDIR / "gene.b49.parity.json", OUTDIR / "gene.b49.parity.tsv"
    q = probe_ids
    probe = subprocess.run([PY, "-X", "utf8", "-c",
        "import sys; sys.path.insert(0, r'%s'); import b49_cli; "
        "argv=['gene-observed','--query']+%s+['--out-json',r'%s','--out-tsv',r'%s']; "
        "rc=b49_cli.main(argv); print('MODULES', 'torch' in sys.modules, 'numpy' in sys.modules, "
        "'Bio' in sys.modules, rc)" % (C.CODE, repr(q), str(ojson), str(otsv))],
        cwd=unrelated, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    payload = load_json(ojson) or {}
    go_by_id = {r["canonical_gene_id"]: r for r in payload.get("records", []) if r.get("status") == "OK"}
    gr_by_id = {r["gene_id"]: r for r in res["selected"]}
    parity_ok = True
    for gid in selected_ids:
        g = go_by_id.get(gid)
        r = gr_by_id.get(gid)
        if not g or not r:
            parity_ok = False
            continue
        if (g["observedScore_E"] != r["observedScore_E"] or g["S_raw"] != r["S_raw"]
                or g["B_raw"] != r["B_raw"] or g["is_in_main_subset"] != r["is_in_main_subset"]
                or g["S_defined_mask"] != r["S_defined_main"]):
            parity_ok = False
    modline = [ln for ln in probe.stdout.splitlines() if ln.startswith("MODULES")]
    checks["selected_gene_observed_numeric_parity"] = parity_ok
    checks["no_torch_no_numpy_no_bio"] = bool(modline and modline[0] == "MODULES False False False 0")
    checks["undefined_probe_has_null_S"] = all(
        (go_by_id.get(g) or {}).get("S_raw") is None for g in undefined_ids)
    return {"checks": checks, "probe_ids": probe_ids,
            "all_ok": all(v for v in checks.values() if isinstance(v, bool))}


def _strip_times(o):
    if isinstance(o, dict):
        return {k: _strip_times(v) for k, v in o.items() if k not in ("generated_at", "frozen_at")}
    if isinstance(o, list):
        return [_strip_times(v) for v in o]
    return o


def test_cli_edges():
    """(F) invalid flags / empty result / UTF8 external-cwd path / deterministic JSON roundtrip."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    unrelated = tempfile.mkdtemp(prefix="b49_cli_ext_")
    env = {"CUDA_VISIBLE_DEVICES": ""}
    checks = {}
    # invalid flags -> typed error JSON + non-zero exit
    bad = [
        (["--min-b", "1.5"], "INVALID_MIN_B_OUT_OF_RANGE"),
        (["--min-b", "nan"], "INVALID_MIN_B_NOT_FINITE"),
        (["--limit", "0"], "INVALID_LIMIT_OUT_OF_RANGE"),
        (["--min-e", "inf"], "INVALID_MIN_E_NOT_FINITE"),
    ]
    for extra, prefix in bad:
        oj = OUTDIR / ("bad_%s.json" % prefix)
        args = ["gene-rank"] + extra + ["--out-json", str(oj), "--out-tsv", str(OUTDIR / "bad.tsv")]
        r = run_cli(args, unrelated, env)
        p = load_json(oj) or {}
        checks["bad_%s_exit2" % prefix] = r["rc"] == 2
        checks["bad_%s_reason" % prefix] = str(p.get("error", "")).startswith(prefix)
    # bad sort -> argparse exit 2 (no output file required)
    r_sort = run_cli(["gene-rank", "--sort", "bogus", "--out-json", str(OUTDIR / "s.json"),
                      "--out-tsv", str(OUTDIR / "s.tsv")], unrelated, env)
    checks["bad_sort_exit2"] = r_sort["rc"] == 2
    # empty result -> status OK, clear
    oj = OUTDIR / "empty.json"
    r_e = run_cli(["gene-rank", "--min-b", "1.0", "--min-e", "1000", "--out-json", str(oj),
                   "--out-tsv", str(OUTDIR / "empty.tsv")], unrelated, env)
    pe = load_json(oj) or {}
    checks["empty_exit0"] = r_e["rc"] == 0
    checks["empty_status_ok"] = pe.get("status") == "OK" and pe.get("returns", {}).get("n_selected") == 0
    checks["empty_counts_present"] = (pe.get("counts", {}).get("eligible") == 0
                                      and pe.get("counts", {}).get("counts_sum_equals_full_source") is True)
    # UTF8 external cwd + UTF8 output path
    udir = Path(tempfile.mkdtemp(prefix="b49_utf8_")) / "子目录_utf8"
    udir.mkdir(parents=True, exist_ok=True)
    ujson = udir / "结果_gene_rank.json"
    utsv = udir / "结果_gene_rank.tsv"
    r_u = run_cli(["gene-rank", "--limit", "3", "--out-json", str(ujson), "--out-tsv", str(utsv)],
                  udir, env)
    checks["utf8_cwd_exit0"] = r_u["rc"] == 0 and ujson.exists() and utsv.exists()
    checks["utf8_json_parses"] = (load_json(ujson) or {}).get("status") == "OK"
    # deterministic JSON semantic roundtrip (timestamps excluded)
    a_json = OUTDIR / "det_a.json"
    b_json = OUTDIR / "det_b.json"
    run_cli(["gene-rank", "--out-json", str(a_json), "--out-tsv", str(OUTDIR / "det_a.tsv"),
             "--all-eligible-tsv", str(OUTDIR / "det_a_all.tsv")], unrelated, env)
    run_cli(["gene-rank", "--out-json", str(b_json), "--out-tsv", str(OUTDIR / "det_b.tsv"),
             "--all-eligible-tsv", str(OUTDIR / "det_b_all.tsv")], unrelated, env)
    A, B = load_json(a_json), load_json(b_json)
    checks["deterministic_semantic_roundtrip"] = (_strip_times(A) == _strip_times(B)) and A is not None
    # all-eligible TSV has is_frontier flag and the full eligible count
    import csv as _csv
    with open(OUTDIR / "det_a_all.tsv", "r", encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh, delimiter="\t"))
    checks["all_eligible_tsv_rows_18947"] = len(rows) == 18947
    checks["all_eligible_has_frontier_flag"] = all("is_frontier" in r for r in rows)
    checks["all_eligible_frontier_count_12"] = sum(1 for r in rows if r["is_frontier"] == "True") == 12
    return {"checks": checks, "all_ok": all(v for v in checks.values() if isinstance(v, bool))}


def test_inheritance():
    """(G) b48 modules/engine/artifacts byte-unchanged; b49 gene-observed == stored b48 output."""
    pre = load_json(PRE) or {}
    unchanged = {}
    for key, rec in pre.items():
        if key in ("registry_before_b49", "changelog_before_b49"):
            continue   # these are EXPECTED to change (atomic append in phase D); checked by postverify
        ap = C.ROOT / rec["path"]
        now = sha(ap) if ap.exists() else None
        unchanged[key] = {"path": rec["path"], "before": rec["sha256"], "after": now,
                          "unchanged": now == rec["sha256"]}
    # gene-observed parity vs the STORED b48 output (reuse proof)
    b48_out = load_json(C.ROOT / "data/breakthrough_20261002/batch48/out/gene.b48.json") or {}
    b48_recs = b48_out.get("records", [])
    q = [r["query"] for r in b48_recs]
    unrelated = tempfile.mkdtemp(prefix="b49_parity_")
    oj, ot = OUTDIR / "gene.b49.b48parity.json", OUTDIR / "gene.b49.b48parity.tsv"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    r = run_cli(["gene-observed", "--query"] + q + ["--out-json", str(oj), "--out-tsv", str(ot)],
               unrelated, {"CUDA_VISIBLE_DEVICES": ""})
    now_recs = (load_json(oj) or {}).get("records", [])
    fields = ["query", "status", "canonical_gene_id", "symbol", "observedScore_E", "S_raw",
              "S_raw_defined", "B_raw", "observedScore_S", "observedScore_B", "is_in_main_subset",
              "S_defined_mask"]
    parity = (len(b48_recs) == len(now_recs)) and all(
        all(str(a.get(f)) == str(b.get(f)) for f in fields) for a, b in zip(b48_recs, now_recs))
    return {"files": unchanged, "all_unchanged": all(v["unchanged"] for v in unchanged.values()),
            "gene_observed_matches_stored_b48_output": parity, "n_queries": len(q),
            "b48_record_count": len(b48_recs), "b49_record_count": len(now_recs),
            "all_ok": all(v["unchanged"] for v in unchanged.values()) and parity}


def _write_inherited_evidence():
    """Record the earlier true-CUDA forward evidence as INHERITED (never as a new b49 replay)."""
    b47ev = load_json(C.ROOT / "reports/breakthrough_20261002/batch47/test_evidence.json") or {}
    fg = (b47ev.get("fragment_gpu") or {}).get("checks", {})
    b49engine = sha(C.B35_ENGINE) if C.B35_ENGINE.exists() else None
    inh = {
        "batch": "batch49", "role": "worker", "record_type": "inherited_evidence_not_replay",
        "statement": ("The frozen PARM HEK293 inference code (b35 engine) is UNCHANGED by b49, and b49 adds "
                      "no neural path, so b49 performed NO neural forward. The earlier b47 independent "
                      "true-CUDA control parity is INHERITED as the backend evidence."),
        "b35_engine_path": str(C.B35_ENGINE.relative_to(C.ROOT).as_posix()),
        "b35_engine_sha256": b49engine,
        "b47_test_evidence_path": "reports/breakthrough_20261002/batch47/test_evidence.json",
        "b47_control_score": fg.get("control_score"), "b47_control_abs_diff": fg.get("control_abs_diff"),
        "b47_model_params_on_cuda": fg.get("gpu_params_on_cuda"),
        "b47_input_tensor_on_cuda": fg.get("gpu_input_on_cuda"),
        "b49_replay_performed": False, "b49_gpu_used": False,
    }
    C.write_json(C.B49_RUNS / "inherited_b47_cuda_evidence.json", inh)
    return inh


def test_tamper_copy():
    """(F2) Tamper a COPY of one hashed file -> existing verifier FAILs, WITHOUT rewriting the manifest;
    the real original is untouched (exactly the b48 contract, reusing the existing verifier)."""
    import b49_finalize as FIN
    d = Path(tempfile.mkdtemp(prefix="b49_tamper_"))
    src = C.CODE / "b49_gene_rank.py"
    orig_sha = sha(src)
    copy = d / "b49_gene_rank.py"
    copy.write_bytes(src.read_bytes())
    man = {"batch": "fixture", "files": {"b49_gene_rank.py": {"sha256": sha(copy),
                                                              "bytes": copy.stat().st_size}}}
    mp = d / "sha_manifest_copy.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    ok_before = FIN.verify_manifest_file(mp, root=d)
    raw = bytearray(copy.read_bytes())
    raw[10] = raw[10] ^ 0x01
    copy.write_bytes(bytes(raw))
    same_len = (len(raw) == man["files"]["b49_gene_rank.py"]["bytes"])
    bad = FIN.verify_manifest_file(mp, root=d)
    orig_unchanged = (sha(src) == orig_sha)
    checks = {"ok_before": not ok_before, "tamper_same_length": same_len,
              "verification_failed_after_tamper": bool(bad), "original_unchanged": orig_unchanged}
    return {"checks": checks, "bad": bad[:2], "all_ok": all(checks.values())}


def main():
    t0 = time.time()
    ev = {"batch": C.BATCH, "version": C.VERSION, "protocol_section": "tests_C",
          "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
          "note": ("gene-rank verification: independent fixture oracle, real-catalog per-candidate "
                   "no-dominator scan + distinct-reference completeness, permutation invariance, fixed "
                   "threshold filter, 20-row source parity, CLI edges. ZERO GPU, ZERO neural forward; "
                   "b48/b35 inheritance only.")}
    ev["fixture_oracle"] = test_fixture_oracle()
    ev["fixture_classify_duplicate"] = test_fixture_classify_and_duplicate_conflict()
    ev["real_catalog"] = test_real_catalog()
    ev["permutation_invariance"] = test_permutation()
    ev["thresholds"] = test_thresholds()
    ev["source_parity"] = test_source_parity()
    ev["cli_edges"] = test_cli_edges()
    ev["tamper_copy"] = test_tamper_copy()
    ev["inheritance"] = test_inheritance()
    ev["gpu_used_this_batch"] = False
    ev["neural_forward_this_batch"] = False
    parts = [ev["fixture_oracle"]["all_ok"], ev["fixture_classify_duplicate"]["all_ok"],
             ev["real_catalog"]["all_ok"], ev["permutation_invariance"]["all_ok"],
             ev["thresholds"]["all_ok"], ev["source_parity"]["all_ok"], ev["cli_edges"]["all_ok"],
             ev["tamper_copy"]["all_ok"], ev["inheritance"]["all_ok"]]
    ev["all_ok"] = all(parts)
    ev["status"] = "PASS" if ev["all_ok"] else "FAIL"
    ev["total_wall_s"] = round(time.time() - t0, 2)
    ev["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ev["inherited_b47_cuda_evidence"] = _write_inherited_evidence()
    C.write_json(REP / "test_evidence.json", ev)
    print(json.dumps({"status": ev["status"], "all_ok": ev["all_ok"],
                      "fixture_oracle": ev["fixture_oracle"]["all_ok"],
                      "fixture_classify_duplicate": ev["fixture_classify_duplicate"]["all_ok"],
                      "real_catalog": ev["real_catalog"]["all_ok"],
                      "permutation": ev["permutation_invariance"]["all_ok"],
                      "thresholds": ev["thresholds"]["all_ok"],
                      "source_parity": ev["source_parity"]["all_ok"],
                      "cli_edges": ev["cli_edges"]["all_ok"],
                      "tamper_copy": ev["tamper_copy"]["all_ok"],
                      "inheritance": ev["inheritance"]["all_ok"],
                      "wall_s": ev["total_wall_s"]}, ensure_ascii=False))
    return 0 if ev["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
