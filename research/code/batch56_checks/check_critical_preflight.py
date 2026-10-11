# -*- coding: utf-8 -*-
"""batch56 CRITICAL PRE-FIT check (read-only historical bytes; NO fit/solve/gradient/GPU/web)."""
import gzip, json, os, re, time

ROOT = r"E:/promoter_atlas"
COH = os.path.join(ROOT, "data/breakthrough_20261002/batch52/corrected/cohort_v2")
SRC = os.path.join(ROOT, "data/breakthrough_20261002/batch51/GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz")
LEDGER = os.path.join(ROOT, "runs/breakthrough_20261002/batch52/exec_gate_ledger.jsonl")
RUNPOL = os.path.join(ROOT, "runs/breakthrough_20261002/batch52/RUN_POLICY.json")
CORE = os.path.join(ROOT, "runs/breakthrough_20261002/code/batch53_model_core")
DV2 = os.path.join(ROOT, "code/batch52_data_v2")
TARGET = "Log2RPM_HEK293"
OLD = "Log2Norm_HEK293"
OUT = os.path.join(ROOT, "reports/breakthrough_20261002/batch56")
T0 = time.time()
SPLITS = ("TRAIN", "DEV", "TEST")
R = {"batch": "batch56", "kind": "critical_prefit_check",
     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "checks": {}, "findings": []}


def add(sev, key, msg, file_line="", evidence=""):
    R["findings"].append({"severity": sev, "key": key, "msg": msg, "file": file_line, "evidence": evidence})


def c1_target():
    ap = os.path.join(COH, "endpoints.tsv")
    with open(ap, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    ti = header.index(TARGET) if TARGET in header else -1
    spec_txt = open(os.path.join(CORE, "spec.py"), encoding="utf-8").read()
    man = json.load(open(os.path.join(COH, "manifest.json"), encoding="utf-8"))
    cfg = json.load(open(RUNPOL, encoding="utf-8"))
    old_as_target = []
    for base in (CORE, DV2):
        for dp, _, fns in os.walk(base):
            if "__pycache__" in dp:
                continue
            for fn in fns:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dp, fn)
                for line in open(p, encoding="utf-8", errors="replace"):
                    s = line.strip()
                    if s.startswith("TARGET_COLUMN") and OLD in s:
                        old_as_target.append((os.path.relpath(p, ROOT), s[:120]))
    c1 = {"endpoints_header": header, "target_present": TARGET in header, "target_index": ti,
          "old_column_in_header": OLD in header,
          "spec_target_line": [l.strip() for l in spec_txt.splitlines() if l.strip().startswith("TARGET_COLUMN")],
          "manifest_target": man.get("target_column"), "runpolicy_target": cfg.get("target_column"),
          "old_column_used_as_target_anywhere": old_as_target}
    n = len(rows)
    stride = max(1, n // 20)
    idxs = sorted(set([0] + [i * stride for i in range(1, 20)] + [n - 1]))[:20]
    sample, seen = [], set()
    for i in idxs:
        r = rows[i]
        if r[0] not in seen:
            seen.add(r[0])
            sample.append({"coord_id": r[0], "split": r[2], "len": int(r[3]), "cohort_target": float(r[4])})
    mn = min(int(x[3]) for x in rows[:5000])
    for lab, pr in [("minus_strand", lambda r: r[0].endswith("-")), ("chrX", lambda r: r[0].startswith("chrX:")),
                    ("len_min", lambda r: int(r[3]) == mn), ("zero_target", lambda r: float(r[4]) == 0.0),
                    ("negative_target", lambda r: float(r[4]) < 0.0)]:
        for r in rows:
            if pr(r):
                sample.append({"coord_id": r[0], "split": r[2], "len": int(r[3]),
                               "cohort_target": float(r[4]), "stratum": lab})
                break
    samp_ids = {s["coord_id"] for s in sample}
    joined, n_src = {}, 0
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        irpm, inorm = head.index(TARGET), head.index(OLD)
        for line in f:
            a = line.rstrip("\n").split("\t")
            n_src += 1
            cid = "%s:%s-%s:%s" % (a[1], a[2], a[3], a[4])
            if cid in samp_ids:
                try:
                    joined[cid] = {"rpm": float(a[irpm]), "norm": float(a[inorm])}
                except ValueError:
                    joined[cid] = {"rpm": None, "norm": None}
    joins, ok = [], 0
    for s in sample:
        j = joined.get(s["coord_id"])
        if j is None:
            joins.append(dict(s, found_in_source=False, target_match=False)); continue
        match = j["rpm"] is not None and abs(j["rpm"] - s["cohort_target"]) < 1e-4
        ok += 1 if match else 0
        joins.append(dict(s, found_in_source=True, source_rpm=j["rpm"], source_norm=j["norm"],
                          target_match=match,
                          norm_equals_rpm=(j["rpm"] is not None and j["rpm"] == j["norm"])))
    c1.update({"raw_row_joins_n": len(joins), "raw_row_joins_ok": ok,
               "source_rows_scanned": n_src, "raw_row_joins": joins,
               "source_rows_where_rpm_equals_norm": sum(1 for j in joins if j.get("norm_equals_rpm"))})
    R["checks"]["C1_target_and_20_row_joins"] = c1
    if ti < 0 or man.get("target_column") != TARGET:
        add("CRITICAL", "C1-target", "target column not exactly %s" % TARGET)
    if ok != len(joins) or any(not j["found_in_source"] for j in joins):
        add("CRITICAL", "C1-join", "raw-row join mismatch %d/%d" % (ok, len(joins)),
            evidence=json.dumps([j for j in joins if not j.get("target_match")])[:800])
    return rows, header


def c2_full_train(rows):
    cnt = {s: 0 for s in SPLITS}
    for r in rows:
        cnt[r[2]] += 1
    pe = set()
    with open(os.path.join(COH, "pairs.tsv"), encoding="utf-8") as f:
        ph = f.readline().rstrip("\n").split("\t")
        si, sc, lc = ph.index("split"), ph.index("short_coord_id"), ph.index("long_coord_id")
        for ln in f:
            if ln.strip():
                a = ln.rstrip("\n").split("\t")
                if a[si] == "TRAIN":
                    pe.add(a[sc]); pe.add(a[lc])
    ad = open(os.path.join(CORE, "data_adapter.py"), encoding="utf-8").read()
    tn = open(os.path.join(CORE, "train_neural.py"), encoding="utf-8").read()
    uses_all = ('activity_weights("TRAIN")' in tn) and ("split_indices(split)" in ad)
    c2 = {"endpoint_counts": cnt, "pair_only_train_endpoints": len(pe),
          "trainer_uses_all_train_rows_static": uses_all,
          "train_equals_pair_only": cnt["TRAIN"] == len(pe)}
    R["checks"]["C2_all_eligible_train_rows"] = c2
    if cnt["TRAIN"] <= len(pe) or not uses_all:
        add("CRITICAL", "C2-scope", "activity training not using ALL eligible TRAIN rows", evidence=json.dumps(c2))
    return c2


def c3_reference():
    prov = json.load(open(os.path.join(COH, "reference_provenance.json"), encoding="utf-8"))
    gz = os.path.join(ROOT, "data/breakthrough_20261002/batch52/reference/GRCh37.primary_assembly.fa.gz")
    info = {"prov": prov, "exists": os.path.exists(gz), "ref_path": gz}
    if info["exists"]:
        info["bytes"] = os.path.getsize(gz)
        with open(gz, "rb") as f:
            info["gz_magic_ok"] = f.read(2) == b"\x1f\x8b"
        names, got = [], 0
        with gzip.open(gz, "rt", errors="replace") as f:
            for line in f:
                if line.startswith(">"):
                    names.append(line.strip())
                    if len(names) >= 6:
                        break
                got += len(line)
                if got > 2000000:
                    break
        info["leading_fasta_headers"] = names
    R["checks"]["C3_reference_provenance"] = info
    if not info["exists"] or not info.get("gz_magic_ok"):
        add("CRITICAL", "C3-ref", "reference missing/corrupt", evidence=json.dumps(info)[:400])
    return info


def c4_groups(rows):
    gmap, dup = {}, 0
    for r in rows:
        g, s = r[1], r[2]
        if g in gmap and gmap[g] != s:
            dup += 1
        gmap[g] = s
    gs = {s: set() for s in SPLITS}
    for g, s in gmap.items():
        gs[s].add(g)
    disjoint = not ((gs["TRAIN"] & gs["DEV"]) or (gs["TRAIN"] & gs["TEST"]) or (gs["DEV"] & gs["TEST"]))
    gtsv = {}
    with open(os.path.join(COH, "groups.tsv"), encoding="utf-8") as f:
        f.readline()
        for ln in f:
            if ln.strip():
                a = ln.rstrip("\n").split("\t"); gtsv[a[0]] = a[1]
    c4 = {"n_groups": len(gmap), "group_split_counts": {s: len(v) for s, v in gs.items()},
          "group_spans_multiple_splits": dup, "splits_disjoint": disjoint,
          "groups_tsv_consistent": gtsv == gmap,
          "adapter_enforces_disjoint": "split groups are not disjoint" in
          open(os.path.join(CORE, "data_adapter.py"), encoding="utf-8").read()}
    R["checks"]["C4_group_disjointness"] = c4
    if not disjoint or dup or not c4["groups_tsv_consistent"]:
        add("CRITICAL", "C4-groups", "group split leakage", evidence=json.dumps(c4))
    return c4


def c5_gate():
    recs = [json.loads(l) for l in open(LEDGER, encoding="utf-8") if l.strip()]
    consuming = [r for r in recs if r.get("kind") == "reserved" and r.get("consumes")]
    hist = [r for r in recs if r.get("purpose") == "batch52-R0"]
    r0 = open(os.path.join(CORE, "pipeline_coordinator.py"), encoding="utf-8").read()
    fr = r0[r0.index("def fit_r0"):r0.index("def _deadline_epoch")]
    reserve_before_solve = fr.index('gate.reserve("R0")') < fr.index(".fit(")
    tn = open(os.path.join(CORE, "train_neural.py"), encoding="utf-8").read()
    rb = tn[tn.index("def run(self"):]
    reserve_before_setup = rb.index("self._reserve()") < rb.index("self._setup()")
    spec = open(os.path.join(CORE, "spec.py"), encoding="utf-8").read()
    ck = open(os.path.join(CORE, "checkpoint.py"), encoding="utf-8").read()
    # assertion (not mere recording) of a frozen acceptance-file SHA:
    acc = bool(re.search(r'acceptance_sha256"\s*==|EXPECTED_ACCEPTANCE|assert_acceptance|ACCEPTANCE_SHA\s*=', spec + r0))
    c5 = {"ledger_records": recs, "consuming_slots": len(consuming), "budget": 4,
          "remaining": 4 - len(consuming), "historical_ungated_present": bool(hist), "historical": hist,
          "r0_reserve_before_solve": reserve_before_solve, "nabs_reserve_before_setup": reserve_before_setup,
          "config_asserts_target": ("assert_target" in spec), "config_asserts_acceptance_sha": acc,
          "ckpt_optimizer": "optimizer.state_dict" in ck, "ckpt_rng": "capture_rng" in ck,
          "ckpt_sampler": "sampler_state" in ck, "ckpt_gate_attempt_id": "gate_attempt_id" in ck}
    R["checks"]["C5_exec_gate_ledger"] = c5
    if not reserve_before_solve or not reserve_before_setup:
        add("CRITICAL", "C5-gate-order", "gate not reserved before solve/gradient")
    if len(consuming) > 4:
        add("CRITICAL", "C5-budget", "consuming slots exceed MAX 4")
    if not (c5["ckpt_optimizer"] and c5["ckpt_rng"] and c5["ckpt_sampler"]):
        add("MAJOR", "C5-ckpt", "checkpoint missing optimizer/RNG/sampler")
    if not acc:
        add("MINOR", "C5-acc-sha",
            "fit-gate config records but does not ASSERT the acceptance-file SHA; target+source SHA are asserted",
            file_line="runs/breakthrough_20261002/code/batch53_model_core/spec.py:67")
    return c5


def c6_deadline():
    pol = json.load(open(RUNPOL, encoding="utf-8"))
    dv2 = open(os.path.join(DV2, "run_all_v2.py"), encoding="utf-8").read()
    co = open(os.path.join(CORE, "pipeline_coordinator.py"), encoding="utf-8").read()
    fh = re.search(r'"--first-hhmm",\s*default="([^"]+)"', dv2)
    sh = re.search(r'"--second-hhmm",\s*default="([^"]+)"', dv2)
    reads = ("RUN_POLICY" in dv2) or ("RUN_POLICY" in co)
    c6 = {"runpolicy_first_delivery": pol.get("first_model_delivery_deadline"),
          "runpolicy_training_target": pol.get("first_model_training_target"),
          "runpolicy_review_target": pol.get("first_model_review_target"),
          "run_all_v2_first_hhmm_default": fh.group(1) if fh else None,
          "run_all_v2_second_hhmm_default": sh.group(1) if sh else None,
          "coordinator_first_hhmm_default": re.search(r'first_hhmm=\(([^)]*)\)', co).group(1),
          "code_reads_RUN_POLICY_json": reads,
          "first_model_only_flag_present": "--first-model-only" in dv2}
    R["checks"]["C6_deadline_and_freeze"] = c6
    if not reads or c6["run_all_v2_first_hhmm_default"] != "12:00":
        add("MAJOR", "C6-deadline",
            "13:00 delivery / 12:00 training deadline NOT read from RUN_POLICY.json; run_all_v2.py defaults "
            "--first-hhmm=14:00 and the LIVE run omitted it, so the epoch-cap decider uses 14:00",
            file_line="code/batch52_data_v2/run_all_v2.py:40-41,54 ; pipeline_coordinator.py:84",
            evidence=json.dumps({k: c6[k] for k in ("runpolicy_training_target",
                                 "run_all_v2_first_hhmm_default", "code_reads_RUN_POLICY_json")}))
    return c6


def main():
    rows, header = c1_target()
    c2_full_train(rows); c3_reference(); c4_groups(rows); c5_gate(); c6_deadline()
    R["elapsed_s"] = round(time.time() - T0, 2)
    R["verdict"] = {"critical_train_blocking": any(f["severity"] == "CRITICAL" for f in R["findings"]),
                    "major_nonblocking": [f["key"] for f in R["findings"] if f["severity"] == "MAJOR"],
                    "minor": [f["key"] for f in R["findings"] if f["severity"] == "MINOR"]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "critical_preflight.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    print(json.dumps({"verdict": R["verdict"], "findings": R["findings"],
                      "elapsed_s": R["elapsed_s"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
