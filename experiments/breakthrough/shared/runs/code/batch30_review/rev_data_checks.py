# -*- coding: utf-8 -*-
"""Fresh independent reviewer (batch30) PART2: data / source / gate checks. ZERO fits."""
from __future__ import annotations
import gzip, hashlib, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REP = ROOT / "reports/breakthrough_20261002"
REP30 = REP / "batch30"
REP29 = REP / "batch29"
DATA29 = ROOT / "data/breakthrough_20261002/batch29"
SRC = ROOT / "data/breakthrough_20261002/batch27/source/GSE78709_SuRE-counts-K562_B45_B55_LP170105.txt.gz"
OUT = {}


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rec_selection():
    keys = []
    with open(REP29 / "split_manifest.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            keys.append(line.split("\t")[0])
    plus = sorted([k for k in keys if k.endswith(":+")], key=sha_text)
    minus = sorted([k for k in keys if k.endswith(":-")], key=sha_text)
    sel = plus[:50] + minus[:50]
    audit = json.load(open(REP30 / "audit100.json", encoding="utf-8"))
    got = [r["key"] for r in audit["rows"]]
    return sel, got, keys


def main():
    t0 = time.time()
    sel, got, keys = rec_selection()
    OUT["selection_rule_reproduced"] = {
        "n_surviving": len(keys), "n_selected": len(got),
        "plus": sum(1 for k in got if k.endswith(":+")), "minus": sum(1 for k in got if k.endswith(":-")),
        "matches_audit100": set(sel) == set(got), "same_order": sel == got,
        "response_independent": True, "ok": set(sel) == set(got) and len(got) == 100}
    # cached rows
    cached = {}
    with open(DATA29 / "rows.tsv", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            d = dict(zip(hdr, line.rstrip("\n").split("\t"))); cached[d["key"]] = d
    passA = json.load(open(REP29 / "passA.json", encoding="utf-8"))
    N_D = passA["library_totals_fulltable"]["N_D_1plusP"]
    N_R1 = passA["library_totals_fulltable"]["N_R1"]
    N_R2 = passA["library_totals_fulltable"]["N_R2"]
    OUT["library_offsets"] = {"N_D": N_D, "N_R1": N_R1, "N_R2": N_R2,
                              "passA_ok": True,
                              "ok": (N_D, N_R1, N_R2) == (203742844, 37770653, 74081034)}
    target = set(got)
    chroms = sorted(set(k.split(":")[0] for k in target))
    prefixes = tuple((c + "\t").encode() for c in chroms)
    found = {}; nread = 0
    with gzip.open(SRC, "rb") as f:
        f.readline()
        for raw in f:
            nread += 1
            if raw[:1] not in (b"c",):
                continue
            if not raw.startswith(prefixes):
                continue
            p = raw.rstrip(b"\n").split(b"\t")
            if len(p) != 13:
                continue
            k = "%s:%s-%s:%s" % (p[0].decode(), p[1].decode(), p[2].decode(), p[3].decode())
            if k in target:
                found[k] = p
    OUT["source_scan"] = {"n_rows_read": nread, "matches_153562178": nread == 153562178,
                          "n_found": len(found), "elapsed_s": round(time.time() - t0, 1)}
    maxd = {"P": 0.0, "r1": 0.0, "r2": 0.0, "y": 0.0}
    miss = []
    for k in got:
        p = found.get(k)
        c = cached.get(k)
        if p is None or c is None:
            miss.append(k); continue
        num = lambda b: float(b.decode())
        r1 = sum(num(p[i]) for i in (5, 6, 7, 8)); r2 = num(p[9]) + num(p[10])
        P = num(p[11]) + num(p[12]); D = 1.0 + P
        a1 = (r1 / N_R1) / (D / N_D); a2 = (r2 / N_R2) / (D / N_D)
        y = 0.5 * (math.log2(1.0 + a1) + math.log2(1.0 + a2))
        maxd["P"] = max(maxd["P"], abs(P - float(c["P"])))
        maxd["r1"] = max(maxd["r1"], abs(r1 - float(c["r1"])))
        maxd["r2"] = max(maxd["r2"], abs(r2 - float(c["r2"])))
        maxd["y"] = max(maxd["y"], abs(y - float(c["y"])))
    OUT["row_recompute_vs_cached_9013"] = {"n_missing": len(miss), "max_abs_diff": maxd,
                                           "ok": len(miss) == 0 and all(v <= 1e-9 for v in maxd.values())}
    # cross-check audit100 stored values
    audit = json.load(open(REP30 / "audit100.json", encoding="utf-8"))
    arows = {r["key"]: r for r in audit["rows"]}
    maxd2 = 0.0
    for k in got:
        if k in arows and k in cached:
            maxd2 = max(maxd2, abs(float(arows[k]["y"]) - float(arows[k]["cached_y"])))
    OUT["audit100_internal_consistency"] = {"max_abs_y_audit_vs_cached": maxd2, "ok": maxd2 <= 1e-12}
    # prefit gate completeness
    pg = json.load(open(REP30 / "prefit_gate.json", encoding="utf-8"))
    items = pg["items"]
    complete = all(("observed" in v or v.get("file")) and isinstance(v.get("ok"), bool) for v in items.values())
    derived = pg["ready_for_training"] == all(v["ok"] for v in items.values())
    OUT["prefit_gate"] = {"n_items": pg["n_items"], "n_pass": pg["n_pass"], "all_ok": all(v["ok"] for v in items.values()),
                          "ready": pg["ready_for_training"], "complete_no_placeholder": complete,
                          "ready_correctly_derived": derived, "ok": complete and derived and pg["n_pass"] == 7}
    # zero-fit accounting
    led = ROOT / "runs/breakthrough_20261002/batch30/exec_gate/ledger.jsonl"
    att = ROOT / "runs/breakthrough_20261002/batch30/attempts"
    npt = len(list(att.rglob("*.pt"))) if att.exists() else 0
    test_open = (REP30 / "selection_manifest.json").exists() or (REP30 / "test_metrics.json").exists() \
        or (REP30 / "test_predictions.npz").exists()
    OUT["zero_fits"] = {"real_ledger_exists": led.exists(), "checkpoint_pt": npt, "test_opened": bool(test_open),
                        "ok": (not led.exists()) and npt == 0 and not test_open}
    # target_freeze contained TEST target summaries (corrects 'no test labels ever read')
    tf = REP29 / "target_freeze.json"
    tfd = json.load(open(tf, encoding="utf-8")) if tf.exists() else {}
    OUT["test_label_summaries"] = {"target_freeze_exists": tf.exists(),
                                   "keys": list(tfd.keys())[:8],
                                   "mentions_test": "test" in json.dumps(tfd).lower(),
                                   "ok": tf.exists() and "test" in json.dumps(tfd).lower()}
    out = REP / "batch30_review" / "rev_data_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(OUT, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
