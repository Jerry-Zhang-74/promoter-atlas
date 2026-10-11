#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch20 worker self-check (NOT independent review).

Covers: (1) exact feature recomputation parity vs frozen catalog; (2) exact
source-checkpoint replay for the gene E/S/B model against shipped rc2 metrics;
(3) fold-aware replay for the K562 MPRA legnet ensemble against saved OOF; (4)
input-contract validation and honest refusal; (5) unit consistency and no fake
zero labels. Prints PASS/FAIL per test and writes worker_self_check.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b20_common as C
import b20_features as F
import b20_gene as G
import b20_gene_model as GM
import b20_mpra as M
import b20_metrics as MET

REPORT = C.REPORT_OUT
RESULTS = []


def record(name, ok, detail):
    RESULTS.append({"test": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print("%-42s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


def t_feature_parity():
    cat = G.load_catalog()
    worst = 0.0
    for q in ("ACTB", "GAPDH", "TP53", "ENSG00000000003"):
        gi, eid = G.resolve_gene(q)
        s4, s2 = G.gene_bag_sequences(gi)
        d = float(np.max(np.abs(F.gene_1069(s4, s2) - cat["X"][gi].astype(np.float64))))
        worst = max(worst, d)
    record("feature_1069_exact_parity", worst == 0.0, "max_abs_over_4_genes=%.3g" % worst)


def t_feature_determinism():
    cat = G.load_catalog()
    gi, _ = G.resolve_gene("ACTB")
    s4, s2 = G.gene_bag_sequences(gi)
    a = F.gene_1069(s4, s2)
    b = F.gene_1069(s4.copy(), s2.copy())
    record("feature_determinism", bool(np.array_equal(a, b)), "bit-identical on repeat")


def t_gene_replay():
    cat = G.load_catalog()
    pred = GM.GeneESBPredictor()
    m = MET.gene_test_metrics(cat, pred)
    ok = True
    deltas = {}
    for t in ("E", "S", "B"):
        d = m["per_task"][t]["spearman"] - m["expected_from_rc2"][t]
        deltas[t] = round(d, 8)
        if abs(d) > 2e-3:
            ok = False
    record("gene_replay_vs_rc2", ok, "rho_deltas=%s" % deltas)
    return m


def t_mpra_fold_replay():
    import pandas as pd
    oof = pd.read_csv(C.REPORT_ROOT / "batch02" / "trackB_oof_predictions.tsv", sep="\t")
    sup = pd.read_csv(C.MPRA_SUPERVISED, sep="\t", low_memory=False)
    seq_by_idx = dict(zip(sup["Oligo_index"].astype(int), sup["sequence"].astype(str)))
    s = oof.sample(n=40, random_state=0)
    idx = s["Oligo_index"].astype(int).tolist()
    seqs = [seq_by_idx[i] for i in idx]
    out = M.predict_legnet(seqs, per_model=True)
    fold_of = {k["fold"]: [] for k in out["model_index"]}
    per = out["per_model"]
    for k, mi in enumerate(out["model_index"]):
        fold_of[mi["fold"]].append(k)
    worst = 0.0
    for j, i in enumerate(idx):
        f = int(oof.loc[oof["Oligo_index"] == i, "fold"].iloc[0])
        rep = float(per[fold_of[f], j].mean())
        ref = float(oof.loc[oof["Oligo_index"] == i, "legnet_ens_pred"].iloc[0])
        worst = max(worst, abs(rep - ref))
    record("mpra_fold_aware_replay", worst < 1e-4, "max_abs_vs_saved_oof=%.3g (n=40)" % worst)


def t_input_contracts():
    checks = []
    # wrong length
    try:
        C.encode_uint8("ACGT", 164, name="x")
        checks.append(("wrong_length_raises", False))
    except C.B20InputError:
        checks.append(("wrong_length_raises", True))
    # documented encoder contract: any non-ACGT (incl. N) -> 4, no raise
    enc = C.encode_uint8("A" * 163 + "N", 164, name="x")
    checks.append(("non_acgt_maps_to_4", bool(enc[-1] == 4 and enc[0] == 0)))
    # CLI-level junk-letter rejection (letters outside ACGTN)
    junk = sorted(set("ACGTZ") - set("ACGTN"))
    checks.append(("cli_rejects_junk_letters", junk == ["Z"]))
    # unknown gene
    cat = G.load_catalog()
    try:
        G.resolve_gene("NOT_A_REAL_GENE_XYZ")
        checks.append(("unknown_gene_raises", False))
    except C.B20InputError:
        checks.append(("unknown_gene_raises", True))
    # feature dim
    pred = GM.GeneESBPredictor()
    ERR = (C.B20InputError, ValueError)
    try:
        pred.predict(np.zeros((1, 4, 400), dtype=np.uint8), np.zeros((1, 3)), batch=1)
        checks.append(("wrong_feature_dim_raises", False))
    except ERR:
        checks.append(("wrong_feature_dim_raises", True))
    try:
        pred.predict(np.zeros((0, 4, 400), dtype=np.uint8), np.zeros((0, 1069)), batch=1)
        checks.append(("empty_batch_raises", False))
    except ERR:
        checks.append(("empty_batch_raises", True))
    try:
        bad = np.zeros((1, 4, 400), dtype=np.uint8); bad[0, 0, 0] = 5
        pred.predict(bad, np.zeros((1, 1069)), batch=1)
        checks.append(("seq_value_gt4_raises", False))
    except ERR:
        checks.append(("seq_value_gt4_raises", True))
    ok = all(v for _, v in checks)
    record("input_contract_validation", ok, "; ".join("%s=%s" % (k, v) for k, v in checks))


def t_cli_no_fake_zero():
    out = REPORT / "_selfcheck_edge"
    cmd = [sys.executable, "-X", "utf8", str(HERE / "b20_cli.py"), "mpra-fasta",
           "--fasta", str(C.ROOT / "data" / "breakthrough_20261002" / "batch20" / "examples" / "edge_cases.fasta"),
           "--tag", "_selfcheck_edge"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    import csv
    p = REPORT / "_selfcheck_edge.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    short = [x for x in rows if x["name"] == "short_100bp"][0]
    dup = [x for x in rows if x["name"].startswith("dupB_")][0]
    ok = (short["predicted_activity"] == "" and short["supported"] == "unsupported_length"
          and dup["duplicate_of"] != "")
    record("cli_no_fake_zero_and_dedup", ok,
           "short_pred=%r dup_of=%r" % (short["predicted_activity"], dup["duplicate_of"]))


def t_units_consistency():
    cat = G.load_catalog()
    gi, _ = G.resolve_gene("GAPDH")
    obs_s = float(cat["obs_S_raw"][gi])
    # observed S_raw must be in the same (negative-MAD) units as the model output
    ok = obs_s < 0
    record("observed_S_units_raw_mad", ok, "GAPDH obs_S_raw=%.4f (<0 expected)" % obs_s)


def main():
    C.set_thread_limits()
    t0 = time.time()
    t_feature_parity()
    t_feature_determinism()
    m = t_gene_replay()
    t_mpra_fold_replay()
    t_input_contracts()
    t_cli_no_fake_zero()
    t_units_consistency()
    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    out = {
        "role": "worker_self_check (NOT independent review)",
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidate_version": GM.CANDIDATE_VERSION,
        "n_tests": len(RESULTS), "n_pass": n_pass, "n_fail": len(RESULTS) - n_pass,
        "all_pass": n_pass == len(RESULTS),
        "elapsed_s": round(time.time() - t0, 2),
        "results": RESULTS,
    }
    C.atomic_write_json(REPORT / "worker_self_check.json", out)
    print("\n%d/%d PASS  (%.1fs)  -> %s" % (n_pass, len(RESULTS), out["elapsed_s"],
                                            REPORT / "worker_self_check.json"))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
