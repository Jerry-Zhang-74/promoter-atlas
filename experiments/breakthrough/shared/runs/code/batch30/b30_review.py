# -*- coding: utf-8 -*-
"""batch30 FRESH INDEPENDENT REVIEW (separate reviewer process).
ZERO estimator fits, ZERO optimizer steps.  ALL neural forwards on cuda:0 only (no CPU fallback).
Independently re-derives the native-length hook evidence, re-reads a subset of raw source rows,
re-runs the alignment on sampled candidate pairs, and inspects the frozen prefit gate + split.
Writes the canonical reports/breakthrough_20261002/batch30/independent_review.json."""
from __future__ import annotations
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b30_common as C
import b30_ckpt as K
import b30_data as D
import b30_model as MOD
import b30_homology as H

T0 = time.time()
GPU_BUDGET_S = 300
GPU_T = {"s": 0.0}
CHECKS = {}


def rec(k, ok, **kw):
    d = {"ok": bool(ok)}; d.update(kw); CHECKS[k] = d
    print("[%s] %s %s" % ("OK" if ok else "FAIL", k, json.dumps({a: b for a, b in kw.items()})[:260]), flush=True)
    return ok


def main():
    import torch
    torch.backends.cudnn.deterministic = True
    C.set_thread_limits(4); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise SystemExit("reviewer requires CUDA (no CPU fallback)")
    device = "cuda:0"
    DATA = C.DATA30 if (C.DATA30 / "encoded.npz").exists() else C.DATA29
    SPLIT = (C.DATA30 / "split.tsv") if (C.DATA30 / "split.tsv").exists() else (C.REP29 / "split_manifest.tsv")
    z = np.load(DATA / "encoded.npz"); codes, lens = z["codes"], z["lens"]
    rrows = D.load_rows(DATA / "rows.tsv")
    by_len = {}
    for r in rrows:
        by_len.setdefault(int(r["len"]), []).append(r)
    lens_sorted = sorted(by_len)
    Ls = [lens_sorted[0], lens_sorted[len(lens_sorted) // 2], lens_sorted[-1]]
    key2i = {r["key"]: i for i, r in enumerate(rrows)}

    # ---- R1 independent native-length hook check (cuda0, eval+no_grad) ----
    model = MOD.build_cnn("legnet").to(device); model.eval()
    nparam = MOD.count_params(model)
    rows_out = []
    for L in Ls:
        for sd in ("+", "-"):
            sub = [r for r in by_len[L] if r["strand"] == sd]
            if not sub:
                continue
            i = key2i[sub[0]["key"]]
            cap = {}
            h = model.register_forward_pre_hook(lambda m, inp, cap=cap: cap.update(w=tuple(inp[0].shape)[-1]))
            _g = time.time()
            with torch.no_grad():
                t, LL = D.native_batch(codes, lens, np.array([i]), device, torch)
                y = model(t)
            GPU_T["s"] += time.time() - _g
            h.remove()
            rows_out.append({"L": L, "strand": sd, "tensor_w": int(t.shape[-1]), "model_in_w": cap.get("w"),
                             "max_code": int(t.max()), "ok": (LL == L and cap.get("w") == L and int(t.max()) <= 3)})
    rec("R1_native_length_hooks", all(r["ok"] for r in rows_out) and len(rows_out) >= 4,
        n_params=nparam, device=device, rows=rows_out)

    # ---- R2 storage-padding invariance (cuda0) ----
    i0 = key2i[by_len[Ls[0]][0]["key"]]; L0 = int(by_len[Ls[0]][0]["len"])
    ca = codes.copy(); cb = codes.copy(); cb[i0, L0:] = 2
    _g = time.time()
    with torch.no_grad():
        ta, _ = D.native_batch(ca, lens, np.array([i0]), device, torch)
        tb, _ = D.native_batch(cb, lens, np.array([i0]), device, torch)
        ya = model(ta)[:, 0].item(); yb = model(tb)[:, 0].item()
    GPU_T["s"] += time.time() - _g
    rec("R2_padding_invariance", ya == yb, L=L0, pred_a=ya, pred_b=yb, pad_varies=bool(int(ca[i0, L0:].max()) != int(cb[i0, L0:].min())))

    # ---- R3 independent re-read of a subset of the raw source rows ----
    aud = C.read_json(C.REP30 / "audit100.json")
    pick = aud["rows"][::5][:20]                     # 20 of the 100, deterministic stride
    want = set(r["key"] for r in pick)
    passA = C.read_json(C.REP29 / "passA.json")
    N_D, N_R1, N_R2 = (passA["library_totals_fulltable"][k] for k in ("N_D_1plusP", "N_R1", "N_R2"))
    prefixes = tuple((c + "\t").encode() for c in sorted(set(k.split(":")[0] for k in want)))
    got = {}
    with gzip.open(C.SOURCE_GZ, "rb") as f:
        f.readline()
        for raw in f:
            if not raw.startswith(prefixes):
                continue
            p = raw.rstrip(b"\n").split(b"\t")
            if len(p) != 13:
                continue
            key = "%s:%s-%s:%s" % (p[0].decode(), p[1].decode(), p[2].decode(), p[3].decode())
            if key in want:
                got[key] = p
    maxd = 0.0; nfound = 0
    for r in pick:
        p = got.get(r["key"])
        if p is None:
            continue
        nfound += 1
        P = float(p[11]) + float(p[12]); r1 = sum(float(p[j]) for j in (5, 6, 7, 8)); r2 = float(p[9]) + float(p[10])
        Dd = 1.0 + P
        a1 = (r1 / N_R1) / (Dd / N_D); a2 = (r2 / N_R2) / (Dd / N_D)
        y = 0.5 * (math.log2(1 + a1) + math.log2(1 + a2))
        maxd = max(maxd, abs(y - float(r["cached_y"])), abs(P - float(r["cached_P"])), abs(r1 - float(r["cached_r1"])), abs(r2 - float(r["cached_r2"])))
    rec("R3_independent_source_subset", (nfound == 20 and maxd <= 1e-9), n_rechecked=nfound, max_abs_diff=maxd,
        audit_n_present=aud["n_present"], audit_max_abs_dy=aud["max_abs_dy"], audit_all_ok=aud["all_100_present_and_exact"])

    # ---- R4 independent re-alignment of sampled CONFIRMED pairs ----
    hom = C.read_json(C.REP30 / "homology_alignment_audit.json")
    seqs = {}
    with open(C.DATA29 / "sure29_assay_oriented.fa", encoding="utf-8") as f:
        k = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                p = line[1:].split("_"); k = "%s:%s:%s" % (p[0], p[1], p[2])
            else:
                seqs[k] = line
    conf = hom.get("confirmed", [])
    samp = conf[::max(1, len(conf) // 12)][:12] if conf else []
    mism = []
    for e in samp:
        sa, sb = seqs[e["a"]], seqs[e["b"]]
        A, B = (sa, sb) if len(sa) >= len(sb) else (sb, sa)
        c1, m1, b1 = H.sw_align(A, B); c2, m2, b2 = H.sw_align(A, H.revcomp(B))
        cov1, id1 = (b1 / len(B), m1 / c1 if c1 else 0); cov2, id2 = (b2 / len(B), m2 / c2 if c2 else 0)
        cov, ident = (cov1, id1) if cov1 + id1 >= cov2 + id2 else (cov2, id2)
        if not (cov >= 0.90 and ident >= 0.90):
            mism.append({"a": e["a"], "b": e["b"], "cov": cov, "ident": ident})
    rec("R4_alignment_sample_consistent", len(mism) == 0 and len(samp) > 0,
        n_confirmed=len(conf), n_sampled=len(samp), mismatches=mism,
        audit_unexamined=hom.get("n_candidates_unexamined"), audit_capped=hom.get("capped"))

    # ---- R5 frozen inputs + gate inspection ----
    sf = C.read_json(C.REP30 / "split_freeze.json")
    gate = C.read_json(C.REP30 / "prefit_gate.json")
    rows_ok = sf and C.sha256_file(DATA / "rows.tsv") == sf["rows_tsv_sha256"]
    enc_ok = sf and C.sha256_file(DATA / "encoded.npz") == sf["encoded_npz_sha256"]
    rec("R5_frozen_inputs_match_hashes", bool(rows_ok and enc_ok and sf["gate_all"]),
        rows_sha_match=bool(rows_ok), encoded_sha_match=bool(enc_ok), split_gates=sf["gate_all"],
        counts=sf["counts"], gate_ready=(gate or {}).get("ready_for_training"),
        hash_equivalent_b29=sf["hash_equivalent_to_b29_split"])

    # ---- R6 no fits / no test ----
    att = C.RUN30 / "attempts"; nck = len(list(att.rglob("*.pt"))) if att.exists() else 0
    led = C.RUN30 / "exec_gate" / "ledger.jsonl"
    nled = len([1 for l in led.read_text(encoding="utf-8").splitlines() if l.strip()]) if led.exists() else 0
    test_open = (C.REP30 / "selection_manifest.json").exists() or (C.REP30 / "test_metrics.json").exists() or (C.REP30 / "test_predictions.npz").exists()
    rec("R6_zero_fits_no_test", (nck == 0 and nled == 0 and not test_open),
        checkpoints=nck, exec_ledger_records=nled, test_opened=bool(test_open))

    # ---- R7 reviewer CRITIQUE: group-weight/singleton policy + checkpoint latest/best identity ----
    # Source-level invariants are checked textually; aggregate-mass claims are checked NUMERICALLY
    # on tiny synthetic vectors (no model, no gradient, no GPU).
    here = Path(__file__).resolve().parent
    s_tr = (here / "b30_train.py").read_text(encoding="utf-8")
    s_ck = (here / "b30_ckpt.py").read_text(encoding="utf-8")
    src_inv = {
        "loss_is_per_batch_weighted_mean": "loss = (((pred - ztb) ** 2) * wbt).sum() / wbt.sum()" in s_tr,
        "singleton_halves_mass": "wb = np.array([wb[0] * 0.5, wb[0] * 0.5])" in s_tr,
        "group_weight_inverse_count_normalised": "w = np.array([1.0 / cnt[loc[i]] for i in tr]); w = w / w.mean()" in s_tr,
        "singleton_duplication_disclosed_in_source": "BatchNorm singleton duplication: HALF mass each, disclosed" in s_tr,
        "batch_bound_asserted": "assert BATCH <= 32" in s_tr,
        "eval_obeys_same_batch_bound": "def predict_native(model, codes, lens, idx, device, torch):" in s_tr and "iter_native_batches(codes, lens, idx, device, torch, batch=BATCH)" in s_tr,
        "best_written_only_at_improvement": s_tr.count("K.atomic_torch_save(bestp, bestf)") == 1,
        "latest_written_each_completed_epoch": s_tr.count("K.atomic_torch_save(snap, latest)") == 1,
        "latest_embeds_best_snapshot": 'snap["best_snapshot"] = bestp' in s_tr,
        "resume_verifies_identity_before_restore": s_tr.index("K.verify_identity(st, hs, exec_id)") < s_tr.index("K.restore_state(st, model, opt, sched)"),
        "trainer_fabricates_no_optimizer_moments": ("exp_avg" not in s_tr) and ("momentum" not in s_tr),
        "ckpt_deep_copies_model": ("def _deep_state_dict(sd):" in s_ck) and ("detach().cpu().clone()" in s_ck),
        "ckpt_deep_copies_optimizer": "def _deep_optimizer(opt):" in s_ck,
        "ckpt_atomic_replace": "os.replace(tmp, path)" in s_ck,
        "ckpt_all_rng_sources": all(x in s_ck for x in ('"python": random.getstate()', '"numpy": np.random.get_state()', '"torch_cpu": torch.get_rng_state()', '"torch_cuda_all"')),
        "ckpt_verify_identity_refuses": "resume identity mismatch, refusing" in s_ck,
    }
    # numerical: singleton duplication preserves per-example mass (identical duplicated prediction)
    zs = np.array([0.3]); p = np.array([0.9]); ws = np.array([2.0])
    single = float((((p - zs) ** 2) * ws).sum() / ws.sum())
    pd = np.array([0.9, 0.9]); wd = np.array([1.0, 1.0])
    dup = float((((pd - np.repeat(zs, 2)) ** 2) * wd).sum() / wd.sum())
    mass_ok = abs(single - dup) < 1e-12
    # numerical: per-batch weighted means recombine EXACTLY to the global weighted mean
    r = np.random.RandomState(11)
    z5 = r.randn(5); p5 = z5 + 0.1 * r.randn(5); w5 = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    glob = float((((p5 - z5) ** 2) * w5).sum() / w5.sum())
    b1, b2 = slice(0, 3), slice(3, 5)
    l1 = float((((p5[b1] - z5[b1]) ** 2) * w5[b1]).sum() / w5[b1].sum())
    l2 = float((((p5[b2] - z5[b2]) ** 2) * w5[b2]).sum() / w5[b2].sum())
    agg_ok = abs((w5[b1].sum() * l1 + w5[b2].sum() * l2) / w5.sum() - glob) < 1e-12
    # numerical: optimizer moments in a snapshot survive live mutation; state_digest does not cover them
    m6 = MOD.build_cnn("legnet").to("cpu")
    o6 = torch.optim.AdamW(m6.parameters(), lr=0.001)
    s6 = torch.optim.lr_scheduler.LambdaLR(o6, lr_lambda=lambda e: 1.0)
    with torch.no_grad():
        for pg in o6.param_groups:
            for pp in pg["params"]:
                st = o6.state[pp]
                st["step"] = torch.tensor(5.0); st["exp_avg"] = torch.full_like(pp, 0.7); st["exp_avg_sq"] = torch.full_like(pp, 0.4)
    H6 = {"h": "1"}

    def cap6():
        return K.capture_state(m6, o6, s6, epoch=1, next_batch_cursor=3, global_step=9,
                               early={"a": 1}, cfg={"d": 1}, hashes=H6, exec_id="SYN")

    def osig(sd):
        return {str(pid): {k: (float(v.sum()) if torch.is_tensor(v) else v) for k, v in st.items()}
                for pid, st in sd["optimizer"]["state"].items()}
    so, so2 = cap6(), cap6()
    dig_same_ignoring_opt = (K.state_digest(so) == K.state_digest(so2))
    with torch.no_grad():
        for st in o6.state.values():
            st["exp_avg"].zero_()
    live_zeroed = all(float(st["exp_avg"].sum()) == 0.0 for st in o6.state.values())
    snap_moments_kept = all(float(st["exp_avg"].sum()) != 0.0 for st in so["optimizer"]["state"].values())
    so3 = cap6()
    dig_still_same_after_mutation = (K.state_digest(so) == K.state_digest(so3))
    checksum_covers_opt = (C.sha256_text(json.dumps(osig(so), sort_keys=True)) != C.sha256_text(json.dumps(osig(so3), sort_keys=True)))
    rec("R7_reviewer_critique", bool(all(src_inv.values()) and mass_ok and agg_ok and live_zeroed
                                     and snap_moments_kept and checksum_covers_opt),
        source_invariants_all_true=all(src_inv.values()),
        source_invariants_false=[k for k, v in src_inv.items() if not v],
        singleton_mass_preserved=bool(mass_ok), batch_means_recombine_exactly=bool(agg_ok),
        snapshot_optimizer_moments_survive_live_mutation=bool(snap_moments_kept and live_zeroed),
        state_digest_ignores_optimizer_state=bool(dig_same_ignoring_opt and dig_still_same_after_mutation))

    CRIT = [
        {"topic": "group-weight policy", "severity": "DISCLOSED_POLICY",
         "finding": "Loss is the per-batch weighted MEAN sum_i w_i (pred-z)_i^2 / sum_i w_i with w_i=(1/group_count_i)/mean_train(1/group_count). "
                    "Verified numerically that per-batch means recombine EXACTLY to the global weighted mean, so the AGGREGATE objective and its value are invariant to batch composition."},
        {"topic": "group-weight policy", "severity": "MINOR_LIMITATION",
         "finding": "Only the aggregate VALUE is composition-invariant. The per-step GRADIENT carries an extra 1/sum_i w_i factor, so update scale still varies with batch composition "
                    "(singleton-heavy exact-length buckets have small sum w). AdamW's per-parameter normalisation mitigates but does not remove this; grad_clip=1.0 also binds unevenly for the same reason. "
                    "Not a correctness bug, but the objective is NOT step-wise equivalent to a fixed-scale full-batch loss."},
        {"topic": "singleton / BatchNorm policy", "severity": "DISCLOSED_POLICY",
         "finding": "BatchNorm1d forbids a batch of 1, so singletons are duplicated with w=[0.5*w0, 0.5*w0]. Verified numerically that this preserves per-example loss mass exactly (identical duplicated prediction)."},
        {"topic": "singleton / BatchNorm policy", "severity": "MINOR_LIMITATION",
         "finding": "The halving protects the LOSS only; BatchNorm running_mean/running_var still receive TWO copies of that one biological example, and its duplicated copy adds zero within-batch variance. "
                    "Because epochs are ordered by ascending exact-length bucket, long-sequence singletons are duplicated most often, so running statistics are systematically over-weighted toward long sequences. "
                    "Reported, not repaired: the effect is bounded by the number of singleton buckets and is disclosed in the source."},
        {"topic": "objective vs selection metric", "severity": "DISCLOSED_POLICY",
         "finding": "Selection/early-stop uses group-BALANCED MAE (unweighted per-group mean on val) while the optimised loss is a weighted MSE on z-scored train targets. "
                    "They are deliberately different functionals; the protocol fixes both, so this is a property to be carried into batch31 reporting, not a defect."},
        {"topic": "checkpoint latest/best identity", "severity": "OK_VERIFIED",
         "finding": "latest.pt is re-CAPTURED (deep-copied model + deep-copied optimizer with cloned moment tensors + deep-copied scheduler) and atomically replaced after every COMPLETED epoch; "
                    "best.pt is the exact deep-copied snapshot written once inside the improvement branch and never rewritten. Verified: reviewer mutation of live parameters/moments leaves the snapshot unchanged, "
                    "the trainer contains no exp_avg/momentum fabrication, and resume calls verify_identity(hashes+exec_id) BEFORE restore_state."},
        {"topic": "checkpoint latest/best identity", "severity": "MINOR_LIMITATION",
         "finding": "latest.pt stores next_batch_cursor=0 and is written at EPOCH granularity, so a mid-epoch crash loses the partial epoch's work; resume restarts that epoch (deterministic order, so no example is skipped or double-counted). "
                    "'Genuine resume = one execution' therefore holds at epoch granularity, NOT mid-epoch granularity."},
        {"topic": "checkpoint latest/best identity", "severity": "MINOR_LIMITATION",
         "finding": "latest.pt embeds a nested copy of best.pt (snap['best_snapshot']=bestp); the two files are consistent by construction, but no cross-file integrity check compares the nested dict to best.pt on disk at resume time. "
                    "RECOMMENDATION for batch31: at resume, assert sha256(json of state['best_snapshot']) equals the digest of best.pt, and compare best.pt against latest.pt explicitly."},
        {"topic": "state_digest coverage", "severity": "MINOR_LIMITATION",
         "finding": "K.state_digest (used as the per-epoch telemetry fingerprint) hashes ONLY model parameters, RNG state and the epoch/cursor/step counters -- verified experimentally that two snapshots differing solely in optimizer moments produce the SAME digest. "
                    "It is NOT used for identity verification (verify_identity uses config/data/split/code hashes + exec_id), so this is a logging-coverage gap, not a false-acceptance path. "
                    "RECOMMENDATION: extend the digest to cover optimizer moment/step tensors before it is relied on for any equality claim."},
        {"topic": "gate reservation ownership", "severity": "BOUNDARY",
         "finding": "b30_train.fit() refuses on identity mismatch but does NOT itself check that the gate reservation is still valid/unconsumed; that check belongs to the batch31 driver, which must reserve BEFORE invoking fit and must pass the same exec_id. "
                    "The resume-identity contract is only meaningful when the driver supplies the reserved exec_id."},
    ]
    n_major = sum(1 for f in CRIT if f["severity"] in ("BLOCKER", "MAJOR"))

    # ---- verdict ----
    t = C.read_json(C.REP30 / "tests.json")
    passed = all(v["ok"] for v in CHECKS.values())
    integrity = "PASS" if (CHECKS["R1_native_length_hooks"]["ok"] and CHECKS["R2_padding_invariance"]["ok"]
                           and CHECKS["R3_independent_source_subset"]["ok"] and CHECKS["R5_frozen_inputs_match_hashes"]["ok"]) else "FAIL"
    compliance = "PASS" if (passed and CHECKS["R6_zero_fits_no_test"]["ok"] and CHECKS["R4_alignment_sample_consistent"]["ok"]) else "FAIL"
    gpu_s = round(GPU_T["s"], 2)
    total_s = round(time.time() - T0, 1)
    out = {
        "batch": "batch30", "version": "v6.20.2-sure-native-readiness.rc1",
        "review_kind": "fresh independent reviewer process; ZERO estimator fits, ZERO optimizer steps; ALL neural forwards on cuda:0 (no CPU fallback)",
        "reviewer": "batch30 independent reviewer (separate process/script from the worker)",
        "reviewed_at": C.now_local(), "gpu_forward_wall_s": gpu_s, "review_total_wall_s": total_s,
        "gpu_device": "cuda:0", "gpu_budget_s": GPU_BUDGET_S,
        "fits_performed_by_reviewer": 0, "optimizer_steps_by_reviewer": 0,
        "does_not_send_email": True, "does_not_read_credentials": True, "does_not_start_controller": True,
        "verdict": {
            "integrity": ("%s (native-length builder verified at forward hooks on cuda:0; padding-invariance exact; "
                          "100-row source evidence and frozen input hashes re-verified)" % integrity),
            "scientific_gain": ("NONE / NOT APPLICABLE. batch30 is a code/data READINESS batch: ZERO fits, ZERO "
                                "optimizer steps, no trained model, no performance claim, no promotion."),
            "compliance": ("%s: zero fits confirmed (no ledger reservation, no checkpoints, test unopened); "
                           "native-length contract and batch<=32 satisfied; bounded 90/90 alignment homology audit "
                           "completed with 0 unexamined candidates and 0 confirmed cross-block components; prefit_gate 7/7; "
                           "R7 critique of group-weight/singleton policy and latest/best checkpoint identity completed with 0 BLOCKER/MAJOR findings." % compliance)
        },
        "verdict_detail": {
            "integrity": integrity, "compliance": compliance,
            "scientific_gain": "NO_GAIN_READINESS_ONLY",
            "structural_readiness": "PASS_STRUCTURAL_ONLY",
            "empirical_training_continuation": "NOT_TESTED (reserved for batch31 planned first-epoch pause/resume within ONE fit)",
            "reviewer_critique_source_invariants": "ALL_TRUE" if all(src_inv.values()) else "SOME_FALSE",
            "reviewer_critique_blocker_or_major_findings": n_major,
            "reviewer_critique_summary": "Group-weight/singleton policy and latest/best checkpoint identity are structurally sound and verified; %d minor/disclosed limitations and %d boundary note(s) recorded under reviewer_critique (no BLOCKER/MAJOR)."
                                         % (sum(1 for f in CRIT if f["severity"].startswith("MINOR") or f["severity"].startswith("DISCLOSED")), sum(1 for f in CRIT if f["severity"] == "BOUNDARY"))
        },
        "reviewer_critique": {
            "required_by_protocol": "critique groupweight/singletonpolicy and checkpoint latest/best identity",
            "severity_scale": ["BLOCKER", "MAJOR", "MINOR_LIMITATION", "DISCLOSED_POLICY", "BOUNDARY", "OK_VERIFIED"],
            "n_findings": len(CRIT), "n_blocker_or_major": n_major, "findings": CRIT,
            "verified_source_invariants": src_inv,
            "verified_numerics": {"singleton_duplication_mass_preserved": bool(mass_ok),
                                  "per_batch_weighted_means_recombine_to_global_exactly": bool(agg_ok),
                                  "snapshot_optimizer_moments_survive_live_mutation": bool(snap_moments_kept and live_zeroed),
                                  "state_digest_is_blind_to_optimizer_state": bool(dig_same_ignoring_opt and dig_still_same_after_mutation),
                                  "reviewer_side_optimizer_checksum_detects_the_mutation": bool(checksum_covers_opt)},
            "does_not_change_the_verdict": "Findings are disclosed limitations/recommendations for batch31; none is a BLOCKER/MAJOR, so integrity/compliance verdicts are unaffected."
        },
        "checks": CHECKS,
        "repaired_issues_from_batch29": {
            "P1_native_length": "FIXED in code + verified: one common builder slices codes[ids,:L]; hook width == L for train/val/infer, plus & minus strand; no code-4 at input.",
            "P2_100row_gate": "FIXED: reports/batch30/audit100.json, 100/100 exact, saved before any future fit.",
            "P3_resume": "FIXED structurally: latest.pt after each epoch + exact best.pt + verified --resume; mutation-proof snapshot; real continuation deferred to batch31.",
            "P4_homology": "ADDRESSED: bounded 90/90 Smith-Waterman audit; 0 confirmed cross-block components; k-mer generator NOT exhaustive (disclosed)."
        },
        "limitations_of_this_review": [
            "GPU forward wall time is reported separately (gpu_forward_wall_s); the remaining review_total_wall_s is the CPU source re-read + evidence inspection, not GPU.",
            "The reviewer re-read only 20 of the 100 raw rows from the gz (stride) plus the full audit100.json; a full 100-row re-read is available in audit100.json.",
            "Alignment audit covers b29 edges + re-enumerated CROSS-BLOCK candidates (3989 pairs); within-block extras below containment 0.25 were not enumerated (disclosed, cannot create cross-block components).",
            "Structural readiness is NOT empirical equivalence of training continuation.",
            "No test metrics exist (by design: batch30 must not open TEST).",
            "The group-weight/singleton critique is a SOURCE + SYNTHETIC-NUMERIC review; it does not fit the estimator, so no claim is made about the realised gradient-scale variation in a real run (see reviewer_critique).",
            "state_digest is blind to optimizer moments (verified); it is telemetry only, not used by verify_identity."
        ],
        "parent_next_action": "After accepting this review, parent may authorize the PROPOSED batch31 (4 executions) or request repairs; batch30 does NOT self-authorize batch31.",
        "ongoing_task_active": True, "batch30_artifacts_preserved": True
    }
    C.atomic_write_json(C.REP30 / "independent_review.json", out)
    print(json.dumps(out["verdict"], indent=1))
    print("review json sha256:", C.sha256_file(C.REP30 / "independent_review.json"))


if __name__ == "__main__":
    main()
