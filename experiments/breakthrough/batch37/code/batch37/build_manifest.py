#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch37 step 1 - freeze the immutable diagnostic_manifest BEFORE any PARM forward.

ZERO fits / ZERO neural inference here (pure data assembly + hashing).
Pins: official PARM code + 5 HEK293 fold weights (SHA recomputed), official bare-FASTA
central zero-pad (L_max=600), equal mean of 5 folds. Builds label-free paired (alt,ref)
variable inserts from the author design FASTA + batch36 frozen cohorts. No RC ensemble,
no tiling, no length clipping, no flank fill.
"""
import gzip, hashlib, json, os, csv, sys, time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCH = "batch37"; VERSION = "v6.26.0-frozen-variant-diagnostic.rc1"
ASSET = os.path.join(REPO, "data/breakthrough_20261002/batch31/_external")
DESIGN = os.path.join(REPO, "data/breakthrough_20261002/batch35/mapt/gsm9640449_VariantMPRAdesignFile.fa.gz")
DESIGN_SHA = "4f7243b83f4b9df08e64c1b0f3f2d7bedc9ca8fc270a424331e270dc9862f9e9"
B36 = os.path.join(REPO, "data/breakthrough_20261002/batch36/mapt")
OUTDIR = os.path.join(REPO, "data/breakthrough_20261002/batch37")
RPT = os.path.join(REPO, "reports/breakthrough_20261002/batch37")
FROZEN_MAN = os.path.join(REPO, "reports/breakthrough_20261002/batch35/frozen_assets.json")
FLANK = 15; VIN = 219

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def sha256_str(s): return hashlib.sha256(s.encode()).hexdigest()

def load_fasta_gz(path):
    seqs = {}
    with gzip.open(path, "rt") as f:
        cur = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                cur = line[1:]; seqs[cur] = ""
            elif cur is not None:
                seqs[cur] += line
    return seqs

def load_tsv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def base_ref_header(oid):
    parts = oid.split("_")
    for i, tok in enumerate(parts):
        if tok.startswith("chr"):
            return "_".join(parts[:i + 2]) + "_referenceseq"
    raise ValueError("no chr token: " + oid)

def intervals_group(positions, half=VIN // 2):
    iv = sorted((p - half, p + half) for p in positions)
    groups = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            groups.append((cs, ce)); cs, ce = s, e
    groups.append((cs, ce))
    return groups

def gap_clusters(positions, gap=2000):
    ps = sorted(positions); n = 1
    for a, b in zip(ps, ps[1:]):
        if b - a > gap: n += 1
    return n

def build_pairs(rows, arm, design):
    out = []
    for r in rows:
        oid = r["original_id"]; ah = r["design_header"]
        alt_full = design[ah]
        rh = base_ref_header(oid); ref_full = design[rh]
        alt_ins = alt_full[FLANK:len(alt_full) - FLANK]
        ref_ins = ref_full[FLANK:len(ref_full) - FLANK]
        dele = r["deletion"] == "1"
        if len(alt_full) == len(ref_full):
            d = [i for i in range(len(alt_full)) if alt_full[i] != ref_full[i]]
            if len(d) == 0:
                status = "DEGENERATE_ALT_EQ_REF"
                reason = "author alt oligo identical to reference oligo (no encoded variant)"
            elif len(d) == 1 and not dele:
                status = "VALID_SNV"
                reason = "single substitution in matched reference window"
            else:
                status = "UNRESOLVED_GEOMETRY"; reason = "n_diff=%d" % len(d)
        else:
            k = len(ref_ins) - len(alt_ins)
            ok = False; off = None
            if k > 0:
                for o in range(0, len(ref_ins) - k + 1):
                    if ref_ins[:o] + ref_ins[o + k:] == alt_ins:
                        ok = True; off = o; break
            if ok:
                status = "VALID_DEL"
                reason = "centred %dbp deletion of variable insert at offset %d" % (k, off)
            else:
                status = "UNRESOLVED_GEOMETRY"; reason = "deletion geometry not matched"
        lab = r["logFC"].strip()
        try:
            labv = float(lab); lab_missing = False
        except Exception:
            labv = None; lab_missing = True
        out.append({
            "arm": arm, "cell": r["cell"], "original_id": oid, "chr": r["chr"], "pos": int(r["pos"]),
            "variant_class": "deletion" if dele else "SNV",
            "alt_header": ah, "ref_header": rh,
            "base_ref": r["base_ref"], "base_alt": r["base_alt"], "orientation": r["orientation"],
            "label_logFC": labv, "label_missing": lab_missing, "label_adjP": r["adjP_Value"],
            "design_len_alt": len(alt_full), "design_len_ref": len(ref_full),
            "insert_len_alt": len(alt_ins), "insert_len_ref": len(ref_ins),
            "alt_insert": alt_ins, "ref_insert": ref_ins,
            "design_sha_alt": sha256_str(alt_full), "design_sha_ref": sha256_str(ref_full),
            "pair_status": status, "pair_reason": reason,
            "alt_seq_id": sha256_str(alt_ins)[:20], "ref_seq_id": sha256_str(ref_ins)[:20],
        })
    return out

def add_groups(pairs):
    positions = sorted({p["pos"] for p in pairs})
    groups = intervals_group(positions)
    gid = {}
    for gi, (s, e) in enumerate(groups):
        for p in pairs:
            if s <= p["pos"] <= e: gid[p["original_id"]] = gi
    for p in pairs: p["group_id"] = gid[p["original_id"]]
    return len(groups)

def arm_audit(pairs, n_groups):
    a = Counter(p["pair_status"] for p in pairs)
    n = len(pairs)
    n_missing = sum(1 for p in pairs if p["label_missing"])
    valid = a.get("VALID_SNV", 0) + a.get("VALID_DEL", 0)
    return {"n_rows": n, "pair_status_counts": dict(a), "valid_pairs": valid,
            "valid_fraction": round(valid / n, 4),
            "degenerate": a.get("DEGENERATE_ALT_EQ_REF", 0),
            "unresolved": a.get("UNRESOLVED_GEOMETRY", 0),
            "label_missing": n_missing, "label_finite_for_metric": n - n_missing,
            "n_distinct_positions": len({p["pos"] for p in pairs}),
            "n_groups_interval_union": n_groups}

def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True); os.makedirs(RPT, exist_ok=True)
    created = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    with open(FROZEN_MAN, encoding="utf-8") as f: fman = json.load(f)
    weights_sha = {}
    for rel, exp in fman["weights_sha256"].items():
        got = sha256_file(os.path.join(ASSET, "src", rel))
        weights_sha[rel] = {"sha256": got, "matches_frozen": got == exp}
    code_sha = {}
    for rel in ["parm_pkg/PARM/PARM_predict.py", "parm_pkg/PARM/PARM_utils_load_model.py",
                "parm_pkg/PARM/PARM_misc.py", "parm_pkg/PARM/version.py",
                "parm_pkg/PARM/PARM_utils_data_loader.py", "parm_pkg/PARM/PARM_mutagenesis.py"]:
        code_sha[rel] = sha256_file(os.path.join(ASSET, rel))
    design_sha_now = sha256_file(DESIGN)
    assert design_sha_now == DESIGN_SHA, "design FASTA SHA mismatch " + design_sha_now

    design = load_fasta_gz(DESIGN)
    sat = load_tsv(os.path.join(B36, "cohort_SAT_HEK293FT.tsv"))
    kv = load_tsv(os.path.join(B36, "cohort_KV_SNV_HEK293FT.tsv"))

    sat_pairs = build_pairs(sat, "SAT", design)
    kv_pairs = build_pairs(kv, "KV", design)
    sat_groups = add_groups(sat_pairs)
    kv_groups = add_groups(kv_pairs)
    kv_positions = sorted({p["pos"] for p in kv_pairs})
    kv_gap_clusters = gap_clusters(kv_positions)
    sat_gap_clusters = gap_clusters(sorted({p["pos"] for p in sat_pairs}))

    uniq = {}
    for p in sat_pairs + kv_pairs:
        for role, sid in (("alt", p["alt_seq_id"]), ("ref", p["ref_seq_id"])):
            seq = p["alt_insert"] if role == "alt" else p["ref_insert"]
            if sid in uniq:
                assert uniq[sid]["sequence"] == seq
                uniq[sid]["n_uses"] += 1; uniq[sid]["roles"].add(role)
            else:
                uniq[sid] = {"seq_id": sid, "sequence": seq, "length": len(seq), "n_uses": 1, "roles": set([role])}
    n_unique = len(uniq)
    len_dist = Counter(v["length"] for v in uniq.values())

    sat_audit = arm_audit(sat_pairs, sat_groups)
    kv_audit = arm_audit(kv_pairs, kv_groups)

    cols = ["arm", "cell", "original_id", "chr", "pos", "variant_class", "base_ref", "base_alt",
            "orientation", "alt_header", "ref_header", "pair_status", "pair_reason", "group_id",
            "label_logFC", "label_missing", "label_adjP",
            "design_len_alt", "design_len_ref", "insert_len_alt", "insert_len_ref",
            "alt_insert", "ref_insert", "design_sha_alt", "design_sha_ref"]
    paired_path = os.path.join(OUTDIR, "paired_inputs.tsv")
    with open(paired_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for p in sat_pairs + kv_pairs: w.writerow(p)

    uniq_fa = os.path.join(OUTDIR, "unique_sequences.fasta")
    with open(uniq_fa, "w", encoding="utf-8") as f:
        for sid in sorted(uniq):
            f.write(">%s\n%s\n" % (sid, uniq[sid]["sequence"]))

    map_path = os.path.join(OUTDIR, "pair_sequence_ids.tsv")
    with open(map_path, "w", encoding="utf-8", newline="") as f:
        f.write("arm\toriginal_id\trole\tseq_id\tpos\tpair_status\tgroup_id\tvariant_class\tlabel_logFC\tlabel_missing\n")
        for p in sat_pairs + kv_pairs:
            lab = "" if p["label_logFC"] is None else p["label_logFC"]
            f.write("%s\t%s\talt\t%s\t%d\t%s\t%d\t%s\t%s\t%d\n" % (
                p["arm"], p["original_id"], p["alt_seq_id"], p["pos"], p["pair_status"],
                p["group_id"], p["variant_class"], lab, int(p["label_missing"])))
            f.write("%s\t%s\tref\t%s\t%d\t%s\t%d\t%s\t%s\t%d\n" % (
                p["arm"], p["original_id"], p["ref_seq_id"], p["pos"], p["pair_status"],
                p["group_id"], p["variant_class"], lab, int(p["label_missing"])))

    thresholds = {
        "KV_rho_warrants_separate_path": 0.20,
        "group_bootstrap_lower_must_exceed": 0.0,
        "min_effective_groups_for_bootstrap": 30,
        "per_region_min_pairs": 10,
        "SAT_secondary_abs_logFC_strata": [0.1],
        "SAT_secondary_adjP_threshold": 0.05,
        "max_unique_sequences": 30000,
        "max_figures": 3,
        "frozen": True, "retrochange_allowed": False,
    }
    metric_plan = {
        "model_output": "per-sequence raw PARM HEK293 score; equal mean of 5 official HEK293 folds",
        "predicted_effect_d": "mean_score(alt_insert) - mean_score(ref_insert); fixed before any label",
        "true_effect": "author bcalm logFC (variant vs reference control); log base NOT assumed",
        "forbidden": ["log of possibly-negative output", "linear calibration",
                      "MAE/R2 across incompatible units", "choosing best fold",
                      "flipping globally-negative correlation", "setting undefined rho to 0",
                      "outcome-chosen RC", "subsetting by significance or label"],
        "primary": {
            "KV_SNV": {
                "pooled_spearman": "Spearman(d, logFC); NaN-label rows excluded from numeric metric after row-count disclosure",
                "per_region_spearman": "where >=10 pairs and non-constant",
                "group_balanced_mean_strict_sign_accuracy": True,
                "group_bootstrap": "paired group bootstrap 2000 with multiplicity retained, only if >=30 effective groups and deltaGC baseline finite; else descriptive",
            },
            "SAT": {
                "descriptive_only": ["all_finite_rho", "SNV_rho", "deletion_rho", "strict_sign_accuracy"],
                "no_generalization_ci_or_pvalues": True,
            },
        },
        "secondary": {"SAT_allrows_p_and_abs_effect_strata": "descriptive interpretation only; never replaces primary"},
        "baselines_zero_fit": {
            "delta_gc_fraction": "GC(alt_insert) - GC(ref_insert) on the SAME paired inserts",
            "delta_length": "len(alt_insert) - len(ref_insert)",
            "constant_predictor_rho": "undefined",
            "direction_baselines": "fixed always-positive and always-negative sign rules; model strict sign accuracy with predicted ties counted wrong; true-zero effects excluded from sign denominator, all other finite effects retained",
        },
        "decision_rule_operational_only": {
            "warrants_separate_variant_path_if": ["KV pooled rho >= 0.20",
                "valid group-bootstrap lower bound > 0",
                "macro strict sign accuracy > both fixed sign baselines", "no audit failure"],
            "SAT_cannot_rescue_KV": True,
            "if_groups_lt_30_or_CI_undefined": "verdict LIMITED, not PASS",
            "no_model_promotion": True,
            "old_HEK_MAE_gate_and_local_D28_fail_unchanged": True,
        },
    }

    man = {
        "batch": BATCH, "version": VERSION, "generated_at": created, "phase": "manifest",
        "protocol": "runs/breakthrough_20261002/batch37/PROTOCOL.txt",
        "parent_acceptance": "runs/breakthrough_20261002/batch37/PARENT_ACCEPTANCE.txt",
        "target_requested_cell": "293FT", "model_source_cell": "HEK293",
        "pinned_model": {
            "name": "PARM", "repo": "https://github.com/vansteensellab/PARM",
            "commit": "260a27082942060364c1c4e2393018a0fa128844", "parm_version": "0.2.6",
            "asset_root": os.path.relpath(ASSET, REPO).replace("\\", "/"),
            "weights_sha256": weights_sha, "code_sha256": code_sha,
            "folds": 5, "aggregation": "equal mean of 5 official HEK293 folds",
            "input_encoding": "official bare FASTA; official central zero-pad to L_max=600; filter_size=125; type_loss=poisson",
        },
        "input_construction": {
            "synthesized_design_bp": 249, "deletion_design_bp": 244,
            "flank_left_bp": FLANK, "flank_right_bp": FLANK,
            "variable_insert_bp": VIN, "deletion_insert_bp": 214,
            "rule": "variable insert = design_seq[15:len-15]; constant flanks stripped; variant centered",
            "no_rc_ensemble": True, "no_tiling": True, "no_length_clipping": True, "no_flank_fill": True,
            "orientation": "single author-supplied sequence only; source orientation field UNKNOWN; no outcome-chosen RC",
        },
        "source_corroboration_preflight": {
            "status": "CORROBORATED_FROM_PRIMARY_METHODS_TEXT",
            "source": "PMC13060879 Europe PMC full XML (local copy)",
            "quotes": [
                "MAPT Promoter Mutagenesis: 219 bp oligos were designed for a 2,000 bp region surrounding and including the promoter and ordered from Twist Biosciences.",
                "For every base in this region, an oligo was designed where the reference sequence is centered on the position of interest, plus 3 alternate-base oligos and a centered 5 bp deletion.",
                "For SNVs, 219 bp oligos were designed with the variant centered for both the reference sequence and the alternate allele.",
                "The variant sequences were the length of 219 or 218 bp minus the number of bases being deleted.",
                "Oligos were inserted into pLS-SceI following the published protocol.",
            ],
            "empirical": "30 constant columns across the author 2000 reference oligos (15+15 flank); SAT SNV ref-vs-alt single-base mismatch at design offset 124; KV SNV alt-vs-ref mismatch at design offset 125 with alt base equal to header alt allele.",
            "distinction": "249 bp synthesized design = 15+15 constant flank + 219 bp measured variable insert; the measured variable insert is scored; exact cloning/vector role beyond pLS-SceI is UNKNOWN (a fixed vector does NOT prove non-autonomy).",
            "network_requests_used": 0, "budget_requests_allowed": 3, "budget_MB_allowed": 10,
        },
        "arms": {
            "SAT_HEK293FT": {"n_rows": len(sat_pairs), "audit": sat_audit, "strata": "6000 SNV + 2000 deletion kept separate"},
            "KV_SNV_HEK293FT": {"n_rows": len(kv_pairs), "audit": kv_audit,
                                "groups_2kb_gap_clusters": kv_gap_clusters,
                                "note": "2kb-gap clusters are EXPLORATORY components, not verified independent CRE families"},
        },
        "not_scored_this_batch": ["KV InDel (HEK293FT)", "neurons (SAT/KV)", "local 14 constructs", "HEK 673"],
        "design_fasta": {"path": os.path.relpath(DESIGN, REPO).replace("\\", "/"),
                         "sha256": design_sha_now, "n_records": len(design)},
        "cohort_sources": {
            "SAT_HEK293FT": sha256_file(os.path.join(B36, "cohort_SAT_HEK293FT.tsv")),
            "KV_SNV_HEK293FT": sha256_file(os.path.join(B36, "cohort_KV_SNV_HEK293FT.tsv")),
        },
        "unique_sequences": {"n_unique": n_unique,
                             "length_distribution": {str(k): v for k, v in len_dist.items()},
                             "max_allowed": thresholds["max_unique_sequences"],
                             "within_budget": n_unique <= thresholds["max_unique_sequences"]},
        "metric_plan": metric_plan, "thresholds_frozen": thresholds,
        "license": "PARM local academic evaluation only; weights/source NOT redistributed.",
        "gpu": "none in this step (CPU assembly only)",
        "outputs": {"paired_inputs.tsv": sha256_file(paired_path),
                    "unique_sequences.fasta": sha256_file(uniq_fa),
                    "pair_sequence_ids.tsv": sha256_file(map_path)},
    }
    man_path = os.path.join(RPT, "diagnostic_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": "OK", "manifest": man_path, "n_unique": n_unique,
                      "sat_valid_frac": sat_audit["valid_fraction"],
                      "kv_valid_frac": kv_audit["valid_fraction"],
                      "kv_gap_clusters": kv_gap_clusters, "kv_groups": kv_groups,
                      "elapsed_s": round(time.time() - t0, 2)}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
