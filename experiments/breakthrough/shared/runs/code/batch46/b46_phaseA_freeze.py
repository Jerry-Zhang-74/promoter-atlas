# -*- coding: utf-8 -*-
"""batch46 Phase A: metadata-only freeze BEFORE joining published score / activity.
Triplets are built purely from info grammar + query length; no prediction/expression column read.
"""
import sys, json, zlib, hashlib, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b46_common as C


def main():
    raw = C.SRC.read_bytes()
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    dec = d.decompress(raw)
    src_sha_c = hashlib.sha256(raw).hexdigest()
    src_sha_d = hashlib.sha256(dec).hexdigest()
    integrity = {
        "compressed_bytes": len(raw), "decompressed_bytes": len(dec),
        "gzip_eof": bool(d.eof), "gzip_unused_tail_bytes": len(d.unused_data),
        "gzip_crc_ok": bool(d.eof) and len(d.unused_data) == 0,
        "sha256_compressed": src_sha_c, "sha256_decompressed": src_sha_d,
        "compressed_sha256_matches_expected": src_sha_c == C.SRC_SHA256_COMPRESSED,
        "decompressed_sha256_matches_b45": src_sha_d == C.SRC_SHA256_DECOMPRESSED,
        "expected_compressed_sha256": C.SRC_SHA256_COMPRESSED,
    }
    rows = C.parse()
    assert len(rows) == 3556, len(rows)
    b45_freeze = json.loads((C.B45 / "freeze_integrity.json").read_text(encoding="utf-8"))
    b45_ident = json.loads((C.B45 / "identity_metadata.json").read_text(encoding="utf-8"))
    b45_agg = json.loads((C.B45 / "aggregation.json").read_text(encoding="utf-8"))
    uniq_query = {r[C.CI["query.seq"]] for r in rows}
    uniq_info = {r[C.CI["info"]] for r in rows}
    uniq_bc = {r[C.CI["barcode"]] for r in rows}
    b45_identity_ok = (
        b45_ident["n_unique_query_seq"] == len(uniq_query) == 889
        and b45_ident["n_unique_info"] == len(uniq_info) == 889
        and b45_ident["n_unique_barcode"] == len(uniq_bc) == 3556
        and b45_freeze["sha256_compressed"] == src_sha_c
        and b45_agg["n_unique_fragments"] == 889)
    groups = collections.defaultdict(dict)
    other_infos = collections.Counter()
    for r in rows:
        info = r[C.CI["info"]]
        g = C.parse_info(info)
        if g is None or C.category(info) != "design":
            other_infos[C.category(info)] += 1
            continue
        key = (g["lib"], g["rep"], g["gen"])
        rec = {"info": info, "seq": r[C.CI["query.seq"]],
               "len": len(r[C.CI["query.seq"]]), "library": r[C.CI["library"]]}
        if g["sel"] in groups[key] and groups[key][g["sel"]]["info"] != info:
            groups[key].setdefault("__conflict__", []).append(rec)
        else:
            groups[key][g["sel"]] = rec

    triplets = []
    incomplete = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        v = groups[key]
        conflict = v.get("__conflict__", [])
        present = [s for s in C.SELS if s in v]
        if conflict or sorted(present) != ["max", "median", "min"]:
            reason = ("duplicate_selection_label" if conflict
                      else "missing_selection_label:" + ",".join(s for s in C.SELS if s not in v))
            incomplete.append({"key": list(key), "present": sorted(present),
                               "missing": [s for s in C.SELS if s not in v], "reason": reason})
            continue
        seqs = {s: v[s]["seq"] for s in C.SELS}
        lens = {s: v[s]["len"] for s in C.SELS}
        distinct = len(set(seqs.values())) == 3
        same_len = len(set(lens.values())) == 1
        if not distinct or not same_len:
            incomplete.append({"key": list(key), "present": sorted(present),
                               "reason": ("repeated_DNA" if not distinct else "") +
                                         ("+mixed_actual_length" if not same_len else "")})
            continue
        triplets.append({
            "triplet_id": "%s:rep%s:gen%d" % (key[0], key[1], key[2]),
            "library": key[0], "lineage": key[1], "generation": key[2],
            "actual_query_length": lens["max"], "length_uniform": same_len,
            "candidates": {s: {"info": v[s]["info"],
                               "query_sha256": hashlib.sha256(seqs[s].encode()).hexdigest(),
                               "query_length": lens[s], "library_label": v[s]["library"]}
                           for s in C.SELS}})
    code_hashes = {p.name: C.sha256_file(p) for p in sorted(C.CODE.glob("b46_*.py"))}
    C.write_json(C.OUT / "source_identity.json", {
        "batch": C.BATCH, "version": C.VERSION, "phase": "A_source_identity",
        "source_file": "data/breakthrough_20261002/batch44/syn.txt.gz",
        "official_filename": "GSE301246_Synthetic_Promoter_Libraries_Small_and_Large_Data.txt.gz",
        "integrity": integrity, "n_data_rows": len(rows),
        "n_unique_query_DNA": len(uniq_query), "n_unique_info": len(uniq_info),
        "n_unique_barcode": len(uniq_bc),
        "b45_source_and_table_identity_ok": b45_identity_ok,
        "b45_cross_reference": {
            "b45_freeze_sha256_compressed": b45_freeze["sha256_compressed"],
            "b45_aggregation_n_unique_fragments": b45_agg["n_unique_fragments"],
            "b45_identity_n_unique_query_seq": b45_ident["n_unique_query_seq"]},
        "non_design_row_counts": dict(other_infos)})
    C.write_json(C.OUT / "freeze_triplets.json", {
        "batch": C.BATCH, "version": C.VERSION, "phase": "A_freeze_metadata_only",
        "frozen_before_activity_join": True,
        "note": ("Activity-blind metadata construction. Prior b45 labels were already seen; the design "
                 "category and this construction do NOT make the analysis prospective."),
        "eligibility_rule": ("unmodified design construct; group key = (library, lineage R, generation G); "
                             "qualify iff exactly one distinct query DNA per selection label {max,median,min}, "
                             "the three DNA pairwise distinct, and identical actual query length within the group. "
                             "No pooling across generation or lineage; no subset chosen by outcome."),
        "n_eligible_triplets": len(triplets),
        "n_eligible_by_library": dict(collections.Counter(t["library"] for t in triplets)),
        "n_eligible_by_generation": {"%s:gen%d" % (k[0], k[1]): sum(
            1 for t in triplets if t["library"] == k[0] and t["generation"] == k[1])
            for k in sorted({(t["library"], t["generation"]) for t in triplets})},
        "n_incomplete_or_conflict_groups_retained": len(incomplete),
        "incomplete_or_conflict_groups": incomplete,
        "code_sha256_frozen": code_hashes,
        "source_sha256_compressed": src_sha_c,
        "metric_definitions_frozen": {
            "measured_summary": ("per author RNA replicate = arithmetic mean of expression.normalized.log2rpm."
                                 "<cell>.replicate<k> over FINITE barcode cells; assay summary = mean of the two "
                                 "replicate means."),
            "contrast_max_minus_median": "measured_summary(_max) - measured_summary(_median)",
            "contrast_max_minus_min": "measured_summary(_max) - measured_summary(_min)",
            "score_selector": "argmax published <cell>.query.PARM.prediction; tie -> lexicographically smallest info id",
            "gc_selector": "argmax query-DNA GC fraction; tie -> lexicographically smallest info id",
            "uniform_expectation": "mean of the three candidates measured summaries",
            "oracle_best": "max of the three candidates measured summaries",
            "regret": "chosen measured summary - oracle_best (<=0; hindsight only)"},
        "triplets": triplets})
    print("source compressed sha", src_sha_c, "match", integrity["compressed_sha256_matches_expected"])
    print("b45 identity ok", b45_identity_ok)
    print("eligible triplets", len(triplets),
          "by library", dict(collections.Counter(t["library"] for t in triplets)))
    print("incomplete/conflict retained", len(incomplete))
    print("code frozen", list(code_hashes))


if __name__ == "__main__":
    main()
