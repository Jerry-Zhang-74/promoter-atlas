"""Apply the user's CMVcore identity clarification without changing v1 outputs."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

from extract_cellline_constructs import read_snapgene, extract_feature, write_table
from extract_cellline_measurements import sequence_endpoint_tables, file_info

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "reports/cellline_v1_20261001"
OUT = ROOT / "reports/cellline_v2_20261001"


def main():
    OUT.mkdir(exist_ok=True)
    for name in ["observations_raw.tsv", "measurements_ready.tsv", "diagnostic_candidate_summary.tsv", "extraction_validation.json"]:
        shutil.copy2(OLD / name, OUT / name)
    with (OLD / "construct_sequences.tsv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    core = next(row for row in rows if row["construct_id"] == "CMVcore")
    excluded = dict(core, construct_id="CMV_SP1_81_untransfected", sequence_status="user_confirmed_not_transfected")
    write_table(OUT / "excluded_untransfected_fragment.tsv", [excluded])
    source = Path("E:/BIO/promoter_truncation/plasmid/pool_1_SCP/AP77-CMV - complete.dna")
    molecule, features, sha = read_snapgene(source)
    chosen = [feature for feature in features if feature["name"] == "CMV promoter"]
    assert len(chosen) == 1
    sequence = extract_feature(molecule, chosen[0])
    full = next(row for row in rows if row["construct_id"] == "CMVfull")["sequence"]
    assert len(sequence) == 204 and full.endswith(sequence)
    core.update(sequence_length_bp=204, sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                source_path=str(source), source_sha256=sha, source_molecule_length_bp=len(molecule),
                feature_names="CMV promoter", feature_ranges_1based_inclusive=";".join(chosen[0]["segments_1based_inclusive"]),
                feature_directionality=chosen[0]["directionality"],
                mapping_evidence="user_clarified_CMVcore_without_enhancer;CMV_promoter_feature_204bp;historical_length_label_200bp",
                eligible_exploratory_sequence_fit=1, sequence_status="user_clarified_core_promoter_feature", sequence=sequence)
    write_table(OUT / "construct_sequences.tsv", rows)
    (OUT / "construct_sequences.fasta").write_text("".join(f'>{row["construct_id"]}\n{row["sequence"]}\n' for row in rows), encoding="ascii")
    provenance = json.loads((OLD / "construct_sequence_provenance.json").read_text(encoding="utf-8"))
    provenance["revision"] = {"version": "v2", "supersedes": str(OLD),
        "user_clarification": "CMVcore is CMV promoter without enhancer; the 81 bp literature fragment was not transfected.",
        "selected_feature": "CMV promoter", "actual_length_bp": 204, "historical_documented_length_bp": 200,
        "coordinates_1based_inclusive": "667-870", "assay_clone_sequence_verified": False,
        "comparison_rule": "same frozen encoder and model protocols as v1; only corrected CMVcore sequence/endpoint is added"}
    provenance["limitations"] = [value for value in provenance["limitations"] if not value.startswith("The AP77-CMV candidate")]
    provenance["limitations"].append("User clarified control identity; exact design feature is 204 bp, while historical documents call it 200 bp. No clone sequencing evidence was supplied.")
    (OUT / "construct_sequence_provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    inventory = json.loads((OLD / "source_inventory.json").read_text(encoding="utf-8"))
    inventory["construct_endpoint_link"] = sequence_endpoint_tables(OUT)
    assert inventory["construct_endpoint_link"]["D28_construct_endpoint_rows"] == 14
    inventory["missing_metadata"][0]["current_resolution"] = "CMVcore identity clarified by user: CMV promoter feature without enhancer (204 bp); 81 bp literature fragment was never transfected. Assay-to-plasmid barcode/clone sequencing remains unverified."
    inventory["revision_provenance"] = provenance["revision"]
    inventory["outputs"] = {name: file_info(OUT / name) for name in ["observations_raw.tsv", "measurements_ready.tsv", "diagnostic_candidate_summary.tsv", "construct_endpoints_D28.tsv", "construct_time_diagnostics.tsv"]}
    (OUT / "source_inventory.json").write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OLD / "construct_endpoints_D28.tsv").open(encoding="utf-8-sig",newline="") as handle:
        prior = {r["construct_id"]:r for r in csv.DictReader(handle,delimiter="\t")}
    with (OUT / "construct_endpoints_D28.tsv").open(encoding="utf-8-sig",newline="") as handle:
        current = {r["construct_id"]:r for r in csv.DictReader(handle,delimiter="\t")}
    assert set(current) - set(prior) == {"CMVcore"}
    for name,row in prior.items():
        assert current[name] == row
    check = {"status":"passed","unchanged_prior_endpoint_rows":len(prior),"new_endpoint":current["CMVcore"],
             "core_is_exact_CMVfull_suffix":True,"no_original_source_modified":True,"prior_version_preserved":True}
    (OUT / "revision_validation.json").write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Prepared 14 constructs; CMVcore 204 bp; previous 13 endpoints unchanged.")


if __name__ == "__main__":
    main()
