"""Count actual frozen training input records and distinguish label origins."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / "labels/build/promoter_atlas_v6_input.tsv"
    df = pd.read_csv(source, sep="\t", usecols=["species", "split", "label_mask", "Score_E", "Score_S", "Score_B", "ortholog_E"])
    df["split"] = df["split"].fillna("train")
    rows = []
    for species, group in df.groupby("species", sort=True):
        human = species == "homo_sapiens"
        if not human:
            assert group[["Score_E", "Score_S", "Score_B"]].isna().all().all()
        rows.append({"species": species, "all_records": len(group),
                     **{split + "_records": int(group["split"].eq(split).sum()) for split in ["train", "val", "test"]},
                     **{axis + "_direct_label_records": int(group["Score_" + axis].notna().sum()) for axis in "ESB"},
                     "human_ortholog_E_records": int(group["ortholog_E"].notna().sum()),
                     "label_origin": "human_GTEx_gene_labels" if human else "human_ortholog_E_when_available;no_independent_species_ESB_labels"})
    with source.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    audit = json.loads((ROOT / "reports/rounds/round04_split_audit.json").read_text(encoding="utf-8"))
    assert sha == audit["training_input_sha256"]
    assert len(df) == audit["training_rows"] == 1746138 and len(rows) == 19
    out = ROOT / "reports/rounds/round04_training_sources"
    pd.DataFrame(rows).to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    result = {"source": str(source.relative_to(ROOT)), "source_sha256": sha,
              "source_bytes": source.stat().st_size, "species_count": len(rows),
              "all_records": len(df), "train_records": int(df["split"].eq("train").sum()),
              "unit": "input sequence-pair record, not unique gene or independent expression measurement",
              "MPRA_included_in_these_counts": False,
              "scope": "Human and nonhuman sequence input pool; nonhuman expression-E targets transfer human labels. This is not 19 independently measured species-specific ESB datasets.",
              "species": rows}
    out.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["species_count", "all_records", "train_records", "source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
