#!/usr/bin/env python3
"""Fetch human -> panel-species one2one orthologs via Ensembl BioMart (batched).

Input : labels/build/promoter_model_input_core.tsv (human gene ids with labels)
Output: features/multispecies/ortholog_human_to_species.tsv
        columns: human_gene_id, <species>, <species>_orth_type, <species>_peptide
"""
import time
import urllib.request
import urllib.parse
import pandas as pd
from pathlib import Path

ROOT = Path(r"E:\promoter_atlas")
INPUT = ROOT / "labels" / "build" / "promoter_model_input_core.tsv"
OUT = ROOT / "features" / "multispecies" / "ortholog_human_to_species.tsv"
OUT.parent.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "pan_troglodytes": "ptroglodytes_gene_ensembl",
    "macaca_mulatta": "mmulatta_gene_ensembl",
    "callithrix_jacchus": "cjacchus_gene_ensembl",
    "mus_musculus": "mmusculus_gene_ensembl",
    "rattus_norvegicus": "rnorvegicus_gene_ensembl",
    "cavia_porcellus": "cporcellus_gene_ensembl",
    "oryctolagus_cuniculus": "ocuniculus_gene_ensembl",
    "bos_taurus": "btaurus_gene_ensembl",
    "sus_scrofa": "sscrofa_gene_ensembl",
    "ovis_aries": "oaries_gene_ensembl",
    "capra_hircus": "chircus_gene_ensembl",
    "equus_caballus": "ecaballus_gene_ensembl",
    "canis_lupus_familiaris": "clfamiliaris_gene_ensembl",
    "felis_catus": "fcatus_gene_ensembl",
    "pteropus_vampyrus": "pvampyrus_gene_ensembl",
    "loxodonta_africana": "lafricana_gene_ensembl",
    "monodelphis_domestica": "mdomestica_gene_ensembl",
    "ornithorhynchus_anatinus": "oanatinus_gene_ensembl",
}

MARTSERVICE = "https://www.ensembl.org/biomart/martservice/result"
BATCH = 100
HUMAN_DATASET = "hsapiens_gene_ensembl"

def biomart_query(dataset, gene_ids, attrs):
    xml = '<?xml version="1.0" encoding="UTF-8"?>'
    xml += '<!DOCTYPE Query><Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" datasetConfigVersion="0.6">'
    xml += f'<Dataset name="{dataset}" interface="default">'
    xml += f'<Filter name="ensembl_gene_id" value="{",".join(gene_ids)}"/>'
    for a in attrs:
        xml += f'<Attribute name="{a}"/>'
    xml += "</Dataset></Query>"
    url = MARTSERVICE + "?" + urllib.parse.urlencode({"query": xml})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            print("  retry", attempt, repr(e)[:100], flush=True)
            time.sleep(5)
    return ""

def main():
    core = pd.read_csv(INPUT, sep="\t", usecols=["gene_id"], low_memory=False)
    genes = sorted(set(g.split(".")[0] for g in core["gene_id"].dropna() if isinstance(g, str) and g.startswith("ENSG")))
    print("human genes:", len(genes), flush=True)
    frames = [pd.DataFrame({"human_gene_id": genes})]
    for sp, ds in DATASETS.items():
        prefix = ds.split("_gene_ensembl")[0]
        rows = []
        for i in range(0, len(genes), BATCH):
            chunk = genes[i:i + BATCH]
            txt = biomart_query(HUMAN_DATASET, chunk,
                                ["ensembl_gene_id", f"{prefix}_homolog_ensembl_gene",
                                 f"{prefix}_homolog_orthology_type"])
            for line in txt.strip().split("\n"):
                p = line.split("\t")
                if len(p) == 3 and p[0] and p[1]:
                    rows.append((p[0], p[1], p[2]))
            print(sp, i + len(chunk), "/", len(genes), "pairs", len(rows), flush=True)
            time.sleep(0.5)
        d = pd.DataFrame(rows, columns=["human_gene_id", f"{sp}_gene", f"{sp}_orth"])  # cols named by species key
        d = d[d[f"{sp}_orth"] == "ortholog_one2one"].drop(columns=[f"{sp}_orth"])
        d = d.drop_duplicates(subset="human_gene_id", keep="first")
        frames.append(d)
        print(sp, "one2one:", len(d), flush=True)
        frames[0].to_csv(OUT, sep="\t", index=False)  # partial save
        merged = frames[0]
        for d2 in frames[1:]:
            merged = merged.merge(d2, on="human_gene_id", how="left")
        merged.to_csv(OUT, sep="\t", index=False)
    print("wrote", OUT, flush=True)

if __name__ == "__main__":
    main()
