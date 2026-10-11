"""
Build motif PWM manifest v1 for promoter_atlas.

This script performs four tasks:

1. Download JASPAR CORE vertebrates non-redundant PFMs in JASPAR format.
2. Parse the downloaded batch file into a local motif index.
3. Build an initial motif PWM manifest for motif_panel_v1.
4. Write motif scanning rules documentation.

Notes
-----
1. This step is conservative by design.
2. Only exact unique TF-name matches are auto-assigned.
3. Ambiguous or family-level motif names remain pending_manual_review.
4. All paths are relative to the project root.
"""

from pathlib import Path
import urllib.request
import pandas as pd


ROOT = Path(".")

JASPAR_URL = "https://jaspar.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt"

JASPAR_DIR = ROOT / "data/external/jaspar"
JASPAR_TXT = JASPAR_DIR / "JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
JASPAR_INDEX = JASPAR_DIR / "JASPAR2026_CORE_vertebrates_non-redundant_index.tsv"

MOTIF_PANEL = ROOT / "features/motif/motif_panel_v1.tsv"
PWM_MANIFEST = ROOT / "features/motif/motif_pwm_manifest_v1.tsv"
SCAN_RULE_DOC = ROOT / "docs/notes/motif_scanning_rules_v1.md"


def ensure_dirs():
    JASPAR_DIR.mkdir(parents=True, exist_ok=True)
    PWM_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SCAN_RULE_DOC.parent.mkdir(parents=True, exist_ok=True)


def download_jaspar_txt():
    if JASPAR_TXT.exists():
        print(f"download skipped, file exists: {JASPAR_TXT}")
        return

    print(f"downloading JASPAR batch file to: {JASPAR_TXT}")
    urllib.request.urlretrieve(JASPAR_URL, JASPAR_TXT)
    print("download finished")


def parse_jaspar_batch(path):
    records = []
    header = None
    matrix_lines = []

    def finalize_record(current_header, current_lines):
        if current_header is None:
            return None
        parts = current_header.split(maxsplit=1)
        matrix_id = parts[0].strip()
        tf_name = parts[1].strip() if len(parts) > 1 else matrix_id

        rows = {}
        for line in current_lines:
            line = line.strip()
            if not line:
                continue
            base = line[0]
            if base not in {"A", "C", "G", "T"}:
                continue
            left = line.find("[")
            right = line.find("]")
            nums = line[left + 1:right].strip().split()
            rows[base] = [float(x) for x in nums]

        motif_length = None
        if rows:
            motif_length = len(next(iter(rows.values())))

        return {
            "matrix_id": matrix_id,
            "tf_name": tf_name,
            "motif_length": motif_length,
            "raw_header": current_header,
        }

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                rec = finalize_record(header, matrix_lines)
                if rec is not None:
                    records.append(rec)
                header = line[1:].strip()
                matrix_lines = []
            else:
                matrix_lines.append(line)

    rec = finalize_record(header, matrix_lines)
    if rec is not None:
        records.append(rec)

    if len(records) == 0:
        raise ValueError("No JASPAR records parsed from downloaded batch file")

    df = pd.DataFrame(records)
    return df


def build_index():
    jaspar_df = parse_jaspar_batch(JASPAR_TXT)
    jaspar_df = jaspar_df.sort_values(["tf_name", "matrix_id"]).reset_index(drop=True)
    jaspar_df.to_csv(JASPAR_INDEX, sep="\t", index=False)
    return jaspar_df


def build_pwm_manifest(panel_df, jaspar_df):
    panel_df = panel_df.copy()
    jaspar_df = jaspar_df.copy()

    panel_df["motif_id_upper"] = panel_df["motif_id"].astype(str).str.upper()
    jaspar_df["tf_name_upper"] = jaspar_df["tf_name"].astype(str).str.upper()

    tf_name_counts = jaspar_df["tf_name_upper"].value_counts().to_dict()

    rows = []

    for row in panel_df.itertuples(index=False):
        motif_id = row.motif_id
        motif_upper = str(motif_id).upper()

        exact_matches = jaspar_df.loc[jaspar_df["tf_name_upper"] == motif_upper].copy()

        if len(exact_matches) == 1:
            match_status = "exact_unique"
            jaspar_matrix_id = exact_matches.iloc[0]["matrix_id"]
            jaspar_tf_name = exact_matches.iloc[0]["tf_name"]
            motif_length = exact_matches.iloc[0]["motif_length"]
        elif len(exact_matches) > 1:
            match_status = "exact_ambiguous"
            jaspar_matrix_id = ""
            jaspar_tf_name = ""
            motif_length = ""
        else:
            match_status = "pending_manual_review"
            jaspar_matrix_id = ""
            jaspar_tf_name = ""
            motif_length = ""

        rows.append(
            {
                "motif_id": motif_id,
                "tf_family": row.tf_family,
                "tier": row.tier,
                "design_relevance": row.design_relevance,
                "source_db": "JASPAR",
                "source_release": "2026_CORE_vertebrates_non_redundant",
                "jaspar_matrix_id": jaspar_matrix_id,
                "jaspar_tf_name": jaspar_tf_name,
                "motif_length": motif_length,
                "match_status": match_status,
                "threshold_loose_type": "relative_score",
                "threshold_loose_value": 0.80,
                "threshold_strict_type": "relative_score",
                "threshold_strict_value": 0.90,
                "score_norm_rule": "observed_score_over_theoretical_max",
                "reverse_complement_scanned": True,
                "notes": row.notes,
            }
        )

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(PWM_MANIFEST, sep="\t", index=False)
    return manifest_df


def write_scanning_rules_doc():
    text = """# Motif scanning rules v1

## Purpose

This document freezes the first motif PWM sourcing and scanning rules for promoter_atlas.

## External PWM source

- source database: JASPAR
- release used in this step: 2026 CORE vertebrates non-redundant
- matrix format: JASPAR batch text
- local raw file:
  - `data/external/jaspar/JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt`

## Matching policy

`motif_panel_v1.tsv` contains project-level motif identifiers.
These identifiers are not always one-to-one with a unique matrix in JASPAR.

Therefore v1 uses a conservative manifest build policy:

- exact unique TF-name match:
  - auto-assign matrix ID
- exact ambiguous TF-name match:
  - keep unresolved
- no exact match:
  - keep unresolved

Unresolved rows are marked as:

- `pending_manual_review`

## Threshold policy

For v1 motif scanning, both loose and strict thresholds are reserved:

- loose threshold:
  - type: relative_score
  - value: 0.80
- strict threshold:
  - type: relative_score
  - value: 0.90

These are initial defaults and may be revised after pilot scan QC.

## Score normalization

Planned normalized score rule:

- observed_score / theoretical_max_score

## Strand policy

- scan forward strand
- scan reverse complement strand
- keep best hit and count features after strand-aware scanning

## Current scope

This step does not scan promoters yet.
It only freezes the PWM source manifest and scanning defaults.

## Planned next outputs

After manual review and matrix freeze, later steps will build:

- `features/motif/promoter_motif_hits.tsv`
- `features/motif/promoter_motif_agg.tsv`
- `features/motif/promoter_motif_grammar.tsv`
"""
    with open(SCAN_RULE_DOC, "w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    ensure_dirs()

    if not MOTIF_PANEL.exists():
        raise FileNotFoundError(f"Missing motif panel file: {MOTIF_PANEL}")

    download_jaspar_txt()
    jaspar_df = build_index()

    panel_df = pd.read_csv(MOTIF_PANEL, sep="\t", low_memory=False)
    manifest_df = build_pwm_manifest(panel_df, jaspar_df)
    write_scanning_rules_doc()

    print("motif PWM manifest build finished")
    print(f"jaspar raw: {JASPAR_TXT}")
    print(f"jaspar index: {JASPAR_INDEX}")
    print(f"pwm manifest: {PWM_MANIFEST}")
    print(f"scan rules doc: {SCAN_RULE_DOC}")
    print(f"jaspar_record_count: {len(jaspar_df)}")
    print(f"manifest_row_count: {len(manifest_df)}")
    print("match_status counts:")
    print(manifest_df["match_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
