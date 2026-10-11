"""
Build motif PWM review candidates v1 for promoter_atlas.

This script generates candidate JASPAR matrix matches for unresolved motifs in
`features/motif/motif_pwm_manifest_v1.tsv`.

Outputs
-------
- features/motif/motif_pwm_review_candidates_v1.tsv

Notes
-----
1. This step does not overwrite the formal manifest.
2. It only prepares review candidates for manual freeze.
3. All paths are relative to the project root.
"""

from pathlib import Path
import re
import pandas as pd


ROOT = Path(".")

JASPAR_INDEX = ROOT / "data/external/jaspar/JASPAR2026_CORE_vertebrates_non-redundant_index.tsv"
PWM_MANIFEST = ROOT / "features/motif/motif_pwm_manifest_v1.tsv"
REVIEW_OUT = ROOT / "features/motif/motif_pwm_review_candidates_v1.tsv"


def normalize_token(text):
    text = str(text).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return text


def tokenize_family(text):
    text = str(text).strip().upper()
    chunks = re.split(r"[^A-Z0-9]+", text)
    return [x for x in chunks if x]


def build_candidates(manifest_df, jaspar_df):
    jaspar_df = jaspar_df.copy()
    jaspar_df["tf_name_upper"] = jaspar_df["tf_name"].astype(str).str.upper()
    jaspar_df["tf_name_norm"] = jaspar_df["tf_name"].map(normalize_token)

    rows = []

    unresolved = manifest_df.loc[
        manifest_df["match_status"].isin(["pending_manual_review", "exact_ambiguous"])
    ].copy()

    for m in unresolved.itertuples(index=False):
        motif_id = str(m.motif_id)
        motif_norm = normalize_token(motif_id)
        family_tokens = tokenize_family(m.tf_family)

        exact_norm = jaspar_df.loc[jaspar_df["tf_name_norm"] == motif_norm].copy()
        family_contains = pd.DataFrame(columns=jaspar_df.columns)
        if family_tokens:
            mask = jaspar_df["tf_name_upper"].apply(
                lambda x: any(tok in x for tok in family_tokens)
            )
            family_contains = jaspar_df.loc[mask].copy()

        prefix_like = jaspar_df.loc[
            jaspar_df["tf_name_norm"].str.startswith(motif_norm) |
            jaspar_df["tf_name_norm"].str.contains(motif_norm, regex=False)
        ].copy()

        candidate_frames = [
            ("exact_norm", exact_norm),
            ("family_contains", family_contains),
            ("name_contains", prefix_like),
        ]

        seen = set()
        rank = 0

        for strategy, frame in candidate_frames:
            frame = frame.sort_values(["tf_name", "matrix_id"]).reset_index(drop=True)
            for row in frame.itertuples(index=False):
                key = row.matrix_id
                if key in seen:
                    continue
                seen.add(key)
                rank += 1
                rows.append(
                    {
                        "motif_id": motif_id,
                        "tf_family": m.tf_family,
                        "current_match_status": m.match_status,
                        "candidate_rank": rank,
                        "candidate_strategy": strategy,
                        "candidate_matrix_id": row.matrix_id,
                        "candidate_tf_name": row.tf_name,
                        "candidate_motif_length": row.motif_length,
                        "candidate_header": row.raw_header,
                    }
                )

        if rank == 0:
            rows.append(
                {
                    "motif_id": motif_id,
                    "tf_family": m.tf_family,
                    "current_match_status": m.match_status,
                    "candidate_rank": 1,
                    "candidate_strategy": "no_candidate_found",
                    "candidate_matrix_id": "",
                    "candidate_tf_name": "",
                    "candidate_motif_length": "",
                    "candidate_header": "",
                }
            )

    return pd.DataFrame(rows)


def main():
    if not JASPAR_INDEX.exists():
        raise FileNotFoundError(f"Missing JASPAR index: {JASPAR_INDEX}")
    if not PWM_MANIFEST.exists():
        raise FileNotFoundError(f"Missing motif PWM manifest: {PWM_MANIFEST}")

    jaspar_df = pd.read_csv(JASPAR_INDEX, sep="\t", low_memory=False)
    manifest_df = pd.read_csv(PWM_MANIFEST, sep="\t", low_memory=False)

    review_df = build_candidates(manifest_df, jaspar_df)
    REVIEW_OUT.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(REVIEW_OUT, sep="\t", index=False)

    print("motif PWM review candidates build finished")
    print(f"review candidates: {REVIEW_OUT}")
    print(f"rows: {len(review_df)}")
    print("motif counts:")
    print(review_df['motif_id'].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
