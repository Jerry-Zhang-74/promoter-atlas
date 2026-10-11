"""
Freeze Phase 6 model decision for promoter_atlas.

Purpose
-------
Create a formal Phase 6 model registry and decision note using the
current best validated model artifacts.

Outputs
-------
- reports/model_registry_phase6.tsv
- docs/notes/phase6_model_decision_v1.md

Rules
-----
- relative paths only
- no in-place patching of existing artifacts
"""

from pathlib import Path
import pandas as pd


ROOT = Path(".")

MODEL_SPECS = [
    {
        "model_name": "promoter_model_v1",
        "route": "sequence_baseline",
        "checkpoint": "models/checkpoints/promoter_model_v1.pt",
        "eval": "models/eval/promoter_model_v1_metrics.tsv",
        "status": "official_baseline",
        "selection_rule": "best_existing_checkpoint",
        "notes": "Current official sequence-only baseline.",
    },
    {
        "model_name": "promoter_model_v4_arch_only_pilot",
        "route": "architecture_route",
        "checkpoint": "models/checkpoints/promoter_model_v4_arch_only_pilot.pt",
        "eval": "models/eval/promoter_model_v4_arch_only_pilot_metrics.tsv",
        "status": "best_architecture_pilot",
        "selection_rule": "best_test_row_in_eval",
        "notes": "Current best architecture-route pilot.",
    },
    {
        "model_name": "promoter_model_v4_arch2_reduced_pilot",
        "route": "architecture_route",
        "checkpoint": "models/checkpoints/promoter_model_v4_arch2_reduced_pilot.pt",
        "eval": "models/eval/promoter_model_v4_arch2_reduced_pilot_metrics.tsv",
        "status": "arch_route_reference",
        "selection_rule": "best_summary_or_best_test_row",
        "notes": "Did not exceed v4_arch_only_pilot on Score_S.",
    },
]

REGISTRY_OUT = ROOT / "reports/model_registry_phase6.tsv"
DECISION_NOTE_OUT = ROOT / "docs/notes/phase6_model_decision_v1.md"


def read_best_metrics(eval_path: Path):
    if not eval_path.exists():
        return {
            "best_source": "missing_eval",
            "Score_E_r2": None,
            "Score_S_r2": None,
            "Score_B_r2": None,
            "predEScorr": None,
        }

    df = pd.read_csv(eval_path, sep="\t", low_memory=False)

    if "epoch" not in df.columns:
        return {
            "best_source": "eval_without_epoch",
            "Score_E_r2": None,
            "Score_S_r2": None,
            "Score_B_r2": None,
            "predEScorr": None,
        }

    best_test = df.loc[df["epoch"].astype(str) == "best_test"].copy()
    if len(best_test) > 0:
        row = best_test.iloc[0]
        return {
            "best_source": "best_test",
            "Score_E_r2": row.get("Score_E_r2"),
            "Score_S_r2": row.get("Score_S_r2"),
            "Score_B_r2": row.get("Score_B_r2"),
            "predEScorr": row.get("pred_E_pred_S_corr_defined"),
        }

    numeric_epoch = df[pd.to_numeric(df["epoch"], errors="coerce").notna()].copy()
    if len(numeric_epoch) == 0:
        return {
            "best_source": "no_numeric_epoch",
            "Score_E_r2": None,
            "Score_S_r2": None,
            "Score_B_r2": None,
            "predEScorr": None,
        }

    if "val_mean_mse" in numeric_epoch.columns:
        numeric_epoch = numeric_epoch.sort_values("val_mean_mse", ascending=True)
        row = numeric_epoch.iloc[0]
        return {
            "best_source": "lowest_val_mean_mse_epoch",
            "Score_E_r2": row.get("Score_E_r2"),
            "Score_S_r2": row.get("Score_S_r2"),
            "Score_B_r2": row.get("Score_B_r2"),
            "predEScorr": row.get("pred_E_pred_S_corr_defined"),
        }

    row = numeric_epoch.iloc[0]
    return {
        "best_source": "first_numeric_epoch",
        "Score_E_r2": row.get("Score_E_r2"),
        "Score_S_r2": row.get("Score_S_r2"),
        "Score_B_r2": row.get("Score_B_r2"),
        "predEScorr": row.get("pred_E_pred_S_corr_defined"),
    }


def fmt_float(x):
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.4f}"


def build_registry():
    rows = []
    for spec in MODEL_SPECS:
        ckpt = ROOT / spec["checkpoint"]
        eval_path = ROOT / spec["eval"]
        metrics = read_best_metrics(eval_path)

        rows.append(
            {
                "model_name": spec["model_name"],
                "route": spec["route"],
                "status": spec["status"],
                "checkpoint_path": spec["checkpoint"],
                "checkpoint_exists": ckpt.exists(),
                "eval_path": spec["eval"],
                "eval_exists": eval_path.exists(),
                "selection_rule": spec["selection_rule"],
                "best_metric_source": metrics["best_source"],
                "Score_E_r2": metrics["Score_E_r2"],
                "Score_S_r2": metrics["Score_S_r2"],
                "Score_B_r2": metrics["Score_B_r2"],
                "predEScorr": metrics["predEScorr"],
                "notes": spec["notes"],
            }
        )

    df = pd.DataFrame(rows)

    REGISTRY_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REGISTRY_OUT, sep="\t", index=False)

    return df


def write_decision_note(registry_df: pd.DataFrame):
    v1 = registry_df.loc[registry_df["model_name"] == "promoter_model_v1"].iloc[0]
    v4_arch_only = registry_df.loc[
        registry_df["model_name"] == "promoter_model_v4_arch_only_pilot"
    ].iloc[0]
    v4_arch2_reduced = registry_df.loc[
        registry_df["model_name"] == "promoter_model_v4_arch2_reduced_pilot"
    ].iloc[0]

    lines = []
    lines.append("# Phase 6 model decision v1")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Freeze the current model decision after the Phase 6 pilot series.")
    lines.append("")
    lines.append("## Current official baseline")
    lines.append("")
    lines.append(f"- model: {v1['model_name']}")
    lines.append(f"- route: {v1['route']}")
    lines.append(f"- checkpoint: {v1['checkpoint_path']}")
    lines.append(f"- Score_E_r2: {fmt_float(v1['Score_E_r2'])}")
    lines.append(f"- Score_S_r2: {fmt_float(v1['Score_S_r2'])}")
    lines.append(f"- Score_B_r2: {fmt_float(v1['Score_B_r2'])}")
    lines.append(f"- predEScorr: {fmt_float(v1['predEScorr'])}")
    lines.append("")
    lines.append("## Current best architecture-route pilot")
    lines.append("")
    lines.append(f"- model: {v4_arch_only['model_name']}")
    lines.append(f"- checkpoint: {v4_arch_only['checkpoint_path']}")
    lines.append(f"- Score_E_r2: {fmt_float(v4_arch_only['Score_E_r2'])}")
    lines.append(f"- Score_S_r2: {fmt_float(v4_arch_only['Score_S_r2'])}")
    lines.append(f"- Score_B_r2: {fmt_float(v4_arch_only['Score_B_r2'])}")
    lines.append(f"- predEScorr: {fmt_float(v4_arch_only['predEScorr'])}")
    lines.append("")
    lines.append("## Architecture reduction reference")
    lines.append("")
    lines.append(f"- model: {v4_arch2_reduced['model_name']}")
    lines.append(f"- checkpoint: {v4_arch2_reduced['checkpoint_path']}")
    lines.append(f"- Score_E_r2: {fmt_float(v4_arch2_reduced['Score_E_r2'])}")
    lines.append(f"- Score_S_r2: {fmt_float(v4_arch2_reduced['Score_S_r2'])}")
    lines.append(f"- Score_B_r2: {fmt_float(v4_arch2_reduced['Score_B_r2'])}")
    lines.append(f"- predEScorr: {fmt_float(v4_arch2_reduced['predEScorr'])}")
    lines.append("")
    lines.append("## Current scientific judgment")
    lines.append("")
    lines.append("- single motif aggregation route did not improve the main objective")
    lines.append("- pair-lite grammar route did not improve the main objective")
    lines.append("- promoter architecture features carry real independent signal for Score_S")
    lines.append("- current best Score_S among pilot models comes from the architecture route")
    lines.append("- future upgrades should prioritize stronger sequence/structure modeling rather than adding weak motif summaries")
    lines.append("")
    lines.append("## Versioning rule")
    lines.append("")
    lines.append("- major version increases only when the model architecture family changes")
    lines.append("- route-internal iterations remain under the same major version")
    lines.append("- architecture route remains under v4")
    lines.append("")

    DECISION_NOTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    DECISION_NOTE_OUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    registry_df = build_registry()
    write_decision_note(registry_df)

    print("phase6 model decision freeze finished")
    print(f"registry: {REGISTRY_OUT}")
    print(f"decision_note: {DECISION_NOTE_OUT}")


if __name__ == "__main__":
    main()
