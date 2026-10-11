"""
Update Phase 6 registry after unified v1 evaluation.

Outputs
-------
- reports/model_registry_phase6.tsv
- docs/notes/phase6_model_decision_v2.md
"""

from pathlib import Path
import pandas as pd

ROOT = Path(".")

REGISTRY_PATH = ROOT / "reports/model_registry_phase6.tsv"
V1_UNIFIED = ROOT / "models/eval/promoter_model_v1_metrics_unified.tsv"
DECISION_NOTE_OUT = ROOT / "docs/notes/phase6_model_decision_v2.md"


def fmt_float(x):
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.4f}"


def main():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY_PATH}")
    if not V1_UNIFIED.exists():
        raise FileNotFoundError(f"Missing unified v1 eval: {V1_UNIFIED}")

    reg = pd.read_csv(REGISTRY_PATH, sep="\t", low_memory=False)
    v1u = pd.read_csv(V1_UNIFIED, sep="\t", low_memory=False)

    test_row = v1u.loc[v1u["split"] == "test"]
    if len(test_row) != 1:
        raise ValueError("Expected exactly one test row in unified v1 eval")
    row = test_row.iloc[0]

    mask = reg["model_name"] == "promoter_model_v1"
    if mask.sum() != 1:
        raise ValueError("Expected exactly one promoter_model_v1 row in registry")

    reg.loc[mask, "best_metric_source"] = "unified_test"
    reg.loc[mask, "Score_E_r2"] = float(row["Score_E_r2"])
    reg.loc[mask, "Score_S_r2"] = float(row["Score_S_r2"])
    reg.loc[mask, "Score_B_r2"] = float(row["Score_B_r2"])
    reg.loc[mask, "predEScorr"] = float(row["pred_E_pred_S_corr_defined"])
    reg.loc[mask, "notes"] = "Current official sequence-only overall baseline, updated with unified test metrics."

    reg.to_csv(REGISTRY_PATH, sep="\t", index=False)

    v1 = reg.loc[reg["model_name"] == "promoter_model_v1"].iloc[0]
    v4 = reg.loc[reg["model_name"] == "promoter_model_v4_arch_only_pilot"].iloc[0]

    lines = []
    lines.append("# Phase 6 model decision v2")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Refresh the Phase 6 model decision after unified evaluation of promoter_model_v1.")
    lines.append("")
    lines.append("## Dual baseline decision")
    lines.append("")
    lines.append("### Official overall baseline")
    lines.append("")
    lines.append(f"- model: {v1['model_name']}")
    lines.append(f"- route: {v1['route']}")
    lines.append(f"- checkpoint: {v1['checkpoint_path']}")
    lines.append(f"- Score_E_r2: {fmt_float(v1['Score_E_r2'])}")
    lines.append(f"- Score_S_r2: {fmt_float(v1['Score_S_r2'])}")
    lines.append(f"- Score_B_r2: {fmt_float(v1['Score_B_r2'])}")
    lines.append(f"- predEScorr: {fmt_float(v1['predEScorr'])}")
    lines.append("")
    lines.append("### Official S-focused architecture reference")
    lines.append("")
    lines.append(f"- model: {v4['model_name']}")
    lines.append(f"- route: {v4['route']}")
    lines.append(f"- checkpoint: {v4['checkpoint_path']}")
    lines.append(f"- Score_E_r2: {fmt_float(v4['Score_E_r2'])}")
    lines.append(f"- Score_S_r2: {fmt_float(v4['Score_S_r2'])}")
    lines.append(f"- Score_B_r2: {fmt_float(v4['Score_B_r2'])}")
    lines.append(f"- predEScorr: {fmt_float(v4['predEScorr'])}")
    lines.append("")
    lines.append("## Current scientific judgment")
    lines.append("")
    lines.append("- motif aggregation route is not the main path for Score_S")
    lines.append("- pair-lite grammar route is not the main path for Score_S")
    lines.append("- promoter architecture carries independent signal for Score_S")
    lines.append("- next model upgrades should focus on stronger sequence-structure fusion rather than adding weak motif summaries")
    lines.append("")

    DECISION_NOTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    DECISION_NOTE_OUT.write_text("\n".join(lines), encoding="utf-8")

    print("phase6 registry update finished")
    print(f"registry: {REGISTRY_PATH}")
    print(f"decision_note: {DECISION_NOTE_OUT}")


if __name__ == "__main__":
    main()
