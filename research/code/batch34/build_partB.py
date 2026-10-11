# -*- coding: utf-8 -*-
"""Batch34 B: bounded primary-methods closure evidence matrix + conclusion."""
import json, time
from pathlib import Path

B = Path(r"E:/promoter_atlas/reports/breakthrough_20261002/batch34")
NOW = time.strftime("%Y-%m-%dT%H:%M:%S%z")
COLS = ["source", "title", "cell_line_as_named", "cell_role", "assay_type",
        "integration_selection_timepoint", "sequence_availability", "label_availability",
        "groups_replicates", "license", "verified_vs_unknown", "usable_quant_cohort"]

rows = [
 ["PMID 17195153 / PMC1976183 (J Cell Physiol 2007)",
  "Transcription and epigenetic profile of the promoter, first exon and first intron of the human tyrosine hydroxylase (TH) gene",
  "human renal carcinoma 293FT (+ neuroblastoma BE(2)-C-16)",
  "BOTH: 293FT is a lentiviral packaging/transfection host AND was directly transfected (FuGENE 6) with pGR435/pGR436 for promoter-driven GFP",
  "reporter GFP in self-inactivating lentiviral vector; immunofluorescence/microscopy readout; ChIP for chromatin",
  "defined constructs; no selection/timepoint table; single-timepoint reporter imaging",
  "construct derived from pMAK288 (Kessler 2003); 6.3 kb minimal TH promoter (-6244/+35) > 600; NOT deposited in this paper",
  "qualitative GFP expression; no quantitative sequence+label table",
  "cell-line comparisons; no replicate level table reported",
  "Wiley / NIH author manuscript (PMC); reuse restricted",
  "VERIFIED: 293FT serves as an ACTIVITY-ASSAY host (not only packaging)",
  "NO"],
 ["PMID 23847683 / PMC3701987 (Arch Med Sci 2013)",
  "Therapeutic anti-tumor effect of exogenous apoptin driven by human survivin gene promoter in a lentiviral construct",
  "293FT labelled 'lentivirus packaging cell line' (+ SW480, Hela, MCF-7, BMSC)",
  "BOTH: named as packaging cell line, AND transiently transfected with FG-pSur-ApoHis to compare pSur promoter activity",
  "transient transfection; pMIR-Report internal control + apoptin-6xHis Western blot; lentivirus titred on 293FT by GFP",
  "transient, 48 h; no genomic integration/selection; lentiviral context later",
  "pSur 161/272/990 bp: primers given, fragments PCR-amplified from genomic DNA and sequenced but NOT deposited (no accession)",
  "Western blot band (semi-quantitative); no numeric table",
  "length-variant comparison; biological replicates not tabulated",
  "Arch Med Sci (open access)",
  "VERIFIED: 293FT activity-assay host for pSur length comparison",
  "NO"],
 ["PMID 41959502 / PMC13060879 (bioRxiv 2026, preprint)",
  "Promoter mutagenesis and a massively parallel reporter screen of the MAPT locus identifies cis-regulatory elements and genetic variation effects",
  "HEK293FT cells (+ KOLF2.1J h-NGN2 neurons)",
  "activity-assay host: MPRA performed IN HEK293FT cells",
  "massively parallel reporter assay (2 strategies) over ~3 Mb; barcoded saturation mutagenesis of a 2,000 bp MAPT promoter region",
  "MPRA (episomal/plasmid library per MPRA design); details not yet read (metadata-only)",
  "library sequences expected (preprint supplement / GEO); not verified this batch",
  "MPRA activity counts expected; not verified this batch",
  "not read this batch",
  "bioRxiv preprint",
  "UNKNOWN this batch: strongest discovered lead; requires a bounded data-availability check",
  "UNDETERMINED (lead for next bounded batch)"],
]

lines = ["\t".join(COLS)]
for r in rows:
    lines.append("\t".join(x.replace("\t", " ") for x in r))
(B / "primary_source_evidence.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

concl = {
 "batch": "batch34", "version": "v6.23.0-scoped-inference.rc1", "generated_at": NOW,
 "question": "Was 293FT merely a vector-packaging host, or an actual promoter-activity assay host?",
 "answer": "Within the two named primary studies, 293FT was used as an ACTUAL promoter-activity assay host (TH reporter GFP in 17195153; survivin pSur length-variant comparison in 23847683), in addition to packaging/transfection duties. 293FT is therefore not merely a packaging host in the literature.",
 "usable_cohort_found": False,
 "why_not": "Both full studies use qualitative/semi-quantitative readouts (GFP imaging; Western blot) and do NOT provide accessible matched sequence + quantitative label tables; the TH construct is 6.3 kb (> PARM L_max).",
 "strongest_lead": {"pmid": "41959502", "pmcid": "PMC13060879",
                    "why": "HEK293FT MPRA with sequence libraries and activity readouts (incl. 2,000 bp MAPT promoter saturation mutagenesis)",
                    "status": "metadata-only this batch; needs a bounded data-availability check (<=50MB, 0 fits)"},
 "bound_used": {"targeted_queries": "2 esearch (Q3,Q4) + 2 abstract efetch + 2 full-text retrieval",
                "full_studies_reviewed": 2, "bytes_retrieved_total": 287239, "cap_bytes": 10485760,
                "note": "PMC1976183 full-text XML blocked by publisher; HTML article page used. PMC3701987 full XML."},
 "limitation_statement": "Within this bounded review NO qualified matched sequence + quantitative label 293FT cohort was found; this is a bounded-search result, NOT proof that no such data exist. Current 293FT performance remains UNRESOLVED; this does not block other computational work.",
 "no_negative_overreach": True,
}
(B / "primary_source_evidence.json").write_text(json.dumps(concl, ensure_ascii=False, indent=2), encoding="utf-8")
print("partB written")
