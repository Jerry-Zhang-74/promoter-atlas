# batch33 code (v6.22.1-assay-context.rc1)

Evidence-only. ZERO fits / ZERO model inference / ZERO GPU.

| file | purpose |
|---|---|
| `build_context.py` | parse the ORIGINAL exported CytoFLEX CSVs + frozen cellline_v2 tables; rebuild the unchanged endpoint formula for every context; write `frozen_eligibility_matrix.tsv`, `context_registry.tsv`, `exclusions.tsv`. Column names (Chinese) matched by ASCII substrings; source paths read from `source_inventory.json`. |
| `build_compare.py` | observed family/timepoint comparison vs D28 and the frozen-model direction grid from UNCHANGED saved scores. |
| `partB_search.py` | bounded PubMed E-utilities search (<=6 queries, <=30MB). |
| `partB_evidence.py` | build the evidence matrix + conclusion from the bounded search. |
| `make_figures.py` | the <=3 figures (matplotlib Agg). |

Run with `PYTHONUTF8=1 python -X utf8 <script>` from `E:/promoter_atlas`.
Endpoints are never fit; the D28 FAIL is preserved; no production pointer is changed.
