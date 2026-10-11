# -*- coding: utf-8 -*-
"""batch26 minimal raw-FASTA inference for the TASK-NATIVE STR+flank predictor.

SCOPE / LIMITS (read before use):
  * Input = the FULL VARIABLE INSERT ONLY (the STR + its hg38 flank), 108-168 nt,
    one sequence per FASTA record, ACGT only (N/other -> hard error).
  * The measured context is FIXED and NOT part of the input: a minimal promoter
    (minP) - GFP reporter in a pGL4.23 backbone, transient plasmid in HEK293T,
    3 biological replicates, endpoint = mean log2(cDNA/gDNA) over replicates and
    eligible barcodes.  The model therefore predicts RELATIVE reporter activity
    of that fragment in THAT context.  It is NOT a calibrated GFP measurement.
  * NOT supported for: arbitrary promoters, long truncations, autonomous-promoter
    activity, other cell lines, or fragments outside 108-168 nt.  No padding is
    added; each sequence is scored at its native length.
  * Internal retrospective locus-holdout model: v6.19.0-strvariant.rc1, experimental,
    not promoted, no transfer claim.

Usage:
  .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch26/b26_infer_fasta.py in.fa out.tsv [--model A1]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b26_common as C
import b26_data as D
import b26_train as TR
sys.path.insert(0, str(C.BATCH02_CODE))
import trackB_grouped_cv as T  # noqa: E402

MIN_LEN, MAX_LEN = 108, 168
MODELS = {
    "A1": "A1_cnn_true_seed20261026",
    "A2": "A2_cnn_true_seed20261027",
    "A3": "A3_cnn_shuffle_seed20261026",
}


def read_fasta(path):
    recs, name, buf = [], None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(buf)))
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.upper())
    if name is not None:
        recs.append((name, "".join(buf)))
    if not recs:
        raise SystemExit("no FASTA records parsed")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta"); ap.add_argument("out")
    ap.add_argument("--model", default="A1", choices=sorted(MODELS))
    a = ap.parse_args()
    aid = MODELS[a.model]
    seqs = read_fasta(a.fasta)
    for n, s in seqs:
        if not (MIN_LEN <= len(s) <= MAX_LEN):
            raise SystemExit("record %s length %d outside supported domain %d-%d nt (unsupported; no padding)"
                             % (n, len(s), MIN_LEN, MAX_LEN))
    codes, lens, bad = D.encode([s for _, s in seqs])
    if bad:
        raise SystemExit("unknown base(s) at %s -- refusing (ACGT only)" % bad[:5])
    if aid == "A0_not_supported":
        pass
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; refusing CPU fallback")
    dev = "cuda:0"
    ck = torch.load(TR.ATTEMPTS / aid / "ckpt.pt", map_location="cpu", weights_only=False)
    model = T.build_cnn("legnet")
    model.load_state_dict(ck["model"]); model.eval().to(dev)
    assert next(model.parameters()).is_cuda
    idx = np.arange(len(seqs))
    p = TR.predict_all(model, codes, lens, idx, dev, torch) * ck["config"]["sd_y"] + ck["config"]["mu_y"]
    lines = ["record\tlength_nt\tpred_log2_cDNA_gDNA\tmodel"]
    for (n, s), v in zip(seqs, p):
        lines.append("%s\t%d\t%.6f\t%s" % (n, len(s), v, aid))
    C.atomic_write_text(Path(a.out), "\n".join(lines) + "\n")
    print("[infer] %d sequences -> %s (model %s; context = fixed minP-GFP, not calibrated)"
          % (len(seqs), a.out, aid))


if __name__ == "__main__":
    main()
