#!/usr/bin/env python3
"""Predict v6 targets with checkpoint scales and explicit input coverage.

Paired mode: --fasta400 local.fa --fasta2000 upstream.fa [--fasta10000 long.fa].
IDs must be unique and matching; window lengths must be exact. Sequences must
already follow training BED orientation. FASTA alone cannot verify the TSS.
Human sym400 is genomic [tss0-200,tss0+200), reverse complemented on minus:
TSS index 200 (+), 199 (-). Human up2000 excludes the TSS on both strands.
Historical human up10000 on minus strands includes the TSS. Other species use
their own training BED. Use actual BED/memmap windows, not guessed slices.

Legacy mode: --fasta input.fa --window 400|2000 [--tss-position 200|1999].
Position is an old alignment parameter, not evidence of a genomic TSS. The
prefix through that position is right-aligned; the suffix follows it. Missing
model windows are N. Short/missing windows have unvalidated applicability.
Even structurally complete inputs do not establish reporter-assay reliability.

*_pred use original TRAINING TARGET scales (not raw assay units); *_pred_z
retain standardized outputs. Predictions outside a label range are not clipped.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_v6_training_input import SPECIES as SPECIES_NAMES
from train_promoter_atlas_v6 import ESB_COLS, OMICS_COLS, infer_arch
from v6_evaluation_utils import load_evaluation_model

ROOT = Path(__file__).resolve().parents[2]
TABLE = {base: i for i, base in enumerate("ACGTN")}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_fasta(path):
    records, seen, name, parts = [], set(), None, []

    def append():
        if name is None:
            return
        seq = "".join(parts).upper()
        if not seq:
            raise ValueError(f"Empty sequence: {name} in {path}")
        invalid = set(seq) - set(TABLE)
        if invalid:
            raise ValueError(f"Unsupported bases for {name}: {sorted(invalid)}; use A/C/G/T/N")
        records.append((name, seq))

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                append()
                fields = line[1:].split()
                if not fields or fields[0] in seen:
                    raise ValueError(f"Missing or duplicate FASTA ID in {path}: {line}")
                name, parts = fields[0], []
                seen.add(name)
            else:
                if name is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                parts.append(line)
    append()
    if not records:
        raise ValueError(f"No FASTA records in {path}")
    return records


def encode(seq, length, tss_pos):
    """Preserve legacy placement; reject truncation and unsupported positions."""
    if length not in (400, 2000):
        raise ValueError("Legacy window must be 400 or 2000")
    allowed = {199, 200} if length == 400 else {1999}
    if tss_pos not in allowed:
        raise ValueError(f"Legacy position for window {length} must be one of {sorted(allowed)}")
    if len(seq) > length:
        raise ValueError(f"Sequence length {len(seq)} exceeds window {length}; explicit extraction required")
    seq = seq.upper()
    if set(seq) - set(TABLE):
        raise ValueError("Sequence must use A/C/G/T/N")
    x = np.full(length, 4, dtype=np.uint8)
    up, down = seq[:tss_pos + 1], seq[tss_pos + 1:]
    x[tss_pos + 1 - len(up):tss_pos + 1] = [TABLE[b] for b in up]
    x[tss_pos + 1:tss_pos + 1 + len(down)] = [TABLE[b] for b in down]
    return x


def prepare_inputs(args, ctx10k):
    paired = any(getattr(args, f"fasta{n}", None) for n in (400, 2000, 10000))
    windows = (400, 2000, 10000) if ctx10k else (400, 2000)
    sources = {}
    if args.fasta:
        if paired:
            raise ValueError("Use either --fasta or paired --fasta400/--fasta2000")
        length = args.window if args.window is not None else 400
        pos = args.tss_position if args.tss_position is not None else (200 if length == 400 else 1999)
        records = read_fasta(args.fasta)
        sources[length] = (str(Path(args.fasta).resolve()), sha256(args.fasta))
        items = []
        for name, sequence in records:
            seqs = {n: np.full(n, 4, dtype=np.uint8) for n in windows}
            seqs[length] = encode(sequence, length, pos)
            items.append((name, seqs, {length: len(sequence)}))
        mode, placement = "legacy_single_window", f"legacy_alignment_parameter_{pos};genomic_TSS_unverified"
        print("WARNING: legacy single-window input N-pads missing context. Outputs are exploratory; "
              "reporter activity and TSS placement are unvalidated.", file=sys.stderr)
    else:
        if not (args.fasta400 and args.fasta2000):
            raise ValueError("Provide --fasta, or both --fasta400 and --fasta2000")
        if args.window is not None or args.tss_position is not None:
            raise ValueError("--window/--tss-position apply only to legacy --fasta mode")
        if ctx10k and not args.fasta10000:
            raise ValueError("This checkpoint uses a 10 kb tower; paired mode requires --fasta10000")
        if not ctx10k and args.fasta10000:
            raise ValueError("This checkpoint has no 10 kb tower; --fasta10000 would be ignored")
        records_by_window = {}
        for n in windows:
            path = getattr(args, f"fasta{n}")
            records_by_window[n] = dict(read_fasta(path))
            sources[n] = (str(Path(path).resolve()), sha256(path))
            for name, seq in records_by_window[n].items():
                if len(seq) != n:
                    raise ValueError(f"Paired window {n}, record {name}: expected exactly {n} bases, got {len(seq)}")
        ids = list(records_by_window[400])
        for n in windows:
            if set(records_by_window[n]) != set(ids):
                raise ValueError(f"FASTA IDs do not match between 400 bp and {n} bp inputs")
        items = [(name, {n: np.array([TABLE[b] for b in records_by_window[n][name]], dtype=np.uint8)
                         for n in windows}, {n: n for n in windows}) for name in ids]
        mode, placement = "paired_exact_length_windows", "caller_supplied_training_BED_orientation;genomic_TSS_unverified"
        print("NOTE: paired lengths and IDs are checked. Genomic coordinates, TSS identity, "
              "and reporter-assay applicability are not established by FASTA structure.", file=sys.stderr)

    rows = []
    for name, seqs, supplied in items:
        row = {"record_id": name, "length": sum(supplied.values()), "input_mode": mode,
               "tss_placement_status": placement,
               "context_complete": int(all(supplied.get(n) == n and np.all(seqs[n] < 4) for n in windows)),
               "prediction_scope": "exploratory_reporter_applicability_unvalidated",
               "input_window_lengths_complete": int(all(supplied.get(n) == n for n in windows))}
        for n in windows:
            row[f"window{n}_supplied_bp"] = supplied.get(n, 0)
            row[f"window{n}_acgt_bp"] = int((seqs[n] < 4).sum())
            row[f"window{n}_unknown_or_padding_bp"] = int((seqs[n] == 4).sum())
            row[f"window{n}_input_sha256"] = sources.get(n, ("", ""))[1]
            row[f"window{n}_input_path"] = sources.get(n, ("", ""))[0]
        rows.append(row)
    if any(not r["context_complete"] for r in rows):
        print("WARNING: at least one record has incomplete A/C/G/T context; inspect coverage columns.", file=sys.stderr)
    return items, rows


def target_scales(ck):
    """Refuse original-scale predictions without stored normalization provenance."""
    stats = ck.get("zstats", {})
    n_omics = int(ck.get("n_omics", 0))
    omics = ck.get("omics_cols")
    if omics is None:
        omics = [name for name in OMICS_COLS if name in stats]
    omics = list(omics)
    if len(omics) != n_omics or len(set(omics)) != len(omics) or set(omics) - set(OMICS_COLS):
        raise ValueError(f"Cannot establish omics target order: n_omics={n_omics}, names={omics}")
    targets = list(ESB_COLS) + omics
    scales = {}
    for name in targets:
        value = stats.get(name)
        if value is None or len(value) != 2:
            raise ValueError(f"Missing checkpoint zstats for {name}; original-scale prediction unavailable")
        mu, sd = map(float, value)
        if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid checkpoint zstats for {name}: {value}")
        scales[name] = (mu, sd)
    return omics, scales


def add_target_outputs(rows, outputs, omics, scales):
    for head, names in (("esb", ESB_COLS), ("omics", omics)):
        if not names:
            continue
        values = outputs[head].detach().cpu().numpy()
        if values.shape != (len(rows), len(names)):
            raise ValueError(f"Unexpected {head} output shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {head} prediction")
        for i, row in enumerate(rows):
            for j, name in enumerate(names):
                z = float(values[i, j])
                mu, sd = scales[name]
                row[name + "_pred_z"] = z
                row[name + "_pred"] = z * sd + mu
                row[name + "_scale_mean"] = mu
                row[name + "_scale_sd"] = sd


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fasta")
    ap.add_argument("--fasta400")
    ap.add_argument("--fasta2000")
    ap.add_argument("--fasta10000")
    ap.add_argument("--out")
    ap.add_argument("--window", type=int, choices=[400, 2000])
    ap.add_argument("--tss-position", type=int)
    ap.add_argument("--species", default="homo_sapiens", choices=SPECIES_NAMES)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return ap


def predict(args):
    if args.batch < 1:
        raise ValueError("Batch size must be positive")
    if args.species not in SPECIES_NAMES:
        raise ValueError(f"Unknown species: {args.species}")
    checkpoint_path = Path(args.checkpoint)
    before_sha = sha256(checkpoint_path)
    ck = torch.load(checkpoint_path, map_location="cpu")
    if sha256(checkpoint_path) != before_sha:
        raise RuntimeError("Checkpoint changed while loading; use a frozen checkpoint snapshot")
    omics, scales = target_scales(ck)
    items, rows = prepare_inputs(args, infer_arch(ck)["ctx10k"])
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 2))
    model = load_evaluation_model(ck, device)
    species_id = SPECIES_NAMES.index(args.species)
    with torch.inference_mode():
        for start in range(0, len(items), args.batch):
            chunk = items[start:start + args.batch]
            seqs = {n: torch.from_numpy(np.stack([item[1][n] for item in chunk])).long().to(device)
                    for n in chunk[0][1]}
            sp = torch.full((len(chunk),), species_id, dtype=torch.long, device=device)
            outputs = model(seqs[400], None, seqs[2000], sp, seqs.get(10000))
            add_target_outputs(rows[start:start + len(chunk)], outputs, omics, scales)
    for row in rows:
        row.update(checkpoint_sha256=before_sha, checkpoint_path=str(checkpoint_path.resolve()),
                   standardization_source="checkpoint.zstats;original_training_target_scale",
                   omics_order_source="checkpoint.omics_cols" if "omics_cols" in ck else "trainer_OMICS_COLS_filtered_by_checkpoint_zstats",
                   species=args.species, model_name=ck.get("name", checkpoint_path.stem))
    return pd.DataFrame(rows)


def main():
    ap = make_parser()
    args = ap.parse_args()
    try:
        df = predict(args)
    except (ValueError, RuntimeError, KeyError) as exc:
        ap.error(str(exc))
    out = Path(args.out) if args.out else ROOT / "models/infer" / (Path(args.checkpoint).stem + "_predictions.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(df[["record_id", "Score_E_pred", "Score_S_pred", "Score_B_pred", "context_complete"]].to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
