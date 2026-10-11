#!/usr/bin/env python3
"""Batch05 scope-aware inference entrypoint (no automatic production pointer).

Exposes a narrow, version-pinned inference interface for the batch05 frozen-split
gene-bag candidates (B1 repeat_bag / B2 diverse_bag). It refuses out-of-scope
inputs instead of silently over-interpreting them.

Input contract:
  * gene_promoter_bag : one or more 400bp sym400 promoter sequences for a gene
                        -> gene-level E ranking score (numeric ranking, gene scope).
  * single_promoter / synthetic_construct : REFUSED (unsupported_scope). A single
                        construct is NOT a validated activity prediction and the
                        model is NOT a single-construct activity scorer.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b5_common import sha256_file  # noqa: E402
from b5_model import GeneBagModel, onehot  # noqa: E402

L400 = 400
_VALID_ARMS = ("repeat_bag", "diverse_bag")


class ScopeError(ValueError):
    pass


def _encode(seqs, device):
    arr = np.zeros((len(seqs), L400), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s.strip().upper()
        if len(s) != L400:
            raise ScopeError("sequence length must be %d (got %d)" % (L400, len(s)))
        codes = []
        for c in s:
            if c == "A":
                codes.append(0)
            elif c == "C":
                codes.append(1)
            elif c == "G":
                codes.append(2)
            elif c == "T":
                codes.append(3)
            elif c == "N":
                codes.append(4)
            else:
                raise ScopeError("invalid base %r" % c)
        arr[i] = codes
    x = torch.from_numpy(arr).to(device)
    return onehot(x, L400)


class ScopeAwareScorer:
    def __init__(self, checkpoint, device="cpu", expected_sha256=None):
        self.checkpoint = Path(checkpoint)
        self.device = device
        if expected_sha256 is not None:
            actual = sha256_file(self.checkpoint)
            if actual != expected_sha256:
                raise ScopeError("checkpoint SHA mismatch: expected %s got %s" % (expected_sha256, actual))
        ck = torch.load(self.checkpoint, map_location=device, weights_only=False)
        self.arm = ck.get("arm", None)
        self.model = GeneBagModel().to(device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()
        self.meta = {
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "n_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "arm": self.arm,
            "seed": ck.get("seed"),
            "lr": ck.get("lr"),
            "split_scope": "exact/RC + detected-near-identity blocked (NOT homology-free); retrospective only",
        }

    def predict_gene_bag(self, promoter_sequences):
        if not promoter_sequences:
            raise ScopeError("empty promoter bag")
        if not isinstance(promoter_sequences, (list, tuple)):
            raise ScopeError("promoter_sequences must be a list/tuple of 400bp strings")
        x = _encode(list(promoter_sequences), self.device)
        with torch.no_grad():
            z = self.model.embed(x)
            mean = z.mean(dim=0, keepdim=True)
            maxv = z.max(dim=0).values.unsqueeze(0)
            out = self.model.head(torch.cat([mean, maxv], dim=1)).squeeze(-1)
        return float(out.cpu().item())

    def predict_single_construct(self, sequence, **kwargs):
        return {
            "status": "unsupported_scope",
            "reason": "single-promoter/synthetic-construct activity is not a validated output of the gene-bag model; a single construct is not a gene promoter bag and cannot be silently interpreted as validated activity",
            "contract": "gene_promoter_bag",
        }

    def predict(self, input_type, payload):
        if input_type == "gene_promoter_bag":
            return {"status": "ok", "scope": "gene_level_ranking",
                    "score_E_rank": self.predict_gene_bag(payload),
                    "note": "numeric gene-level ranking only; NOT a validated single-construct activity prediction"}
        if input_type in ("single_promoter", "synthetic_construct", "single_construct"):
            return self.predict_single_construct(payload)
        return {"status": "missing_context",
                "reason": "unknown input_type %r; supported: gene_promoter_bag" % input_type}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--expected-sha256", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--input-type", default="gene_promoter_bag",
                    choices=["gene_promoter_bag", "single_promoter", "synthetic_construct"])
    ap.add_argument("--sequences", default=None, help="comma-separated 400bp sequences or a file path with one per line")
    args = ap.parse_args()

    scorer = ScopeAwareScorer(args.checkpoint, args.device, args.expected_sha256)
    if args.sequences is None:
        print(json.dumps({"meta": scorer.meta, "status": "ready"}, indent=2))
        return
    path = Path(args.sequences)
    if path.exists():
        seqs = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        seqs = [s for s in args.sequences.split(",") if s]
    result = scorer.predict(args.input_type, seqs)
    result["meta"] = scorer.meta
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
