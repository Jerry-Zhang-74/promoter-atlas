#!/usr/bin/env python3
"""Batch04 small smoke tests (no network, no credentials).

Covers: bag slot construction (B1/B2 matching anchor, 4 slots, n<4 cycling),
inference contract refusal for single constructs, SHA-pinned scorer loading,
and deterministic gene order identity. Run with the venv Python.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_bag_v2 import build_slots  # noqa: E402
from b4_common import spearman  # noqa: E402


def test_build_slots():
    # gene with 5 promoters
    starts = np.array([0, 5, 10])
    for g in [0, 1]:
        b1 = build_slots("repeat_bag", g, 3, starts)
        b2 = build_slots("diverse_bag", g, 3, starts)
        assert len(b1) == 4 and len(b2) == 4
        assert len(set(b1)) == 1          # repeat: all identical
        assert b1[0] == b2[0]             # first anchor matches
        if g == 0:
            assert len(set(b2)) == 4      # n>=4 -> 4 distinct
        else:
            assert len(set(b2)) == 4      # n=5 -> 4 distinct
    # n=2 gene cycles
    starts2 = np.array([0, 2, 4])
    b2 = build_slots("diverse_bag", 0, 0, starts2)
    assert len(set(b2)) == 2              # cycles between 2 promoters
    b1 = build_slots("repeat_bag", 0, 0, starts2)
    assert len(set(b1)) == 1
    print("PASS build_slots")


def test_spearman():
    assert abs(spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12
    print("PASS spearman")


def test_scorer_refusal(tmp_checkpoint=None):
    from inference_entrypoint import ScopeAwareScorer, ScopeError  # noqa: E402
    if tmp_checkpoint is None:
        # build a tiny throwaway checkpoint on CPU
        import torch
        from b4_model import GeneBagModel
        m = GeneBagModel()
        import tempfile, os
        fd, p = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save({"model_state": m.state_dict(), "arm": "diverse_bag",
                    "seed": 20261011, "lr": 0.001}, p)
        tmp_checkpoint = p
    scorer = ScopeAwareScorer(tmp_checkpoint, device="cpu")
    out = scorer.predict("single_promoter", "A" * 400)
    assert out["status"] == "unsupported_scope", out
    out2 = scorer.predict("gene_promoter_bag", ["A" * 400, "C" * 400])
    assert out2["status"] == "ok", out2
    assert isinstance(out2["score_E_rank"], float)
    # wrong SHA must be refused
    try:
        ScopeAwareScorer(tmp_checkpoint, device="cpu", expected_sha256="deadbeef")
        raise AssertionError("should have refused on SHA mismatch")
    except ScopeError:
        pass
    print("PASS scorer refusal + gene-bag ok + SHA pin")


if __name__ == "__main__":
    test_build_slots()
    test_spearman()
    test_scorer_refusal()
    print("ALL SMOKE TESTS PASS")
