"""Unit tests for the batch24 fixed kernels. NO ridge fits / NO solves here."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch24_kernels as K  # noqa: E402

TOL = 1e-9
rng = np.random.default_rng(7)


def _random_seqs(n, length):
    return ["".join(rng.choice(list("ACGT"), size=length)) for _ in range(n)]


def test_vocab_order():
    c = K.kmer_counts("AAAA", 4)
    assert c.shape == (256,) and c[0] == 1 and c.sum() == 1
    c = K.kmer_counts("TTTT", 4)
    assert c[255] == 1 and c.sum() == 1
    # "ACGT" one 4-mer -> code 0*64+1*16+2*4+3 = 27
    c = K.kmer_counts("ACGT", 4)
    assert c[27] == 1
    assert K.kmer_counts("ACGT", 5).shape == (1024,)


def test_counts_sum_and_nonneg():
    s = _random_seqs(3, 133)
    for k in (4, 5):
        for seq in s:
            c = K.kmer_counts(seq, k)
            assert c.sum() == len(seq) - k + 1
            assert (c >= 0).all()


def test_dimension_and_norm():
    seqs = _random_seqs(4, 133)
    for k in (4, 5):
        F = np.vstack([K.feature_vector(s, k) for s in seqs])
        assert F.shape == (4, 4 ** k)
        assert np.allclose(np.linalg.norm(F, axis=1), 1.0, atol=TOL)


def test_empty_vector_stays_zero():
    # sequence shorter than k -> no k-mers -> zero vector -> cosine 0
    assert np.allclose(K.feature_vector("AC", 4), 0.0)
    K_small = K.global_kernel(["AC", "ACGTACGT"])
    assert abs(K_small[0, 0]) < TOL
    assert abs(K_small[0, 1]) < TOL


def test_symmetry_and_diagonal():
    seqs = _random_seqs(12, 133)
    res = K.kernels_for(seqs)
    for name, M in res.items():
        assert M.shape == (12, 12)
        assert np.allclose(M, M.T, atol=TOL), name
        assert np.allclose(np.diag(M), 1.0, atol=TOL), name
        assert (M >= -TOL).all(), name
        assert (M <= 1 + TOL).all(), name


def test_psd_tolerance():
    seqs = _random_seqs(40, 133)
    res = K.kernels_for(seqs)
    for name, M in res.items():
        ev = np.linalg.eigvalsh((M + M.T) / 2.0)
        assert ev.min() >= -1e-8, (name, ev.min())


def test_h2_identity():
    seqs = _random_seqs(10, 133)
    res = K.kernels_for(seqs)
    assert np.allclose(res["H2"], 0.5 * res["S_global"] + 0.5 * res["S_position"], atol=TOL)
    assert np.allclose(res["H1"], res["S_global"], atol=TOL)


def test_orientation_matters():
    seq = _random_seqs(1, 133)[0]
    rcseq = K.rc(seq)
    assert seq != rcseq
    for k in (4, 5):
        a = K.feature_vector(seq, k)
        b = K.feature_vector(rcseq, k)
        assert not np.allclose(a, b, atol=1e-6)
    M = K.global_kernel([seq, rcseq])
    assert abs(M[0, 0] - 1.0) < TOL
    # palindromic-free random sequence: an RC duplicate is a near-miss, not identity
    assert M[0, 1] < 0.5


def test_rc_is_not_palindrome_identity():
    s = "ACGTACGTAA"
    assert K.rc(s) == "TTACGTACGT"


def test_intervals_fixed():
    assert K.POSITION_INTERVALS == ((0, 44), (44, 89), (89, 133))
    assert K.POSITION_LENGTH == 133
    assert K.GLOBAL_KS == (4, 5)


def test_spec_dimensions():
    spec = K.KERNEL_SPEC
    assert spec["components"]["counts_k4_global"]["dim"] == 256
    assert spec["components"]["counts_k5_global"]["dim"] == 1024
    assert spec["ridge"]["alpha"] == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL %d TESTS PASS" % len(fns))
