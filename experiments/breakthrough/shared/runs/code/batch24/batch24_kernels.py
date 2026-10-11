"""
batch24 fixed sequence kernels (v6.18.0-hek-kernel.rc1).

KERNELS ARE FIXED BEFORE ANY FIT (batch24/PROTOCOL.txt).

Feature mapping (per sequence, per-sequence, no population-estimated parameters):
  - forward-orientation k-mer counts over the ACTUAL full insert, complete fixed
    ACGT vocabulary of size 4**k (order fixed: A<C<G<T, most-significant first).
  - NO reverse-complement merging: orientation matters.
  - L2-normalize each count vector; an empty vector stays all-zeros (cosine 0).

Kernels:
  S_global(a,b)   = 0.5 * cos(c4_a, c4_b) + 0.5 * cos(c5_a, c5_b)
  H1              = S_global
  S_position(a,b) = mean over the three fixed intervals of
                    cos(c4_a[interval], c4_b[interval])
                    intervals (zero-based half-open, 133bp contract):
                        [0,44), [44,89), [89,133)
  H2              = 0.5 * S_global + 0.5 * S_position

Weights, dimensions and fixed alpha are declared in KERNEL_SPEC below.
"""
import numpy as np

ALPHABET = "ACGT"
BASE = {c: i for i, c in enumerate(ALPHABET)}

# Fixed intervals for the validated 133 bp positional contract (zero-based half-open)
POSITION_INTERVALS = ((0, 44), (44, 89), (89, 133))
POSITION_LENGTH = 133
POSITION_K = 4
GLOBAL_KS = (4, 5)

KERNEL_SPEC = {
    "name": "batch24_fixed_spectrum_kernels",
    "version": "v6.18.0-hek-kernel.rc1",
    "encoding": {
        "alphabet": ALPHABET,
        "order": "lexicographic A<C<G<T, most significant position first",
        "orientation": "forward only; reverse-complement NOT merged (orientation matters)",
        "normalization": "L2 per feature vector; empty vector -> zeros",
    },
    "components": {
        "counts_k4_global": {"k": 4, "dim": 4 ** 4, "scope": "full insert"},
        "counts_k5_global": {"k": 5, "dim": 4 ** 5, "scope": "full insert"},
        "counts_k4_positional": {
            "k": 4,
            "dim": 4 ** 4,
            "scope": "within each of intervals %s" % (POSITION_INTERVALS,),
            "per_interval_normalization": "independent L2 per interval",
            "validated_input_length": POSITION_LENGTH,
        },
    },
    "equations": {
        "cos": "cos(u,v) = <u,v> / (||u||*||v||); 0 if either norm is 0",
        "S_global": "0.5*cos(k4_full) + 0.5*cos(k5_full)",
        "S_position": "mean_{i in intervals} cos(k4_within_interval_i)",
        "H1": "S_global",
        "H2": "0.5*S_global + 0.5*S_position",
    },
    "ridge": {
        "alpha": 1.0,
        "alpha_origin": "fixed a priori, not tuned",
        "target_centering": "TRAIN-fold mean only (no full-cohort centering/scaling)",
        "solve": "(K_train + alpha*I) w = y_train_centered ; predict K_holdout_train @ w + train_mean",
        "dtype": "float64",
        "solver": "numpy.linalg.solve on the symmetric positive-definite system",
    },
}


def _kmer_index(k):
    """Return lookup array of shape (4**k,) mapping base-4 code -> lexicographic index."""
    # lexicographic: A<C<G<T, first character most significant -> code is base-BASE
    # with the FIRST character most significant. Bincount index = code.
    return np.arange(4 ** k, dtype=np.int64)


def kmer_counts(seq, k):
    """Forward-orientation k-mer counts, complete fixed vocabulary, dim 4**k.

    Characters outside ACGT are not expected (inputs are validated ACGT); they
    raise ValueError rather than being silently binned.
    """
    n = len(seq)
    if n < k:
        return np.zeros(4 ** k, dtype=np.float64)
    if k > 32:
        raise ValueError("k too large for the rolling base-4 encoder")
    mask = (4 ** k) - 1
    codes = np.empty(n - k + 1, dtype=np.int64)
    code = 0
    # rolling base-4 code; first char most significant
    for i, ch in enumerate(seq):
        try:
            d = BASE[ch]
        except KeyError:
            raise ValueError("non-ACGT character %r in sequence" % (ch,))
        code = ((code * 4) + d) & mask
        if i >= k - 1:
            codes[i - k + 1] = code
    counts = np.bincount(codes, minlength=4 ** k).astype(np.float64)
    return counts


def l2_normalize(v):
    nrm = np.linalg.norm(v)
    if nrm <= 0:
        return np.zeros_like(v)
    return v / nrm


def feature_vector(seq, k):
    return l2_normalize(kmer_counts(seq, k))


def cosine_matrix(F):
    """F: (n, d) row-normalized features -> (n, n) cosine Gram."""
    return F @ F.T


def global_kernel(seqs):
    """S_global = 0.5*cos(k4) + 0.5*cos(k5). Returns (n,n) float64."""
    F4 = np.vstack([feature_vector(s, 4) for s in seqs])
    F5 = np.vstack([feature_vector(s, 5) for s in seqs])
    return 0.5 * cosine_matrix(F4) + 0.5 * cosine_matrix(F5)


def position_kernel(seqs):
    """S_position = mean of the three interval cosine kernels (k=4)."""
    n = len(seqs)
    K = np.zeros((n, n), dtype=np.float64)
    for (a, b) in POSITION_INTERVALS:
        F = np.vstack([feature_vector(s[a:b], POSITION_K) for s in seqs])
        K += cosine_matrix(F)
    K /= float(len(POSITION_INTERVALS))
    return K


def kernels_for(seqs):
    """Return dict with the (label-free, per-sequence) Gram matrices Kg, Kp, H1, H2."""
    Kg = global_kernel(seqs)
    Kp = position_kernel(seqs)
    return {
        "S_global": Kg,
        "S_position": Kp,
        "H1": Kg,
        "H2": 0.5 * Kg + 0.5 * Kp,
    }


def rc(seq):
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return "".join(comp[c] for c in reversed(seq))
