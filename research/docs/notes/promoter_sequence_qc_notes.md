# Promoter Sequence QC Notes

## Purpose

- build sequence-level QC before motif scan and model input
- separate promoter definition issues from downstream modeling issues
- support wet-lab friendly candidate gating

## Recommended QC Gates

- high_N: n_frac > 0.10
- low_entropy: entropy_base4 < 1.20
- homopolymer: max_homopolymer_len >= 12
- dinuc_repeat: dinuc_repeat_frac >= 0.30
- recommended_pass = no QC failure above

## Window Summary

### up200
- record_count: `507365`
- pass_recommended_count: `492108`
- pass_recommended_frac: `0.969929`
- flag_high_n_count: `13`
- flag_low_entropy_count: `107`
- flag_homopolymer_count: `14212`
- flag_dinuc_repeat_count: `1115`
- mean_gc_frac: `0.575897`
- mean_n_frac: `1.5e-05`
- mean_entropy_base4: `1.898611`

### up2000
- record_count: `507365`
- pass_recommended_count: `315445`
- pass_recommended_frac: `0.621732`
- flag_high_n_count: `147`
- flag_low_entropy_count: `4`
- flag_homopolymer_count: `191770`
- flag_dinuc_repeat_count: `281`
- mean_gc_frac: `0.486149`
- mean_n_frac: `0.000143`
- mean_entropy_base4: `1.97294`

## Interpretation

- up200 is the cleaner core-promoter view and should usually be the first modeling baseline
- up2000 may contain additional regulatory context but also more reference-boundary and low-complexity burden
- recommended_pass should be treated as an entry gate, not as the final biological truth
- later E/S/B modeling should compare the two promoter definitions under the same QC gate
