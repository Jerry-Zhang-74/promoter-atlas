# Promoter Analysis Set Notes

## Purpose

- build an analysis-ready transcript-level promoter table
- separate clean core-promoter inputs from extended-context inputs
- keep risky promoter records visible instead of silently dropping them
- prepare later E / S / B label integration

## Tier Definitions

- core_clean: up200 and strict_pass = 1
- extended_clean: up2000 and strict_pass = 1
- qc_pass_but_review: recommended_pass = 1 but not strict_pass
- qc_fail: failed sequence QC

## New Flags

- flag_boundary_near_start: promoter start0 < 2000
- flag_shorter_than_nominal: extracted length shorter than nominal window
- flag_chm13_rescue_candidate: up2000 record with boundary/highN/shortening risk

## Recommended Usage

- first promoter-only baseline model should use core_clean
- extended_clean should be used only as a paired comparison against core_clean
- qc_pass_but_review can be retained for sensitivity analysis
- flag_chm13_rescue_candidate records should not enter main design library directly

## Window Summary

### up200
- record_count: `507365`
- recommended_pass_count: `492108`
- recommended_pass_frac: `0.969929`
- strict_pass_count: `492104`
- strict_pass_frac: `0.969921`
- core_model_pass_count: `492104`
- extended_context_pass_count: `0`
- qc_fail_count: `15257`
- flag_boundary_near_start_count: `4`
- flag_shorter_than_nominal_count: `0`
- flag_chm13_rescue_candidate_count: `0`

### up2000
- record_count: `507365`
- recommended_pass_count: `315445`
- recommended_pass_frac: `0.621732`
- strict_pass_count: `315436`
- strict_pass_frac: `0.621714`
- core_model_pass_count: `0`
- extended_context_pass_count: `315436`
- qc_fail_count: `191920`
- flag_boundary_near_start_count: `6`
- flag_shorter_than_nominal_count: `7`
- flag_chm13_rescue_candidate_count: `156`
