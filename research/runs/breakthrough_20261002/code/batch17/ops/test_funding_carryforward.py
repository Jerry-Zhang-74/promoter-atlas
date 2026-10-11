#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch17 ops (deliverable 5a tests): carry-forward funding basis.

Validates the proposed fix against (1) the real control_24 funding_basis.json,
(2) the real control_23 manifest that lost its top-up, (3) manifest-only legacy
shapes. Proves that a fresh controller can recover
original 142.67 + carried 100.00 = effective 242.67 with spent staying continuous.
Run: python -X utf8 test_funding_carryforward.py
"""
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from funding_carryforward import load_funding_basis, fresh_status_funding, DEFAULT_ORIGINAL_BASELINE_CNY

ROOT = Path(__file__).resolve().parents[5]
CTRL23 = ROOT / "runs" / "breakthrough_20261002" / "control_23" / "status.json"
CTRL24 = ROOT / "runs" / "breakthrough_20261002" / "control_24" / "status.json"
FB24 = ROOT / "runs" / "breakthrough_20261002" / "control_24" / "funding_basis.json"

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))


def main():
    # 1. real control_24 with explicit funding_basis.json (preferred path)
    if FB24.is_file():
        b24 = load_funding_basis(manifest_path=CTRL24)
        check("ctrl24 original baseline is 142.67", b24["original_baseline_cny"] == Decimal("142.67"), str(b24["original_baseline_cny"]))
        check("ctrl24 carried topup is 100.00", b24["carried_topup_cny"] == Decimal("100.00"), str(b24["carried_topup_cny"]))
        check("ctrl24 effective baseline is 242.67", b24["effective_baseline_cny"] == Decimal("242.67"), str(b24["effective_baseline_cny"]))
        check("ctrl24 last prior balance 114.04", b24["last_balance_cny"] == Decimal("114.04"), str(b24["last_balance_cny"]))
    else:
        check("ctrl24 funding_basis.json present", False, str(FB24))

    # 2. real control_23 manifest: legacy controller that recorded original 142.67
    #    with topup 100 and balance 114.04 -> must reconstruct the SAME basis.
    m23 = json.loads(CTRL23.read_text(encoding="utf-8"))
    b23 = load_funding_basis(manifest=m23)
    check("ctrl23 reconstructs original 142.67", b23["original_baseline_cny"] == Decimal("142.67"), str(b23["original_baseline_cny"]))
    check("ctrl23 reconstructs carried 100.00", b23["carried_topup_cny"] == Decimal("100.00"), str(b23["carried_topup_cny"]))
    check("ctrl23 reconstructs effective 242.67", b23["effective_baseline_cny"] == Decimal("242.67"), str(b23["effective_baseline_cny"]))
    check("ctrl23 last balance 114.04", b23["last_balance_cny"] == Decimal("114.04"), str(b23["last_balance_cny"]))
    check("ctrl23 prior spent 128.63", b23["prior_spent_cny"] == Decimal("128.63"), str(b23["prior_spent_cny"]))

    # 3. carry-forward continuity: fresh status fields equal the effective basis
    fs = fresh_status_funding(b23)
    check("fresh status baseline is effective", fs["baseline_cny"] == "242.67", fs["baseline_cny"])
    check("fresh status keeps original separate", fs["baseline_original_cny"] == "142.67", fs["baseline_original_cny"])
    check("fresh status carries topup", fs["cumulative_topup_cny"] == "100.00", fs["cumulative_topup_cny"])

    # 4. empty manifest -> default original baseline, zero carried
    b0 = load_funding_basis(manifest={})
    check("empty manifest default original", b0["original_baseline_cny"] == DEFAULT_ORIGINAL_BASELINE_CNY, str(b0["original_baseline_cny"]))
    check("empty manifest zero carried", b0["carried_topup_cny"] == Decimal("0"), str(b0["carried_topup_cny"]))
    check("empty manifest no balance", b0["last_balance_cny"] is None, str(b0["last_balance_cny"]))

    # 5. explicit manifest override wins
    b1 = load_funding_basis(manifest={"baseline_original_cny": "150.00", "cumulative_topup_cny": "50.00",
                                      "balance_cny": "175.00"})
    check("explicit original honored", b1["original_baseline_cny"] == Decimal("150.00"), str(b1["original_baseline_cny"]))
    check("explicit effective 200.00", b1["effective_baseline_cny"] == Decimal("200.00"), str(b1["effective_baseline_cny"]))
    check("explicit prior spent 25.00", b1["prior_spent_cny"] == Decimal("25.00"), str(b1["prior_spent_cny"]))

    # 6. accounting identity: effective - last_balance == prior_spent
    check("identity effective-balance==spent (ctrl23)",
          b23["effective_baseline_cny"] - b23["last_balance_cny"] == b23["prior_spent_cny"],
          "%s-%s=%s" % (b23["effective_baseline_cny"], b23["last_balance_cny"], b23["prior_spent_cny"]))

    npass = sum(1 for _, ok, _ in results if ok)
    print("\n%d/%d checks passed" % (npass, len(results)))
    out = {"module": "funding_carryforward", "n_checks": len(results), "n_passed": npass,
           "all_pass": npass == len(results),
           "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in results]}
    outp = Path(__file__).resolve().parent / "test_funding_carryforward.results.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
