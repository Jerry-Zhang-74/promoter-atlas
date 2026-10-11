#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch17 ops (deliverable 5a): minimal carry-forward funding fix for future controllers.

Problem (diagnosed read-only):
  ``budget_controller.py`` fresh-start path ``_init_status`` hardcodes
  ``cumulative_topup_cny = "0"`` and computes ``spent`` from the CLI-supplied
  ``baseline_cny``. So a NEW controller launch loses the cumulative prior
  top-ups and the last known balance unless the operator manually re-supplies an
  inflated effective baseline each time. This is why control_24 currently needs
  the temporary EFFECTIVE-242.67 workaround instead of a genuine carry-forward.

Fix (this module, imported by the proposed patch): derive the funding basis from
the previous manifest / funding_basis.json so a fresh controller continues the
same accounting:
    effective_baseline = original_account_baseline + cumulative_prior_topups
    spent              = effective_baseline - current_balance
originating original baseline and carried top-ups remain separately reported.

No top-ups, no account changes, no live-controller mutation here.
"""
import json
from decimal import Decimal
from pathlib import Path

DEFAULT_ORIGINAL_BASELINE_CNY = Decimal("142.67")


def _dec(v):
    if v is None or v == "":
        return None
    try:
        return Decimal(str(v))
    except Exception:
        return None


def _funding_basis_from_path(manifest_path):
    if manifest_path is None:
        return None
    p = Path(manifest_path)
    cand = (p.parent if p.name == "status.json" else p) / "funding_basis.json"
    if cand.is_file():
        try:
            return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def load_funding_basis(manifest=None, manifest_path=None, funding_basis=None):
    """Return the carry-forward funding basis as Decimals.

    Precedence: explicit funding_basis.json -> explicit manifest fields ->
    reconstruction from recorded (spent_cny, balance_cny, cumulative_topup_cny).
    """
    manifest = manifest or {}
    fb = funding_basis
    if fb is None:
        fb = _funding_basis_from_path(manifest_path)
    fb = fb or {}

    carried = _dec(fb.get("carried_observed_topup_cny"))
    if carried is None:
        carried = _dec(manifest.get("cumulative_topup_cny"))
    if carried is None:
        carried = Decimal("0")

    original = _dec(fb.get("original_account_baseline_cny"))
    if original is None:
        original = _dec(manifest.get("baseline_original_cny"))
    if original is None:
        spent_rec = _dec(manifest.get("spent_cny"))
        bal_rec = _dec(manifest.get("balance_cny"))
        if spent_rec is not None and bal_rec is not None:
            # spent = effective - balance  =>  effective = spent + balance
            original = spent_rec + bal_rec - carried
        else:
            base_rec = _dec(manifest.get("baseline_cny"))
            original = (base_rec - carried) if base_rec is not None else DEFAULT_ORIGINAL_BASELINE_CNY

    effective = original + carried

    last_balance = _dec(fb.get("last_prior_balance_cny"))
    if last_balance is None:
        last_balance = _dec(manifest.get("balance_cny"))

    prior_spent = _dec(fb.get("prior_estimated_spent_cny"))
    if prior_spent is None:
        prior_spent = _dec(manifest.get("spent_cny"))
    if prior_spent is None and last_balance is not None:
        prior_spent = effective - last_balance

    return {
        "original_baseline_cny": original,
        "carried_topup_cny": carried,
        "effective_baseline_cny": effective,
        "last_balance_cny": last_balance,
        "prior_spent_cny": prior_spent,
    }


def fresh_status_funding(basis):
    """Fields a corrected fresh controller should write into status.json."""
    return {
        "baseline_cny": str(basis["effective_baseline_cny"]),
        "baseline_original_cny": str(basis["original_baseline_cny"]),
        "carried_topup_cny": str(basis["carried_topup_cny"]),
        "cumulative_topup_cny": str(basis["carried_topup_cny"]),
    }
