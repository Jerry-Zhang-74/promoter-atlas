#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""batch55 submission STAGER: package the new first model into a self-contained, verified,
redistributable bundle - with HARD refusals instead of any false "completed" claim.

What it stages (whitelist ONLY):
    MODEL_MANIFEST.json   enriched provenance (source/target/checkpoint SHAs, honest epoch)
    model/<ckpt>          real trained weights + correct .sha256.json sidecar
    code/predict.py       standalone predictor (path-independent)
    code/pinned_core/*    pinned model-core modules + PINNED_CORE_SHA256.json
    config/config.json    frozen config (spec.config_dict())
    normalizer/normalizer.json
    results/*             held-out results JSON WHEN AVAILABLE (never placeholder)
    requirements.txt  LICENSE_NOTES.md  README.md  USER_GUIDE.md
    SHA256SUMS.txt        every staged file (excluding itself)
    STAGING_STATUS.json   STAGED_COMPLETE or STAGER_PARTIAL + actionable blockers

It NEVER copies and actively scans against: API/email keys, credentials, .env, private
tokens, huge genomes/raw data, old failed wrong-target R0 artifacts, or proprietary external
weights. It refuses (nonzero exit, STAGING_STATUS status=REFUSED) when:
    * weights are missing / checkpoint hash mismatch        -> "missing_weights"
    * target/label mismatch (Log2Norm or anything != Log2RPM)-> "label_target_mismatch"
    * untrained checkpoint (init bytes == fresh model)       -> "untrained_checkpoint"
    * source SHA mismatch                                    -> "source_sha_mismatch"
    * placeholder/absent-when-required results               -> "placeholder_results"
    * a SOFTWAREONLY fixture is presented as a real model    -> "fixture_as_real"
A refusal or missing (optional) held-out results yields an ACTIONABLE PARTIAL status, never
a completed claim.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PINNED = os.path.join(_HERE, "pinned_core")
if _PINNED not in sys.path:
    sys.path.insert(0, _PINNED)

import spec  # noqa: E402
from model_core import build_model, param_sha256  # noqa: E402
from checkpoint import load_checkpoint  # noqa: E402
from io_utils import sha256_file  # noqa: E402

TARGET = spec.TARGET_COLUMN                       # Log2RPM_HEK293
EXPECTED_SOURCE_SHA = spec.SOURCE_SHA256

LICENSE_FILES = ["requirements.txt", "LICENSE_NOTES.md", "README.md", "USER_GUIDE.md"]
CORE_MODULES = ["spec.py", "model_core.py", "normalizer.py", "checkpoint.py", "io_utils.py"]

FORBIDDEN_NAME_RE = re.compile(
    r"(api[_-]?key|email[_-]?key|credential|secret|passwd|password|token|\.env$|"
    r"id_rsa|\.pem$|\.key$|private[_-]?key)", re.IGNORECASE)
FORBIDDEN_CONTENT_RE = [
    re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"(?i)api[_-]?key\s*[:=]\s*\S+"),
    re.compile(rb"(?i)password\s*[:=]\s*\S+"),
]
PLACEHOLDER_TOKENS = ["PLACEHOLDER", "TODO", "TBD", "FIXME", "NOT_A_REAL", "<fill",
                      "REPLACE_ME", "EXAMPLE_ONLY"]


class StageRefusal(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__("%s: %s" % (reason, detail))
        self.reason = reason
        self.detail = detail


# ------------------------------------------------------------------- checks ---
def load_bundle_manifest(bundle_dir: str) -> dict:
    p = os.path.join(bundle_dir, "MODEL_MANIFEST.json")
    if not os.path.exists(p):
        raise StageRefusal("missing_weights", "no MODEL_MANIFEST.json in %s" % bundle_dir)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ckpt(bundle_dir: str, man: dict) -> str:
    ck = (man.get("checkpoint") or {}).get("path")
    if not ck:
        raise StageRefusal("missing_weights", "manifest has no checkpoint.path")
    p = ck if os.path.isabs(ck) else os.path.join(bundle_dir, ck)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        raise StageRefusal("missing_weights", "checkpoint file absent/empty: %s" % p)
    want = (man.get("checkpoint") or {}).get("sha256")
    if want:
        got = sha256_file(p)
        if got != want:
            raise StageRefusal("checkpoint_sha_mismatch", "%s != %s" % (got, want))
    return p


def check_target(man: dict, obj: dict) -> None:
    mt = man.get("target_column")
    ct = obj.get("config", {}).get("target_column")
    if mt != TARGET or ct != TARGET:
        raise StageRefusal("label_target_mismatch",
                           "manifest target=%r checkpoint target=%r required=%r"
                           % (mt, ct, TARGET))


def check_source(man: dict, cohort_dir: str | None) -> None:
    ms = (man.get("source_sha256") or "").lower()
    if ms != EXPECTED_SOURCE_SHA:
        raise StageRefusal("source_sha_mismatch",
                           "manifest source_sha256=%r != accepted %r"
                           % (man.get("source_sha256"), EXPECTED_SOURCE_SHA))
    if cohort_dir:
        mpath = os.path.join(cohort_dir, "manifest.json")
        if os.path.exists(mpath):
            with open(mpath, "r", encoding="utf-8") as f:
                cm = json.load(f)
            cs = (cm.get("source_sha256") or "").lower()
            if cs != EXPECTED_SOURCE_SHA:
                raise StageRefusal("source_sha_mismatch",
                                   "cohort manifest source_sha256=%r != %r" % (cs, EXPECTED_SOURCE_SHA))


def check_trained(obj: dict, fixture: bool) -> dict:
    """Return honest epoch status. Refuse an untrained checkpoint unless fixture mode."""
    fresh = build_model()
    fresh_sha = param_sha256(fresh)
    state = obj["model"]
    tmp = build_model()
    tmp.load_state_dict(state)
    got_sha = param_sha256(tmp)
    epoch = int(obj.get("epoch", 0))
    gstep = int(obj.get("global_step", 0))
    extra = obj.get("extra", {}) or {}
    trained = (got_sha != fresh_sha)
    status = {"epoch": epoch, "global_step": gstep,
              "param_sha256": got_sha, "fresh_init_sha256": fresh_sha,
              "looks_trained": trained,
              "train_loss": extra.get("train_loss"),
              "dev": extra.get("dev"), "dev_pair_mae": extra.get("dev_pair_mae"),
              "epoch_cap": extra.get("epoch_cap")}
    if not fixture:
        if not trained:
            raise StageRefusal("untrained_checkpoint",
                               "weights equal fresh init (%s): no training detected" % fresh_sha)
        if epoch <= 0 and gstep <= 0 and extra.get("train_loss") is None:
            raise StageRefusal("untrained_checkpoint",
                               "epoch=%d global_step=%d and no train_loss recorded" % (epoch, gstep))
    return status


def _find_activity_rhos(obj, path="") -> list:
    """Recursively collect (path, rho) for any dict {'activity': {'rho': <number>}}.
    Accepts both flat evaluate_split output and nested final_compare output
    ({'Nabs':{'activity':...}, 'Njoint':{'activity':...}})."""
    found = []
    if isinstance(obj, dict):
        act = obj.get("activity")
        if isinstance(act, dict) and isinstance(act.get("rho"), (int, float)):
            found.append((path or "$", float(act["rho"])))
        for k, v in obj.items():
            found.extend(_find_activity_rhos(v, "%s.%s" % (path, k) if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_find_activity_rhos(v, "%s[%d]" % (path, i)))
    return found


def check_results(results_path: str | None, ckpt_sha: str, version: str) -> dict:
    """Validate held-out results. Absent -> PARTIAL blocker (returned, not raised).
    Present but placeholder/malformed -> StageRefusal('placeholder_results')."""
    if not results_path:
        return {"present": False, "reason": "held-out results not yet available"}
    if not os.path.exists(results_path):
        raise StageRefusal("placeholder_results", "results path does not exist: %s" % results_path)
    raw = open(results_path, "rb").read()
    for tok in PLACEHOLDER_TOKENS:
        if tok.encode() in raw:
            raise StageRefusal("placeholder_results", "results contain placeholder token %r" % tok)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise StageRefusal("placeholder_results", "results not valid JSON: %s" % e)
    rhos = _find_activity_rhos(obj)
    if not rhos:
        raise StageRefusal("placeholder_results",
                           "results contain no numeric activity.rho (flat or Nabs/Njoint nested)")
    if isinstance(obj.get("checkpoint_sha256"), str) and obj["checkpoint_sha256"] != ckpt_sha:
        raise StageRefusal("placeholder_results", "results checkpoint_sha256 != staged checkpoint")
    primary = rhos[0]
    return {"present": True, "path": results_path,
            "activity_rho": primary[1], "activity_rho_at": primary[0],
            "all_activity_rhos": {p: r for p, r in rhos},
            "split": obj.get("split"), "scope": obj.get("scope"),
            "verdict": obj.get("verdict")}


# --------------------------------------------------------------- staging io ---
def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def secret_scan(root: str) -> list:
    """Return list of violations (name or content) under root."""
    bad = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if FORBIDDEN_NAME_RE.search(fn):
                bad.append({"file": rel, "why": "forbidden filename"})
                continue
            try:
                data = open(full, "rb").read()
            except OSError:
                continue
            for rx in FORBIDDEN_CONTENT_RE:
                if rx.search(data):
                    bad.append({"file": rel, "why": "secret-like content: %s" % rx.pattern.decode("latin1")})
                    break
    return bad


def write_sums(root: str) -> int:
    lines = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel == "SHA256SUMS.txt":
                continue
            lines.append("%s  %s" % (sha256_file(full), rel))
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    with open(os.path.join(root, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def _status(status_dir: str, payload: dict) -> None:
    os.makedirs(status_dir, exist_ok=True)
    with open(os.path.join(status_dir, "STAGING_STATUS.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


# -------------------------------------------------------------------- stage ---
def stage(bundle_dir: str, out_dir: str, version: str, cohort_dir=None, results=None,
          fixture=False, status_dir=None) -> dict:
    bundle_dir = os.path.abspath(bundle_dir)
    out_dir = os.path.abspath(out_dir)
    status_dir = os.path.abspath(status_dir) if status_dir else out_dir + ".status"
    man = load_bundle_manifest(bundle_dir)
    kind = man.get("kind")

    if fixture:
        if kind != "SOFTWAREONLY_FIXTURE":
            raise StageRefusal("fixture_as_real", "fixture mode requires kind=SOFTWAREONLY_FIXTURE, got %r" % kind)
        if "release" in out_dir.lower():
            raise StageRefusal("fixture_as_real", "SOFTWAREONLY fixture must never be written under a release path")
    else:
        if kind == "SOFTWAREONLY_FIXTURE":
            raise StageRefusal("fixture_as_real", "SOFTWAREONLY fixture cannot be staged as a real submission")

    ckpt = resolve_ckpt(bundle_dir, man)
    obj = load_checkpoint(ckpt, expect_config_sha=man.get("config_sha256"), expect_datahash=None)
    check_target(man, obj)
    check_source(man, cohort_dir)
    epoch_status = check_trained(obj, fixture=fixture)

    blockers = []
    res = check_results(results, man.get("checkpoint", {}).get("sha256"), version)
    if not res.get("present"):
        blockers.append({"code": "heldout_results_unavailable",
                         "detail": res.get("reason"),
                         "action": "run the independent evaluator (batch54/eval_replay) on TEST, then re-stage with --results"})
    if epoch_status["epoch"] <= 0:
        blockers.append({"code": "epoch_status_incomplete",
                         "detail": "checkpoint epoch=%d (honest: may be a partial/early stop)"
                                   % epoch_status["epoch"],
                         "action": "report honest epoch status; do not claim a final-quality model"})

    if os.path.exists(out_dir):
        raise StageRefusal("output_exists", "refusing to overwrite existing staging dir: %s" % out_dir)

    staging = out_dir + ".tmp.%d" % os.getpid()
    if os.path.exists(staging):
        shutil.rmtree(staging)
    try:
        # model
        _copy(ckpt, os.path.join(staging, "model", os.path.basename(ckpt)))
        side = ckpt + ".sha256.json"
        if os.path.exists(side):
            _copy(side, os.path.join(staging, "model", os.path.basename(side)))
        # code (pinned)
        _copy(os.path.join(_HERE, "predict.py"), os.path.join(staging, "code", "predict.py"))
        for m in CORE_MODULES:
            _copy(os.path.join(_PINNED, m), os.path.join(staging, "code", "pinned_core", m))
        _copy(os.path.join(_PINNED, "PINNED_CORE_SHA256.json"),
              os.path.join(staging, "code", "pinned_core", "PINNED_CORE_SHA256.json"))
        # config + normalizer
        cfg = {"format": "batch55.config.v1", "config": spec.config_dict(),
               "config_sha256": spec.config_sha256()}
        os.makedirs(os.path.join(staging, "config"), exist_ok=True)
        with open(os.path.join(staging, "config", "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.makedirs(os.path.join(staging, "normalizer"), exist_ok=True)
        with open(os.path.join(staging, "normalizer", "normalizer.json"), "w", encoding="utf-8") as f:
            json.dump({"format": "batch55.normalizer.v1",
                       "normalizer": obj.get("normalizer", {})}, f, ensure_ascii=False, indent=2, sort_keys=True)
        # license / docs
        for fn in LICENSE_FILES:
            src = os.path.join(_HERE, fn)
            if os.path.exists(src):
                _copy(src, os.path.join(staging, fn))
        # results
        if res.get("present"):
            _copy(results, os.path.join(staging, "results", os.path.basename(results)))

        staged_manifest = dict(man)
        staged_manifest.update({
            "format": "batch55.model_bundle.v1",
            "version": version,
            "kind": kind,
            "staged_at_epoch_status": epoch_status,
            "staging": {"bundle_dir": bundle_dir, "status": "STAGED_COMPLETE" if not blockers else "STAGER_PARTIAL",
                        "blockers": blockers},
        })
        with open(os.path.join(staging, "MODEL_MANIFEST.json"), "w", encoding="utf-8") as f:
            json.dump(staged_manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

        # final SHA256SUMS then secret scan then STAGING_STATUS
        write_sums(staging)
        violations = secret_scan(staging)
        if violations:
            raise StageRefusal("secret_detected", "forbidden assets in staged tree: %s" % violations)

        status = {
            "status": "STAGED_COMPLETE" if not blockers else "STAGER_PARTIAL",
            "version": version, "kind": kind, "out_dir": out_dir,
            "checkpoint_sha256": man.get("checkpoint", {}).get("sha256"),
            "target_column": TARGET, "source_sha256": EXPECTED_SOURCE_SHA,
            "epoch_status": epoch_status, "results": res,
            "blockers": blockers,
            "completed_claim": False if blockers else True,
        }
        with open(os.path.join(staging, "STAGING_STATUS.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2, sort_keys=True)
        write_sums(staging)

        os.replace(staging, out_dir)
        _status(status_dir, status)
        return status
    except StageRefusal:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="batch55 submission stager (hard refusals)")
    ap.add_argument("--bundle", required=True, help="model bundle dir (MODEL_MANIFEST.json + ckpt)")
    ap.add_argument("--out", required=True, help="staging/release output dir (must not exist)")
    ap.add_argument("--version", required=True)
    ap.add_argument("--cohort-dir", default=None)
    ap.add_argument("--results", default=None, help="held-out results JSON (optional -> PARTIAL)")
    ap.add_argument("--fixture", action="store_true", help="stage a SOFTWAREONLY fixture (never a release)")
    ap.add_argument("--status-dir", default=None)
    a = ap.parse_args(argv)
    try:
        st = stage(a.bundle, a.out, a.version, cohort_dir=a.cohort_dir, results=a.results,
                   fixture=a.fixture, status_dir=a.status_dir)
    except StageRefusal as e:
        payload = {"status": "REFUSED", "reason": e.reason, "detail": e.detail,
                   "completed_claim": False,
                   "next_action": "resolve the reported condition before (re)staging"}
        d = a.status_dir or (os.path.abspath(a.out) + ".status")
        _status(d, payload)
        sys.stderr.write("REFUSED [%s] %s\n" % (e.reason, e.detail))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 3
    print(json.dumps(st, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
