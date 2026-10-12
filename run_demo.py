#!/usr/bin/env python3
"""Run the three inference tasks on the bundled examples.

Creates a fresh output directory, invokes ``predict.py`` via subprocess for
``esb``, ``activity`` and ``profile``, records the real execution log, then
summarises the actual CSV outputs and their run metadata.  Cross-file
consistency (task, row count, run_id/version, checkpoint SHA) is verified
against each prediction's ``.run.json`` sidecar and the current weights file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True

from _bootstrap import project_root, resolve_path

_ROOT = project_root()
_HERE = os.path.dirname(os.path.abspath(__file__))
_PREDICT = os.path.join(_HERE, "predict.py")

TASKS = (
    ("esb", os.path.join("examples", "esb_demo.tsv"), "esb_predictions.csv"),
    ("activity", os.path.join("examples", "fragment_demo.tsv"), "fragment_activity.csv"),
    ("profile", os.path.join("examples", "esb_demo.tsv"), "tissue_predictions.csv"),
)

_METADATA_COLUMNS = frozenset(
    ["candidate_id", "species", "model_version", "run_id", "length_bp",
     "predicted_Log2RPM_HEK293"]
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sanitize(text: str, replacements: list[tuple[str, str]]) -> str:
    """Replace exact machine paths (roots, CWD, weights, output) in log text."""
    out = text
    for needle, repl in replacements:
        if needle:
            out = out.replace(needle, repl)
    return out


def _path_replacements(paths: list[str]) -> list[tuple[str, str]]:
    reps = []
    for p in paths:
        if not p:
            continue
        reps.append((p, os.path.basename(p.rstrip("\\/")) or "<path>"))
    # Longest first so nested paths are replaced before their parents.
    reps.sort(key=lambda kv: len(kv[0]), reverse=True)
    return reps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the bundled demo end to end.")
    parser.add_argument("--outdir", default="demo_results")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    cwd = os.getcwd()
    outdir = resolve_path(args.outdir, cwd)
    if os.path.exists(outdir):
        if not os.path.isdir(outdir):
            print("error: --outdir exists and is not a directory", file=sys.stderr)
            return 2
        if os.listdir(outdir):
            print("error: --outdir must be absent or empty", file=sys.stderr)
            return 2
    else:
        os.makedirs(outdir)

    # Explicit --weights resolve against the caller CWD; only an omitted
    # --weights falls back to the project-root default.
    if args.weights:
        weights = resolve_path(args.weights, cwd)
    else:
        weights = resolve_path(os.path.join("models", "v6.6.0.pt"), _ROOT)

    log_lines = []
    results = {}
    failures = 0
    started = time.time()

    for task, rel_input, out_name in TASKS:
        input_path = os.path.join(_ROOT, rel_input)
        output_path = os.path.join(outdir, out_name)
        cmd = [
            sys.executable, _PREDICT,
            "--task", task,
            "--input", input_path,
            "--output", output_path,
            "--weights", weights,
            "--device", args.device,
        ]
        log_lines.append("$ predict.py --task %s --input %s --output %s" % (task, rel_input, out_name))
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            rc = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except Exception as exc:  # noqa: BLE001
            rc = -1
            stdout = ""
            stderr = "subprocess failed: %s" % exc
        reps = _path_replacements([_ROOT, cwd, weights, outdir, input_path, output_path])
        if stdout:
            log_lines.append(_sanitize(stdout.rstrip(), reps))
        if stderr:
            log_lines.append(_sanitize(stderr.rstrip(), reps))
        log_lines.append("exit_code=%d" % rc)
        if rc != 0:
            failures += 1
            results[task] = {"ok": False, "output": out_name, "exit_code": rc}
        else:
            results[task] = {"ok": True, "output": out_name, "exit_code": rc}

    log_path = os.path.join(outdir, "demo_run.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")

    summary = {
        "completion": False,
        "tasks": {},
        "elapsed_seconds": round(time.time() - started, 3),
    }

    # Recompute the current weights SHA before and after the consistency pass.
    weights_sha_before = None
    weights_sha_after = None
    if os.path.isfile(weights):
        try:
            weights_sha_before = _sha256_file(weights)
        except OSError as exc:
            summary["weights_error"] = "cannot hash weights: %s" % exc

    versions, run_ids, checkpoint_shas = set(), set(), set()
    consistency_errors = []

    for task, rel_input, out_name in TASKS:
        info = results[task]
        entry = {"ok": info["ok"], "output": out_name, "exit_code": info["exit_code"]}
        if info["ok"]:
            csv_path = os.path.join(outdir, out_name)
            run_json_path = csv_path + ".run.json"
            try:
                header, rows = _read_csv(csv_path)
                entry["rows"] = len(rows)
                if not rows:
                    consistency_errors.append("%s: CSV has no data rows" % task)
                else:
                    row_versions = set()
                    row_run_ids = set()
                    for r in rows:
                        row_versions.add(r.get("model_version"))
                        row_run_ids.add(r.get("run_id"))
                    if len(row_versions) != 1:
                        consistency_errors.append("%s: CSV rows disagree on model_version" % task)
                    if len(row_run_ids) != 1:
                        consistency_errors.append("%s: CSV rows disagree on run_id" % task)
                    versions |= row_versions
                    run_ids |= row_run_ids
                if task == "profile":
                    tissue_cols = [c for c in header if c not in _METADATA_COLUMNS]
                    entry["tissue_count"] = len(tissue_cols)
                    entry["tissue_names"] = tissue_cols
            except Exception as exc:  # noqa: BLE001
                entry["ok"] = False
                entry["error"] = "cannot read CSV: %s" % exc
                consistency_errors.append("%s: cannot read CSV: %s" % (task, exc))
                failures += 1

            if not os.path.isfile(run_json_path):
                entry["ok"] = False
                entry["error"] = "missing run metadata sidecar"
                consistency_errors.append("%s: missing run metadata sidecar" % task)
                failures += 1
            else:
                try:
                    meta = _read_json(run_json_path)
                except Exception as exc:  # noqa: BLE001
                    meta = None
                    entry["ok"] = False
                    entry["error"] = "cannot read run metadata: %s" % exc
                    consistency_errors.append("%s: cannot read run metadata: %s" % (task, exc))
                    failures += 1
                if meta is not None:
                    entry["units"] = meta.get("units")
                    entry["elapsed_seconds"] = meta.get("elapsed_seconds")
                    entry["batch_size"] = meta.get("batch_size")
                    if meta.get("task") != task:
                        consistency_errors.append("%s: sidecar task mismatch (%r)" % (task, meta.get("task")))
                    if entry.get("rows") is not None and meta.get("rows") != entry["rows"]:
                        consistency_errors.append(
                            "%s: sidecar rows %r != CSV rows %r" % (task, meta.get("rows"), entry["rows"])
                        )
                    if meta.get("version") is not None:
                        versions.add(meta.get("version"))
                    if meta.get("run_id") is not None:
                        run_ids.add(meta.get("run_id"))
                    sha = meta.get("checkpoint_sha256")
                    if sha:
                        checkpoint_shas.add(sha)
                    else:
                        consistency_errors.append("%s: sidecar missing checkpoint_sha256" % task)
                    if task == "activity":
                        entry["fragment_padding"] = meta.get("fragment_padding")
        summary["tasks"][task] = entry

    if os.path.isfile(weights):
        try:
            weights_sha_after = _sha256_file(weights)
        except OSError as exc:
            summary["weights_error"] = "cannot hash weights: %s" % exc

    if weights_sha_before is not None:
        summary["checkpoint_sha256"] = weights_sha_before
        summary["checkpoint_basename"] = os.path.basename(weights)

    if weights_sha_before is None or weights_sha_after is None:
        consistency_errors.append("weights file missing or unhashable")
    elif weights_sha_before != weights_sha_after:
        consistency_errors.append("weights file changed during the run")

    if len(versions) == 1:
        summary["model_version"] = next(iter(versions))
    if len(run_ids) == 1:
        summary["run_id"] = next(iter(run_ids))

    if len(versions) != 1:
        consistency_errors.append("inconsistent model_version across outputs")
    if len(run_ids) != 1:
        consistency_errors.append("inconsistent run_id across outputs")
    if len(checkpoint_shas) != 1:
        consistency_errors.append("inconsistent checkpoint_sha256 across sidecars")
    elif weights_sha_before is not None and next(iter(checkpoint_shas)) != weights_sha_before:
        consistency_errors.append("sidecar checkpoint_sha256 does not match current weights")

    summary["consistent_checkpoint"] = not consistency_errors
    summary["consistency_errors"] = consistency_errors
    summary["completion"] = (failures == 0) and not consistency_errors

    # Optional E/S Pareto filtering on the actual ESB CSV, gated on B >= 0.5.
    # Preserves all actual numeric values; no fabricated rows or counts.
    esb_path = os.path.join(outdir, "esb_predictions.csv")
    selected_path = os.path.join(outdir, "selected_candidates.csv")
    selected_rows = []
    if results["esb"]["ok"] and os.path.isfile(esb_path):
        try:
            _, esb_rows = _read_csv(esb_path)
            candidates = []
            for row in esb_rows:
                try:
                    e = float(row["Score_E"])
                    s = float(row["Score_S"])
                    b = float(row["Score_B"])
                except (KeyError, TypeError, ValueError):
                    continue
                if b >= 0.5:
                    candidates.append((row["candidate_id"], row.get("species", ""), e, s, b))
            for cid, species, e, s, b in candidates:
                dominated = any(
                    (o_e >= e and o_s >= s and (o_e > e or o_s > s))
                    for o_cid, _o_sp, o_e, o_s, _o_b in candidates
                    if o_cid != cid
                )
                if not dominated:
                    selected_rows.append([cid, species, repr(e), repr(s), repr(b)])
        except Exception as exc:  # noqa: BLE001
            summary["pareto_error"] = "cannot compute Pareto selection: %s" % exc
    with open(selected_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["candidate_id", "species", "Score_E", "Score_S", "Score_B"])
        writer.writerows(selected_rows)
    summary["selected_candidates"] = len(selected_rows)

    with open(os.path.join(outdir, "demo_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print("demo outdir: %s" % os.path.basename(outdir))
    print("completion: %s" % summary["completion"])
    return 0 if summary["completion"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
