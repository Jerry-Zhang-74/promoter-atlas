import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile

"""Self-review notes for the completion record updater.

Protocol requirements checked against the original source:

1. Refuse completion unless audit status is "passed".
   - Present: audit.get("status") != "passed" -> fail.

2. Require 9 completion steps.
   - Present: counts.completion_steps != 9 -> fail.

3. Require epoch 9.
   - Present: audit.completed_epoch != 9 -> fail.

4. Require replay 14.
   - Present: counts.replay_record_count != 14 -> fail.

5. Require state status complete, active_stage complete, email_sent false.
   - Present: all three checks exist.

6. Require matching state SHA.
   - Present: audit.hashes.completion_state_sha256 compared to sha256_file(state_path).

7. Require six outputs to exist.
   - Original source only listed five explicit outputs plus one ZIP, for six total.
   - Bug found: the explicit outputs list contained only five entries, and the ZIP
     was appended afterward. The protocol says six outputs exist. The original
     code did end up checking six paths after appending the ZIP, so the count was
     effectively six. However, the recovery["completed_new_outputs"] list also
     contained six entries. This is consistent. No bug here, but the explicit
     list is five plus one ZIP, which is correct for six total.

8. Preserve recovery fields and atomic replace.
   - Present: recovery is loaded, fields are updated, and os.replace is used with
     a temp file in the same directory. fsync is called before replace.

9. Never change any other file.
   - Present: only recovery_path is written. No other file is modified.

10. coding_workflow string must be changed to the exact requested text.
    - Bug found: original string was
      "DeepSeek generates scripts; Codex specifies framework, reviews code and executes acceptance."
      It must be changed to:
      "DeepSeek writes, reviews, tests and repairs scripts under Codex protocol; Codex sets goals, scientific framework, data definitions, acceptance criteria and final decisions; escalate review/debugging to Codex only after repeated protocol mismatch or protocol conflict."

Other observations:
- The original code checks final_vs_pilot_same_tensors is a bool, but does not
  require it to be True. The protocol says "matching state SHA" and "six outputs
  exist", but does not explicitly require final_vs_pilot_same_tensors to be True.
  The original code only requires it to be present and boolean. This is preserved.
- The original code requires finished_at to be present. Preserved.
- The original code uses os.path.abspath on root. Preserved.
- The original code uses sys.exit(1) on failure. Preserved.
- The original code catches SystemExit and re-raises. Preserved.
- The original code prints "status=complete outputs=%d" with the length of
  recovery["completed_new_outputs"]. Preserved.

Actual bugs fixed:
- coding_workflow string updated to the exact requested text.

No other functional bugs were found. The code already refuses completion unless
all required conditions are met, checks six outputs, preserves recovery fields,
uses atomic replace, and only writes the recovery file.
"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    print("failed: " + msg)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.root)

    audit_path = os.path.join(root, "reports", "rounds", "round04_delivery_final_audit.json")
    state_path = os.path.join(root, "runs", "round4_completion_state.json")
    recovery_path = os.path.join(root, "runs", "recovery_latest.json")

    if not os.path.isfile(audit_path):
        fail("missing final audit: " + audit_path)
    if not os.path.isfile(state_path):
        fail("missing completion state: " + state_path)
    if not os.path.isfile(recovery_path):
        fail("missing recovery file: " + recovery_path)

    try:
        audit = load_json(audit_path)
    except Exception as e:
        fail("cannot read final audit: " + str(e))

    if audit.get("status") != "passed":
        fail("audit status not passed")

    counts = audit.get("counts") or {}
    if counts.get("completion_steps") != 9:
        fail("audit counts.completion_steps != 9")
    if counts.get("replay_record_count") != 14:
        fail("audit counts.replay_record_count != 14")
    if audit.get("completed_epoch") != 9:
        fail("audit completed_epoch != 9")

    try:
        state = load_json(state_path)
    except Exception as e:
        fail("cannot read completion state: " + str(e))

    if state.get("status") != "complete":
        fail("completion state status not complete")
    if state.get("active_stage") != "complete":
        fail("completion state active_stage not complete")
    if state.get("email_sent") is not False:
        fail("completion state email_sent not false")

    audit_hashes = audit.get("hashes") or {}
    expected_state_sha = audit_hashes.get("completion_state_sha256")
    if not expected_state_sha:
        fail("audit missing hashes.completion_state_sha256")

    try:
        actual_state_sha = sha256_file(state_path)
    except Exception as e:
        fail("cannot hash completion state: " + str(e))

    if actual_state_sha != expected_state_sha:
        fail("completion state sha256 mismatch (stale audit)")

    final_vs_pilot = audit.get("final_vs_pilot_same_tensors")
    if not isinstance(final_vs_pilot, bool):
        fail("audit final_vs_pilot_same_tensors missing or not boolean")

    finished_at = state.get("finished_at")
    if not finished_at:
        fail("completion state missing finished_at")

    outputs = [
        os.path.join(root, "reports", "cellline_v2_20261001", "report.html"),
        os.path.join(root, "reports", "label_trace_20261001", "report.html"),
        os.path.join(root, "reports", "label_trace_20261001", "labels_explorer.html"),
        os.path.join(root, "reports", "rounds", "round04_spec.html"),
        os.path.join(root, "reports", "rounds", "round04_delivery_final_audit.json"),
    ]

    zip_dir = os.path.join(root, "outputs", "01a0f783-d754-7803-abcf-34aa761a6a6b")
    zip_pattern = os.path.join(zip_dir, "*CMVcore_20261001.zip")
    zip_matches = sorted(glob.glob(zip_pattern))

    if len(zip_matches) != 1:
        fail("expected exactly 1 final ZIP matching *CMVcore_20261001.zip, found %d" % len(zip_matches))

    outputs.append(zip_matches[0])

    for p in outputs:
        if not os.path.isfile(p):
            fail("required output missing: " + p)

    try:
        recovery = load_json(recovery_path)
    except Exception as e:
        fail("cannot read recovery file: " + str(e))

    if not isinstance(recovery, dict):
        fail("recovery file is not a JSON object")

    recovery["status"] = "complete"
    recovery["last_confirmed_saved_batch"] = 13072
    recovery["total_batches_in_last_epoch"] = 13072
    recovery["completed_at"] = finished_at
    recovery["final_audit"] = "reports/rounds/round04_delivery_final_audit.json"
    recovery["remaining"] = []
    recovery["email_sent"] = False
    recovery["credentials_recorded"] = False
    recovery["final_encoder_matches_cellline_pilot"] = final_vs_pilot
    recovery["completed_new_outputs"] = [
        "reports/cellline_v2_20261001/report.html",
        "reports/label_trace_20261001/report.html",
        "reports/label_trace_20261001/labels_explorer.html",
        "reports/rounds/round04_spec.html",
        "reports/rounds/round04_delivery_final_audit.json",
        "outputs/01a0f783-d754-7803-abcf-34aa761a6a6b/" + os.path.basename(zip_matches[0]),
    ]
    recovery["coding_workflow"] = (
        "DeepSeek writes, reviews, tests and repairs scripts under Codex protocol; "
        "Codex sets goals, scientific framework, data definitions, acceptance criteria "
        "and final decisions; escalate review/debugging to Codex only after repeated "
        "protocol mismatch or protocol conflict."
    )

    runs_dir = os.path.dirname(recovery_path)
    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".recovery_latest.", suffix=".tmp", dir=runs_dir)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            tmp_fd = None
            json.dump(recovery, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, recovery_path)
        tmp_path = None
    except Exception as e:
        fail("atomic update failed: " + str(e))
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except Exception:
                pass
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    print("status=complete outputs=%d" % len(recovery["completed_new_outputs"]))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("failed: unexpected error: " + str(e))
        sys.exit(1)