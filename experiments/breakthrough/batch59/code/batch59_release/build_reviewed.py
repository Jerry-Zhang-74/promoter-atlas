# -*- coding: utf-8 -*-
"""batch59 worker: build FINAL reviewed release revisions from the staged bundles.

Concrete packaging fix (new revision only; original staged dirs PRESERVED):
 the staged MODEL_MANIFEST.json kept the SOURCE absolute checkpoint path
 ("E:\promoter_atlas\data\...\<arm>_selected.ckpt"), making the bundle depend on an
 out-of-bundle private file. The reviewed revision rewrites checkpoint.path to the
 BUNDLE-RELATIVE "model/<arm>_selected.ckpt". Weight bytes/SHA are UNCHANGED.
Also drops __pycache__, adds Chinese USER_GUIDE_zh.md + MODEL_CARD_zh.md and compact real
evidence copied from the frozen evaluator outputs, then re-writes SHA256SUMS.
Read-only w.r.t. weights. ZERO fit/solve/GPU/network.
"""
from __future__ import annotations
import hashlib, json, os, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch59")
B54 = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch54")
DOCS = os.path.join(ROOT, "code", "batch59_release", "docs")
SRC = {
    "Nabs": (os.path.join(ROOT, "releases", "batch52_first_model", "v6.38.0-nabs-first"),
             os.path.join(ROOT, "releases", "batch52_first_model", "v6.38.0-nabs-reviewed"),
             "v6.38.0-nabs-reviewed", "Nabs_selected.ckpt"),
    "Njoint": (os.path.join(ROOT, "releases", "batch52_second_model", "v6.38.0-njoint-second"),
               os.path.join(ROOT, "releases", "batch52_second_model", "v6.38.0-njoint-reviewed"),
               "v6.38.0-njoint-reviewed", "Njoint_selected.ckpt"),
}
WEIGHT_SHA = {
    "Nabs": "17a6f622c3d684743d75263af958cb71dad6f13f99e57f6fc69d5336e74c130f",
    "Njoint": "c2698adde33155e28eace574e05dd42d4eef737002e0c68084053ea9dccb0cee",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def copy_tree(src, dst):
    for dirpath, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel = os.path.relpath(dirpath, src)
        tgt = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(tgt, exist_ok=True)
        for fn in files:
            if fn.endswith(".pyc"):
                continue
            shutil.copy2(os.path.join(dirpath, fn), os.path.join(tgt, fn))


def write_sums(root):
    lines = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
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


def evidence():
    J = lambda n: json.load(open(os.path.join(B54, n), encoding="utf-8"))
    return {
        "kind": "batch59_release_evidence",
        "note": "Real frozen held-out TEST metrics copied verbatim from the independent evaluator "
                "outputs. Public HEK293 (GSE301246 author target Log2RPM_HEK293), retrospective frozen "
                "group split; NOT prospective; NOT local 293FT.",
        "split": "TEST",
        "arms": {"R0": J("R0_TEST_parent_metrics.json"),
                 "Nabs": J("Nabs_TEST_parent_metrics.json"),
                 "Njoint": J("Njoint_TEST_parent_metrics.json")},
        "njoint_minus_nabs_pair_groupMAE_bootstrap": J("Njoint_vs_Nabs_TEST_parent_bootstrap.json"),
        "sources": {
            "R0": "reports/breakthrough_20261002/batch54/R0_TEST_parent_metrics.json",
            "Nabs": "reports/breakthrough_20261002/batch54/Nabs_TEST_parent_metrics.json",
            "Njoint": "reports/breakthrough_20261002/batch54/Njoint_TEST_parent_metrics.json",
            "bootstrap": "reports/breakthrough_20261002/batch54/Njoint_vs_Nabs_TEST_parent_bootstrap.json",
            "independent_review": "reports/breakthrough_20261002/batch54/independent_review.json",
        },
    }


def build_arm(arm):
    src_dir, dst_dir, version, ckpt_name = SRC[arm]
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    copy_tree(src_dir, dst_dir)
    man_path = os.path.join(dst_dir, "MODEL_MANIFEST.json")
    man = json.load(open(man_path, encoding="utf-8"))
    old_path = man.get("checkpoint", {}).get("path")
    man["checkpoint"]["path"] = "model/" + ckpt_name
    man["version"] = version
    man["staging"] = {"bundle_dir": None, "status": "REVIEWED_COMPLETE", "blockers": [],
                      "basis": "batch59 reviewed revision of the batch55 staged bundle",
                      "packaging_fix": {"field": "checkpoint.path", "from": old_path,
                                        "to": man["checkpoint"]["path"],
                                        "reason": "remove absolute out-of-bundle dependency; weight bytes unchanged"}}
    json.dump(man, open(man_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
    st_path = os.path.join(dst_dir, "STAGING_STATUS.json")
    st = json.load(open(st_path, encoding="utf-8"))
    st.update({"status": "REVIEWED_COMPLETE", "version": version, "reviewed_by": "batch59 worker",
               "packaging_fix": man["staging"]["packaging_fix"]})
    json.dump(st, open(st_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
    ug = open(os.path.join(DOCS, "USER_GUIDE_zh.md"), encoding="utf-8").read()
    mc = open(os.path.join(DOCS, "MODEL_CARD_zh.md"), encoding="utf-8").read()
    mc = mc.replace("{ARM}", arm).replace("{VERSION}", version).replace("{SHA}", WEIGHT_SHA[arm])
    open(os.path.join(dst_dir, "USER_GUIDE_zh.md"), "w", encoding="utf-8", newline="\n").write(ug)
    open(os.path.join(dst_dir, "MODEL_CARD_zh.md"), "w", encoding="utf-8", newline="\n").write(mc)
    os.makedirs(os.path.join(dst_dir, "results"), exist_ok=True)
    json.dump(evidence(), open(os.path.join(dst_dir, "results", "batch59_release_evidence.json"),
                               "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
    shutil.copy2(os.path.join(B54, "independent_review.json"),
                 os.path.join(dst_dir, "results", "batch54_independent_review.json"))
    n = write_sums(dst_dir)
    got = sha256_file(os.path.join(dst_dir, "model", ckpt_name))
    return {"arm": arm, "dir": os.path.relpath(dst_dir, ROOT).replace(os.sep, "/"), "version": version,
            "files_hashed": n, "weight_sha256": got, "weight_sha_match": got == WEIGHT_SHA[arm],
            "checkpoint_path_old": old_path, "checkpoint_path_new": man["checkpoint"]["path"]}


def main():
    out = {"kind": "batch59_reviewed_release_build",
           "arms": {a: build_arm(a) for a in ("Nabs", "Njoint")}}
    out["all_weight_sha_match"] = all(a["weight_sha_match"] for a in out["arms"].values())
    os.makedirs(REP, exist_ok=True)
    json.dump(out, open(os.path.join(REP, "reviewed_release_build.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
